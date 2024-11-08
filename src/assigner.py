import pickle
import random
from logging import getLogger
from pathlib import Path

import numpy as np
import torch

from environment import Env
from io_models import CostConstraintMatrix
from model import Model
from utils.logging import *


class Assigner:

    __slots__ = [
        "env_params",
        "model_params",
        "assign_params",
        "logger",
        "device",
        "model",
        "time_estimator",
        "env",
        "load_data_from_file",
    ]

    def __init__(
        self, env_params, model_params, assign_params, load_data_from_file: bool
    ):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.assign_params = assign_params
        self.load_data_from_file = load_data_from_file

        max_num_resources = self.env_params["num_resources"]

        # result folder, logger
        self.logger = getLogger(name="assigner")

        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        if torch.backends.cuda.is_built():
            cuda_device_num = self.assign_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            device = torch.device("cuda", cuda_device_num)
            torch.set_default_dtype(torch.float32)
            torch.set_default_device(device)

        torch.set_default_dtype(torch.float32)

        self.device = device

        self.model = Model(max_num_resources, **self.model_params)

        # Restore
        model_load = assign_params["model_load"]
        checkpoint_fullname = Path(model_load["path"]) / Path(
            f"checkpoint-{model_load['epoch']}.pt"
        )
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, cost_constraint_matrices: list[CostConstraintMatrix] | None):

        # ENV
        self.env = Env(
            env_params=self.env_params,
            load_data_from_file=self.load_data_from_file,
            cost_constraint_matrices=cost_constraint_matrices,
        )

        self.time_estimator.reset()

        score_AM = AverageMeter()

        episode = 0
        score_values: Optional[torch.Tensor] = None

        batch_size = (
            cost_constraint_matrices[0].batch_size
            if cost_constraint_matrices
            else self.assign_params["batch_size"]
        )
        num_batched_problems = (
            len(cost_constraint_matrices)
            if cost_constraint_matrices
            else self.assign_params["num_batched_problems"]
        )
        num_episodes = num_batched_problems * batch_size
        mappings: list[torch.Tensor] = []
        max_costs: list[float] = []
        # shape: (0, num_tasks, num_resources)

        elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
            episode, num_episodes
        )

        while episode < num_episodes:

            remaining = num_episodes - episode
            batch_size = min(batch_size, remaining)

            score, score_values_batch, batch_mappings, batch_max_costs = (
                self._assign_one_batch(batch_size)
            )
            # (_, shape: (batch))

            if score_values is None:
                score_values = score_values_batch
                # shape: (batch)
            else:
                score_values = torch.cat([score_values, score_values_batch], dim=0)
                # shape: (batch)

            score_AM.update(score, batch_size)

            episode += batch_size

            mappings += batch_mappings
            max_costs += batch_max_costs

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                episode, num_episodes
            )
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}".format(
                    episode, num_episodes, elapsed_time_str, remain_time_str, score
                )
            )

        if score_values is None:
            raise ValueError("'num_episodes' should be greater than 0")

        self.logger.info(" *** Test Done *** ")
        self.logger.info(" SCORE: {:.4f} ".format(score_AM.avg))

        result = {
            "avg_score": score_AM.avg,
            "elapsed_time": elapsed_time_str,
        }

        # saving generated problems
        if not self.load_data_from_file:
            for i in range(len(self.env.generated_problems[0])):
                self.env.generated_problems[0][i] = (
                    self.env.generated_problems[0][i].cpu().detach().numpy()
                )
                self.env.generated_problems[1][i] = (
                    self.env.generated_problems[1][i].cpu().detach().numpy()
                )
            with open("./internal/POMO_generated_problems.pkl", "wb") as file:
                pickle.dump(self.env.generated_problems, file)
                print(
                    "dumped generated problems to file ./internal/POMO_generated_problems.pkl"
                )

        return result, mappings, max_costs

    def _assign_one_batch(self, batch_size):
        # this case is only handled if we want to generate mixed size problems with constraint matrices
        if (
            self.assign_params["mixed_problem_size_testing"]
            and not self.load_data_from_file
        ):
            num_tasks_max = self.env_params["num_tasks"]
            random_num_tasks = random.randint(10, num_tasks_max)
            random_problem_size = random_num_tasks * self.env_params["num_resources"]
            self.env.problem_size = random_problem_size
            self.env.num_tasks = random_num_tasks

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size)
            reset_state, _, _ = self.env.reset(batch_size)
            assert self.env.step_state.ninf_mask is not None, "ninf_mask is None"
            rollouts2ignore: list[torch.Tensor] = [
                torch.where(torch.isinf(subtensor.diagonal()))[0]
                for subtensor in self.env.step_state.ninf_mask
            ]
            # list[shape: (0~problem_size)]
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, problem_size)
            state, reward, done = self.env.step(batch_size, selected)
            # StepState, # shape: (batch, problem_size), bool
        # Return
        ###############################################

        # Save the pomo scores to a file for analysis
        self.save_pomo_scores_to_file(reward)

        filtered_reward = [
            reward[i][
                torch.ones(reward.size(1), dtype=torch.bool).scatter_(
                    0, rollouts2ignore[i], False
                )
            ]
            for i in range(batch_size)
        ]

        sorted_reward_idx = [
            filtered_reward[i].argsort(descending=True) for i in range(batch_size)
        ]

        best_rollout_idx = [sorted_reward_idx[i][0] for i in range(batch_size)]

        max_pomo_reward = [
            filtered_reward[i][best_rollout_idx[i]] for i in range(batch_size)
        ]

        max_pomo_reward = torch.Tensor(max_pomo_reward).reshape(batch_size)
        # shape: (batch)

        # negative sign to make positive value
        score = -max_pomo_reward.float().mean()

        score_values = -max_pomo_reward.float()
        # shape: (batch)

        assert self.env.selected_job_list is not None, "selected_job_list is None"
        best_rollout = [
            self.env.selected_job_list[i, best_rollout_idx[i]]
            for i in range(batch_size)
        ]
        # shape: (batch, problem_size)

        mappings = [
            torch.zeros(self.env.num_tasks, self.env.num_resources)
            .flatten()
            .index_fill_(0, best_rollout[i], 1)
            .reshape(self.env.num_resources, self.env.num_tasks)
            .T
            for i in range(batch_size)
        ]
        # shape: (batch, num_tasks, num_resources)

        # convert mappings to shape (batch, num_tasks, num_resources, 1)
        mappings = [mapping.unsqueeze(-1) for mapping in mappings]

        costs = (
            reset_state.problems[:, :, 0]
            .reshape(batch_size, self.env.num_resources, self.env.num_tasks, 1)
            .transpose(1, 2)
        )

        mappings = [
            torch.cat([mappings[i], costs[i]], dim=-1) for i in range(batch_size)
        ]

        return (
            score.item(),
            score_values,
            mappings,
            (
                -max_pomo_reward.float()
            ).tolist(),  # TODO refactor max_costs passing to output-model
        )

    def save_pomo_scores_to_file(self, reward):
        reward = np.array(reward.cpu())
        reward = reward.squeeze()
        reward = reward[:100]
        # np.savetxt('internal/pomo_scores.txt', reward)
