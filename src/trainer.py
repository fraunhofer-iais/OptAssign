import contextlib
import os
import pickle
import random
import time
from logging import getLogger

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from environment import Env
from model import Model
from utils.logging import *
from utils.metric_progress_tracker import MetricProgressTracker


class Trainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name="trainer")
        process_start_time = datetime.now(pytz.timezone("Europe/Berlin"))

        num_resources = self.env_params["num_resources"]
        if self.trainer_params["mixed_resource_num"]:
            num_resources = "mixed"

        if trainer_params["debug_mode"]:
            self.result_folder = f"models/debug_train_{num_resources}_resources"
        else:
            timestamp = process_start_time.strftime("%Y%m%d_%H%M%S")
            self.result_folder = (
                f"models/{timestamp}_train_{num_resources}_resources"
            )

        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        self.result_log = LogData()

        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        if torch.backends.cuda.is_built():
            cuda_device_num = self.trainer_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            device = torch.device("cuda", cuda_device_num)
            torch.set_default_dtype(torch.float32)
            torch.set_default_device(device)
        
        torch.set_default_dtype(torch.float32)

        # Main Components
        self.model = Model(num_resources, **self.model_params)
        self.env = Env(
            env_params=self.env_params,
            load_data_from_file=self.env_params["load_data_from_file"],
        )
        self.optimizer = Optimizer(
            self.model.parameters(), **self.optimizer_params["optimizer"]
        )
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params["scheduler"])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params["model_load"]
        if model_load["enable"]:
            checkpoint_fullname = "{path}/checkpoint-{epoch}.pt".format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.start_epoch = 1 + model_load["epoch"]
            self.result_log.set_raw_data(checkpoint["result_log"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.last_epoch = model_load["epoch"] - 1
            self.logger.info("Saved Model Loaded !!")

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        model_configuration = {
            "env_params": self.env_params,
            "model_params": self.model_params,
            "optimizer_params": self.optimizer_params,
            "trainer_params": self.trainer_params,
        }
        with open(f"{self.result_folder}/model_configuration.json", "w") as file:
            json.dump(model_configuration, file, indent=4)
            file.close()

        start = time.time()
        self.time_estimator.reset(self.start_epoch)

        loss_progress_tracker = MetricProgressTracker(
            f"{self.result_folder}/loss_progress.json"
        )
        norm_of_gradient_progress_tracker = MetricProgressTracker(
            f"{self.result_folder}/norm_of_gradient_progress.json"
        )

        for epoch in range(self.start_epoch, self.trainer_params["epochs"] + 1):
            self.logger.info(
                "================================================================="
            )

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append("train_score", epoch, train_score)
            self.result_log.append("train_loss", epoch, train_loss)

            loss_progress_tracker.track({"epoch": epoch, "loss": train_loss})
            norm_of_gradient_progress_tracker.track(
                {"epoch": epoch, "norm": self.model.get_norm_of_gradients()}
            )

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(
                epoch, self.trainer_params["epochs"]
            )
            self.logger.info(
                "Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                    epoch,
                    self.trainer_params["epochs"],
                    elapsed_time_str,
                    remain_time_str,
                )
            )

            all_done = epoch == self.trainer_params["epochs"]
            model_save_interval = self.trainer_params["logging"]["model_save_interval"]

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "result_log": self.result_log.get_raw_data(),
                }
                torch.save(
                    checkpoint_dict, f"{self.result_folder}/checkpoint-{epoch}.pt"
                )

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
                end = time.time()
                run_time = end - start
                if not os.path.isfile(path=f"{self.result_folder}/internal"):
                    os.makedirs(f"{self.result_folder}/internal")
                with open(f"{self.result_folder}/internal/run_time.pkl", "wb") as file:
                    pickle.dump(run_time, file)
                    file.close()

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode: int = self.trainer_params["train_episodes"]
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params["train_batch_size"], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        "Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}".format(
                            epoch,
                            episode,
                            train_num_episode,
                            100.0 * episode / train_num_episode,
                            score_AM.avg,
                            loss_AM.avg,
                        )
                    )
                    self.logger.info("Now logging only every epoch.")

        # Log Once, for each epoch
        self.logger.info(
            "Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}".format(
                epoch, 100.0 * episode / train_num_episode, score_AM.avg, loss_AM.avg
            )
        )

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        if self.trainer_params["mixed_task_num"]:
            max_task_num = self.env_params["num_tasks"]
            self.env.num_tasks = random.randint(10, max_task_num)

        if self.trainer_params["mixed_resource_num"]:
            max_resource_num = self.env_params["num_resources"]
            self.env.num_resources = random.randint(3, max_resource_num)

        self.env.problem_size = self.env.num_resources * self.env.num_tasks

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset(batch_size)
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.problem_size, 0))
        # shape: (batch, problem_size, 0~problem_size)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, problem_size)
            state, reward, done = self.env.step(batch_size, selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage: torch.Tensor = reward - reward.float().mean(dim=1, keepdim=True)
        # shape: (batch, problem_size)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, problem_size)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, problem_size)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = (
            -max_pomo_reward.float().mean()
        )  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()
