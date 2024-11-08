import itertools
from dataclasses import dataclass
from enum import unique
from typing import Literal, Optional, Tuple

import numpy as np
import torch

from exceptions import FunctionOrderError
from io_models import CostConstraintMatrix
from problem import get_random_problems


@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem_size, 3)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, problem_size)

    current_node: Optional[torch.Tensor] = None
    # shape: (batch, problem_size)

    already_selected_nodes: Optional[list[torch.Tensor]] = None

    ninf_mask: Optional[torch.Tensor] = None
    # shape: (batch, problem_size, job)


class Env:

    __slots__ = (
        "cost_constraint_matrices",
        "env_params",
        "num_tasks",
        "num_resources",
        "problem_size",
        "load_data_from_file",
        "BATCH_IDX",
        "POMO_IDX",
        "problems",
        "constraint_matrices",
        "generated_problems",
        "selected_count",
        "current_node",
        "already_selected_nodes",
        "selected_job_list",
        "current_data_index",
        "step_state",
    )

    def __init__(
        self,
        env_params,
        load_data_from_file: bool,
        cost_constraint_matrices: list[CostConstraintMatrix] | None = None,
    ):
        self.cost_constraint_matrices = cost_constraint_matrices

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.num_tasks = env_params["num_tasks"]
        self.num_resources = int(env_params["num_resources"])
        self.problem_size = self.num_resources * self.num_tasks
        self.load_data_from_file = load_data_from_file
        # self.data_file_path = env_params['data_file_path']
        # Const @Load_Problem
        ####################################
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, problem_size)
        self.problems = None
        # shape: (batch, problem_size, job)
        self.constraint_matrices = None
        # shape: (batch, num_tasks, num_resources)
        self.generated_problems: list[list] = []
        # list for generated problems / only used for testing
        # Dynamic
        ####################################
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, problem_size)
        self.already_selected_nodes: list[torch.Tensor] = []
        self.selected_job_list: Optional[torch.Tensor] = None
        # shape: (batch, problem_size, 0~problem_size)

        self.current_data_index = 0

    def get_matrix(self, index: int):
        if self.cost_constraint_matrices is None:
            raise ValueError("Cost constraint matrices not loaded")
        cost_constraint_matrix = self.cost_constraint_matrices[index]
        cost_matrix = cost_constraint_matrix.cost_matrix
        constraint_matrix = cost_constraint_matrix.constraint_matrix

        input_matrix = torch.zeros(cost_matrix.shape[0] * cost_matrix.shape[1], 3)
        idx = 0

        for j in range(cost_matrix.shape[1]):
            for i in range(cost_matrix.shape[0]):
                input_matrix[idx, 0] = cost_matrix[i, j, 0]
                input_matrix[idx, 1] = j + 1  # resources
                input_matrix[idx, 2] = i + 1  # tasks
                idx += 1

        return cost_matrix, constraint_matrix

    def load_problems(self, batch_size):

        if self.load_data_from_file:
            idx = int(self.current_data_index / batch_size)

            self.problems, self.constraint_matrices = self.get_matrix(idx)
            self.current_data_index += batch_size

            # mixed_num_resources
            self.num_resources = self.constraint_matrices.shape[2]

            # mixed_problem_sizes
            self.problem_size = self.problems.shape[1]
            self.num_tasks = self.problem_size // self.num_resources
            # mixed_problem_sizes
        else:
            self.problems, self.constraint_matrices = get_random_problems(
                batch_size, self.problem_size, self.num_resources
            )

        self.BATCH_IDX = torch.arange(batch_size)[:, None].expand(
            batch_size, self.problem_size
        )
        self.POMO_IDX = torch.arange(self.problem_size)[None, :].expand(
            batch_size, self.problem_size
        )

    def reset(self, batch_size: int):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, problem_size)
        self.already_selected_nodes = []
        self.selected_job_list = torch.zeros(
            (batch_size, self.problem_size, 0), dtype=torch.long
        )
        # shape: (batch, problem_size, 0~problem_size)

        # CREATE STEP STATE
        if self.BATCH_IDX is None or self.POMO_IDX is None:
            raise FunctionOrderError(self.reset, self.load_problems)
        self.step_state = Step_State(
            BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX, already_selected_nodes=[]
        )
        self.step_state.ninf_mask = torch.zeros(
            (batch_size, self.problem_size, self.problem_size)
        )
        # shape: (batch, problem_size, problem_size)

        if self.constraint_matrices is None:
            raise FunctionOrderError(self.reset, self.load_problems)
        """update ninf_mask according to constraints here"""

        flatten_inverted_constraints = 1 - self.constraint_matrices.mT.flatten(1)
        # shape: (batch, problem_size)

        ninf_mask_single_rollout = flatten_inverted_constraints.where(
            flatten_inverted_constraints == 0, torch.tensor(float("-inf"))
        )
        # shape: (batch, problem_size)

        self.step_state.ninf_mask = ninf_mask_single_rollout.unsqueeze(1).expand(
            batch_size, self.problem_size, self.problem_size
        )
        # shape: (batch, problem_size, problem_size)

        """update ninf_mask according to constraints here"""

        if self.BATCH_IDX is None:
            raise FunctionOrderError(self.reset, self.load_problems)
        reward = torch.empty(self.BATCH_IDX.shape[0])
        done = False
        if self.problems is None:
            raise FunctionOrderError(self.reset, self.load_problems)
        return Reset_State(self.problems), reward, done

    def pre_step(self) -> Tuple[Step_State, torch.Tensor, Literal[False]]:
        if self.BATCH_IDX is None:
            raise FunctionOrderError(self.pre_step, self.load_problems)
        reward = torch.empty(self.BATCH_IDX.shape[0])
        done = False
        return self.step_state, reward, done

    def step(
        self, batch_size: int, selected: torch.Tensor
    ) -> Tuple[Step_State, torch.Tensor, bool]:
        # selected.shape: (batch, problem_size)

        self.selected_count += 1
        self.current_node = selected
        self.already_selected_nodes += selected
        # shape: (batch, problem_size)

        if self.selected_job_list is None:
            raise FunctionOrderError(self.step, self.reset)
        self.selected_job_list = torch.cat(
            (self.selected_job_list, selected[:, :, None]), dim=2
        )
        # shape: (batch, problem_size, 0~problem_size)

        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        self.step_state.already_selected_nodes = self.already_selected_nodes
        # shape: (batch, problem_size)
        if (
            self.step_state.ninf_mask is None
            or self.BATCH_IDX is None
            or self.POMO_IDX is None
        ):
            raise FunctionOrderError(self.step, self.reset)

        # update ninf_mask
        ninf_mask = self.step_state.ninf_mask.clone()
        selected = selected.unsqueeze(2)
        additions = torch.arange(1, self.num_resources) * self.num_tasks
        selected = torch.cat(
            (selected, (selected + additions) % self.problem_size), dim=2
        )
        ninf_mask[self.BATCH_IDX.unsqueeze(2), self.POMO_IDX.unsqueeze(2), selected] = (
            float("-inf")
        )
        self.step_state.ninf_mask = ninf_mask

        # returning values
        done = self.selected_count == self.num_tasks

        reward = -self._get_total_cost(batch_size) if done else torch.empty(batch_size)

        return self.step_state, reward, done

    def _get_total_cost(self, batch_size):
        if self.selected_job_list is None:
            raise FunctionOrderError(self._get_total_cost, self.step)
        gathering_index = self.selected_job_list.unsqueeze(3).expand(
            batch_size, self.problem_size, self.num_tasks, 3
        )
        # shape: (batch, problem_size, problem_size, 3)
        if self.problems is None:
            raise FunctionOrderError(self._get_total_cost, self.step)
        problems_expanded = self.problems[:, None, :, :].expand(
            batch_size, self.problem_size, self.problem_size, 3
        )
        # shape: (batch, problem_size, problem_size, 3)
        jobs = problems_expanded.gather(dim=2, index=gathering_index)
        # shape: (batch, problem_size, job, 3)

        job_cost = jobs[:, :, :, 0]
        job_resource = jobs[:, :, :, 1]
        # shape: (batch, problem_size, job)

        unique_resource, inverse_indices = torch.unique(
            job_resource, return_inverse=True
        )

        summed_costs_per_resource = torch.zeros(
            batch_size, self.problem_size, len(unique_resource)
        ).scatter_add_(2, inverse_indices, job_cost)

        return summed_costs_per_resource.max(2).values
