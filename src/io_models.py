from enum import Enum
from typing import NamedTuple, Optional

import torch
from pydantic import BaseModel
from returns.result import Failure, Result, Success
from torch import Tensor

class IncorrectInput(Exception):
    
    __slots__ = ("message", "input")
    
    message: str
    input: "UserInput"
    
    def __init__(self, input: "UserInput", message: Optional[str] = None):
        self.message = message or "The problem is unsolvable."
        self.input = input
        
        super().__init__(self.message)

    def __str__(self):
        return self.message

    def __repr__(self) -> str:
        return super().__repr__() + f" {self.message}"


class Unsolvable(IncorrectInput):
    def __init__(self, input: "UserInput", message: Optional[str] = None):
        super().__init__(input, f"UserInput is not solvable{f": {message}" if message else "!"}")


class Constraint(BaseModel):
    resource: str
    cost: float


class Task(BaseModel):
    name: str
    constraints: list[Constraint]


class CostConstraintMatrix(NamedTuple):
    cost_matrix: Tensor
    # shape: (batch_size, problem_size, 3)
    constraint_matrix: Tensor
    # shape: (batch_size, num_tasks, num_resources)

    # TODO additional validator to validate the batch sizes of the cost matrix and constraint matrix are equal

    @property
    def batch_size(self) -> int:
        return self.cost_matrix.shape[0]


class UserInput(BaseModel):
    resources: list[str]
    tasks: list[Task]

    def to_cost_constraint_matrix(self) -> CostConstraintMatrix:
        num_resources = len(self.resources)
        num_tasks = len(self.tasks)
        problem_size = num_resources * num_tasks
        batch_size = 1

        cost_matrix = torch.zeros(batch_size, problem_size, 3).cuda()
        for i, resource in enumerate(self.resources):
            for j, task in enumerate(self.tasks):
                cost = next(
                    (c.cost for c in task.constraints if c.resource == resource), 1.0
                )
                cost_matrix[0, i * num_tasks + j, 0] = cost
                cost_matrix[0, i * num_tasks + j, 1] = i
                cost_matrix[0, i * num_tasks + j, 2] = j

        constraint_matrix = torch.zeros(batch_size, num_tasks, num_resources).cuda()

        for i, task in enumerate(self.tasks):
            for constraint in task.constraints:
                resource_idx = self.resources.index(constraint.resource)
                constraint_matrix[0, i, resource_idx] = 1

        return CostConstraintMatrix(cost_matrix, constraint_matrix)

    def solvable(self) -> None | Unsolvable:
        for task in self.tasks:
            if not task.constraints:
                return Unsolvable(self, f"Task {task.name} has no constraints")

            for constraint in task.constraints:
                if constraint.cost < 0:
                    return Unsolvable(self, f"Task {task.name} has negative cost")

                if constraint.resource not in self.resources:
                    return Unsolvable(self, 
                        f"Task {task.name} has unknown resource {constraint.resource}"
                    )

    def correct(self) -> None | IncorrectInput:

        match self.solvable():
            case Unsolvable() as e:
                return e

        if not self.tasks:
            return IncorrectInput(self, "No tasks provided")


class SolvedResource(BaseModel):
    resource: str
    tasks: list[str]
    cost: float


class UserOutputResult(BaseModel):
    max_cost: float
    allocations: list[SolvedResource]

    @classmethod
    def from_result_matrix(
        cls,
        result_matrix: Tensor,  # shape: (num_tasks, num_resources, 2)
        max_cost: float,
        user_input: Optional[UserInput] = None,
    ) -> "UserOutputResult":
        allocations = []
        tasks = (
            [task.name for task in user_input.tasks]
            if user_input
            else [f"Task_{i}" for i in range(result_matrix.shape[0])]
        )
        resources = (
            user_input.resources
            if user_input
            else [f"Resource_{i}" for i in range(result_matrix.shape[1])]
        )

        for i, task in enumerate(tasks):
            for j, resource in enumerate(resources):
                selected = bool(result_matrix[i, j, 0].bool().item())
                cost = round(float(result_matrix[i, j, 1].float().item()), 6)
                if selected:
                    allocations.append(
                        SolvedResource(resource=resource, tasks=[task], cost=cost)
                    )

        return cls(max_cost=round(max_cost, 6), allocations=allocations)


class UserOutputType(str, Enum):
    Success = "SUCCESS"
    Failure = "FAILURE"


class UserOutput(BaseModel):
    type: UserOutputType
    msg: str = ""
    result: Optional[UserOutputResult] = None

    @classmethod
    def from_result_matrix(
        cls,
        result_matrix: Tensor,  # shape: (num_tasks, num_resources, 2)
        max_cost: float,
        user_input: Optional[UserInput] = None,
    ) -> "UserOutput":
        return cls(
            type=UserOutputType.Success,
            result=UserOutputResult.from_result_matrix(
                result_matrix, max_cost, user_input=user_input
            ),
        )

    @classmethod
    def failure(cls, msg: str) -> "UserOutput":
        return cls(type=UserOutputType.Failure, msg=msg)


class UserOutputs(BaseModel):
    outputs: list[UserOutputResult]

    @classmethod
    def from_result_matrices(
        cls,
        result_matrices: list[Tensor],  # shape: (x, num_tasks, num_resources, 2)
        max_costs: list[float],
        user_inputs: Optional[list[UserInput]] = None,
    ) -> "UserOutputs":
        outputs = []
        num_outputs = len(result_matrices)

        for result_matrix, max_cost, user_input in zip(
            result_matrices,
            max_costs,
            user_inputs if user_inputs else [None for _ in range(num_outputs)],
            strict=True,
        ):
            outputs.append(
                UserOutput.from_result_matrix(
                    result_matrix, max_cost, user_input=user_input
                )
            )
        return cls(outputs=outputs)
