import argparse
import json
import logging
import pickle
import random
import shutil
from os import makedirs
from pathlib import Path
import tempfile
from typing import Any, TypeGuard, overload

import torch
from pydantic import BaseModel, ValidationError
from returns.result import Failure, Result, Success

from assigner import Assigner
from exceptions import *
from io_models import CostConstraintMatrix, IncorrectInput, UserInput, UserOutput
from utils.logging import LogFile, create_logger, process_start_time
from utils.types import CostConstraintMatricesPath, UserInputPath
from utils.utils import random_string

logger_params = {"log_file": LogFile(desc="assign")}


type InputPath = UserInputPath | CostConstraintMatricesPath
type PossiblePickleOutputType = list[CostConstraintMatrix] | list[list[torch.Tensor]]


class ArgNamespace(argparse.Namespace):
    user_input_json_path: Optional[UserInputPath]
    cost_constraint_matrices_path: Optional[CostConstraintMatricesPath]


class Args(BaseModel):
    user_input_json_path: Optional[UserInputPath] = None
    cost_constraint_matrices_path: Optional[CostConstraintMatricesPath] = None
    store: bool = True


def run(
    cost_constraint_matrices: list[CostConstraintMatrix] | None,
    assign_params: dict[str, Any],
    model_load_path: Optional[str] = None,
):

    if model_load_path is not None:
        assign_params["model_load"]["path"] = model_load_path

    model_config_path = (
        Path(assign_params["model_load"]["path"]) / "model_configuration.json"
    )

    with open(model_config_path, "r") as file:
        json_config = json.load(file)
        env_params = json_config["env_params"]
        model_params = json_config["model_params"]

    assert (
        assign_params["load_data_from_file"] != cost_constraint_matrices is not None
    ), "if load_data_from_file is True, cost_constraint_matrices should be provided"

    if env_params["num_tasks"] <= 1:  # problem_size = num_tasks * num_resources
        raise ProblemSize("Problem size should be greater that number of resources")

    if assign_params["debug_mode"]:
        _set_debug_mode(assign_params)

    create_logger(**logger_params)
    _print_config(assign_params)

    assigner = Assigner(
        env_params=env_params,
        model_params=model_params,
        assign_params=assign_params,
        load_data_from_file=assign_params["load_data_from_file"],
    )

    # copy_all_src(tester.result_folder)

    result, mappings, max_costs = assigner.run(cost_constraint_matrices)

    return result, mappings, max_costs


def _set_debug_mode(assign_params: dict[str, Any]):
    assign_params["test_episodes"] = 100


def _print_config(assign_params: dict[str, Any]):
    logger = logging.getLogger("root")
    logger.info(f"DEBUG_MODE: {assign_params['debug_mode']}")
    logger.info(
        f"USE_CUDA: {assign_params['use_cuda']}, CUDA_DEVICE_NUM: {assign_params['cuda_device_num']}"
    )
    [
        logger.info(f"{g_key}{globals()[g_key]}")
        for g_key in globals()
        if g_key.endswith("params")
    ]


##########################################################################################


def is_tensor_list_list(obj: Any) -> TypeGuard[list[list[torch.Tensor]]]:
    if not isinstance(obj, list):
        return False

    if not all(
        isinstance(item, list) for item in obj
    ):  # pyright: ignore [reportUnknownVariableType]
        return False

    return all(
        all(isinstance(tensor, torch.Tensor) for tensor in sublist)
        for sublist in obj  # pyright: ignore [reportUnknownVariableType]
    )


def is_cost_constraint_matrix_list(obj: Any) -> TypeGuard[list[CostConstraintMatrix]]:
    return all(isinstance(item, CostConstraintMatrix) for item in obj)


def is_possible_pickle_output_type(obj: Any) -> TypeGuard[PossiblePickleOutputType]:
    return is_tensor_list_list(obj) or is_cost_constraint_matrix_list(obj)


@overload
def load_input(
    input_path: UserInputPath,
) -> Result[
    UserInput, TypeError | FileExistsError | OSError | ValueError | ValidationError
]: ...


@overload
def load_input(
    input_path: CostConstraintMatricesPath,
) -> Result[
    PossiblePickleOutputType,
    TypeError
    | FileExistsError
    | OSError
    | ValueError
    | pickle.UnpicklingError
    | ValidationError,
]: ...


def load_input(
    input_path: UserInputPath | CostConstraintMatricesPath,
) -> Result[
    UserInput | PossiblePickleOutputType,
    TypeError | FileExistsError | OSError | ValueError | pickle.UnpicklingError,
]:
    try:
        if isinstance(input_path, UserInputPath):

            with open(input_path, "r") as file:
                try:
                    return Success(UserInput.model_validate_json(file.read()))
                except (ValueError, ValidationError) as e:
                    return Failure(e)

        else:  # isinstance(input_path, CostConstraintMatricesPath)
            with open(input_path, "rb") as file:
                try:
                    if is_possible_pickle_output_type(
                        pickle_output := pickle.load(file)
                    ):
                        return Success(pickle_output)
                    else:
                        return Failure(
                            ValueError(
                                f"Invalid pickle output type: {type(pickle_output)} but expected {PossiblePickleOutputType}"
                            )
                        )
                except pickle.UnpicklingError as e:
                    return Failure(e)
    except (FileExistsError, OSError) as e:
        return Failure(e)


@overload
def load_cost_constraint_matrices(
    input: PossiblePickleOutputType,
) -> Result[list[CostConstraintMatrix], TypeError]: ...


@overload
def load_cost_constraint_matrices(
    input: UserInput,
) -> Result[list[CostConstraintMatrix], IncorrectInput]: ...


def load_cost_constraint_matrices(
    input: PossiblePickleOutputType | UserInput,
) -> Result[list[CostConstraintMatrix], TypeError | IncorrectInput]:
    if is_tensor_list_list(input):
        # (num_batched_problems, 2, ...)
        # [0] cost matrix:            (..., ..., batch_size, problem_size, 3)
        # [1] constraint matrix:      (..., ..., batch_size, num_tasks, num_resources)

        return Success(
            [
                CostConstraintMatrix(
                    input[i][0].cuda(),
                    input[i][1].cuda(),
                )
                for i in range(len(input))
            ]
        )
    elif is_cost_constraint_matrix_list(input):
        return Success(input)

    elif isinstance(input, UserInput):
        match input.correct():
            case IncorrectInput() as e:
                return Failure(e)
            case None:
                pass

        return Success([input.to_cost_constraint_matrix()])

    raise TypeError(
        f"Invalid input type: {type(input)} but expected {PossiblePickleOutputType} or UserInput"
    )


def main(
    args: Args,
):
    match (args.user_input_json_path, args.cost_constraint_matrices_path):
        case (None, None):
            # use default input path
            path = UserInputPath("assign_input/input.json")
        case (UserInputPath() as path, None) | (
            None,
            CostConstraintMatricesPath() as path,
        ):
            pass
        case _:
            raise ValueError(
                "Only one of user_input_json_path or cost_constraint_matrices_path should be provided!"
            )

    config, assign_params = load_assigner_config()

    folder_path = (
        Path(assign_params["result_output_path"])
        / process_start_time.strftime("%Y%m%d_%H%M%S")
        if args.store
        else Path(tempfile.gettempdir()) / random_string(10)
    )

    folder_path.mkdir(parents=True, exist_ok=True)

    match load_input(path):
        case Success(raw_input):
            pass
        case Failure(exception):
            raise exception
        case _:
            raise ValueError("Invalid input type")

    match load_cost_constraint_matrices(raw_input):
        case Success(cost_constraint_matrices):
            result, mappings, max_costs = run(cost_constraint_matrices, assign_params)
            user_outputs, user_input = (
                (
                    UserOutput.from_result_matrix(mappings[0], max_costs[0], raw_input),
                    raw_input,
                )
                if isinstance(raw_input, UserInput)
                else (None, None)
            )

        case Failure(exception):
            result = {}
            match exception:
                case IncorrectInput() as e:
                    user_outputs, user_input = (UserOutput.failure(str(e)), e.input)
                case _ as e:
                    raise e
        case _:
            raise ValueError("Invalid cost constraint matrices type")

    if user_outputs is not None:
        save_user_output(folder_path, user_outputs)
    if user_input is not None:
        save_user_input(user_input, folder_path)

    save_meta_result(folder_path, result)

    backup_config(config, folder_path)

    # copy whole output folder to last_assign folder
    if args.store:
        shutil.rmtree(Path.cwd() / "last_assign")
        makedirs(Path.cwd() / "last_assign")
        shutil.copytree(folder_path, Path.cwd() / "last_assign")

    return user_outputs


def backup_config(config: Any, folder_path: Path):
    with open(
        folder_path / "config.json",
        "x",
    ) as f:
        json.dump(config, f, indent=4)


def save_meta_result(folder_path: Path, result: dict[str, Any]):
    with open(
        folder_path / "meta-results.json",
        "x",
    ) as f:
        json.dump(result, f, indent=4)


def load_assigner_config():
    with open("./config/assigner.json", "r") as file:
        config = json.load(file)
        assign_params = config["assign_params"]
        file.close()

    with open("./config/advanced/assigner.json", "r") as file:
        config = json.load(file)
        advanced_assign_params = config["assign_params"]
        file.close()

    assign_params = {**assign_params, **advanced_assign_params}

    assert (
        assign_params["batch_size"] > 0
    ), "batch size of the assigner configuration should be greater 0"

    assign_params["episodes"] = 10 * assign_params["batch_size"]

    return config, assign_params


def save_user_input(user_input: UserInput, folder_path: Path):
    with open(
        folder_path / "user_input.json",
        "x",
    ) as f:
        f.write(user_input.model_dump_json(indent=4))


def save_user_output(folder_path: Path, user_output: UserOutput):
    with open(
        folder_path / "user_output.json",
        "x",
    ) as f:
        f.write(user_output.model_dump_json(indent=4))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read cost constraint matrices")

    input_group = parser.add_mutually_exclusive_group()

    input_group.add_argument(
        "--cost-constraint-matrices",
        "-c",
        dest="cost_constraint_matrices_path",
        metavar="path",
        type=str,
        help="cost constraint matrices as pickle file",
        default=None,
    )

    input_group.add_argument(
        "--user-input-json",
        "-u",
        dest="user_input_json_path",
        metavar="path",
        type=str,
        help="user input json file",
        default=None,
    )

    args = parser.parse_args(namespace=ArgNamespace())
    main(Args(**vars(args)))
