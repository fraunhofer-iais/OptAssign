import argparse
import pickle
import random
from dataclasses import dataclass
from datetime import datetime

from problem import get_random_problems


@dataclass
class Args:
    num_to_gen: int
    batch_size: int
    num_resources_max: int
    mixed_num_resources: bool
    num_tasks_max: int
    mixed_num_tasks: bool
    save_path: str


def main(args: Args):

    timestamp = int(datetime.timestamp(datetime.now()))

    path = f"{args.save_path}{args.num_to_gen}_{args.batch_size}_{args.num_resources_max}_{args.mixed_num_resources}_{args.num_tasks_max}_{args.mixed_num_tasks}_{timestamp}.pkl"

    with open(path, "wb") as file:

        generated_problems: list[list] = []

        num_resources = args.num_resources_max
        problem_size = args.num_tasks_max * num_resources

        for i in range(args.num_to_gen):

            if args.mixed_num_resources:
                random_num_resources = random.randint(3, args.num_resources_max)
                num_resources = random_num_resources

            if args.mixed_num_tasks:
                random_num_task = random.randint(10, args.num_tasks_max)
                problem_size = random_num_task * num_resources

            problems, constraint_matrix = get_random_problems(
                args.batch_size, problem_size, num_resources
            )

            generated_problems.append([problems, constraint_matrix])

        """ for i in range(len(generated_problems[0])):
            generated_problems[0][i] = generated_problems[0][i].cpu().detach().numpy()
            generated_problems[1][i] = generated_problems[1][i].cpu().detach().numpy() """

        pickle.dump(generated_problems, file)
        print(f"dumped generated problems to file {path}'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generates random problems and constraint matrices."
    )

    parser.add_argument(
        "--num-to-gen", "-n", dest="num_to_gen", type=int, required=True
    )
    parser.add_argument(
        "--batch-size", "-b", dest="batch_size", type=int, required=True
    )
    parser.add_argument(
        "--num-resources-max", "-nm", dest="num_resources_max", type=int, required=True
    )
    parser.add_argument(
        "--mixed-num-resources",
        "-mnm",
        dest="mixed_num_resources",
        action="store_true",
    )
    parser.add_argument(
        "--num-tasks-max",
        "-nc",
        dest="num_tasks_max",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--mixed-num-tasks",
        "-mnc",
        dest="mixed_num_tasks",
        action="store_true",
    )
    parser.add_argument("--save-path", "-s", dest="save_path", type=str, required=True)

    args = parser.parse_args()
    main(Args(**vars(args)))
