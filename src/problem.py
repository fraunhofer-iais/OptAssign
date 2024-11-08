import numpy as np
import torch


def get_random_problems(batch_size, problem_size, num_resources):
    min_cost = 0.2  # TODO: read from config
    max_cost = 1  # TODO: read from config
    min_mean = 0.25  # TODO: read from config
    max_mean = 0.75  # TODO: read from config
    variance_factor = 8  # TODO: read from config

    # Sample costs for jsp1
    assert (
        problem_size % num_resources == 0
    ), "problem_size: needs to be a multiple of num_resources"
    num_tasks = int(problem_size / num_resources)
    problems = torch.zeros((batch_size, problem_size, 3), dtype=torch.float)

    # with vectorization
    # for every resource draw a mean value for costs
    means = torch.FloatTensor(batch_size * num_resources).uniform_(min_mean, max_mean)
    # for every resource draw a variance for costs
    variances = (
        torch.FloatTensor(batch_size * num_resources).uniform_(0.01, 1)
        / variance_factor
    )

    # draw cost after normal distribution
    costs = (
        torch.distributions.normal.Normal(means, torch.sqrt(variances))
        .sample(torch.Size([num_tasks]))
        .swapaxes(0, 1)
        .reshape(1, batch_size, problem_size)
    )

    # cut costs with minimal and maximal value
    costs[costs.ge(max_cost)] = max_cost
    costs[costs.le(min_cost)] = min_cost

    # indices for input values
    resource_idx = (
        torch.arange(num_resources, dtype=torch.float)
        .repeat_interleave(num_tasks)
        .view(1, problem_size)
        + 1
    ).repeat(batch_size, 1)
    task_idx = (
        torch.arange(num_tasks, dtype=torch.float)
        .repeat(num_resources)
        .view(1, problem_size)
        + 1
    ).repeat(batch_size, 1)

    # arange in data
    problems[:, :, 0] = costs
    problems[:, :, 1] = resource_idx
    problems[:, :, 2] = task_idx  # tasks

    constraint_matrix = torch.ones(
        batch_size, num_tasks, num_resources
    )

    for i in range(batch_size):
        percentage = np.random.uniform(0.25, 0.3)
        constraint_count = round(num_tasks * percentage)
        choice = np.random.choice(np.arange(num_tasks), constraint_count)

        remove_list = []
        for j in range(num_resources - 1):
            if j > 0:
                choice = np.delete(choice, remove_list)
            remove_list = []

            for k in range(choice.shape[0]):
                if j == 0 or np.random.randint(0, 2) == 1:
                    constraint_matrix[i, choice[k], j] = 0
                else:
                    remove_list.append(k)
    return problems, constraint_matrix
