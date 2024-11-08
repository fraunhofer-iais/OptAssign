# Opt-Assign


## Description
The Python module OptAssign uses Reinforcement Learning to solve an assignment problem: A set of independent tasks should be assigned to simultaneously available resources in such a way that the completion time is minimized. It includes Python code for training and evaluating Reinforcement Learning models. The Reinforcement Learning approach allows for a similar solution quality compared to mathematical solvers with significantly reduced inference time. Training and evaluation of models are provided via services of an asynchronous OPC UA server.

## Background
OptAssign is built on a Reinforcement Learning approach for solving routing problems \[1,2]. It views the assignment problem as a special routing problem through the assignment cost matrix, where nodes are given by (task-resource)-tuples. Visiting a node $v= (i,j)$ induces the assignment of task i to resource j and incurs a cost of $c_{i,j}$. The goal of the algorithm is to optimally utilize the resources by minimizing the completion time: 

$$min_y max_{i=1,...,m} \sum_{j=1,...,n}c_{i,j}y_{i,j}$$

where $y_{i,j} \in {0,1}$ denotes the binary decision variable of assigning task $i$ to resource $j$ and $c_{i,j} \in \R_{\geq 0}$ the corresponding processing cost. An assignment solution is represented as a permutation over the nodes. The Reinforcement Learning model has an encoder-decoder structure, where the encoder embeds nodes via an attention mechanism and the decoder produces a probability distribution over the currently unvisited nodes. During training, the model learns about the symmetries in the solution representation by performing multiple parallel rollouts where each rollout starts from a different node. For more information on the algorithm, see Section 4.2 in \[3].


## Features

The assign module can assign tasks, that do not depend on each other, to resources. The tasks are assigned to the resources with the goal of minimizing throughput time. The assignment does not set the order of the tasks, but only assigns the tasks to the resources. 

The assign module uses a trained model to assign the tasks to the resources which was trained in a mixed setup of up to 10 resources and up to 50 tasks.


## Installation

1. Install pyenv: https://github.com/pyenv/pyenv

2. Install python 3.12
```bash
pyenv install 3.12
```

3. Select python 3.12
```bash
pyenv local 3.12
```

4. Confirm that python 3.12 is selected
```bash
pyenv versions
```
If python 3.12 is not selected, run the following command instead of step 5:
```bash
pyenv exec python -m venv .venv
```

5. Create a virtual environment
```bash
python -m venv .venv
```

6. Activate the virtual environment
```bash
source .venv/bin/activate
```

7. Install the required packages
```bash
pip install -r requirements.txt
```


## Usage

### generate random problems

To generate some random assignment problems (in terms of random cost and constraint matrices) for testing purposes, run the following command:
```bash
python src/generate_random_problems.py  --num-to-gen NUM_TO_GEN --batch-size BATCH_SIZE --num-resources-max NUM_RESOURCES_MAX [--mixed-num-resources] --num-tasks-max NUM_TASKS_MAX [--mixed-num-tasks] --save-path SAVE_PATH
```


### Training

To train a model, run the following command:
```bash
python src/train.py
```

### Assign

To assign tasks to resources using a trained model specified in config/assigner.json, run the following command:
```bash
python src/assign.py (--cost-constraint-matrices path | --user-input-json path)
```
- The **user-input-json** file defines the assignment problem in terms of the given resources and the given tasks. For each task, the constraints have to be specified in terms of on which resources a task can be processed and which processing cost it implies. See /assign_input/input.json for an example file. 
- Alternatively, the assignment problem definition can be handed-over as a **cost-constraint-matrix**. It contains the cost matrix of size $m  \times n$ consisting of the cost values $c_{i,j}$ for all tasks $i$ and resources $j$, as well as the binary constraint matrix of equal size which defines whether task $i$ can be assigned to resource $j$ (value=1) or not (value=0). See /source/generate_random_problems.py for the generation of multiple assignment problems in terms of const-constraint-matrices (=entry in generated_problems).



## License

Our license can be found in the LICENSE.md file. 

## References
\[1] Kwon, Y.D., Choo, J., Kim, B., Yoon, I., Gwon , Y., Min, S.: Pomo: Policy optimization with multiple optima for reinforcement learning. Advances in Neural Information Processing Systems 33,21188 21198 (2020)

\[2] Kool, W., van Hoof, H., Welling, M.: Attention, learn to solve routing problems! In: International Conference on Learning Representations (2019)

\[3] Paul, N., Kister, A., Schnellhardt, T., Fetz, M., Hecker, D., Wirtz, T.: Reinforcement learning for segmented manufacturing. Presented at the workshop AI for Manufacturing (AI4M), European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (2023)
