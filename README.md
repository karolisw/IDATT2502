### IDATT2502 Project - Reinforcement Learning using Bio-inspired methods
#  Training procgen environment using Population-Based Training (PBT) for PPO agent vs Using baseline3 PPO agent 


For this research-project, we have combined two separate repositories. One on Population Based Training (PBT), which is our bio-inspired method of choice, as well as one that only contains our algorithm of choice. Our algorithm of choice is Proximal Policy Optimization (PPO), a policy gradient method that has proven to function well, but is sensitive to hyperparameter-tuning. Thus, using a hyperparameter-tuning optimization technique such as PBT should garner better result than a baseline PPO algorithm. 

In order to give both projects an equal starting point, both use the Stable Baselines3 PPO algorithm. Originally, the project that trains PPO without PBT had its own pre-defined PPO model with hand-picked hyperparameters, environment-builder, logger -and storage classes. The project is now rewritten such that it also relies on the SB3 PPO-model. 

To ensure greater chance of achieving a trained agent able to complete unseen levels (AKA. generalization), our environmental benchmark is ProcGen. SB3 is also compatible with ProcGen and has its own gym wrapper called ProcGenEnv. We have implemented ProcGenEnv in both our repositores. 


<p>
    <img src="images/ppo.png"  width=300 />
</p>



## Proximal Policy Optimization


<p>
    <img src="images/ppo_alg.png"  width=300 />
</p>




## Population Based Training

Population-Based Training is a novel approach to hyperparameter optimisation by jointly optimising a population of models and their hyperparameters to maximise performance. PBT takes its inspiration from genetic algorithms where each member of the population can exploit information from the remainder of the population.



<p>
    <img src="images/pbt.jpg"  width=300 />
</p>

<p>
    <em>Illustration of PBT training process (Liebig, Jan Frederik, Evaluating Population based Reinforcement Learning for Transfer Learning, 2021)</em>
</p>



# How to run the PBT project


## Prerequisites

- Python 3.8
- Conda
- (Poetry)
- (Pytorch)[^1]

[^1]: Please use **cpu-only** version if possible 


## Creating environment
Create `conda` virtual environment:
```
conda create -p ./venv python==3.8
```

## Installation
Use poetry to install all required Python packages:

```
poetry install
```

If `mpi4py` was not installed, use pip or poetry to install:
```
pip install mpi4py
``` 

or

```
poetry add mpi4py
```

Do not use Conda install as that may lead to some unknown issues.


Tensorboard should be installed if logging is preferable:
```
pip install tensorboard
```

## Activating environment
Activate `conda` environment:
```
conda activate ./venv
```

## Running the experiment
Please use `mpiexec` or `mpirun` to run experiments. 
To run the project using 8 cores and 8 agents (must be 1 agent per core) Please verify how many cores you have as the program will crash if you choose more than your CPU has. 
"pbt_rl_wta.py" is the program for running PBT locally. 
"pbt_rl_truct_collective.py" is the program for running PBT online (asynchronously), which is supported by MPI. 
The env id "bigfish" is the ProcGen env we've chosen to train our agents on. 
--tb-writer is a boolean value that states whether or not to log the Monitor's results (our env is wrapped in a monitor).
```
mpiexec -n 8 python pbt_rl_wta.py --num-agents 8 --env-id bigfish --tb-writer True
```

Due to the loggers compatibility with tensorboard, the logs can be displayed:
```
tensorboard --logdir=logs
```

To display, simle click the link that appears in the terminal after writing the command.



# How to run the single agent project
===============

This repository initially contained code to train a single ppo agent in Procgen with the Pytorch framework. 
Currently, the repository is using a Stable Baselines3 and its baseline PPO model instead of the previous model.
This is, as previously stated, to ensure equal beginnings when running this trained agent against the PBT agent in testing.


Procgen differs from other Benchmarks within the RL domain:

- The convolutional layers are initialized differently (Xavier uniform initialization instead of orthogonal initialization).
- Do not use observation normalization
- Gradient accumulation to [handle large mini-batch size](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255).


## Prerequisites

- python>=3.6
- torch 1.3
- procgen
- pyyaml (for the hyperparameters)

## Parameters

Use `train.py` to train the agent in procgen environment. It has the following arguments:
- `--exp_name`: ID to designate your expriment. 
- `--env_name`: Name of the Procgen environment.
- `--start_level`: Start level for for environment.
- `--num_levels`: Number of training levels for environment.
- `--distribution_mode`: Mode of your environ
- `--param_name`: Configurations name for your training. By default, the training loads hyperparameters from `config.yml/procgen/param_name`.
- `--num_timesteps`: Number of total timesteps to train your agent.

After you start training your agent, log and parameters are automatically stored in `logs/procgen/env-name/exp-name/`

## Running the experiment

Generalization on easy environments as provided by the [PROCGEN](https://cdn.openai.com/procgen.pdf) paper. 
The hyperparameters are present under "baseline" in the ./hyperparams/procgen/config.yml - file.

`python train.py --exp_name easy-run-200 --env_name bigfish --param_name baseline --num_levels 200 --distribution_mode easy --num_timesteps 25000000`

To run the experiment using the default hyperparameters from the SB3 PPO-model, run the project using:

`python train.py --exp_name easy-run-200 --env_name bigfish --param_name baseline3 --num_levels 200 --distribution_mode easy --num_timesteps 25000000`


If your GPU device cannot handle the mini-batch size, the size can be configured in the yml file. 
However, the mini-batch size should be a factor of n_envs * n_steps. 



## References

[1] [PBT project: EVO-PopulationBasedTraining ](https://github.com/yyzpiero/EVO-PopulationBasedTraining/tree/master/utils) <br>
[2] [Single agent project: Training Procgen environment with Pytorch ](https://github.com/joonleesky/train-procgen-pytorch) <br>
[3] [PBT: Population Based Training of Neural Network ](https://arxiv.org/pdf/1711.09846.pdf) <br>
[4] [Procgen: Leveraging Procedural Generation to Benchmark Reinforcement Learning ](https://cdn.openai.com/procgen.pdf) <br>
[5] [PPO: Human-level control through deep reinforcement learning ](https://arxiv.org/abs/1707.06347) <br>




























# EVO: Population-Based Training (PBT) for Reinforcement Learning using MPI 

<p>
    <img src="./logo.png"  width=300 />
</p>

## Overview

Population-Based Training is a novel approach to hyperparameter optimisation by jointly optimising a population of models and their hyperparameters to maximise performance. PBT takes its inspiration from genetic algorithms where each member of the population can exploit information from the remainder of the population.



<p>
    <img src="https://i.imgur.com/hvfgzyf.png" alt="PBT Illustration" style="zoom:30%;" />
</p>
<p>
    <em>Illustration of PBT training process (Liebig, Jan Frederik, Evaluating Population based Reinforcement Learning for Transfer Learning, 2021)</em>
</p>


To extend the population of agents to extreme-scale using High-Performance Computer, this repo, namely **EVO** provide a PBT implementation for RL using Message Passing Interface. 

## MPI (Message Passing Interface) and mpi4py

[Message passing interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface) provides a powerful, efficient, and portable way to express parallel programs.  It is the dominant model used in [high-performance computing](https://en.wikipedia.org/wiki/High-performance_computing). MPI is a programmable communication protocol on parallel computing nodes, which supports point-to-point communication and collective communication. Socket and TCP protocol communication are used in the transport layer. It is the main communication method on distributed memory supercomputers, and it can also run on shared computers.

[mpi4py](https://mpi4py.readthedocs.io/en/stable/) provides a Python interface that resembles the [message passing interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface), and hence allows Python programs to exploit multiple processors on multiple compute nodes. 

## Get Started
Prerequisites:

- Python 3.8
- Conda
- (Poetry)
- (Pytorch)[^1]

[^1]: Please use **cpu-only** version if possible, as most HPC clusters don't have GPUs

Clone this repo: 

```
git clone https://github.com/yyzpiero/evo.git
```

Create `conda` environment:
```
conda create -p ./venv python==X.X
```
and use poetry to install all Python packages:

```
poetry install
```

Please use pip or poetry to install `mpi4py` :  
```
pip install mpi4py
``` 

or

```
poetry add mpi4py
```

Using Conda install may lead to some unknown issues.


### Basic Usage
Activate `conda` environment:
```
conda activate ./venv
```

Please use `mpiexec` or `mpirun` to run experiments:
```
mpiexec -n 4 python pbt_rl_truct.py --num-agents 4 --env-id bigfish --tb-writer True
```


### Example

#### Tensorboard support
EVO also supports experiment monitoring with Tensorboard. Example command line to run an experiment with Tensorboard monitoring:
```
mpiexec -n 4 python pbt_rl_truct_collective.py --num-agents 4 --env-id CartPole-v1 --tb-writer True
```
## Toy Model
The toy example was reproduced from Fig. 2 in the [PBT paper](https://arxiv.org/abs/1711.09846)

<p>
    <img src="https://i.imgur.com/bbJ12k5.png" alt="PBT Illustration" style="zoom:50%;" />
</p>

## Reinforcement Learning Agent

[PPO agent]() from [`stable-baselines 3`](https://github.com/DLR-RM/stable-baselines3) with default settings are used as reinforcement learning agent.

` self.model = PPO("MlpPolicy", env=self.env, verbose=0, create_eval_env=True)` 

However, it can also be replaced by any other reinforcement learning algorithms.

### Reference: 

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [PPO in Stable Baseline 3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## Selection Mechanism

### "Winner-takes-all"

A simply selection mechanism, that for each generation, only the best-performed agent is kept, and its NN parameters are copied to all other agents. 
.py provides an implementation of such a mechanism using collective communications.

### Truncation selection

> It is the default selection strategy in [PBT paper](https://arxiv.org/abs/1711.09846) for RL training, and is widely used in other PBT-based methods.

All agents in the entire population are ranked by their episodic rewards. If the agent is in the bottom $25\%$ of the entire population, another agent from the top $25\%$ is sampled and its NN parameters and hyperparameters are copied to the current agent.  **Different <u>MPI communication methods</u>[^note] are implemented.**

#### Implemented Variants

| Variants | Description                                                  |
| -------- | ------------------------------------------------------------ |
|   `pbt_rl_truct.py`       | implementation using point-2-point communications via `send` and `recv`. |
|     `pbt_rl_truct_collective.py`     | implementation using collective communications.              |

For small clusters with a limited number of nodes, we suggest the point-2-point method, which is faster than the collective method. However, for large HPC clusters, the collective method is much faster and more robust.

[^note]: [This article](https://www.futurelearn.com/info/courses/python-in-hpc/0/steps/65143) briefly introduces the difference between point-2-point communications and collective communications in MPI.

## Benchmarks

We used continuous control `AntBulletEnv-v0` scenario in [PyBullet environments](https://pybullet.org/wordpress/) to test our implementations. 

Results of the experiments are presented on the Figure below:


<p>
    <img src="https://i.imgur.com/Mi5Giit.png" alt="Benchmark Results" style="zoom:50%;" />
</p>

<p>
    <em>Left Figure: Reward per generation using PBT | Right Figure: Reward per step using single SB3 agent</em>
</p>




**Some key observations:**

- By using PBT to train PPO agents can achieve better results than a SAC agent(single agent)

  - Note: SAC should outperforms PPO (see [OpenRL](https://wandb.ai/cleanrl/cleanrl.benchmark/reports/Open-RL-Benchmark-0-6-0---Vmlldzo0MDcxOA)) in most *PyBullet* environments

- "Winner-takes-all" outperforms the Truncation Selection mechanism in this scenario.


## Acknowledgements
This repo is inspired by [graf](https://github.com/PytLab/gaft), [angusfung's population based training repo](https://github.com/angusfung/population-based-training). 
