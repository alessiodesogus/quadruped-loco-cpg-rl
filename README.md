# Quadruped-Sim
This repository contains an environment for simulating a quadruped robot.

## Installation

Recommend using a virtualenv (or conda) with python3.6 or higher. After installing virtualenv with pip, this can be done as follows:

`virtualenv {quad_env, or choose another name venv_name} --python=python3`

To activate the virtualenv: 

`source {PATH_TO_VENV}/bin/activate` 

Your command prompt should now look like: 

`(venv_name) user@pc:path$`

The repository depends on recent versions of pybullet, gym, numpy, stable-baselines3, matplotlib, etc., `pip install [PACKAGE]` 



## Code structure

- [env](./env) for the quadruped environment files, please see the gym simulation environment [quadruped_gym_env.py](./env/quadruped_gym_env.py), the robot specific functionalities in [quadruped.py](./env/quadruped.py), and config variables in [configs_a1.py](./env/configs_a1.py). You will need to make edits in [quadruped_gym_env.py](./env/quadruped_gym_env.py), and review [quadruped.py](./env/quadruped.py) carefully for accessing robot states and calling functions to solve inverse kinematics, return the leg Jacobian, etc. 
- [a1_description](./a1_description) contains the robot mesh files and urdf.
- [utils](./utils) for some file i/o and plotting helpers.
- [hopf_network.py](./env/hopf_network.py) provides a CPG class skeleton for various gaits, and [run_cpg.py](run_cpg.py) maps these joint commands to be executed on an instance of the  [quadruped_gym_env](./env/quadruped_gym_env.py) class. Please fill in these files carefully. 
- [run_sb3.py](./run_sb3.py) and [load_sb3.py](./load_sb3.py) provide an interface to training RL algorithms based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). You should review the documentation carefully for information on the different algorithms and training hyperparameters. 

## Code resources
- The [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3) is the current up-to-date documentation for interfacing with the simulation. 
- The quadruped environment took inspiration from [Google's motion-imitation repository](https://github.com/google-research/motion_imitation) based on [this paper](https://xbpeng.github.io/projects/Robotic_Imitation/2020_Robotic_Imitation.pdf). 
- Reinforcement learning algorithms from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). Also see for example [ray[rllib]](https://github.com/ray-project/ray) and [spinningup](https://github.com/openai/spinningup). 

## Conceptual resources
The CPG and RL framework are based on the following papers:
- G. Bellegarda and A. Ijspeert, "CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion," in IEEE Robotics and Automation Letters, 2022, doi: 10.1109/LRA.2022.3218167. [IEEE](https://ieeexplore.ieee.org/abstract/document/9932888), [arxiv](https://arxiv.org/abs/2211.00458)
- G. Bellegarda, Y. Chen, Z. Liu, and Q. Nguyen, "Robust High-speed Running for Quadruped Robots via Deep Reinforcement Learning," in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2022. [arxiv](https://arxiv.org/abs/2103.06484)

## Tips
- If your simulation is very slow, remove the calls to time.sleep() and disable the camera resets in [quadruped_gym_env.py](./env/quadruped_gym_env.py).
- The camera viewer can be modified in `_render_step_helper()` in [quadruped_gym_env.py](./env/quadruped_gym_env.py) to track the hopper.