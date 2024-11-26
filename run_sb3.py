# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

"""
Run stable baselines 3 on quadruped env 
Check the documentation! https://stable-baselines3.readthedocs.io/en/master/
"""
import os
from datetime import datetime
# stable baselines 3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
# utils
from utils.utils import CheckpointCallback
from utils.file_utils import get_latest_model
# gym environment
from env.quadruped_gym_env import QuadrupedGymEnv


LEARNING_ALG = "PPO" # or "SAC"
LOAD_NN = False # if you want to initialize training with a previous model 
NUM_ENVS = 1    # how many pybullet environments to create for data collection
USE_GPU = False # make sure to install all necessary drivers 

LEARNING_ALG = "SAC";  USE_GPU = True
# after implementing, you will want to test how well the agent learns with your MDP: 
# env_configs = {"motor_control_mode":"CPG",
#                "task_env": "FWD_LOCOMOTION", #  "LR_COURSE_TASK",
#                "observation_space_mode": "LR_COURSE_OBS"}
env_configs = {}

if USE_GPU and LEARNING_ALG=="SAC":
    gpu_arg = "auto" 
else:
    gpu_arg = "cpu"

if LOAD_NN:
    interm_dir = "./logs/intermediate_models/"
    log_dir = interm_dir + '' # add path
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    model_name = get_latest_model(log_dir)

# directory to save policies and normalization parameters
SAVE_PATH = './logs/intermediate_models/'+ datetime.now().strftime("%m%d%y%H%M%S") + '/'
os.makedirs(SAVE_PATH, exist_ok=True)
# checkpoint to save policy network periodically
checkpoint_callback = CheckpointCallback(save_freq=30000, save_path=SAVE_PATH,name_prefix='rl_model', verbose=2)
# create Vectorized gym environment
env = lambda: QuadrupedGymEnv(**env_configs)  
env = make_vec_env(env, monitor_dir=SAVE_PATH,n_envs=NUM_ENVS)
# normalize observations to stabilize learning (why?)
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=100.)

if LOAD_NN:
    env = lambda: QuadrupedGymEnv(**env_configs)
    env = make_vec_env(env, monitor_dir=SAVE_PATH, n_envs=NUM_ENVS)
    env = VecNormalize.load(stats_path, env)

# Multi-layer perceptron (MLP) policy of two layers of size _,_ 
policy_kwargs = dict(net_arch=[256,256])
# What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
n_steps = 4096 
learning_rate = lambda f: 1e-4 
ppo_config = {  "gamma":0.99, 
                "n_steps": int(n_steps/NUM_ENVS), 
                "ent_coef":0.0, 
                "learning_rate":learning_rate, 
                "vf_coef":0.5,
                "max_grad_norm":0.5, 
                "gae_lambda":0.95, 
                "batch_size":128,
                "n_epochs":10, 
                "clip_range":0.2, 
                "clip_range_vf":1,
                "verbose":1, 
                "tensorboard_log":None, 
                "_init_setup_model":True, 
                "policy_kwargs":policy_kwargs,
                "device": gpu_arg}

# What are these hyperparameters? Check here: https://stable-baselines3.readthedocs.io/en/master/modules/sac.html
sac_config={"learning_rate":1e-4,
            "buffer_size":300000,
            "batch_size":256,
            "ent_coef":'auto', 
            "gamma":0.99, 
            "tau":0.005,
            "train_freq":1, 
            "gradient_steps":1,
            "learning_starts": 10000,
            "verbose":1, 
            "tensorboard_log":None,
            "policy_kwargs": policy_kwargs,
            "seed":None, 
            "device": gpu_arg}

if LEARNING_ALG == "PPO":
    model = PPO('MlpPolicy', env, **ppo_config)
elif LEARNING_ALG == "SAC":
    model = SAC('MlpPolicy', env, **sac_config)
else:
    raise ValueError(LEARNING_ALG + 'not implemented')

if LOAD_NN:
    if LEARNING_ALG == "PPO":
        model = PPO.load(model_name, env)
    elif LEARNING_ALG == "SAC":
        model = SAC.load(model_name, env)
    print("\nLoaded model", model_name, "\n")

# Learn and save (may need to train for longer)
model.learn(total_timesteps=1000000, log_interval=1,callback=checkpoint_callback, progress_bar=True)
# Don't forget to save the VecNormalize statistics when saving the agent
model.save( os.path.join(SAVE_PATH, "rl_model" ) ) 
env.save(os.path.join(SAVE_PATH, "vec_normalize.pkl" )) 
if LEARNING_ALG == "SAC": # save replay buffer 
    model.save_replay_buffer(os.path.join(SAVE_PATH,"off_policy_replay_buffer"))

