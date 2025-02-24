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


import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


#LEARNING_ALG = "SAC"
LEARNING_ALG = "PPO"

interm_dir = "./logs/intermediate_models/"
# interm_dir = "./rl_test_models/"
#interm_dir = "./rl_final_models/"
#path to saved models, i.e. interm_dir + rl_model_5: '120824001750'; rl_model_6: '120824085440'; ; rl_model_7'120824231026'
# lr_course_1: '121624231717'; ; lr_course_2: '121624232812'; lr_course_3:'121724083431' 
#pd_lr_course_1: '121824012911' pp0_pd_lr_course: '121824172255', pp0_pd_fwd_course_default_obs: '121924010159', sac_pd_lrcoursetask_obsmodifié: 122024012206
#sac_pd_fwdreward_lrcourseobs: '122024012206' 122024113331, ppo_cartesianpd_fwd_lrcourseobs: '122124051447'
#rl_slope_1 : SLOPES = 0.2; cpg; lr_course_obs; lr_course = original fwdlocomotion; : 122224121932 -> marche pas trop mal mais reward mean bof -> tente une plus grande pente
#rl_slope_2 : SLOPES = 0.5; cpg; lr_course_obs; lr_course = original fwdlocomotion; : 122224161257 -> pente trop grande la simu s'arrête quand le robot entame la pente -> a cause de dotmin sans doute -> coupe la poire en deux pour le prochain test 
#rl_slope_3 : SLOPES = 0.35; cpg; lr_course_obs; lr_course = originql fwdlocomotion; 122224185149 -> pente trop grande, apres quelques essais, on remarque que rl_slope_1 marche pas trop mal pour une pente allant jusque 0.25
# en affichant la reward avec rl_slope_1, on s'aperçoit que le robot tombe quand la reward devient négative(jusqu'à -0.1 !!!) (plus d'info car reward fixé à zéro) -> décide de repartir sur ce modèle en ajoutant une goal reward positive (=1 quand le robot atteint l'objectif fixé en (x,y) =(10,0))
#rl_slope_4 : SLOPES = 0.2; cpg; lr_course_obs; lr_course = original_fwd+goal_reward; 122224223115 -> linalgnorm de la diff entre goal et position (mauvaise idée car le robot est recompense s'il recule)
# on décide d'utiliser np.sum plutot que linalg, on load le model rl_slope_1 plusieurs fois en ajustant le coef devant la somme; la erward reste toujours négative au moment de tomber
#surtout près de l'origine -> la contribution du déplacement en y semble être trop importante à ce moment, on enlève la composante en y du goal -> goal = [10]
# note : reward négative = robot n'a plus d'info et tombe
# list(self.robot.GetBasePosition()[:2]) -> list(self.robot.GetBasePosition()[:1])  ( (x,y) -> (x) )
# on réajuste le coef en lancant le modele rl_slope_1 et en affichant la reward avec et sans la goal reward; jusqu'à ce que la nouvelle reward soit toujours positive (meme quand lautre est négative et que le robot tombe )
# on trouve un coef satisfaisant avec 1/50; (1/100 toujours négatif et 1/10 rique de trop influencer le modèle)
# rl_slope_5 : SLOPES = 0.2; cpg; lr_course_obs; lr_course + g=goal_reward; 122224233348 -> modèle marche bien, tombe quelque fois -> augmenter le nombre d'itération du training pour rl_slope_6
# ideas for next model : augmenter le nombre d'iteration pour atteindre rew_mean = 1000; (le modèle commence à deveenir intéresssant autour de 900k itérations)
#                        pouvoir imposer une vitesse 
# rl_slope_6 = rl_slope_5 with 100 000 000 itérations en plus: 122324012912 -> vitesse assez élevée dans la pente, on aimerait bien la maitriser un peu plus -> imposer une vitesse 
# marche parfois avec une pente = 0.275, presque avec 0.29  -> peut être essayer de train avec cette valeur (en tout cas 0.30 semble etre la limite aveec le dotprod) +  ok pour els pente plus faible
# ideas for next model : penaliser l'ecart en y avec une reward du genre "alpha * (abs(y_max_rl_slope_6) - abs(y)) " NON car déja la drift reward
#                        imposer une vitesse souhaitée - ok  
#                        ?? reduire le dotprodmin dans is_fallen semble permettre de passer plus facilement les pentes à 0.29 ??
# rl_slopes_7 : SLOPES = 0.29; cpg; lr_course_obs + goal_reward + vitesse tracking speed: 0.8m*s^-1 ie on change aussi le goal vers x = 8m; 122324034628 efnaite s'est initialisé avec l'autre modèle !!!!rl_slope_5
# next model : rl_slope_5 avec velocity tracking
                         

################ TEST FINAL #########################
# forward loc : 010425201721
# slopes : 010425224908
# slopes comme forward + dist, 010525003919
# slopes comme forward + dist + 0.8*orientation : 010525075217
# slopes comme dessus mais avec 500 000 de step en plus : 010525100748
# slopes (ecrire reward) : 010525105257
log_dir = interm_dir + '010525131034' 
#log_dir = interm_dir + 'rl_model_11'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training

env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
env_config["task_env"] = "LR_COURSE_TASK"
env_config["observation_space_mode"] = "LR_COURSE_OBS"

# config terrain
# env_config["terrain"] = False
env_config["terrain"] = "SLOPES"

# config control mode -- don't forget to change observation space (3)
env_config["motor_control_mode"] = "CPG"
# env_config["motor_control_mode"] = "PD"
# env_config["motor_control_mode"] = "CARTESIAN_PD"


# env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env  
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
#
steps_nb = 2000

base_position = np.zeros((steps_nb,3))
x_base_position = np.zeros(steps_nb)
y_base_position = np.zeros(steps_nb)
z_base_position = np.zeros(steps_nb)

x_base_vel = np.zeros(steps_nb)
y_base_vel = np.zeros(steps_nb)
z_base_vel = np.zeros(steps_nb)

roll = np.zeros(steps_nb)
pitch = np.zeros(steps_nb)
yaw = np.zeros(steps_nb)

roll_rate=np.zeros(steps_nb)
pitch_rate=np.zeros(steps_nb)
yaw_rate=np.zeros(steps_nb)

leg_index = 0
threshold = -0.29
stance_duration = 0
swing_duration = 0

distance_to_origin = 0
total_energy = 0


#for i in range(2000):
for i in range(steps_nb):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        episode_reward = 0

        # # CoT
        # dx = x_base_position[0]-info[0]['base_pos'][0]
        # dy = y_base_position[0]-info[0]['base_pos'][1]
        # dz = 0.8
        # distance_to_origin += np.sqrt((dx)**2 + (dy)**2 + (dz)**2)
 
        # CoT = distance_to_origin/total_energy
        # print("CoT :", CoT)

        

    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 


    # base position 
    base_position[i]=info[0]['base_pos']
    x_base_position[i]=info[0]['base_pos'][0]
    y_base_position[i]=info[0]['base_pos'][1]
    z_base_position[i]=info[0]['base_pos'][2]
    #print("position en x:",x_base_position[i] )
    #print("position en y:",y_base_position[i] )
    #print("position en z:",z_base_position[i] )

    # base velocity 
    x_base_vel[i]=info[0]['base_vel'][0]
    y_base_vel[i]=info[0]['base_vel'][1]
    z_base_vel[i]=info[0]['base_vel'][2]

    # Roll, pitch Yaw
    roll[i]=info[0]['base_orientation'][0]
    pitch[i]=info[0]['base_orientation'][1]
    yaw[i]=info[0]['base_orientation'][2]

    #Roll, Pitch, Yaw rates 
    roll_rate[i]=info[0]['base_orientation_rate'][0]
    pitch_rate[i]=info[0]['base_orientation_rate'][1]
    yaw_rate[i]=info[0]['base_orientation_rate'][2]

    # foot position 
    foot_pos = info[0]['foot_pos']

    # swing and stance 
    if foot_pos[leg_index, 2] < threshold:  # Foot in stance phase
        stance_duration += 1/100 # 1/100 pour conversion en secondes
    else:  # Foot in swing phase
        swing_duration += 1/100

    # CoT
    total_energy += info[0]['energy']

    
# duty cycle 
stride_duration = stance_duration + swing_duration
if stride_duration > 0: 
    duty_cycle = stance_duration / stride_duration
else : 
    duty_cycle=0

# Cost Of Transport COT
difference = np.zeros((steps_nb,3))
dist = 0
for i in range(0, steps_nb-1):
    difference[i,:]=base_position[i+1,:]-base_position[i,:]
#     print("difference:", difference[i,:])
#     dist = dist + np.linalg.norm(difference)
#     print("linalgnorm:", np.linalg.norm(difference))
#     print("dist", dist)
# print("distance:", dist)
dist = np.zeros(steps_nb)
total_dist = 0
for i in range(0,steps_nb-1): 
    dist[i] = np.linalg.norm(difference[i, :])
total_dist = np.sum(dist)
#print("total_dist:", total_dist)
CoT = total_dist/total_energy

# average speed (x axis, y, z axis)
avg_vel_x = np.sum(x_base_vel)/steps_nb
avg_vel_y = np.sum(y_base_vel)/steps_nb
avg_vel_z = np.sum(z_base_vel)/steps_nb
# print("avg velocity along x:", avg_vel_x)
# print("avg velocity along y:", avg_vel_y)
# print("avg velocity along z:", avg_vel_z)

# [TODO] make plots:

timesteps = np.linspace(1,steps_nb/100, num=steps_nb)
#print("timesteps =", timesteps)
#print("x_base_position = ",x_base_position)

print("##### RESULTS ########")
print("avg velocity along x:", avg_vel_x)
print("stance_duration:", stance_duration)
print("swing_duration:", swing_duration)
print("duty_cycle:", duty_cycle)
print("CoT :", CoT)
print("##### END RESULTS ########")

# def HelPlot(suptitle, x, y1, y2, y3, Xlabel, Ylabel): 
#     fig, axs = plt.subplots(3)
#     fig.suptitle('Base Position')
#     axs[0].plot(x, y1, label='x-pos', color='b')
#     axs[1].plot(x, y2, label='y-pos', color='g')
#     axs[2].plot(x, y3, label='z-pos',color='r')
#     for ax in axs.flat:
#         ax.set(xlabel=Xlabel, ylabel=Ylabel)
#     plt.show()

# #  base position 
# HelPlot('Base position', timesteps, x_base_position, y_base_position, z_base_position, 'Time[s]', 'Position[m]')
# HelPlot('Base velocity', timesteps, x_base_vel, y_base_vel, z_base_vel, 'Time[s]', 'Speed[m/s]' )


# base position
fig1, axs1 = plt.subplots(3)
fig1.suptitle('Base Position')
axs1[0].plot(timesteps, x_base_position, label='x-pos', color='b')
axs1[0].grid(True)
axs1[0].set(ylabel='x-position [m]')
axs1[1].plot(timesteps, y_base_position, label='y-pos', color='g')
axs1[1].grid(True)
axs1[1].set(ylabel='y-position [m]')
axs1[2].plot(timesteps, z_base_position, label='z-pos',color='r')
axs1[2].grid(True)
axs1[2].set(ylabel='z-position [m]')
for ax in axs1.flat:
    ax.set(xlabel='Time[s]')

plt.show()

# base velocity 
fig, axs = plt.subplots(3)
fig.suptitle('Base Velocity')
axs[0].plot(timesteps, x_base_vel, label='x-vel', color='b')
axs[0].grid(True)
axs[0].set(ylabel='x-velocity [m/s]')
axs[1].plot(timesteps, y_base_vel, label='y-vel', color='g')
axs[1].grid(True)
axs[1].set(ylabel='y-velocity [m]')
axs[2].plot(timesteps, z_base_vel, label='z-vel',color='r')
axs[2].grid(True)
axs[2].set(ylabel='z-vel [m]')
for ax in axs.flat:
    ax.set(xlabel='Time[s]')
plt.show()

# base orientation
fig, axs = plt.subplots(3)
fig.suptitle('Base Orientation')
axs[0].plot(timesteps, roll, label='roll', color='b')
axs[0].grid(True)
axs[0].set(ylabel='Roll [deg]')
axs[1].plot(timesteps, pitch, label='pitch', color='g')
axs[1].grid(True)
axs[1].set(ylabel='Pitch [deg]')
axs[2].plot(timesteps, yaw, label='yaw',color='r')
axs[2].grid(True)
axs[2].set(ylabel='Yaw [deg]')
for ax in axs.flat:
    ax.set(xlabel='Time[s]')
plt.show()


#base orientation rates
fig, axs = plt.subplots(3)
fig.suptitle('Base Orientation rates')
axs[0].plot(timesteps, roll_rate, label='roll rate', color='b')
axs[0].grid(True)
axs[0].set(ylabel='Roll rate [deg/s]')
axs[1].plot(timesteps, pitch_rate, label='pitch rate', color='g')
axs[1].grid(True)
axs[1].set(ylabel='Pitch [deg/s]')
axs[2].plot(timesteps, yaw_rate, label='yaw rate',color='r')
axs[2].grid(True)
axs[2].set(ylabel='Yaw [deg/s]')
for ax in axs.flat:
    ax.set(xlabel='Time[s]')
plt.show()

# plt.show()
# plt.plot(timesteps, roll_rate)
# plt.plot(timesteps, pitch_rate)
# plt.plot(timesteps, yaw_rate)
# plt.title('Base orientation rates')
# plt.show()




