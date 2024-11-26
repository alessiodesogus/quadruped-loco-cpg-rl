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

""" Run CPG """
import time
import numpy as np
import matplotlib

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


ADD_CARTESIAN_PD = False
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP, gait="TROT")
# cpg = HopfNetwork(time_step=TIME_STEP, gait="WALK")
# cpg = HopfNetwork(time_step=TIME_STEP, gait="BOUND")
# cpg = HopfNetwork(time_step=TIME_STEP, gait="PACE")

SECONDS_OF_SIMULATION = 10
TEST_STEPS = int(SECONDS_OF_SIMULATION / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
# Initialize storage for joint positions and other data
joint_vel = np.zeros((12, TEST_STEPS)) # joint velocities
CPG_states_legs = np.zeros((4, TEST_STEPS, 4))  # (4 variables, TEST_STEPS, 4 legs)
actual_foot_pos = np.zeros((4, 3, TEST_STEPS))  # 4 legs, 3D (x, y, z), over time
desired_foot_pos = np.zeros((4, 3, TEST_STEPS))  # To store desired positions
desired_joint_angles = np.zeros((3, TEST_STEPS))  # To store desired joint angles for each leg
actual_joint_angles = np.zeros((3, TEST_STEPS))   # To store actual joint angles for each leg3


############## Sample Gains
# joint PD gains
kp=np.array([100,100,100])
kd=np.array([2,2,2])
# Cartesian PD gains
kpCartesian = np.diag([500]*3)
kdCartesian = np.diag([20]*3)

for j in range(TEST_STEPS):
  # initialize torque array to send to motors
  action = np.zeros(12) 
  # get desired foot positions from CPG 
  xs,zs = cpg.update()
  # print("xs: ", xs)
  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()

  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for leg i 
    tau = np.zeros(3)
    
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
    desired_foot_pos[i, :, j] = leg_xyz
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
    leg_q = env.robot.ComputeInverseKinematics(xyz_coord=leg_xyz, legID=i) # [TODO] 
    # print("leg_q: ", leg_q)
    
    # Store the desired joint angles for this leg
    desired_joint_angles[:, j] = leg_q

    # Add joint PD contribution to tau for leg i (Equation 4)
    tau += kp * (leg_q - q[3*i:3*i+3]) + kd * ( - dq[3*i:3*i+3]) # [TODO] 

    # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
    J, curr_foot_pos = env.robot.ComputeJacobianAndPosition(i)
    actual_foot_pos[i, :, j] = curr_foot_pos

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current foot velocity in leg frame (Equation 2)
      curr_foot_vel = J @ dq[3*i:3*i+3]
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += J.T @ (kpCartesian @ (leg_xyz - curr_foot_pos) + kdCartesian @ (0 - curr_foot_vel)) # [TODO] 

    # Set tau for leg i in action vector
    action[3*i:3*i+3] = tau

    # Store the CPG state for this leg directly inside the loop
    CPG_states_legs[:, j, i] = xs[i]  # Store the CPG output for leg i at time step j -- TO REVIEW !! 

  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action)

  # print("q: ", q)
  # Save actual joint angles for this time step
  actual_joint_angles[:, j] = q[0:3] 

  # [TODO] save any CPG or robot states for plotting
  joint_vel[:, j] = dq

##################################################### 
# PLOTS
#####################################################

# 1) A plot of the CPG states (r,θ,  ̇r,  ̇θ) for a trot gait (plots for other gaits are encouraged, but not required). We
# suggest making subplots for each leg, and make sure these are at a scale where the states are clearly visible (for
# example 2 gait cycles).

fig, axs = plt.subplots(2, 2, figsize=(10, 8)) # -- TO REVIEW !!

labels = ['r', 'theta', 'r_dot', 'theta_dot']
colors = ['b', 'g', 'r', 'c']

for i in range(4):
    ax = axs[i // 2, i % 2]  
    for state in range(4):  
        ax.plot(CPG_states_legs[state, :, i], label=f'{labels[state]}', color=colors[state])
    ax.set_title(f'Leg {i + 1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('CPG State Value')
    ax.legend()

plt.tight_layout()
plt.show()

# 2) A plot comparing the desired foot position vs. actual foot position with/without joint PD and Cartesian PD (for
# one leg is fine). What gains do you use, and how does this affect tracking performance?
# Plot for Front Right Leg (leg index 0)
leg_index = 0

# Extract data for plotting
desired_x = desired_foot_pos[leg_index, 0, :]
actual_x = actual_foot_pos[leg_index, 0, :]
desired_z = desired_foot_pos[leg_index, 2, :]
actual_z = actual_foot_pos[leg_index, 2, :]

# Create plots
fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# X-axis positions
axs[0].plot(t, desired_x, label="Desired X", linestyle="--")
axs[0].plot(t, actual_x, label="Actual X")
axs[0].set_title("Foot Position X (Front Right Leg)")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Position (m)")
axs[0].legend()

# Z-axis positions
axs[1].plot(t, desired_z, label="Desired Z", linestyle="--")
axs[1].plot(t, actual_z, label="Actual Z")
axs[1].set_title("Foot Position Z (Front Right Leg)")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Position (m)")
axs[1].legend()

plt.tight_layout()
plt.show()

# 3) A plot comparing the desired joint angles vs. actual joint angles with/without joint PD and Cartesian PD (for
# one leg is fine). What gains do you use, and how does this affect tracking performance?
# Plot for Front Right Leg (leg index 0)
leg_index = 0

# Extract desired and actual joint angles for the selected leg
desired_q_leg = desired_joint_angles[:, :]  # All time steps for desired joint angles
actual_q_leg = actual_joint_angles[:, :]    # All time steps for actual joint angles

# Create the plot for the desired vs actual joint angles
fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# Plot for the first joint (hip joint) - IS RIGHT TO HAVE A 0 FOR THE DESIRED JOINT ANGLE?
axs[0].plot(t, desired_q_leg[0, :], label="Desired Hip Joint", linestyle="--")
axs[0].plot(t, actual_q_leg[0, :], label="Actual Hip Joint")
axs[0].set_title("Hip Joint Angle (Front-Right Leg)")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Angle (rad)")
axs[0].legend()

# Plot for the second joint (thigh joint)
axs[1].plot(t, desired_q_leg[1, :], label="Desired Thigh Joint", linestyle="--")
axs[1].plot(t, actual_q_leg[1, :], label="Actual Thigh Joint")
axs[1].set_title("Thigh Joint Angle (Front-Right Leg)")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Angle (rad)")
axs[1].legend()

# Plot for the third joint (calf joint)
axs[2].plot(t, desired_q_leg[2, :], label="Desired Calf Joint", linestyle="--")
axs[2].plot(t, actual_q_leg[2, :], label="Actual Calf Joint")
axs[2].set_title("Calf Joint Angle (Front-Right Leg)")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Angle (rad)")
axs[2].legend()

plt.tight_layout()
plt.show()

# Print controller type and gains
if ADD_CARTESIAN_PD:
    print("------------------------------------") 
    print("Controller: Joint PD + Cartesian PD")
    print(f"Joint PD Gains - Kp: {kp}, Kd: {kd}")
    print(f"Cartesian PD Gains - KpCartesian: diag({kpCartesian[0, 0]}), KdCartesian: diag({kdCartesian[0, 0]})")
    print("------------------------------------") 
else:
    print("------------------------------------") 
    print("Controller: Joint PD Only")
    print(f"Joint PD Gains - Kp: {kp}, Kd: {kd}")
    print("------------------------------------") 

# 4) A discussion on which hyperparameters you found necessary to tune, the highest and lowest resulting body
# velocity you achieved, the resulting duty cycle/ratio, time duration of one step (time in stance/swing), and
# resulting Cost of Transport.

# I) Calculate body velocity
body_velocity_x = np.mean(np.diff(actual_foot_pos[:, 0, :], axis=1) / TIME_STEP, axis=0)
body_velocity_z = np.mean(np.diff(actual_foot_pos[:, 2, :], axis=1) / TIME_STEP, axis=0)

# Total velocity magnitude
body_velocity = np.sqrt(body_velocity_x**2 + body_velocity_z**2)
# print(f"Body Velocity: Max: {np.max(body_velocity):.2f} m/s, Min: {np.min(body_velocity):.2f} m/s")
print(f"Average Body Velocity: {np.mean(body_velocity):.2f} m/s")
print("------------------------------------") 

# II) Calculate stance and swing duration
stance_duration = 0  # Initialize as scalar, not a list
swing_duration = 0   # Initialize as scalar, not a list

# Loop over each time step
threshold = -0.29 # threshold to define stance - BASED ON THE GRAPH 2) !
for j in range(1, TEST_STEPS):
    if actual_foot_pos[leg_index, 2, j] < threshold:  # Foot in stance phase
        stance_duration += TIME_STEP
    else:  # Foot in swing phase
        swing_duration += TIME_STEP

# Output durations
print(f"Stance Duration: {stance_duration:.2f} s")
print(f"Swing Duration: {swing_duration:.2f} s")

# Calculate duty cycle (stance duration / total cycle time)
total_duration = stance_duration + swing_duration
duty_cycle = stance_duration / total_duration if total_duration > 0 else 0  # Avoid division by zero
print(f"Duty Cycle: {duty_cycle:.2f}")
print("------------------------------------") 

# III) Calculate step duration
step_durations = []

# Store the step times based on when foot transitions into stance phase
step_time = []
for j in range(1, TEST_STEPS):
    if actual_foot_pos[leg_index, 2, j] < threshold:  # Stance phase
        if actual_foot_pos[leg_index, 2, j-1] >= threshold:  # Transition from swing to stance
            step_time.append(j * TIME_STEP)

# Calculate average step duration
if len(step_time) > 1:
    step_durations.append(np.mean(np.diff(step_time)))  # Average of step times
print(f"Average Step Duration: {step_durations[0]:.2f} s" if step_durations else "No step transitions detected")
print("------------------------------------") 

# IV) Calculate Cost of Transport (CoT)
# print(f"action shape: {action.shape}")
# print(f"joint_vel shape: {joint_vel.shape}")

action_expanded = np.tile(action, (joint_vel.shape[1], 1)).T  # Expanding action to match joint_vel shape
total_energy_expended = np.sum(np.abs(action_expanded) * joint_vel) * TIME_STEP

# Print energy expended for debugging
# print(f"Total Energy Expended: {total_energy_expended:.2f}")

# Calculate total distance traveled (based on body velocity)
total_distance_traveled = np.sum(body_velocity) * TIME_STEP

# Calculate Cost of Transport (CoT)
CoT = total_energy_expended / total_distance_traveled if total_distance_traveled > 0 else 0
print(f"Cost of Transport (CoT): {CoT:.2f}")
print("------------------------------------") 