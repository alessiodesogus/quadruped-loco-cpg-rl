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

"""This file implements the gym environment for a quadruped. """
import os, inspect
# so we can import files
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import time, datetime
import numpy as np
# gym
import gym
from gym import spaces
from gym.utils import seeding
# pybullet
import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import random
random.seed(10)
# quadruped and configs
import quadruped
import configs_a1 as robot_config
from hopf_network import HopfNetwork

# few helpers 
def unit_vector(vector):
	""" Returns the unit vector of the vector.  """
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	""" Returns the angle in radians between vectors 'v1' and 'v2' """
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotation_matrix(theta):
	return np.array([ [np.cos(theta), -np.sin(theta) ], [np.sin(theta), np.cos(theta)] ])


ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
VIDEO_LOG_DIRECTORY = 'videos/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f")

# Implemented observation spaces for deep reinforcement learning: 
#   "DEFAULT":    motor angles and velocities, body orientation
#   "LR_COURSE_OBS":  [TODO: what should you include? what is reasonable to measure on the real system? CPG states?] 

# Tasks to be learned with reinforcement learning
#     - "FWD_LOCOMOTION"
#         reward forward progress only
#     - "FLAGRUN"
#         move to goal, once reached, a new goal is randomly selected.
#     - "LR_COURSE_TASK" 
#         [TODO: what should you train for?]
#         Ideally we want to command A1 to run in any direction while expending minimal energy
#         How will you construct your reward function? 

# Motor control modes:
#   - "TORQUE": 
#         supply raw torques to each motor (12)
#   - "PD": 
#         supply desired joint positions to each motor (12)
#         torques are computed based on the joint position/velocity error
#   - "CARTESIAN_PD": 
#         supply desired foot positions for each leg (12)
#         torques are computed based on the foot position/velocity error
#   - "CPG": 
#         supply desired CPG state modulations (8), mapped to foot positions
#         torques are computed based on inverse kinematics + joint PD (or you can add Cartesian PD)


EPISODE_LENGTH = 10   # how long before we reset the environment (max episode length for RL)
MAX_FWD_VELOCITY = 1  # to avoid exploiting simulator dynamics, cap max reward for body velocity 

# CPG quantities
MU_LOW = 1
MU_UPP = 2


class QuadrupedGymEnv(gym.Env):
  """The gym environment for a quadruped {Unitree A1}.

  It simulates the locomotion of a quadrupedal robot. 
  The state space, action space, and reward functions can be chosen with:
  observation_space_mode, motor_control_mode, task_env.
  """
  def __init__(
      self,
      robot_config=robot_config,
      isRLGymInterface=True,
      time_step=0.001,
      action_repeat=10,  
      motor_control_mode="CPG",
      task_env="FWD_LOCOMOTION",
      observation_space_mode="LR_COURSE_OBS",
      on_rack=False,
      render=False,
      record_video=False,
      add_noise=True,
      terrain="SLOPES",  
      test_flagrun=False, 
      **kwargs): # any extra arguments from legacy
    """Initialize the quadruped gym environment.

    Args:
      robot_config: The robot config file, contains A1 parameters.
      isRLGymInterface: If the gym environment is being run as RL or not. Affects
        if the actions should be scaled.
      time_step: Simulation time step.
      action_repeat: The number of simulation steps where the same actions are applied.
      motor_control_mode: Whether to use torque control, PD, control, CPG, etc.
      task_env: Task trying to learn (fwd locomotion, standup, etc.)
      observation_space_mode: what should be in here? Check available functions in quadruped.py
        also consider CPG states (amplitudes/phases)
      on_rack: Whether to place the quadruped on rack. This is only used to debug
        the walking gait. In this mode, the quadruped's base is hanged midair so
        that its walking gait is clearer to visualize.
      render: Whether to render the simulation.
      record_video: Whether to record a video of each trial.
      add_noise: vary coefficient of friction
      terrain: string indicating what kind of terrain ("STAIRS", "SLOPES", "GAPS", "RANDOM")
      test_flagrun: follow certain goals in order, fixed coefficient of friction 
    """
    self._robot_config = robot_config
    self._isRLGymInterface = isRLGymInterface
    self._time_step = time_step
    self._action_repeat = action_repeat
    self._motor_control_mode = motor_control_mode
    self._TASK_ENV = task_env
    self._observation_space_mode = observation_space_mode
    self._hard_reset = True # must fully reset simulation at init
    self._on_rack = on_rack
    self._is_render = render
    self._is_record_video = record_video
    self._add_noise = add_noise
    self._using_test_env = test_env
    self._test_flagrun = test_flagrun
    self.goal_id = None
    self._terrain = terrain
    if self._add_noise:
      self._observation_noise_stdev = 0.01 #
    else:
      self._observation_noise_stdev = 0.0

    # other bookkeeping 
    self._num_bullet_solver_iterations = int(300 / action_repeat) 
    self._env_step_counter = 0
    self._sim_step_counter = 0
    self._last_base_position = [0, 0, 0]
    self._last_frame_time = 0.0 # for rendering 
    self._MAX_EP_LEN = EPISODE_LENGTH # max sim time in seconds, arbitrary
    self._action_bound = 1.0

    # if using CPG
    self.setupCPG()

    self.setupActionSpace()
    self.setupObservationSpace()
    if self._is_render:
      self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
    else:
      self._pybullet_client = bc.BulletClient()
    self._configure_visualizer()

    self.videoLogID = None
    self.seed()
    self.reset()
 
  def setupCPG(self):
    self._cpg = HopfNetwork(use_RL=True)

  ######################################################################################
  # RL Observation and Action spaces 
  ######################################################################################
  def setupObservationSpace(self):
    """Set up observation space for RL. """
    if self._observation_space_mode == "DEFAULT":
      observation_high = (np.concatenate((self._robot_config.UPPER_ANGLE_JOINT,
                                         self._robot_config.VELOCITY_LIMITS,
                                         np.array([1.0]*4))) +  OBSERVATION_EPS)
      observation_low = (np.concatenate((self._robot_config.LOWER_ANGLE_JOINT,
                                         -self._robot_config.VELOCITY_LIMITS,
                                         np.array([-1.0]*4))) -  OBSERVATION_EPS)
    elif self._observation_space_mode == "LR_COURSE_OBS":
      # [TODO] Set observation upper and lower ranges. What are reasonable limits? 
      # Note 50 is arbitrary below, you may have more or less 
      # if using CPG-RL, remember to include limits on these
      observation_high = (np.concatenate((
          self._robot_config.UPPER_ANGLE_JOINT, # Joint angle limits
          self._robot_config.VELOCITY_LIMITS, # Velocity limits
          self._robot_config.TORQUE_LIMITS,    # Torque limits
          np.array([50.0, 50.0, 50.0]),                                                    # Position limits (x, y, z)
          np.array([1.0]*4),                                                              # Quaternion limits
          np.array([10.0, 10.0, 10.0]),                                                    # Linear velocity limits
          np.array([50.0, 50.0, 50.0]),                                                    # Angular velocity limits                             # Normal force limits
          np.array([1.0]*4),                                                                # Contact status (binary)
          np.array([4.0]*4),                                                                 # CPG states (r)
          np.array([2*np.pi]*4),                                                                 # CPG states (theta)
          np.array([50.0]*4),                                                               # CPG states (dr)
          np.array([50.0]*4)                                                                # CPG states (dtheta)
      )) + OBSERVATION_EPS)

      observation_low = (np.concatenate((
          self._robot_config.LOWER_ANGLE_JOINT,  # Joint angle limits
          -self._robot_config.VELOCITY_LIMITS, # Velocity limits
          -self._robot_config.TORQUE_LIMITS,    # Torque limits
          np.array([-50.0, -50.0, -50.0]),                                                  # Position limits (x, y, z)
          np.array([-1.0]*4),                                                               # Quaternion limits
          np.array([-10.0, -10.0, -10.0]),                                                  # Linear velocity limits
          np.array([-50.0, -50.0, -50.0]),                                                  # Angular velocity limits                               # Normal force limits
          np.array([0.0]*4),                            # Contact status (binary) 
          np.array([1.0]*4),                                                                  # CPG states (r)
          np.array([0.0]*4),                                                              # CPG states (theta)
          np.array([-50.0]*4),                                                                 # CPG states (dr)
          np.array([-50.0]*4)                                                                  # CPG states (dtheta)
      )) - OBSERVATION_EPS)

    else:
      raise ValueError("observation space not defined or not intended")

    self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

  def setupActionSpace(self):
    """ Set up action space for RL. """
    if self._motor_control_mode in ["PD","TORQUE", "CARTESIAN_PD"]:
      action_dim = 12
    elif self._motor_control_mode in ["CPG"]:
      action_dim = 8
    else:
      raise ValueError("motor control mode " + self._motor_control_mode + " not implemented yet.")
    action_high = np.array([1] * action_dim)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self._action_dim = action_dim


  def _get_observation(self):
    """Get observation, depending on obs space selected. """
    if self._observation_space_mode == "DEFAULT":
      self._observation = np.concatenate((self.robot.GetMotorAngles(), 
                                          self.robot.GetMotorVelocities(),
                                          self.robot.GetBaseOrientation() ))
    elif self._observation_space_mode == "LR_COURSE_OBS":
      # [TODO] Get observation from robot. What are reasonable measurements we could get on hardware?
      # if using the CPG, you can include states with self._cpg.get_r(), for example
      # 50 is arbitrary
      self._observation = np.concatenate((self.robot.GetMotorAngles(),                  # Motor angles
                                          self.robot.GetMotorVelocities(),              # Motor velocities
                                          self.robot.GetMotorTorques(),                 # Motor torques
                                          self.robot.GetBasePosition(),                 # Base position (x, y, z)
                                          self.robot.GetBaseOrientation(),              # Base orientation quaternion
                                          self.robot.GetBaseLinearVelocity(),           # Linear velocity of base
                                          self.robot.GetBaseAngularVelocity(),          # Angular velocity of base
                                          self.robot.GetContactInfo()[3],               # Boolean contact status for each foot
                                          self._cpg.get_r(),                            # CPG states (r)
                                          self._cpg.get_theta(),                        # CPG states (theta)
                                          self._cpg.get_dr(),                           # CPG states (dr)
                                          self._cpg.get_dtheta()))                      # CPG states (dtheta)                  

    else:
      raise ValueError("observation space not defined or not intended")

    self._add_obs_noise = (np.random.normal(scale=self._observation_noise_stdev, size=self._observation.shape) *
          self.observation_space.high)
    return self._observation

  def _noisy_observation(self):
    self._get_observation()
    observation = np.array(self._observation)
    if self._observation_noise_stdev > 0:
      observation += self._add_obs_noise
    return observation

  ######################################################################################
  # Termination and reward
  ######################################################################################
  def is_fallen(self,dot_prod_min=0.85):
    """Decide whether the quadruped has fallen.

    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), the quadruped is considered fallen.

    Returns:
      Boolean value that indicates whether the quadruped has fallen.
    """
    base_rpy = self.robot.GetBaseOrientationRollPitchYaw()
    orientation = self.robot.GetBaseOrientation()
    rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
    local_up = rot_mat[6:]
    pos = self.robot.GetBasePosition()
    return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < dot_prod_min or pos[2] < self._robot_config.IS_FALLEN_HEIGHT)

  def _termination(self):
    """Decide whether we should stop the episode and reset the environment. """
    return self.is_fallen() 

  def _reward_fwd_locomotion(self, des_vel_x=None):
    """Learn forward locomotion"""
    vel_tracking_reward = 0.1 * np.clip(self.robot.GetBaseLinearVelocity()[0], 0.2, 1.0)
    # If you want to track a desired velocity 
    # vel_tracking_reward = 0.05 * np.exp( -1/ 0.25 *  (self.robot.GetBaseLinearVelocity()[0] - des_vel_x)**2 )
    # minimize yaw (go straight)
    yaw_reward = -0.2 * np.abs(self.robot.GetBaseOrientationRollPitchYaw()[2]) 
    # don't drift laterally 
    drift_reward = -0.01 * abs(self.robot.GetBasePosition()[1]) 
    # minimize energy 
    energy_reward = 0 
    for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
      energy_reward += np.abs(np.dot(tau,vel)) * self._time_step

    reward = vel_tracking_reward \
            + yaw_reward \
            + drift_reward \
            - 0.01 * energy_reward \
            - 0.1 * np.linalg.norm(self.robot.GetBaseOrientation() - np.array([0,0,0,1]))

    return max(reward,0) # keep rewards positive

  def get_distance_and_angle_to_goal(self):
    """ Helper to return distance and angle to current goal location. """
    # current object location
    base_pos = self.robot.GetBasePosition()
    yaw = self.robot.GetBaseOrientationRollPitchYaw()[2]
    goal_vec = self._goal_location
    dist_to_goal = np.linalg.norm(base_pos[0:2]-goal_vec)

    # angle to goal (from current heading)
    body_dir_vec = np.matmul( rotation_matrix(yaw), np.array([[1],[0]]) )
    body_goal_vec = goal_vec - base_pos[0:2]
    body_dir_vec = body_dir_vec.reshape(2,)
    body_goal_vec = body_goal_vec.reshape(2,)

    Vn = unit_vector( np.array([0,0,1]) )
    c = np.cross( np.hstack([body_dir_vec,0]), np.hstack([body_goal_vec,0])  )
    angle = angle_between(body_dir_vec, body_goal_vec)
    angle = angle * np.sign( np.dot( Vn , c ) )

    return dist_to_goal, angle
  
  def _reward_flag_run(self):
    """ Learn to move towards goal location. """
    curr_dist_to_goal, angle = self.get_distance_and_angle_to_goal()

    # minimize distance to goal (we want to move towards the goal)
    dist_reward = 10 * ( self._prev_pos_to_goal - curr_dist_to_goal)
    # minimize yaw deviation to goal (necessary?)
    yaw_reward = 0 # -0.01 * np.abs(angle) 

    # minimize energy 
    energy_reward = 0 
    for tau,vel in zip(self._dt_motor_torques,self._dt_motor_velocities):
      energy_reward += np.abs(np.dot(tau,vel)) * self._time_step

    reward = dist_reward \
            + yaw_reward \
            - 0.001 * energy_reward 
    
    return max(reward,0) # keep rewards positive
    
  def _reward_lr_course(self):
      """Reward function for navigating a course with improved efficiency."""
      # [TODO] add your reward function. 
      # 1. Forward locomotion (default terms)
      vel_tracking_reward = np.clip(self.robot.GetBaseLinearVelocity()[0], 0.2, 1.0)
      
      # 2. Minimize yaw (go straight)
      yaw_reward = np.abs(self.robot.GetBaseOrientationRollPitchYaw()[2])

      # 3. Don't drift laterally
      drift_reward = abs(self.robot.GetBasePosition()[1])

      # 4. Minimize energy consumption
      energy_reward = 0
      for tau, vel in zip(self._dt_motor_torques, self._dt_motor_velocities):
          energy_reward += np.abs(np.dot(tau, vel)) * self._time_step

      # 5. Stability reward
      stability_penalty = np.linalg.norm(self.robot.GetBaseOrientation() - np.array([0, 0, 0, 1]))

      # Combine rewards
      reward = (
          0.01 * vel_tracking_reward +
          -0.2 * yaw_reward +
          -0.01 * drift_reward +
          - 0.001 * energy_reward +
          -0.1 * stability_penalty
      )

      return max(reward, 0)  # Keep rewards positive

  def _reward(self):
    """ Get reward depending on task"""
    if self._TASK_ENV == "FWD_LOCOMOTION":
      return self._reward_fwd_locomotion()
    elif self._TASK_ENV == "LR_COURSE_TASK":
      return self._reward_lr_course()
    elif self._TASK_ENV == "FLAGRUN":
      return self._reward_flag_run()
    else:
      raise ValueError("This task mode not implemented yet.")

  ######################################################################################
  # Step simulation, map policy network actions to joint commands, etc. 
  ######################################################################################
  def _transform_action_to_motor_command(self, action):
    """ Map actions from RL (i.e. in [-1,1]) to joint commands based on motor_control_mode. """
    # clip actions to action bounds
    action = np.clip(action, -self._action_bound - ACTION_EPS,self._action_bound + ACTION_EPS)
    if self._motor_control_mode == "PD":
      action = self._scale_helper(action, self._robot_config.LOWER_ANGLE_JOINT, self._robot_config.UPPER_ANGLE_JOINT)
      action = np.clip(action, self._robot_config.LOWER_ANGLE_JOINT, self._robot_config.UPPER_ANGLE_JOINT)
    elif self._motor_control_mode == "CARTESIAN_PD":
      action = self.ScaleActionToCartesianPos(action)
    elif self._motor_control_mode == "CPG":
      action = self.ScaleActionToCPGStateModulations(action)
    else:
      raise ValueError("RL motor control mode" + self._motor_control_mode + "not implemented yet.")
    return action

  def _scale_helper(self, action, lower_lim, upper_lim):
    """Helper to linearly scale from [-1,1] to lower/upper limits. """
    new_a = lower_lim + 0.5 * (action + 1) * (upper_lim - lower_lim)
    return np.clip(new_a, lower_lim, upper_lim)

  def ScaleActionToCartesianPos(self,actions):
    """Scale RL action to Cartesian PD ranges. 
    Edit ranges, limits etc., but make sure to use Cartesian PD to compute the torques. 
    """
    # clip RL actions to be between -1 and 1 (standard RL technique)
    u = np.clip(actions,-1,1)
    # scale to corresponding desired foot positions (i.e. ranges in x,y,z we allow the agent to choose foot positions)
    # [TODO: edit (do you think these should these be increased? How limiting is this?)]
    scale_array = np.array([0.3, 0.15, 0.24]*4) #i suggest we triple these dimensions scale_array = np.array([0.1, 0.05, 0.08]*4)--->scale_array = np.array([0.3, 0.15, 0.24]*4)
    # add to nominal foot position in leg frame (what are the final ranges?)
    des_foot_pos = self._robot_config.NOMINAL_FOOT_POS_LEG_FRAME + scale_array*u

    # get Cartesian kp and kd gains (can be modified)
    kpCartesian = self._robot_config.kpCartesian
    kdCartesian = self._robot_config.kdCartesian
    # get current motor velocities
    qd = self.robot.GetMotorVelocities()

    action = np.zeros(12)
    for i in range(4):
      # get Jacobian and foot position in leg frame for leg i (see ComputeJacobianAndPosition() in quadruped.py)
      # [TODO]
      J, current_foot_pos = self.robot.ComputeJacobianAndPosition(legID=i)
      # desired foot position i (from RL above)
      Pd = des_foot_pos[3 * i:3 * i + 3]  # Extract desired x, y, z for leg i # [TODO]
      # desired foot velocity i
      vd = np.zeros(3) #do we desire a particular foot velocity ? I read that to walk slowly we don't need to control it, but to run it would be nice, so let's start with 0
      # foot velocity in leg frame i (Equation 2) see pdf J*q=v
      # [TODO]
      foot_vel = np.dot(J, qd[3 * i:3 * i + 3])  # Joint velocities to foot velocity 

      # calculate torques with Cartesian PD (Equation 5) [Make sure you are using matrix multiplications] #τCartesian = JT(q)hKp,Cartesian (pd − p) + Kd,Cartesian (vd − v)i
      tau = np.dot(J.T, kpCartesian * (Pd - current_foot_pos) + kdCartesian * (vd - foot_vel)) # [TODO]

      action[3*i:3*i+3] = tau

    return action

  def ScaleActionToCPGStateModulations(self,actions):
    """Scale RL action to CPG modulation parameters."""
    # clip RL actions to be between -1 and 1 (standard RL technique)
    u = np.clip(actions,-1,1)

    # scale omega to ranges, and set in CPG (range is an example)
    omega = self._scale_helper( u[0:4], 5, 4.5*2*np.pi)
    self._cpg.set_omega_rl(omega)

    # scale mu to ranges, and set in CPG (squared since we converge to the sqrt in the CPG amplitude)
    mus = self._scale_helper( u[4:8], MU_LOW**2, MU_UPP**2)
    self._cpg.set_mu_rl(mus)

    # integrate CPG, get mapping to foot positions
    xs,zs = self._cpg.update()

    # IK parameters
    foot_y = self._robot_config.HIP_LINK_LENGTH
    sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)
    # get motor kp and kd gains (can be modified)
    kp = self._robot_config.MOTOR_KP # careful of size!
    kd = self._robot_config.MOTOR_KD
    # get current motor velocities
    q = self.robot.GetMotorAngles()
    dq = self.robot.GetMotorVelocities()

    action = np.zeros(12)
    # loop through each leg
    for i in range(4):
      # get desired foot i pos (xi, yi, zi)
      tau = np.zeros(3)
      x = xs[i]
      y = sideSign[i] * foot_y # careful of sign
      z = zs[i]

      leg_xyz = np.array([x,y,z])
      # call inverse kinematics to get corresponding joint angles
      q_des = self.robot.ComputeInverseKinematics(legID=i, xyz_coord=leg_xyz) # [TODO]
      # Add joint PD contribution to tau 
      tau += kp[3*i:3*i+3] * (q_des - q[3*i:3*i+3]) + kd[3*i:3*i+3] * (-dq[3*i:3*i+3])

      # add Cartesian PD contribution (as you wish) in the big formula we have both contribution this part would add the part from above (fucntion : ScaleActionToCartesianPos ) in a cpg state modulation: see new code:
                  #   # Cartesian PD contribution
                  # # Get current foot position and Jacobian for leg i
                  # J, current_foot_pos = self.robot.ComputeJacobianAndPosition(legID=i)

                  # # Compute position and velocity errors in Cartesian space
                  # position_error = desired_position - current_foot_pos  # Position error
                  # desired_velocity = np.zeros(3)  # Assuming desired foot velocity is zero
                  # current_velocity = np.dot(J, dq[3 * i:3 * i + 3])  # Current Cartesian velocity
                  # velocity_error = desired_velocity - current_velocity

                  # # Compute Cartesian forces
                  # F_cartesian = self._robot_config.kpCartesian @ position_error + \
                  #               self._robot_config.kdCartesian @ velocity_error

                  # # Map Cartesian forces to joint torques using the Jacobian transpose
                  # tau_cartesian = np.dot(J.T, F_cartesian)

                  # # Add Cartesian PD contribution to the total torque
                  # tau += tau_cartesian
     
      action[3*i:3*i+3] = tau

    return action


  def step(self, action):
    """ Step forward the simulation, given the action. """
    curr_act = action.copy()
    # save motor torques and velocities to compute power in reward function
    self._dt_motor_torques = []
    self._dt_motor_velocities = []
    if "FLAGRUN" in self._TASK_ENV:
      self._prev_pos_to_goal, _ = self.get_distance_and_angle_to_goal()
    
    for _ in range(self._action_repeat):
      if self._isRLGymInterface: 
        proc_action = self._transform_action_to_motor_command(curr_act)
      else:
        proc_action = curr_act 
      self.robot.ApplyAction(proc_action)
      self._pybullet_client.stepSimulation()
      self._sim_step_counter += 1
      self._dt_motor_torques.append(self.robot.GetMotorTorques())
      self._dt_motor_velocities.append(self.robot.GetMotorVelocities())

      if self._is_render:
        self._render_step_helper()

    self._last_action = curr_act
    self._env_step_counter += 1
    reward = self._reward()
    done = False
    if self._termination() or (self.get_sim_time() > self._MAX_EP_LEN and not self._test_flagrun ):
      done = True

    if "FLAGRUN" in self._TASK_ENV:
      dist_to_goal, _ = self.get_distance_and_angle_to_goal()
      if dist_to_goal < 0.5:
        self._reset_goal()

    return np.array(self._noisy_observation()), reward, done, {'base_pos': self.robot.GetBasePosition()} 

  ######################################################################################
  # Reset
  ######################################################################################
  def reset(self):
    """ Set up simulation environment. """
    mu_min = 0.5
    if self._hard_reset:
      # set up pybullet simulation
      self._pybullet_client.resetSimulation()
      self._pybullet_client.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
      self._pybullet_client.setTimeStep(self._time_step)
      self.plane = self._pybullet_client.loadURDF(pybullet_data.getDataPath()+"/plane.urdf", 
                                                  basePosition=[80,0,0]) # to extend available running space (shift)
      self._pybullet_client.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 0.9])
      self._pybullet_client.configureDebugVisualizer(
          self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, 0)
      self._pybullet_client.setGravity(0, 0, -9.8)
      if self._terrain == "GAPS":
        self._robot_config.INIT_POSITION[2] = 1.305
        self._robot_config.IS_FALLEN_HEIGHT = 1.18
      self.robot = (quadruped.Quadruped(pybullet_client=self._pybullet_client,
                                         robot_config=self._robot_config,
                                         motor_control_mode=self._motor_control_mode,
                                         on_rack=self._on_rack,
                                         render=self._is_render))
      self._ground_mu_k = 1
      if self._add_noise:
        ground_mu_k = mu_min+(1-mu_min)*np.random.random()
        self._ground_mu_k = ground_mu_k
        self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=ground_mu_k)
        self._add_base_mass_offset()
        if self._is_render:
          print('ground friction coefficient is', ground_mu_k)

      if self._terrain is not None:
        if self._terrain == "SLOPES":
          self.add_slopes(pitch=0.2)
        elif self._terrain == "STAIRS":
          self.add_stairs(num_stairs=12, stair_height=0.05, stair_width=0.25)
        elif self._terrain == "GAPS":
          self.add_gaps(num_gaps=5, gap_width=0.1, between_gaps_width=2)
        elif self._terrain == "RANDOM":
          self.add_random_boxes()
        else:
          print('Terrain',self._terrain,'is not implemented')

      elif self._TASK_ENV == "FLAGRUN":
        self.goal_id = None
        if self._test_flagrun:
          self._ground_mu_k = ground_mu_k = 0.8
          self._pybullet_client.changeDynamics(self.plane, -1, lateralFriction=ground_mu_k)
          self._add_noise = False 
          self._goal_idx = 0
          self.goal_x = np.arange(np.pi/4, 11, np.pi/2)
          self.goal_y = 0.2 * self.goal_x * np.sin(2*self.goal_x)
        self._reset_goal()

    else:
      self.robot.Reset(reload_urdf=False)

    self.setupCPG()
    self._env_step_counter = 0
    self._sim_step_counter = 0
    self._last_base_position = [0, 0, 0]

    if self._is_render:
      self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                       self._cam_pitch, [0, 0, 0])

    self._settle_robot()
    self._last_action = np.zeros(self._action_dim)
    if self._is_record_video:
      self.recordVideoHelper()
    return self._noisy_observation()

  def _reset_goal(self):
    """Reset goal location. """
    try:
      if self.goal_id is not None: 
        self._pybullet_client.removeBody(self.goal_id)
    except:
      pass
    
    if self._test_flagrun:
      self._goal_location = np.array([self.goal_x[self._goal_idx], 
                                      self.goal_y[self._goal_idx]])
      self._goal_idx = min(self._goal_idx+1, len(self.goal_x))
    else:
      self._goal_location = 6 * (np.random.random((2,)) - 0.5) 
      self._goal_location += self.robot.GetBasePosition()[0:2]
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[0.2,0.2,0.2])
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    self.goal_id=self._pybullet_client.createMultiBody(
                          baseMass=0,
                          baseCollisionShapeIndex = sh_colBox,
                          basePosition = [self._goal_location[0],self._goal_location[1],0.6],
                          baseOrientation=orn)
    # print('goal is at ', self._goal_location)

  def _settle_robot(self):
    """ Settle robot and add noise to init configuration. """
    # change to PD control mode to set initial position, then set back..
    tmp_save_motor_control_mode_ENV = self._motor_control_mode
    tmp_save_motor_control_mode_ROB = self.robot._motor_control_mode
    self._motor_control_mode = "PD"
    self.robot._motor_control_mode = "PD"
    try:
      tmp_save_motor_control_mode_MOT = self.robot._motor_model._motor_control_mode
      self.robot._motor_model._motor_control_mode = "PD"
    except:
      pass
    init_motor_angles = self._robot_config.INIT_MOTOR_ANGLES + self._robot_config.JOINT_OFFSETS
    if self._is_render:
      time.sleep(0.2)
    for _ in range(1000):
      self.robot.ApplyAction(init_motor_angles)
      if self._is_render:
        time.sleep(0.001)
      self._pybullet_client.stepSimulation()
    
    # set control mode back
    self._motor_control_mode = tmp_save_motor_control_mode_ENV
    self.robot._motor_control_mode = tmp_save_motor_control_mode_ROB
    try:
      self.robot._motor_model._motor_control_mode = tmp_save_motor_control_mode_MOT
    except:
      pass

  ######################################################################################
  # Render, record videos, bookkeping, and misc pybullet helpers.  
  ######################################################################################
  def startRecordingVideo(self,name):
    self.videoLogID = self._pybullet_client.startStateLogging(
                            self._pybullet_client.STATE_LOGGING_VIDEO_MP4, 
                            name)

  def stopRecordingVideo(self):
    self._pybullet_client.stopStateLogging(self.videoLogID)

  def close(self):
    if self._is_record_video:
      self.stopRecordingVideo()
    self._pybullet_client.disconnect()

  def recordVideoHelper(self, extra_filename=None):
    """ Helper to record video, if not already, or end and start a new one """
    # If no ID, this is the first video, so make a directory and start logging
    if self.videoLogID == None:
      directoryName = VIDEO_LOG_DIRECTORY
      assert isinstance(directoryName, str)
      os.makedirs(directoryName, exist_ok=True)
      self.videoDirectory = directoryName
    else:
      # stop recording and record a new one
      self.stopRecordingVideo()

    if extra_filename is not None:
      output_video_filename = self.videoDirectory + '/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") +extra_filename+ ".MP4"
    else:
      output_video_filename = self.videoDirectory + '/' + datetime.datetime.now().strftime("vid-%Y-%m-%d-%H-%M-%S-%f") + ".MP4"
    logID = self.startRecordingVideo(output_video_filename)
    self.videoLogID = logID


  def configure(self, args):
    self._args = args

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _render_step_helper(self):
    """ Helper to configure the visualizer camera during step(). """
    # Sleep, otherwise the computation takes less time than real time,
    # which will make the visualization like a fast-forward video.
    time_spent = time.time() - self._last_frame_time
    self._last_frame_time = time.time()
    # time_to_sleep = self._action_repeat * self._time_step - time_spent
    time_to_sleep = self._time_step - time_spent
    if time_to_sleep > 0 and (time_to_sleep < self._time_step):
      time.sleep(time_to_sleep)
      
    base_pos = self.robot.GetBasePosition()
    camInfo = self._pybullet_client.getDebugVisualizerCamera()
    curTargetPos = camInfo[11]
    distance = camInfo[10]
    yaw = camInfo[8]
    pitch = camInfo[9]
    targetPos = [
        0.95 * curTargetPos[0] + 0.05 * base_pos[0], 0.95 * curTargetPos[1] + 0.05 * base_pos[1],
        curTargetPos[2]
    ]
    self._pybullet_client.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)

  def _configure_visualizer(self):
    """ Remove all visualizer borders, and zoom in """
    # default rendering options
    self._render_width = 960
    self._render_height = 720
    self._cam_dist = 1.0 
    self._cam_yaw = 0
    self._cam_pitch = -30 
    # get rid of visualizer things
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
    self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_GUI,0)

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos = self.robot.GetBasePosition()
    view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                   aspect=float(self._render_width) /
                                                                   self._render_height,
                                                                   nearVal=0.1,
                                                                   farVal=100.0)
    (_, _, px, _,
     _) = self._pybullet_client.getCameraImage(width=self._render_width,
                                               height=self._render_height,
                                               viewMatrix=view_matrix,
                                               projectionMatrix=proj_matrix,
                                               renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def addLine(self,lineFromXYZ,lineToXYZ,lifeTime=0,color=[1,0,0]):
    """ Add line between point A and B for duration lifeTime"""
    self._pybullet_client.addUserDebugLine(lineFromXYZ,
                                            lineToXYZ,
                                            lineColorRGB=color,
                                            lifeTime=lifeTime)

  def get_sim_time(self):
    """ Get current simulation time. """
    return self._sim_step_counter * self._time_step

  def scale_rand(self,num_rand,low,high):
    """ scale number of rand numbers between low and high """
    return low + np.random.random(num_rand) * (high - low)

  def add_random_boxes(self, num_rand=100, z_height=0.04):
    """Add random boxes in front of the robot in x [0.5, 20] and y [-3,3] """
    # x location
    x_low, x_upp = 0.5, 20
    # y location
    y_low, y_upp = -3, 3
    # block dimensions
    block_x_min, block_x_max = 0.1, 1
    block_y_min, block_y_max = 0.1, 1
    z_low, z_upp = 0.005, z_height
    # block orientations
    roll_low, roll_upp = -0.01, 0.01
    pitch_low, pitch_upp = -0.01, 0.01 
    yaw_low, yaw_upp = -np.pi, np.pi

    x = x_low + np.random.random(num_rand) * (x_upp - x_low)
    y = y_low + np.random.random(num_rand) * (y_upp - y_low)
    z = z_low + np.random.random(num_rand) * (z_upp - z_low)
    block_x = self.scale_rand(num_rand,block_x_min,block_x_max)
    block_y = self.scale_rand(num_rand,block_y_min,block_y_max)
    roll = self.scale_rand(num_rand,roll_low,roll_upp)
    pitch = self.scale_rand(num_rand,pitch_low,pitch_upp)
    yaw = self.scale_rand(num_rand,yaw_low,yaw_upp)
    # loop through
    for i in range(num_rand):
      sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
          halfExtents=[block_x[i]/2,block_y[i]/2,z[i]/2])
      orn = self._pybullet_client.getQuaternionFromEuler([roll[i],pitch[i],yaw[i]])
      block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [x[i],y[i],z[i]/2],baseOrientation=orn)
      # set friction coeff
      self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

    # add walls 
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[x_upp/2,0.5,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,y_low,0.5],baseOrientation=orn)
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,-y_low,0.5],baseOrientation=orn)


  def add_gaps(self, num_gaps=5, gap_width=0.1, between_gaps_width=2):
    """Add N gaps
      -each gap is gap_width wide
      -platforms between gaps are between_gaps_width wide"""
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    # start platform
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
            halfExtents=[2,1,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                              basePosition = [0,0,0.5],baseOrientation=orn)
    # set friction coeff to 1
    self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

    first_gap = 2
    block_0 = first_gap + gap_width + between_gaps_width / 2
    # keep track of gaps (possibly for RL observation space!)
    self._gap_centers = np.zeros(num_gaps) 

    # loop through
    for i in range(num_gaps):
      self._gap_centers[i] = first_gap + gap_width / 2 + i*between_gaps_width
      block_x = block_0 + i * (gap_width + between_gaps_width)
      sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
          halfExtents=[between_gaps_width / 2, 1, 0.5])
      
      block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [block_x,0,0.5],baseOrientation=orn)
      # set friction coeff to 1
      self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)
    # print("gaps are centered at", self._gap_centers)

    # end platform 
    end_platform_size = 2
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
            halfExtents=[end_platform_size,1,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                              basePosition = [block_x+between_gaps_width/2+end_platform_size/2,0,0.5],baseOrientation=orn)
    # set friction coeff to 1
    self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)
  
  def add_stairs(self, num_stairs=12, stair_height=0.05, stair_width=0.25):
    """Add N stairs, with stair_height and stair_width. long so can't get around """
    x_upp = 20
    y_low = -3
    y = 6
    curr_z = 0 
    block_x = stair_width * np.ones(num_stairs)
    curr_x = 1
    # loop through
    for i in range(num_stairs):
      if i < num_stairs / 2:
        curr_z += stair_height
      else:
        curr_z -= stair_height
      if curr_z > 0:
        sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
            halfExtents=[block_x[i]/2,y/2,curr_z/2])
        orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
        block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                              basePosition = [curr_x,0,curr_z/2],baseOrientation=orn)
        # set friction coeff to 1
        self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

      curr_x += block_x[i]

    # add walls 
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[x_upp/2,0.5,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,y_low,0.5],baseOrientation=orn)
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,-y_low,0.5],baseOrientation=orn)

  def add_slopes(self, pitch=0.2):
    """Add slopes with platform in center."""
    y = 6
    slope_len = 2
    box_width = 1
    slope_height = 0.01

    # add first slope UP
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[slope_len/2,y/2,slope_height])
    orn = self._pybullet_client.getQuaternionFromEuler([0,-pitch,0])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex=sh_colBox,
        basePosition = [1+slope_len/2,0,slope_len/2*np.sin(pitch) - slope_height*np.cos(pitch) ],baseOrientation=orn)
    self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

    # add middle box
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[box_width/2,y/2,slope_len/2*np.sin(pitch)])
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex=sh_colBox,
        basePosition = [1+slope_len*np.cos(pitch)+box_width/2,0,slope_len/2*np.sin(pitch)  ],baseOrientation=orn) # + slope_height/2*np.cos(pitch)
    self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

    # add descending box
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[slope_len/2,y/2,slope_height])
    orn = self._pybullet_client.getQuaternionFromEuler([0,pitch,0])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex=sh_colBox,
        basePosition = [1+slope_len*np.cos(pitch)+box_width + slope_len/2 + 2*slope_height*np.sin(-pitch),0,slope_len/2*np.sin(pitch) - slope_height*np.cos(pitch) ],baseOrientation=orn) # + slope_height/2*np.cos(pitch)
    self._pybullet_client.changeDynamics(block2, -1, lateralFriction=self._ground_mu_k)

    self._add_walls()

  def _add_walls(self,x_upp=20,y_low=-3):
    # add walls 
    orn = self._pybullet_client.getQuaternionFromEuler([0,0,0])
    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
        halfExtents=[x_upp/2,0.25,0.5])
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,y_low,0.5],baseOrientation=orn)
    block2=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                          basePosition = [x_upp/2,-y_low,0.5],baseOrientation=orn)

  def _add_base_mass_offset(self, spec_mass=None, spec_location=None):
    """Attach mass to robot base."""
    quad_base = np.array(self.robot.GetBasePosition())
    quad_ID = self.robot.quadruped

    offset_low = np.array([-0.15, -0.05, -0.05])
    offset_upp = np.array([ 0.15,  0.05,  0.05])
    if spec_location is None:
      block_pos_delta_base_frame = self.scale_rand(3,offset_low,offset_upp)
    else:
      block_pos_delta_base_frame = np.array(spec_location)
    if spec_mass is None:
      base_mass = 8*np.random.random()
    else:
      base_mass = spec_mass
    if self._is_render:
      print('=========================== Random Mass:')
      print('Mass:', base_mass, 'location:', block_pos_delta_base_frame)
      # if rendering, also want to set the halfExtents accordingly 
      # 1 kg water is 0.001 cubic meters 
      boxSizeHalf = [(base_mass*0.001)**(1/3) / 2]*3
      translationalOffset = [0,0,0.1]
    else:
      boxSizeHalf = [0.05]*3
      translationalOffset = [0]*3

    sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX, 
                      halfExtents=boxSizeHalf, collisionFramePosition=translationalOffset)
    base_block_ID=self._pybullet_client.createMultiBody(baseMass=base_mass,
                                    baseCollisionShapeIndex = sh_colBox,
                                    basePosition = quad_base + block_pos_delta_base_frame,
                                    baseOrientation=[0,0,0,1])

    cid = self._pybullet_client.createConstraint(quad_ID, -1, base_block_ID, -1, 
          self._pybullet_client.JOINT_FIXED, [0, 0, 0], [0, 0, 0], -block_pos_delta_base_frame)
    # disable self collision between box and each link
    for i in range(-1,self._pybullet_client.getNumJoints(quad_ID)):
      self._pybullet_client.setCollisionFilterPair(quad_ID,base_block_ID, i,-1, 0)


def test_env():
  env = QuadrupedGymEnv(render=True, 
                        on_rack=True,
                        motor_control_mode='PD',
                        action_repeat=100,
                        )

  obs = env.reset()
  print('obs len', len(obs))
  action_dim = env._action_dim
  action_low = -np.ones(action_dim)
  print('act len', action_dim)
  action = action_low.copy()
  while True:
    action = 2*np.random.rand(action_dim)-1
    obs, reward, done, info = env.step(action)


if __name__ == "__main__":
  # test out some functionalities
  test_env()
  sys.exit()
