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
This file implements the functionalities of a quadruped using pybullet.
"""
import os
import numpy as np
import time
import re
import quadruped_motor

class Quadruped(object):
  """The quadruped class to simulate a quadruped robot.
  """
  def __init__(self,
               pybullet_client,
               robot_config=None,
               self_collision_enabled=True,
               accurate_motor_model_enabled=True,
               motor_control_mode="PD",
               on_rack=False,
               render=False):
    """Construct a quadruped and reset it to the initial states.

    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
      robot_config: File with all of the relevant configs for the desired 
        quadruped.
      self_collision_enabled: Whether to enable self collision.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      motor_control_mode: either torque or position control
      on_rack: Whether to place the quadruped on rack. This is only used to debug
        the walking gait. In this mode, the quaruped's base is hanged midair so
        that its walking gait is clearer to visualize.
    """
    self._robot_config = robot_config
    self.num_motors = self._robot_config.NUM_MOTORS
    self.num_legs = self._robot_config.NUM_LEGS 
    self._pybullet_client = pybullet_client
    self._urdf_root = self._robot_config.URDF_ROOT # urdf_root
    self._self_collision_enabled = self_collision_enabled
    self._motor_direction = self._robot_config.JOINT_DIRECTIONS
    self._observed_motor_torques = np.zeros(self.num_motors)
    self._applied_motor_torques = np.zeros(self.num_motors)
    self._accurate_motor_model_enabled = accurate_motor_model_enabled

    # motor control mode for accurate motor model, should only be torque or position at this low level
    if motor_control_mode == "PD":
      self._motor_control_mode = "PD"
    else:
      self._motor_control_mode = "TORQUE"

    self._on_rack = on_rack
    self.render = render
    if self._accurate_motor_model_enabled:
      self._kp = self._robot_config.MOTOR_KP 
      self._kd = self._robot_config.MOTOR_KD
      self._motor_model = quadruped_motor.QuadrupedMotorModel(
                                    motor_control_mode=self._motor_control_mode,
                                    kp=self._kp,
                                    kd=self._kd,
                                    torque_limits=self._robot_config.TORQUE_LIMITS)
    else:
      raise ValueError("Must use accurate motor model")

    self.Reset(reload_urdf=True)


  ######################################################################################
  # Robot states and observation related
  ######################################################################################
  def _GetDefaultInitPosition(self):
    """Get the default initial position of the quadruped's base, to reset simulation

    Returns:
      The initial xyz position of the quadruped's base, either fixed in air, or
        initialized on ground.
    """
    if self._on_rack:
      return self._robot_config.INIT_RACK_POSITION
    else:
      return self._robot_config.INIT_POSITION

  def _GetDefaultInitOrientation(self):
    return self._robot_config.INIT_ORIENTATION

  def GetBasePosition(self):
    """Get the position of the quadruped's base.

    Returns:
      The world xyz position of the quadruped's base.
    """
    position, _ = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    return position

  def GetBaseOrientation(self):
    """Get the orientation of the quadruped's base, represented as quaternion.

    Returns:
      The orientation (quaternion) of the quadruped's base.
    """
    _, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.quadruped))
    _, orientation = self._pybullet_client.multiplyTransforms(
      positionA=[0, 0, 0],
      orientationA=orientation,
      positionB=[0, 0, 0],
      orientationB=self._robot_config.INIT_ORIENTATION_INV)
    return orientation

  def GetBaseOrientationRollPitchYaw(self):
    """Get quadruped's base orientation in euler angle in the world frame.

    Returns:
       (roll, pitch, yaw) of the base in world frame.
    """
    orientation = self.GetBaseOrientation()
    roll_pitch_yaw = self._pybullet_client.getEulerFromQuaternion(orientation)
    return np.asarray(roll_pitch_yaw)

  def GetTrueBaseRollPitchYawRate(self):
    """Get the rate of orientation change of the quadruped's base in euler angle.

    Returns:
      rate of (roll, pitch, yaw) change of the quadruped's base.
    """
    angular_velocity = self.GetBaseAngularVelocity()
    orientation = self.GetBaseOrientation()
    return self.TransformAngularVelocityToLocalFrame(angular_velocity, orientation)

  def TransformAngularVelocityToLocalFrame(self, angular_velocity, orientation):
    """Transform the angular velocity from world frame to robot's frame.

    Args:
    angular_velocity: Angular velocity of the robot in world frame.
    orientation: Orientation of the robot represented as a quaternion.

    Returns:
    angular velocity of based on the given orientation.
    """
    # Treat angular velocity as a position vector, then transform based on the
    # orientation given by dividing (or multiplying with inverse).
    # Get inverse quaternion assuming the vector is at 0,0,0 origin.
    _, orientation_inversed = self._pybullet_client.invertTransform([0, 0, 0], orientation)
    # Transform the angular_velocity at neutral orientation using a neutral
    # translation and reverse of the given orientation.
    relative_velocity, _ = self._pybullet_client.multiplyTransforms([0, 0, 0], 
                            orientation_inversed, 
                            angular_velocity,
                             self._pybullet_client.getQuaternionFromEuler([0, 0, 0]))
    return np.asarray(relative_velocity)

  def GetBaseOrientationMatrix(self):
    """ Get the base orientation matrix (3x3), as numpy array """
    baseOrn = self.GetBaseOrientation()
    return np.asarray(self._pybullet_client.getMatrixFromQuaternion(baseOrn)).reshape((3,3))

  def GetBaseLinearVelocity(self):
    """ Get base linear velocities (dx, dy, dz) """
    linVel,_ = self._pybullet_client.getBaseVelocity(self.quadruped)
    return np.asarray(linVel)

  def GetBaseAngularVelocity(self):
    """ Get base angular velocities (droll, dpitch, dyaw) """
    _, angVel = self._pybullet_client.getBaseVelocity(self.quadruped)
    return np.asarray(angVel)

  def GetMotorAngles(self):
    """Get quadruped motor angles at the current moment.

    Returns:
      Motor angles.
    """
    motor_angles = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[0]
        for motor_id in self._motor_id_list
    ]
    motor_angles = np.multiply(motor_angles, self._motor_direction)
    return motor_angles

  def GetMotorVelocities(self):
    """Get the velocity of all motors.

    Returns:
      Velocities of all motors.
    """
    motor_velocities = [
        self._pybullet_client.getJointState(self.quadruped, motor_id)[1]
        for motor_id in self._motor_id_list
    ]
    motor_velocities = np.multiply(motor_velocities, self._motor_direction)
    return motor_velocities

  def GetMotorTorques(self):
    """Get the torques the motors are exerting.

    Returns:
      Motor torques of all motors.
    """
    if self._accurate_motor_model_enabled:
      return self._observed_motor_torques
    else:
      motor_torques = [
          self._pybullet_client.getJointState(self.quadruped, motor_id)[3]
          for motor_id in self._motor_id_list
      ]
      motor_torques = np.multiply(motor_torques, self._motor_direction)
    return motor_torques


  def GetContactInfo(self):
    """ Returns 
    (1) num valid contacts (only those between a foot and the ground)
    (2) num invalid contacts (either between a limb and the ground (other than the feet) or any self collisions
    (3) normal force at each foot (0 if in the air) 
    (4) boolean for each foot in contact (1) or not (0)
    """
    numValidContacts = 0
    numInvalidContacts = 0
    feetNormalForces = [0,0,0,0]
    feetInContactBool = [0,0,0,0]
    for c in self._pybullet_client.getContactPoints():
      # if bodyUniqueIdA is same as bodyUniqueIdB, self collision
      if c[1] == c[2]:
        # but only check calves for now
        if ( (c[3] in self._calf_ids ) or (c[4] in self._calf_ids)):
          numInvalidContacts += 1
        continue
      # if thighs in contact with anything, this is a failing condition
      if c[3] in self._thigh_ids or c[4] in self._thigh_ids:
        numInvalidContacts += 1
        continue
      # check link indices, one MUST be a foot index, and the other must be -1 (plane)
      # depending on sim and pybullet version, this can happen in either order
      if ( c[3] == -1 and c[4] not in self._foot_link_ids ) or \
         ( c[4] == -1 and c[3] not in self._foot_link_ids ):
        numInvalidContacts += 1
      else:
        numValidContacts += 1
        try:
          footIndex = self._foot_link_ids.index(c[4])
        except:
          footIndex = self._foot_link_ids.index(c[3])
        feetNormalForces[footIndex] += c[9] # if multiple contact locations
        feetInContactBool[footIndex] = 1
    return numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool

  ######################################################################################
  # INPUTS: set torques, ApplyAction, etc.
  ######################################################################################
  def _SetMotorTorqueById(self, motor_id, torque):
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.TORQUE_CONTROL,
                                                force=torque)

  def _SetDesiredMotorAngleById(self, motor_id, desired_angle): 
    # get max_force 
    max_force = self._robot_config.TORQUE_LIMITS[self._joint_ids.index(motor_id)]
    max_vel = self._robot_config.VELOCITY_LIMITS[self._joint_ids.index(motor_id)]
    self._pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.POSITION_CONTROL,
                                                targetPosition=desired_angle,
                                                force=max_force,
                                                maxVelocity=max_vel,
                                                )

  def _SetDesiredMotorAngleByName(self, motor_name, desired_angle):
    self._SetDesiredMotorAngleById(self._joint_name_to_id[motor_name], desired_angle)

  def ApplyAction(self, motor_commands):
    """Apply the desired motor torques to the motors of the quadruped.

    Args:
      motor_commands: The desired motor angles or torques, depending on control mode.
    """
    if self._accurate_motor_model_enabled:
      q = self.GetMotorAngles()
      qdot = self.GetMotorVelocities()

      actual_torque, observed_torque = self._motor_model.convert_to_torque(
          motor_commands, q, qdot)
      self._observed_motor_torques = observed_torque

      # Transform into the motor space when applying the torque.
      self._applied_motor_torque = np.multiply(actual_torque, self._motor_direction)

      for motor_id, motor_torque, motor_enabled in zip(self._motor_id_list,
                                                       self._applied_motor_torque,
                                                       self._motor_enabled_list):
        if motor_enabled:
          self._SetMotorTorqueById(motor_id, motor_torque)
        else:
          self._SetMotorTorqueById(motor_id, 0)


  ######################################################################################
  # Jacobian, IK, etc. 
  ######################################################################################
  def ComputeJacobianAndPosition(self, legID):
    """ Get Jacobian and foot position of leg legID. 
    Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;
    """
    # joint positions of leg legID
    q = self.GetMotorAngles()[legID*3:legID*3+3]

    # rename links
    l1 = self._robot_config.HIP_LINK_LENGTH
    l2 = self._robot_config.THIGH_LINK_LENGTH
    l3 = self._robot_config.CALF_LINK_LENGTH

    sideSign = 1
    if legID == 0 or legID == 2:
      sideSign = -1

    s1 = np.sin(q[0])
    s2 = np.sin(q[1])
    s3 = np.sin(q[2])

    c1 = np.cos(q[0])
    c2 = np.cos(q[1])
    c3 = np.cos(q[2])

    c23 = c2 * c3 - s2 * s3
    s23 = s2 * c3 + c2 * s3

    J = np.zeros((3,3))

    J[1, 0] = -sideSign * l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
    J[2, 0] =  sideSign * l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
    J[0, 1] = -l3 * c23 - l2 * c2
    J[1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
    J[2, 1] = l2 * s2 * c1 + l3 * s23 * c1
    J[0, 2] = -l3 * c23
    J[1, 2] = -l3 * s23 *s1
    J[2, 2] = l3 * s23 * c1  

    # foot pos
    pos = np.zeros(3)
    pos[0] = -l3 * s23 - l2 * s2
    pos[1] = l1 * sideSign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
    pos[2] = l1 * sideSign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

    return J, pos

  def ComputeInverseKinematics(self,legID, xyz_coord):
    """ Get joint angles for leg legID with desired xyz position in leg frame. 

    Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;

    From SpotMicro: 
    https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spotmicro/Kinematics/LegKinematics.py
    """
    # rename links
    shoulder_length = self._robot_config.HIP_LINK_LENGTH
    elbow_length = self._robot_config.THIGH_LINK_LENGTH
    wrist_length =  self._robot_config.CALF_LINK_LENGTH
    # coords
    x = xyz_coord[0]
    y = xyz_coord[1]
    z = xyz_coord[2]

    # get_domain
    D = (y**2 + (-z)**2 - shoulder_length**2 +
        (-x)**2 - elbow_length**2 - wrist_length**2) / (
                 2 * wrist_length * elbow_length)

    D = np.clip(D, -1.0, 1.0)

    # check Right vs Left leg for hip angle
    sideSign = 1
    if legID == 0 or legID == 2:
      sideSign = -1

    # Right Leg Inverse Kinematics Solver
    wrist_angle = np.arctan2(-np.sqrt(1 - D**2), D)
    sqrt_component = y**2 + (-z)**2 - shoulder_length**2
    if sqrt_component < 0.0:
        sqrt_component = 0.0
    shoulder_angle = -np.arctan2(z, y) - np.arctan2(
        np.sqrt(sqrt_component), sideSign*shoulder_length)
    elbow_angle = np.arctan2(-x, np.sqrt(sqrt_component)) - np.arctan2(
        wrist_length * np.sin(wrist_angle),
        elbow_length + wrist_length * np.cos(wrist_angle))
    joint_angles = np.array([-shoulder_angle, elbow_angle, wrist_angle])
    return joint_angles

  ######################################################################################
  # RESET related
  ######################################################################################
  def Reset(self, reload_urdf=False):
    """Reset the quadruped to its initial states.

    Args:
      reload_urdf: Whether to reload the urdf file. If not, Reset() just place
        the quadruped back to its starting position.
    """
    if reload_urdf:
      self._LoadRobotURDF()
      self._BuildJointNameToIdDict()
      self._BuildUrdfIds()
      self._RemoveDefaultJointDamping()
      self._SetLateralFriction()
      self._BuildMotorIdList()
      self._RecordMassAndInertiaInfoFromURDF()
      self._SetMaxJointVelocities()

      self.ResetPose(add_constraint=True)
      if self._on_rack:
        self._pybullet_client.createConstraint(self.quadruped, -1, -1, -1,
                                               self._pybullet_client.JOINT_FIXED, [0, 0, 0],
                                               [0, 0, 0], [0, 0, 1], childFrameOrientation=self._GetDefaultInitOrientation())
    else:
      self._pybullet_client.resetBasePositionAndOrientation(self.quadruped, 
                                                            self._GetDefaultInitPosition(),
                                                            self._GetDefaultInitOrientation())
      self._pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
      self.ResetPose(add_constraint=False)

    self._overheat_counter = np.zeros(self.num_motors)
    self._motor_enabled_list = [True] * self.num_motors


  def ResetPose(self, add_constraint):
    """From laikago.py """
    del add_constraint
    for name in self._joint_name_to_id:
      joint_id = self._joint_name_to_id[name]
      self._pybullet_client.setJointMotorControl2(
          bodyIndex=self.quadruped,
          jointIndex=(joint_id),
          controlMode=self._pybullet_client.VELOCITY_CONTROL,
          targetVelocity=0,
          force=0)
    for i, jointId in enumerate(self._joint_ids):
      angle = self._robot_config.INIT_MOTOR_ANGLES[i] + self._robot_config.JOINT_OFFSETS[i]
      self._pybullet_client.resetJointState(
          self.quadruped, jointId, angle, targetVelocity=0)

  ######################################################################################
  # URDF related
  ######################################################################################
  def _LoadRobotURDF(self):
    """Loads the URDF file for the robot."""
    urdf_file =  os.path.join(self._urdf_root, self._robot_config.URDF_FILENAME)
    if self._self_collision_enabled:
      self.quadruped = self._pybullet_client.loadURDF(
          urdf_file,
          self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation(),
          flags=self._pybullet_client.URDF_USE_SELF_COLLISION)
    else:
      self.quadruped = self._pybullet_client.loadURDF(
          urdf_file, self._GetDefaultInitPosition(),
          self._GetDefaultInitOrientation())

  def _BuildUrdfIds(self):
    """Build the link Ids from its name in the URDF file.

    Raises:
      ValueError: Unknown category of the joint name.
    """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._chassis_link_ids = [-1] # just base link
    self._leg_link_ids = []   # all leg links (hip, thigh, calf)
    self._motor_link_ids = [] # all leg links (hip, thigh, calf)

    self._joint_ids=[]      # all motor joints
    self._hip_ids = []      # hip joint indices only
    self._thigh_ids = []    # thigh joint indices only
    self._calf_ids = []     # calf joint indices only
    self._foot_link_ids = [] # foot joint indices

    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      # print(joint_info)
      joint_name = joint_info[1].decode("UTF-8")
      joint_id = self._joint_name_to_id[joint_name]
      if self._robot_config._CHASSIS_NAME_PATTERN.match(joint_name):
        self._chassis_link_ids = [joint_id] #.append(joint_id) # _should_ be only one!
      elif self._robot_config._HIP_NAME_PATTERN.match(joint_name):
        self._hip_ids.append(joint_id)
      elif self._robot_config._THIGH_NAME_PATTERN.match(joint_name):
        self._thigh_ids.append(joint_id)
      elif self._robot_config._CALF_NAME_PATTERN.match(joint_name):
        self._calf_ids.append(joint_id)
      elif self._robot_config._FOOT_NAME_PATTERN.match(joint_name):
        self._foot_link_ids.append(joint_id)
      else:
        continue
        raise ValueError("Unknown category of joint %s" % joint_name)


    # print('chassis',self._chassis_link_ids)
    # for i in range(-1,num_joints):
    #   print(self._pybullet_client.getDynamicsInfo(self.quadruped, i))

    # everything associated with the leg links
    self._joint_ids.extend(self._hip_ids)
    self._joint_ids.extend(self._thigh_ids)
    self._joint_ids.extend(self._calf_ids)
    # sort in case any weird order
    self._joint_ids.sort()
    self._hip_ids.sort()
    self._thigh_ids.sort()
    self._calf_ids.sort()
    self._foot_link_ids.sort()

    # print('joint ids', self._joint_ids)
    # print('hip ids', self._hip_ids)
    # print('_thigh_ids', self._thigh_ids)
    # print('_calf_ids', self._calf_ids)
    # print('_foot_link_ids', self._foot_link_ids)

  def _RecordMassAndInertiaInfoFromURDF(self):
    """Records the mass information from the URDF file.
    Divide into base, leg, foot, and total (all links)
    """
    # base 
    self._base_mass_urdf = []
    self._base_inertia_urdf = []

    for chassis_id in self._chassis_link_ids:
      self._base_mass_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[0])
      self._base_inertia_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, chassis_id)[2])

    # leg masses
    self._leg_masses_urdf = []
    self._leg_inertia_urdf = []
    for leg_id in self._joint_ids:
      self._leg_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[0])
      self._leg_inertia_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, leg_id)[2])

    # foot masses
    self._foot_masses_urdf = []
    self._foot_inertia_urdf = []
    for foot_id in self._foot_link_ids:
      self._foot_masses_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, foot_id)[0])
      self._foot_inertia_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped, foot_id)[2])


    # all masses present in URDF
    self._total_mass_urdf = []
    self._total_inertia_urdf = []
    # don't forget base (which is at -1)
    for j in range(-1,self._pybullet_client.getNumJoints(self.quadruped)):
      self._total_mass_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped,j)[0])
      self._total_inertia_urdf.append(
          self._pybullet_client.getDynamicsInfo(self.quadruped,j)[2])

    # set originals as needed
    self._original_base_mass_urdf = tuple(self._base_mass_urdf)
    self._original_leg_masses_urdf = tuple(self._leg_masses_urdf)
    self._original_foot_masses_urdf = tuple(self._foot_masses_urdf)
    self._original_total_mass_urdf = tuple(self._total_mass_urdf)

    self._original_base_inertia_urdf = tuple(self._base_inertia_urdf)
    self._original_leg_inertia_urdf = tuple(self._leg_inertia_urdf)
    self._original_foot_inertia_urdf = tuple(self._foot_inertia_urdf)
    self._original_total_inertia_urdf = tuple(self._total_inertia_urdf)

    #print('total mass', sum(self._total_mass_urdf))

  def _BuildJointNameToIdDict(self):
    """_BuildJointNameToIdDict """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    self._joint_name_to_id = {}
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

  def _BuildMotorIdList(self):
    self._motor_id_list = self._joint_ids

  def _RemoveDefaultJointDamping(self):
    """Pybullet convention/necessity  """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    for i in range(num_joints):
      joint_info = self._pybullet_client.getJointInfo(self.quadruped, i)
      self._pybullet_client.changeDynamics(
          joint_info[0], -1, linearDamping=0, angularDamping=0)

  def _SetLateralFriction(self,lateral_friction=1.0):
    """ Lateral friction to be set for every link. 
    NOTE: pybullet default is 0.5 - so call to set to 1 as default
    """
    num_joints = self._pybullet_client.getNumJoints(self.quadruped)
    for i in range(-1,num_joints):
      self._pybullet_client.changeDynamics(self.quadruped, i, lateralFriction=lateral_friction)

  def _SetMaxJointVelocities(self):
    """Set maximum joint velocities from robot_config, the pybullet default is 100 rad/s """
    for i, link_id in enumerate(self._joint_ids):
      self._pybullet_client.changeDynamics(self.quadruped, link_id, maxJointVelocity=self._robot_config.VELOCITY_LIMITS[i]) 

  def GetBaseMassFromURDF(self):
    """Get the mass of the base from the URDF file."""
    return self._base_mass_urdf

  def GetLegMassesFromURDF(self):
    """Get the mass of the legs from the URDF file."""
    return self._leg_masses_urdf

  def GetFootMassesFromURDF(self):
    """Get the mass of the feet from the URDF file."""
    return self._foot_masses_urdf

  def GetTotalMassFromURDF(self):
    """Get the total mass from all links in the URDF."""
    return self._total_mass_urdf