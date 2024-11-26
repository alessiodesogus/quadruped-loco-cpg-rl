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

"""Defines the A1 robot related constants and URDF specs."""
import numpy as np
import re
import pybullet as pyb
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

URDF_ROOT = parentdir 
URDF_FILENAME = "a1_description/urdf/a1.urdf"

##################################################################################
# Default robot configuration (i.e. base and joint positions, etc.)
##################################################################################
NUM_MOTORS = 12
NUM_LEGS = 4
MOTORS_PER_LEG = 3

INIT_RACK_POSITION = [0, 0, 1] # when hung up in air (for debugging)
INIT_POSITION = [0, 0, 0.305]  # normal initial height
IS_FALLEN_HEIGHT = 0.18        # height at which robot is considered fallen

INIT_ORIENTATION = (0, 0, 0, 1) 
_, INIT_ORIENTATION_INV = pyb.invertTransform(
        position=[0, 0, 0], orientation=INIT_ORIENTATION)

# default angles (for init)
DEFAULT_HIP_ANGLE = 0
DEFAULT_THIGH_ANGLE = np.pi/4 
DEFAULT_CALF_ANGLE = -np.pi/2 
INIT_JOINT_ANGLES = np.array([  DEFAULT_HIP_ANGLE, 
                                DEFAULT_THIGH_ANGLE, 
                                DEFAULT_CALF_ANGLE] * NUM_LEGS)
INIT_MOTOR_ANGLES = INIT_JOINT_ANGLES
# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_DIRECTIONS = np.array([1, 1, 1, 1, 1, 1, 
                             1, 1, 1, 1, 1, 1])

# joint offsets 
HIP_JOINT_OFFSET = 0.0
THIGH_JOINT_OFFSET = 0.0
CALF_JOINT_OFFSET = 0.0

# Used to convert the robot SDK joint angles to URDF joint angles.
JOINT_OFFSETS = np.array([  HIP_JOINT_OFFSET, 
                            THIGH_JOINT_OFFSET,
                            CALF_JOINT_OFFSET] * NUM_LEGS)

# Kinematics
HIP_LINK_LENGTH = 0.0838
THIGH_LINK_LENGTH = 0.2
CALF_LINK_LENGTH = 0.2

NOMINAL_FOOT_POS_LEG_FRAME = np.array([ 0, -HIP_LINK_LENGTH, -0.25,
                                        0,  HIP_LINK_LENGTH, -0.25,
                                        0, -HIP_LINK_LENGTH, -0.25,
                                        0,  HIP_LINK_LENGTH, -0.25])

##################################################################################
# Actuation limits/gains, position, and velocity limits
##################################################################################
# joint limits on real system
UPPER_ANGLE_JOINT = np.array([ 0.802851455917,  4.18879020479, -0.916297857297 ] * NUM_LEGS)
LOWER_ANGLE_JOINT = np.array([-0.802851455917, -1.0471975512 , -2.69653369433  ] * NUM_LEGS)

# modified range in simulation (min observation space for RL)
UPPER_ANGLE_JOINT = np.array([ 0.2,  DEFAULT_THIGH_ANGLE + 0.4, DEFAULT_CALF_ANGLE + 0.4 ] * NUM_LEGS)
LOWER_ANGLE_JOINT = np.array([-0.2,  DEFAULT_THIGH_ANGLE - 0.4, DEFAULT_CALF_ANGLE - 0.4 ] * NUM_LEGS)

# torque and velocity limits 
TORQUE_LIMITS   = np.asarray( [33.5] * NUM_MOTORS )
VELOCITY_LIMITS = np.asarray( [21.0] * NUM_MOTORS ) 

# Sample Joint Gains
MOTOR_KP = [100.0, 100.0, 100.0] * NUM_LEGS
MOTOR_KD = [2.0, 2.0, 2.0] * NUM_LEGS

# MOTOR_KP = [55,55,55] * NUM_LEGS
# MOTOR_KD = [0.8,0.8,0.8] * NUM_LEGS

# Sample Cartesian Gains
kpCartesian = np.diag([500,500,500])
kdCartesian = np.diag([10,10,10])

kpCartesian = np.diag([700,700,700])
kdCartesian = np.diag([12,12,12])

# for simulation only 
kpCartesian = np.diag([1000,1000,1000])
kdCartesian = np.diag([20,20,20])

##################################################################################
# Hip, thigh, calf strings, naming conventions from URDF (don't modify)
##################################################################################
JOINT_NAMES = (
    # front right leg
    "FR_hip_joint", 
    "FR_thigh_joint", 
    "FR_calf_joint",
    # front left leg
    "FL_hip_joint", 
    "FL_thigh_joint", 
    "FL_calf_joint",
    # rear right leg
    "RR_hip_joint", 
    "RR_thigh_joint", 
    "RR_calf_joint",
    # rear left leg
    "RL_hip_joint", 
    "RL_thigh_joint", 
    "RL_calf_joint",
)
MOTOR_NAMES = JOINT_NAMES

# standard across all robots
_CHASSIS_NAME_PATTERN = re.compile(r"\w*floating_base\w*")
_HIP_NAME_PATTERN = re.compile(r"\w+_hip_j\w+")
_THIGH_NAME_PATTERN = re.compile(r"\w+_thigh_j\w+")
_CALF_NAME_PATTERN = re.compile(r"\w+_calf_j\w+")
_FOOT_NAME_PATTERN = re.compile(r"\w+_foot_\w+")
