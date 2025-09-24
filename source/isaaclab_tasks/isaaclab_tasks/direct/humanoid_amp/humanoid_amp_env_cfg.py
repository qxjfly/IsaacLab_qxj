# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_assets import HUMANOID_28_CFG,CR01A_noarm_CFG,CR01A_amp_CFG,CR01B_amp_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")


@configclass
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""
    # task reward cfg qxj add
    lin_vel_reward_scale = 1.0
    ang_vel_reward_scale = -0.1 #-0.05
    joint_torque_reward_scale = -2.5e-6 #-2.5e-5
    joint_acc_reward_scale = -1.0e-6 #-5.5e-7
    action_rate_reward_scale = -0.15 # -0.01 
    flat_orientation_reward_scale = -0.5 # -5.0 
    termination_reward_scale = -0.05
    deviation_pos_scale = -0.01 #-0.05
    feet_slide_reward_scale = -0.25
    # none
    rew_termination = -0
    rew_action_l2 = -0.00
    rew_joint_pos_limits = -0
    rew_joint_acc_l2 =-0.00
    rew_joint_vel_l2= -0.00
    # 
    # env
    episode_length_s = 10.0 #baseline 10.0s  sideflip 4s
    decimation = 2

    # spaces
    observation_space = 125 #84 #77#49  #default:81 = 28jointpos+28jointvel+3pos+4quat+3lin+3ang+keybody*3pos
    action_space = 29 #26 #12 #default:28
    num_actions_cr1a = 29 #26 #12
    state_space = 0
    num_amp_observations = 3 #default:2 
    amp_observation_space = 95 #83 #49 #default:81

    early_termination = True
    termination_height = 0.4  # sideflip 0.25

    motion_file: str = MISSING #
    reference_body = "base_link" #"pelvis" #"torso"  usd中的base_link的name
    reference_bodym = "base_link" #"base_link" K188usd  motion_file中base_link的name
    reset_strategy = "default"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60, #default 60
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)
    # qxj_add_contact_sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.01, track_air_time=True
    )
    # CR01a  noarm***************************************************************************
    # robot: ArticulationCfg = CR01A_noarm_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace()
    # robot: ArticulationCfg = CR01A_amp_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace()
    robot: ArticulationCfg = CR01B_amp_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace()
    # robot
    # robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
    #     actuators={
    #         "body": ImplicitActuatorCfg(
    #             joint_names_expr=[".*"],
    #             velocity_limit_sim=100.0,#100 
    #             # stiffness=None,
    #             # damping=None,
    #             armature=0.01,
    #             # effort_limit_sim={
    #             #     "abdomen_.*": 250.0,
    #             #     "neck_.*": 20.0,
    #             #     "right_shoulder_.*": 70.0,
    #             #     "right_elbow": 60.0,
    #             #     "left_shoulder_.*": 70.0,
    #             #     "left_elbow": 60.0,
    #             #     "right_hip_.*": 150.0,
    #             #     "right_knee": 150.0,
    #             #     "right_ankle_.*": 100.0,
    #             #     "left_hip_.*": 150.0,
    #             #     "left_knee": 150.0,
    #             #     "left_ankle_.*": 100.0,
    #             # },
    #             effort_limit_sim={
    #                 "abdomen_.*": 250.0,
    #                 "neck_.*": 20.0,
    #                 "right_shoulder_.*": 70.0,
    #                 "right_elbow": 60.0,
    #                 "left_shoulder_.*": 70.0,
    #                 "left_elbow": 60.0,
    #                 "right_hip_.*": 300.0,
    #                 "right_knee": 300.0,
    #                 "right_ankle_.*": 200.0,
    #                 "left_hip_.*": 300.0,
    #                 "left_knee": 300.0,
    #                 "left_ankle_.*": 200.0,
    #             },
    #             stiffness={
    #                 "abdomen_.*": 200.0,
    #                 "neck_.*": 50.0,
    #                 "right_shoulder_.*": 200.0,
    #                 "right_elbow": 150.0,
    #                 "left_shoulder_.*": 200.0,
    #                 "left_elbow": 150.0,
    #                 "right_hip_.*": 300.0,
    #                 "right_knee": 300.0,
    #                 "right_ankle_.*": 200.0,
    #                 "left_hip_.*": 300.0,
    #                 "left_knee": 300.0,
    #                 "left_ankle_.*": 200.0,
    #             },
    #             damping={
    #                 "abdomen_.*": 5.0,
    #                 "neck_.*": 5.0,
    #                 "right_shoulder_.*": 20.0,
    #                 "right_elbow": 20.0,
    #                 "left_shoulder_.*": 20.0,
    #                 "left_elbow": 20.0,
    #                 "right_hip_.*": 10.0,
    #                 "right_knee": 10.0,
    #                 "right_ankle_.*": 10.0,
    #                 "left_hip_.*": 10.0,
    #                 "left_knee": 10.0,
    #                 "left_ankle_.*": 10.0,
    #             },
    #         ),
    #     },
    


@configclass
class HumanoidAmpDanceEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_dance.npz")


@configclass
class HumanoidAmpRunEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_run.npz")


@configclass
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk_cr01b4.npz")#humanoid_walk_cr01a6
