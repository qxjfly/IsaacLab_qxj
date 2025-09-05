# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.cr1b.mirror_cfg import MirrorCfg
##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG, Wukong4_MINIMAL_CFG, CR01A_MINIMAL_CFG, CR01A_RL_CFG, CR01A_noarm_MINIMAL_CFG # isort: skip
from isaaclab_assets import CR01ADC_MINIMAL_CFG, CR01ADC_noarm_MINIMAL_CFG, CR01B_RL_CFG # 

@configclass
class CR1BRewards(RewardsCfg):
    """Reward terms for the MDP."""
    # 终止惩罚
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0) 

    # 线速度xy跟踪奖励
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # 角速度z跟踪奖励
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # 迈步腾空时间奖励
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
            "vel_threshold": 0.1,
        },
    )

    # 支撑脚部滑动惩罚
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    #惩罚踝关节超限
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )


    # 惩罚hip关节roll yaw ankle关节 roll偏离默认值
    joint_deviation_leg_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_ankle_roll_joint"])},
    )
    # 惩罚hip关节roll yaw ankle关节 roll偏离默认值
    joint_deviation_leg_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )
    
    #惩罚arm关节偏离默认值
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )

    #惩罚腰部关节偏离默认值
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*")},
    )
    ##########################################################################
    #迈步腾空高度奖励
    feet_air_height = RewTerm(
        func=mdp.feet_air_height_exp,
        weight= 0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 0.12, 
            "std": 0.5,
        },
    ) 

    #惩罚root_link高度
    root_height = RewTerm(
        func=mdp.root_height_l1,
        weight=-1.5, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "threshold": 0.89, 
        },
    )

    #惩罚低速迈腿
    feet_air_height_lowvel = RewTerm(
        func=mdp.feet_air_height_lowvel,
        weight=-50,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 0.07, #0.051
        },
    )

    #在低速时惩罚leg_pitch_joint关节偏离默认值
    joint_deviation_zero_lowvel = RewTerm(
        func=mdp.joint_pos_lowvel,
        weight=-0.5,
        params={"command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*" )},
    )

    #双脚腾空惩罚
    feet_both_air = RewTerm(
        func=mdp.feet_both_air,
        weight=-50,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    #############################################################################
    #步长奖励
    step_distance = RewTerm(
        func=mdp.feet_step_distance_zq,
        weight= 0.1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    #############################################################################
    # 惩罚迈步knee angle 不跟踪期望
    step_knee = RewTerm(
        func=mdp.feet_step_knee,
        weight=-2.5, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint"),
            "threshold": 0.05,
        },
    )
    #############################################################################
    #惩罚roll关节力矩法
    joint_hip_roll_torque_l2 = RewTerm(
        func=mdp.joint_torques_hip_roll_l2,
        weight=-3.0e-5,#  -4.0e-5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    #惩罚ankle_pitch关节力矩法
    joint_ankle_pitch_torque_l2 = RewTerm(
        func=mdp.joint_torques_hip_roll_l2,
        weight=-8.0e-6,#  -4.0e-5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"])},
    )

    #惩罚ankle_pitch 不平行身体
    joint_parallel_ankle_pitch = RewTerm(
        func=mdp.joint_parallel_anklepitch_l1,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint"])},
    )

    feet_step_current_airtime = RewTerm(
        func=mdp.feet_step_current_airtime,
        weight=-20, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch_joint"),
            "threshold": 0.55,#
        },
    )

    #############################################################################
    joint_knee_pos_limit_l2 = RewTerm(
        func=mdp.joint_knee_pos_limit,
        weight=-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
                "max": 1.5,
                "min":-0.15},
    )

    #############################################################################
    # 奖励摆动腿在身体前方
    feet_swing_pos = RewTerm(
        func=mdp.feet_stand_pos3,
        weight= 0.15, 
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 0.2, 
        },
    )

    #踏步力1
    # reward_feet_contact_forces = RewTerm(
    #     func=mdp.silent_single_leg_landing,
    #     weight=0.1,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "force_thresh": 880, # 
    #         "speed_adapt_ratio": 0.15,# 
    #     },
    # )
    #踏步力2
    # reward_feet_contact_forces = RewTerm(
    #     func=mdp.feet_contact_forces,
    #     weight=-0.02,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "force_thresh": 800, # A620 B800
    #     },
    # )
    #踏步力3
    reward_feet_contact_vel = RewTerm(
        func=mdp.feet_contact_vel,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    #############################################################################
    # reward_swing_arm_tracking = RewTerm(
    #     func=mdp.reward_swing_arm_tracking,
    #     weight=0.5, #0.35,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_pitch_joint"),
    #     },
    # )

    reward_joint_coordination_hip = RewTerm(
        func=mdp.joint_coordination,
        weight=-0.5,
        params={
            "joint1_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint"]), 
            "joint2_cfg": SceneEntityCfg("robot", joint_names=["right_hip_pitch_joint"]),
            "ratio": -1.0,
        },
    )
    joint_coordination_larm_leg = RewTerm(
        func=mdp.joint_coordination,
        weight=-0.5,
        params={
            "joint1_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_pitch_joint"]), 
            "joint2_cfg": SceneEntityCfg("robot", joint_names=["right_hip_pitch_joint"]),
            "ratio": 1.0, 
        },
    )
    joint_coordination_rarm_leg = RewTerm(
        func=mdp.joint_coordination,
        weight=-0.5,
        params={
            "joint1_cfg": SceneEntityCfg("robot", joint_names=["right_shoulder_pitch_joint"]), 
            "joint2_cfg": SceneEntityCfg("robot", joint_names=["left_hip_pitch_joint"]),
            "ratio": 1.0, 
        },
    )
    # reward_no_double_feet_air = RewTerm(
    #     func=mdp.biped_no_double_feet_air,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    # )

    # body distance
    distance_feet = RewTerm(
        func=mdp.lateral_distance,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), 
            "min_dist": 0.26, # 0.26
            "max_dist": 0.40},
    )
    body_ang_vel_xy_l2 = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["body_link"])},
    )
    
    
@configclass
class CR1BMirrorCfg(MirrorCfg):
    
    def __init__(self):
        # Joint IDs
        lhipPitch = 0
        rhipPitch = 1
        waistYaw = 2
        lhipRoll = 3
        rhipRoll = 4
        waistRoll = 5
        lhipYaw = 6
        rhipYaw = 7
        waistPitch = 8
        lknee = 9
        rknee = 10
        lshoulderPitch = 11
        rshoulderPitch = 12
        lankle1 = 13
        rankle1 = 14
        lshoulderRoll = 15
        rshoulderRoll = 16
        lankle2 = 17
        rankle2 = 18
        lshoulderYaw = 19
        rshoulderYaw = 20
        lelbowPitch = 21
        relbowPitch = 22
        lwristYaw = 23
        rwristYaw = 24
        lwristPitch = 25
        rwristPitch = 26
        lwristRoll = 27
        rwristRoll = 28

        JOINT_NUM = 29

        lhipPitch_a = 0
        lshoulderPitch_a = 1
        rhipPitch_a = 2
        rshoulderPitch_a = 3
        lhipRoll_a = 4
        rhipRoll_a = 5
        lhipYaw_a = 6
        rhipYaw_a = 7
        lknee_a = 8
        lelbowPitch_a = 9
        rknee_a = 10
        relbowPitch_a = 11
        lankle1_a = 12
        rankle1_a = 13
        lankle2_a = 14
        rankle2_a = 15

        # 关节默认排序 对应的aciton id
        # lhipPitch_a = 0
        # rhipPitch_a = 1
        # lhipRoll_a = 2
        # rhipRoll_a = 3
        # lhipYaw_a = 4
        # rhipYaw_a = 5
        # lknee_a = 6
        # rknee_a = 7
        # lshoulderPitch_a = 8
        # rshoulderPitch_a = 9
        # lankle1_a = 10
        # rankle1_a = 11
        # lankle2_a = 12
        # rankle2_a = 13
        # lelbowPitch_a = 14
        # relbowPitch_a = 15

        left_joint_ids  = [
            lhipPitch,
            lhipRoll,
            lhipYaw,
            lknee,
            lankle1,
            lankle2,
        ]
        right_joint_ids = [
            rhipPitch,
            rhipRoll,
            rhipYaw,
            rknee,
            rankle1,
            rankle2,
        ]

        left_joint_ids_a  = [
            lhipPitch_a,
            lhipRoll_a,
            lhipYaw_a,
            lknee_a,
            lankle1_a,
            lankle2_a,
        ]

        right_joint_ids_a = [
            rhipPitch_a,
            rhipRoll_a,
            rhipYaw_a,
            rknee_a,
            rankle1_a,
            rankle2_a,
        ]

        self.action_mirror_id_left  = left_joint_ids_a.copy()
        self.action_mirror_id_right = right_joint_ids_a.copy()

        self.action_opposite_id = [
            lhipRoll,
            rhipRoll,
            lhipYaw,
            rhipYaw,
            lankle2,
            rankle2,
        ]
        self.action_opposite_id_a = [
            lhipRoll_a,
            rhipRoll_a,
            lhipYaw_a,
            rhipYaw_a,
            lankle2_a,
            rankle2_a,
        ]

        BASE_ANG_VEL      = 0
        PROJECTED_GRAVITY = BASE_ANG_VEL + 3
        VELOCITY_COMMANDS = PROJECTED_GRAVITY + 3
        JOINT_POS         = VELOCITY_COMMANDS + 3
        JOINT_VEL         = JOINT_POS + JOINT_NUM
        ACTIONS           = JOINT_VEL + JOINT_NUM

        self.policy_obvs_mirror_id_left = [JOINT_POS + joint_id for joint_id in left_joint_ids]
        self.policy_obvs_mirror_id_left.extend([JOINT_VEL + joint_id for joint_id in left_joint_ids])
        self.policy_obvs_mirror_id_left.extend([ACTIONS + joint_id for joint_id in self.action_mirror_id_left])
        self.policy_obvs_mirror_id_right = [JOINT_POS + joint_id for joint_id in right_joint_ids]
        self.policy_obvs_mirror_id_right.extend([JOINT_VEL + joint_id for joint_id in right_joint_ids])
        self.policy_obvs_mirror_id_right.extend([ACTIONS + joint_id for joint_id in self.action_mirror_id_right])
        
        self.policy_obvs_opposite_id = [
            BASE_ANG_VEL + 0,
            BASE_ANG_VEL + 2,
            PROJECTED_GRAVITY + 1,
            VELOCITY_COMMANDS + 1,
            VELOCITY_COMMANDS + 2,
        ]
        self.policy_obvs_opposite_id.extend([JOINT_POS + joint_id for joint_id in self.action_opposite_id])
        self.policy_obvs_opposite_id.extend([JOINT_VEL + joint_id for joint_id in self.action_opposite_id])
        self.policy_obvs_opposite_id.extend([ACTIONS   + joint_id for joint_id in self.action_opposite_id_a])

@configclass
class CR1BRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: CR1BRewards = CR1BRewards()
    mirror_cfg: CR1BMirrorCfg = CR1BMirrorCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        # self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link" #高度观测器的 base_link
        # self.scene.robot = Wukong4_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = CR01B_RL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = CR01A_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = CR01A_noarm_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = CR01ADC_noarm_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link" #高度观测器的 base_link

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None

        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "body_link"
        self.terminations.arm_contact.params["sensor_cfg"].body_names = ".*_shoulder_.*"
        self.terminations.elbow_contact.params["sensor_cfg"].body_names = ".*_elbow_.*"

@configclass
class CR1BRoughEnvCfg_PLAY(CR1BRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # self.events.reset_base = None