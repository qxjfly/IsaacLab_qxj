# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG, Wukong4_MINIMAL_CFG, CR01A_MINIMAL_CFG, CR01A_noarm_MINIMAL_CFG # isort: skip
from isaaclab_assets import CR01ADC_MINIMAL_CFG, CR01ADC_noarm_MINIMAL_CFG # 

@configclass
class CR1ARewards(RewardsCfg):
    """Reward terms for the MDP."""
    #终止惩罚
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
        func=mdp.feet_air_time_positive_biped,#feet_air_time_positive_biped  #feet_air_time
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,# initial threshold = 0.4
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
        # params={"asset_cfg": SceneEntityCfg("robot", joint_names=["right_wrist_roll_joint"])},
    )


    # 惩罚hip关节roll yaw ankle关节 roll偏离默认值
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_ankle_roll_joint"])},
    )
    
    #惩罚arm关节偏离默认值
    # joint_deviation_arms = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_shoulder_pitch_joint",
    #                 ".*_shoulder_roll_joint",
    #                 ".*_shoulder_yaw_joint",
    #                 ".*_elbow_pitch_joint",
    #                 ".*_wrist_.*",
    #             ],
    #         )
    #     },
    # )

    # joint_deviation_fingers = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_five_joint",
    #                 ".*_three_joint",
    #                 ".*_six_joint",
    #                 ".*_four_joint",
    #                 ".*_zero_joint",
    #                 ".*_one_joint",
    #                 ".*_two_joint",
    #             ],
    #         )
    #     },
    # )

    #惩罚torso关节偏离默认值
    # joint_deviation_torso = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    # )

    #迈步腾空高度奖励
    feet_air_height = RewTerm(
        func=mdp.feet_air_height_exp,
        weight= 0.25, #0.1, #0.25, #0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 0.3,
            "std": 0.5,
        },
    ) 

    #惩罚root_link高度
    root_height = RewTerm(
        func=mdp.root_height_l1,
        weight=-0.85, #-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "threshold": 0.84,
        },
    )

    #奖励 hip_pitch_ang  迈步触地时的角度
    # step_hip_pitch_ang = RewTerm(
    #     func=mdp.step_hip_pitch_ang,
    #     weight=0.1,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.5,# initial threshold = 0.5 步态周期
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch_joint"),
    #     },
    # )

    #惩罚低速迈腿
    feet_air_height_lowvel = RewTerm(
        func=mdp.feet_air_height_lowvel,
        weight=-50,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 0.051, 
            "std": 1.0,
        },
    )

    #在低速时惩罚leg_pitch_joint关节偏离默认值
    joint_deviation_hip_zero = RewTerm(
        func=mdp.joint_pos_lowvel,
        weight=-2.5,
        params={"command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint"])},
    )

    # 奖励腾空时间>期望 稀疏奖励
    # feet_fly = RewTerm(
    #     func=mdp.feet_fly,
    #     weight=0.5,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.2,# initial threshold = 0.4
    #     },
    # )

    # 腾空时间  对称  奖励 
    # feet_sym = RewTerm(
    #     func=mdp.feet_symmetric,
    #     weight=0.25,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.05,# initial threshold = 0.4
    #     },
    # )

    #双脚腾空惩罚
    feet_both_air = RewTerm(
        func=mdp.feet_both_air,
        weight=-50,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    #步长奖励
    step_distance = RewTerm(
        func=mdp.feet_step_distance2,
        weight=0.1,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 0.5,# initial threshold = 0.5 步态周期
            
        },
    )

    #惩罚knee2knee不对称
    # joint_deviation_knee2knee = RewTerm(
    #     func=mdp.joint_deviation_knee2knee,
    #     weight=-0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
    # )

    # 惩罚迈步knee angle 不跟踪期望
    step_knee = RewTerm(
        func=mdp.feet_step_knee,
        weight=-0.85, #-0.85,#-0.5, #-0.25, #-0.025,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint"),
            "threshold": 0.1,#
        },
    )
    # step_knee2 = RewTerm(
    #     func=mdp.feet_step_knee2,
    #     weight=0.001, #-0.85,#-0.5, #-0.25, #-0.025,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint"),
    #         "threshold": 0.08,#
    #     },
    # )

    #惩罚roll关节力矩法
    joint_roll_torque_l2 = RewTerm(
        func=mdp.joint_torques_roll_l2,
        weight=-4.0e-5,# default -5.0e-5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )

    #惩罚ankle_pitch 不平行身体
    joint_deviation_ankle_pitch = RewTerm(
        func=mdp.joint_deviation_anklepitch_l1,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint"])},
    )

    #迈步接触地面力的奖励
    feet_contact_force = RewTerm(
        func=mdp.feet_contact_force_exp,
        weight= 0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "threshold": 0.15,
            "std": 500,
        },
    ) 

    # 
    # feet_step_symmetric_knee = RewTerm(
    #     func=mdp.feet_step_symmetric_knee,
    #     weight=-0.1, #-0.025,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint"),
    #         "threshold": 0.08,#
    #     },
    # )

    feet_step_symmetric_hip = RewTerm(
        func=mdp.feet_step_symmetric_hip,
        weight=-0.05, #-0.025,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch_joint"),
            "threshold": 0.08,#
        },
    )

    feet_step_current_airtime = RewTerm(
        func=mdp.feet_step_current_airtime,
        weight=-20, # -20, #1.0, # -20 #-0.025,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_pitch_joint"),
            "threshold": 0.6,#
        },
    )
    # 新增步态周期一致性奖励
    # step_phase_consistency = RewTerm(
    #     func=mdp.step_phase_consistency,
    #     weight=1.5,  # 增加周期一致性奖励
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "phase_threshold": 0.2,  # 允许20%的相位误差
    #     },
    # )
    
    joint_knee_pos_l2 = RewTerm(
        func=mdp.joint_knee_pos_l2,
        weight=-2,# default -5.0e-5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},
    )

    # step_length_reward = RewTerm(
    #     func=mdp.reward_step_length,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "target_range": (0.4, 0.5),
    #         "reward_scale": 1.0,
    #     },
    # )
    
    # 奖励摆动腿在身体前方
    # feet_swing_pos = RewTerm(
    #     func=mdp.feet_swing_pos,
    #     weight=0.1,
    #     params={
    #         "command_name": "base_velocity",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #         "threshold": 0.4,# initial threshold = 0.5 步态周期
            
    #     },
    # )

@configclass
class CR1ARoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: CR1ARewards = CR1ARewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        # self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link" #高度观测器的 base_link
        # self.scene.robot = Wukong4_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = CR01A_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = CR01A_noarm_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.robot = CR01ADC_noarm_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link" #高度观测器的 base_link

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
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
        self.rewards.flat_orientation_l2.weight = -3.5 # default: -1.0
        self.rewards.action_rate_l2.weight = -0.005
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
        self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        # self.terminations.arm_contact.params["sensor_cfg"].body_names = ".*_shoulder_.*"
        # self.terminations.elbow_contact.params["sensor_cfg"].body_names = ".*_elbow_.*"

@configclass
class CR1ARoughEnvCfg_PLAY(CR1ARoughEnvCfg):
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