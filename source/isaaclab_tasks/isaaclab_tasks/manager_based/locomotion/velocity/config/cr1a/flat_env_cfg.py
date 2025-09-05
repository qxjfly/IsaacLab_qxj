# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import CR1ARoughEnvCfg


@configclass
class CR1AFlatEnvCfg(CR1ARoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        # robot_base
        self.rewards.track_lin_vel_xy_exp.weight = 1.5 #qxj
        self.rewards.track_ang_vel_z_exp.weight = 1.0 
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.5 #default -0.05
        self.rewards.flat_orientation_l2.weight = -4.5 #
        self.rewards.root_height.weight = -1.5
        self.rewards.root_height.params["threshold"] = 0.89

        # Joint deviation
        self.rewards.joint_deviation_hip.weight = -0.5 # default -0.5  hip:yaw roll && ankle: roll
        self.rewards.joint_deviation_arms.weight = -0.5

        # Joint coordination  存在奖励项
        self.rewards.reward_joint_coordination_hip.weight = -0.5

        # Joint limits
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.joint_hip_roll_torque_l2.weight = -3.0e-5

        # Gait related rewards
        self.rewards.feet_slide.weight = -0.25 #qxj 
        self.rewards.feet_air_time.weight = 1.5 #default 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.45 #default 0.4 s 
        self.rewards.feet_air_height.weight = 0.125
        self.rewards.step_distance.weight = 0.1
        self.rewards.step_knee.weight = -2.5
        self.rewards.joint_parallel_ankle_pitch.weight = -0.08
        self.rewards.feet_step_current_airtime.params["threshold"] = 0.55
        self.rewards.feet_swing_pos.weight = 0.15
        # self.rewards.reward_no_double_feet_air.weight = -1.0

        # Joint action
        self.rewards.action_rate_l2.weight = -0.01 #-0.005
        self.rewards.dof_acc_l2.weight = -5.0e-7 # default-1.0e-7
        self.rewards.dof_torques_l2.weight = -2.0e-6 
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )

        # Arm swing rewards
        self.rewards.reward_swing_arm_tracking.weight = 0.5
        
        # lowvel
        self.rewards.joint_deviation_zero_lowvel.weight = -0.5

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0) # default (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5) # default (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0) # default (-1.0, 1.0)
        

class CR1AFlatEnvCfg_PLAY(CR1AFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        #commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_robot_joints.params["position_range"]=(1.0,1.0)
        # self.events.reset_base = None
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
