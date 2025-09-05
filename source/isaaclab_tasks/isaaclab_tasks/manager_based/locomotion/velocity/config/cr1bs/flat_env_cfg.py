# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import CR1BSRoughEnvCfg


@configclass
class CR1BSFlatEnvCfg(CR1BSRoughEnvCfg):
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
        #********************************************************
        # # stand phase
        # # robot_base
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5 #
        # self.rewards.track_ang_vel_z_exp.weight = 1.0 #
        # self.rewards.lin_vel_z_l2.weight = -0.2
        # self.rewards.ang_vel_xy_l2.weight = -0.5 #-2.0 #-0.5qqq
        # self.rewards.flat_orientation_l2.weight = -0.5 #-5.0 #qqq
        # self.rewards.root_height.weight = 1.0#1.25qqq # 1.0
        # self.rewards.root_height.params["threshold"] = 0.85 #0.88qqq #default 0.85qqq
        # self.rewards.body_ang_vel_xy_l2.weight = -0.5
        # # Joint deviation
        # self.rewards.joint_deviation_leg_yaw.weight = -0.5 #-1.0qqq # default -0.5  hip:yaw roll && ankle: roll
        # self.rewards.joint_deviation_leg_roll.weight = -0.5 #-1.0qqq # default -0.5 
        # self.rewards.joint_deviation_ankle_roll.weight = -0.5 #-1.3qqq # default -1.0 
        # self.rewards.joint_deviation_arms.weight = -0.2 # 
        # self.rewards.joint_deviation_torso.weight = -0.5 #-1.0qqq #-0.5 
        # # Joint coordination 
        # self.rewards.reward_joint_coordination_hip.weight = -0.5
        # self.rewards.reward_joint_coordination_lankle.weight = 0.0
        # self.rewards.reward_joint_coordination_rankle.weight = 0.0
        # # Joint limits
        # self.rewards.dof_pos_limits.weight = -1.0
        # self.rewards.joint_hip_roll_torque_max.weight = 0.0 #-0.025qqq
        # self.rewards.joint_ankle_roll_torque_max.weight = 0.0 # -0.02 qqq
        # self.rewards.joint_hip_roll_torque_l2.weight = 0.0 #-6.0e-5 #-3.0e-5
        # self.rewards.joint_knee_torque_l2.weight = 0.0 #-1.0e-5 #-3.0e-5 #-6.0e-5 # knee default -8.0e-6
        # # Joint action
        # self.rewards.action_rate_l2.weight = -0.005 #-0.1qqq #-0.1 #-0.005 #-0.1
        # self.rewards.dof_acc_l2.weight = -1.0e-7 #-1.0e-6qqq #-1.0e-7 # default-1.0e-7
        # # lowvel
        # self.rewards.joint_deviation_zero_lowvel.weight = -0.1 #-1.0 #-0.5
        # self.rewards.feet_both_air.weight = -5
        
        # # Arm swing rewards
        # # self.rewards.reward_swing_arm_tracking.weight = 0.5 #0.5
        # self.rewards.joint_coordination_larm_leg.weight = -0.5 #0.5
        # self.rewards.joint_coordination_larm_leg.params["ratio"]=1.0
        # self.rewards.joint_coordination_rarm_leg.weight = -0.5 #0.5
        # self.rewards.joint_coordination_rarm_leg.params["ratio"]=1.0
        # #walk phase
        # # #Gait related rewards
        # # self.rewards.feet_slide.weight = 0 #qxj
        # # self.rewards.feet_air_time.weight = 0 #1.75 #default 1.75
        # # self.rewards.feet_air_time.params["threshold"] = 0.4 #default 0.45s 
        # # self.rewards.feet_air_height.weight = 0
        # # self.rewards.feet_air_height.params["threshold"] = 0.15 #default 0.12
        # # # Joint action
        # # self.rewards.dof_torques_l2.weight = 0
        # # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        # #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # # )
        # # # Joint limits
        # # self.rewards.joint_knee_pos_limit_l2.weight = 0.0
        # #Gait related rewards
        # self.rewards.feet_slide.weight = -0.25 #-0.25 #-0.25 #qxj
        # self.rewards.feet_air_time.weight = 1.2 #1.5qqq #1.75 #default 1.75
        # self.rewards.feet_air_time.params["threshold"] = 0.55 # 0.45 #default 0.45s
        # self.rewards.feet_air_height.weight = 0.35 #0.3qqq 0.35  0.3
        # self.rewards.feet_air_height.params["threshold"] = 0.15 #0.12qqq #0.1 #default 0.12
        # self.rewards.step_knee.weight = 0 #-8qqq #default -5  -8  -10
        # # Joint action
        # self.rewards.dof_torques_l2.weight = -2.0e-6 
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_yaw.*", ".*_ankle_roll.*"]
        # )
        # # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        # #     "robot", joint_names=[".*_hip_roll.*",".*_hip_yaw.*", ".*_knee_joint", ".*_ankle_roll.*"]
        # # )
        # # Joint limits
        # self.rewards.joint_knee_pos_limit_l2.weight = -1.0
        # #walk phase3
        # self.rewards.reward_feet_contact_vel.weight = 0.0 # -4.0qqq #-2.0 #-0.5 #F3_reward
        # self.rewards.step_distance.weight = 0.0 #0.1qqq
        # self.rewards.feet_swing_pos.weight = 0.0 #0.15 #
        # self.rewards.leg_swing_pos.weight = 0.0 #
        # #joint limit
        # self.rewards.knee_dof_acc_l2.weight = 0.0 #-2.0e-6 #-1.0e-5 #-2.0e-6
        # #lowel
        # self.rewards.feet_air_height_lowvel.weight = 0.0 #-5qqq #-50
        # #None
        
        # #Gait
        # self.rewards.joint_parallel_ankle_pitch.weight = 0.0 #-0.08
        # self.rewards.feet_step_current_airtime.params["threshold"] = 0.75 #0.65 #default 0.55
        # # self.rewards.reward_no_double_feet_air.weight = -1.0
        # self.rewards.distance_feet.weight = 0.0 #-0.85 #
        # # self.rewards.reward_feet_contact_forces.weight = 0.08 #F1_reward
        # # self.rewards.reward_feet_contact_forces.weight = -0.01 #F2_reward
        # # self.rewards.reward_feet_contact_forces.params["force_thresh"] = 880 
        
        # #******************************
        # # robot_base
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5 #
        # self.rewards.track_ang_vel_z_exp.weight = 1.3 #
        # self.rewards.lin_vel_z_l2.weight = -0.2
        # self.rewards.ang_vel_xy_l2.weight = -2.0 #-2.0 #-0.5
        # self.rewards.flat_orientation_l2.weight = -6.0 # -5.0
        # self.rewards.root_height.weight = 1.25# # 1.0
        # self.rewards.root_height.params["threshold"] = 0.88 #default 0.85
        # self.rewards.body_ang_vel_xy_l2.weight = -0.5
        # # Joint deviation
        # self.rewards.joint_deviation_leg_yaw.weight = -0.7  # default -0.5  hip:yaw roll && ankle: roll
        # self.rewards.joint_deviation_leg_roll.weight = -1.0 # default -0.5 
        # self.rewards.joint_deviation_ankle_roll.weight = -1.2# default -1.0 
        # self.rewards.joint_deviation_ankle_pitch.weight = -0.5 #-0.02
        # self.rewards.joint_deviation_arms.weight = -0.2 #
        # self.rewards.joint_deviation_torso.weight = -1.0 #-0.5 
        # # Joint coordination 
        # self.rewards.reward_joint_coordination_hip.weight = -0.5
        # self.rewards.reward_joint_coordination_lankle.weight = 0.0
        # self.rewards.reward_joint_coordination_rankle.weight = 0.0
        # # Joint limits
        # self.rewards.dof_pos_limits.weight = -1.0
        # self.rewards.joint_hip_roll_torque_max.weight = -0.035
        # self.rewards.joint_ankle_roll_torque_max.weight = -0.025
        # self.rewards.joint_knee_torque_max.weight = -0.015
        # self.rewards.joint_ankle_pitch_torque_max.weight = -0.015
        # self.rewards.joint_hip_roll_torque_l2.weight = 0.0 #-6.0e-5 #-3.0e-5
        # self.rewards.joint_knee_torque_l2.weight = 0.0 #-1.0e-5 #-3.0e-5 #-6.0e-5 # knee default -8.0e-6
        # # Joint action
        # self.rewards.action_rate_l2.weight = -0.08 #-0.1 #-0.005 #-0.1
        # self.rewards.dof_acc_l2.weight = -7.0e-7 # default-1.0e-6
        # # lowvel
        # self.rewards.joint_deviation_zero_lowvel.weight = -3.0 #-1.0 #-0.1 #-0.5
        # self.rewards.feet_both_air.weight = -5
        
        # # Arm swing rewards
        # # self.rewards.reward_swing_arm_tracking.weight = 0.5 #0.5
        # self.rewards.joint_coordination_larm_leg.weight = -0.5 #0.5
        # self.rewards.joint_coordination_larm_leg.params["ratio"]=1.0
        # self.rewards.joint_coordination_rarm_leg.weight = -0.5 #0.5
        # self.rewards.joint_coordination_rarm_leg.params["ratio"]=1.0
        # #walk phase
        # # #Gait related rewards
        # # self.rewards.feet_slide.weight = 0 #qxj
        # # self.rewards.feet_air_time.weight = 0 #1.75 #default 1.75
        # # self.rewards.feet_air_time.params["threshold"] = 0.4 #default 0.45s
        # # self.rewards.feet_air_height.weight = 0
        # # self.rewards.feet_air_height.params["threshold"] = 0.15 #default 0.12
        # # # Joint action
        # # self.rewards.dof_torques_l2.weight = 0
        # # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        # #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # # )
        # # # Joint limits
        # # self.rewards.joint_knee_pos_limit_l2.weight = 0.0
        # #Gait related rewards
        # self.rewards.feet_slide.weight = -0.25 #-0.25 #-0.25 #qxj
        # self.rewards.feet_air_time.weight = 1.5 #1.75 #default 1.75
        # self.rewards.feet_air_time.params["threshold"] = 0.55 # 0.45 #default 0.45s
        # self.rewards.feet_air_height.weight = 0.3 # 0.35  0.3
        # self.rewards.feet_air_height.params["threshold"] = 0.12 #0.12 #default 0.12
        # self.rewards.step_knee.weight = -8 #-8 #default -5  -8  -10
        # self.rewards.step_ankle.weight = 5 #-8 #default -5  -8  -10
        # # 增加了ankle_pitch的塑形
        # # 增加了vel_lowvel的惩罚
        # # 减少了抬脚高度目标
        # # Joint action
        # self.rewards.dof_torques_l2.weight = -2.0e-6 
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_yaw.*", ".*_knee_joint", ".*_ankle_roll.*"]
        # )
        # # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        # #     "robot", joint_names=[".*_hip_roll.*",".*_hip_yaw.*", ".*_knee_joint", ".*_ankle_roll.*"]
        # # )
        # # Joint limits
        # self.rewards.joint_knee_pos_limit_l2.weight = -1.0
        # self.rewards.joint_ankle_pitch_pos_limit_l2.weight = -1.0
        # #walk phase3
        # self.rewards.reward_feet_contact_vel.weight = -4.0 #-2.0 #-0.5 #F3_reward
        # self.rewards.step_distance.weight = 0.1
        # self.rewards.feet_swing_pos.weight = 0.5 #0.15 #0.5
        # self.rewards.leg_swing_pos.weight = 0.0 #
        # #joint limit
        # self.rewards.knee_dof_acc_l2.weight = 0.0 #-2.0e-6 #-1.0e-5 #-2.0e-6
        # #lowel
        # self.rewards.feet_air_height_lowvel.weight = -50 #-5 前期鼓励踏步 惩罚较小
        # #None
        
        # #Gait
        # self.rewards.joint_parallel_ankle_pitch.weight = 0.0 #-0.08
        # self.rewards.feet_step_current_airtime.params["threshold"] = 0.7 #0.65 #default 0.55
        # # self.rewards.reward_no_double_feet_air.weight = -1.0
        # self.rewards.distance_feet.weight = 0.0 #-0.85 #
        # # self.rewards.reward_feet_contact_forces.weight = 0.08 #F1_reward
        # # self.rewards.reward_feet_contact_forces.weight = -0.01 #F2_reward
        # # self.rewards.reward_feet_contact_forces.params["force_thresh"] = 880 

        #******************************
        # robot_base
        self.rewards.track_lin_vel_xy_exp.weight = 1.5 #
        self.rewards.track_ang_vel_z_exp.weight = 1.3 #
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -2.0 #-2.0 #-0.5
        self.rewards.flat_orientation_l2.weight = -2.5 # -5.0
        self.rewards.root_height.weight = 1.25# # 1.0
        self.rewards.root_height.params["threshold"] = 0.82 #default 0.85
        self.rewards.body_ang_vel_xy_l2.weight = -0.5
        # Joint deviation
        self.rewards.joint_deviation_leg_yaw.weight = -0.7  # default -0.5  hip:yaw roll && ankle: roll
        self.rewards.joint_deviation_leg_roll.weight = -1.0 # default -0.5 
        self.rewards.joint_deviation_ankle_roll.weight = -1.2# default -1.0 
        self.rewards.joint_deviation_ankle_pitch.weight = 0.0 #-0.02
        self.rewards.joint_deviation_arms.weight = -0.2 #
        self.rewards.joint_deviation_torso.weight = -1.0 #-0.5 
        # Joint coordination 
        self.rewards.reward_joint_coordination_hip.weight = 0.0
        self.rewards.reward_joint_coordination_lankle.weight = 0.0
        self.rewards.reward_joint_coordination_rankle.weight = 0.0
        # Joint limits
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.joint_hip_roll_torque_max.weight = -0.035
        self.rewards.joint_ankle_roll_torque_max.weight = -0.025
        self.rewards.joint_knee_torque_max.weight = -0.015
        self.rewards.joint_ankle_pitch_torque_max.weight = -0.015
        self.rewards.joint_hip_roll_torque_l2.weight = 0.0 #-6.0e-5 #-3.0e-5
        self.rewards.joint_knee_torque_l2.weight = 0.0 #-1.0e-5 #-3.0e-5 #-6.0e-5 # knee default -8.0e-6
        # Joint action
        self.rewards.action_rate_l2.weight = -0.08 #-0.1 #-0.005 #-0.1
        self.rewards.dof_acc_l2.weight = -7.0e-7 # default-1.0e-6
        # lowvel
        self.rewards.joint_deviation_zero_lowvel.weight = -0.5 #-1.0 #-0.1 #-0.5
        self.rewards.feet_both_air.weight = -10
        
        # Arm swing rewards
        # self.rewards.reward_swing_arm_tracking.weight = 0.5 #0.5
        self.rewards.joint_coordination_larm_leg.weight = -0.5 #0.5
        self.rewards.joint_coordination_larm_leg.params["ratio"]=1.0
        self.rewards.joint_coordination_rarm_leg.weight = -0.5 #0.5
        self.rewards.joint_coordination_rarm_leg.params["ratio"]=1.0
        #walk phase
        # #Gait related rewards
        # self.rewards.feet_slide.weight = 0 #qxj
        # self.rewards.feet_air_time.weight = 0 #1.75 #default 1.75
        # self.rewards.feet_air_time.params["threshold"] = 0.4 #default 0.45s
        # self.rewards.feet_air_height.weight = 0
        # self.rewards.feet_air_height.params["threshold"] = 0.15 #default 0.12
        # # Joint action
        # self.rewards.dof_torques_l2.weight = 0
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # )
        # # Joint limits
        # self.rewards.joint_knee_pos_limit_l2.weight = 0.0
        #Gait related rewards
        self.rewards.feet_slide.weight = -0.0 #-0.25 #-0.25 #qxj
        self.rewards.feet_air_time.weight = 0.0 #1.75 #default 1.75
        self.rewards.feet_air_time.params["threshold"] = 0.55 # 0.45 #default 0.45s
        self.rewards.feet_air_height.weight = 0.0 # 0.35  0.3
        self.rewards.feet_air_height.params["threshold"] = 0.12 #0.12 #default 0.12
        self.rewards.step_knee.weight = -0 #-8 #default -5  -8  -10
        self.rewards.step_ankle.weight = 0 #-8 #default -5  -8  -10
        # 增加了ankle_pitch的塑形
        # 增加了vel_lowvel的惩罚
        # 减少了抬脚高度目标
        # Joint action
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_yaw.*", ".*_knee_joint", ".*_ankle_roll.*"]
        )
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_roll.*",".*_hip_yaw.*", ".*_knee_joint", ".*_ankle_roll.*"]
        # )
        # Joint limits
        self.rewards.joint_knee_pos_limit_l2.weight = -1.0
        self.rewards.joint_ankle_pitch_pos_limit_l2.weight = -0.0
        #walk phase3
        self.rewards.reward_feet_contact_vel.weight = -0.0 #-2.0 #-0.5 #F3_reward
        self.rewards.step_distance.weight = 0.0
        self.rewards.feet_swing_pos.weight = 0.0 #0.15 #0.5
        self.rewards.leg_swing_pos.weight = 0.0 #
        #joint limit
        self.rewards.knee_dof_acc_l2.weight = 0.0 #-2.0e-6 #-1.0e-5 #-2.0e-6
        #lowel
        self.rewards.feet_air_height_lowvel.weight = -0 #-5 前期鼓励踏步 惩罚较小
        #None
        
        #Gait
        self.rewards.joint_parallel_ankle_pitch.weight = 0.0 #-0.08
        self.rewards.feet_step_current_airtime.params["threshold"] = 0.7 #0.65 #default 0.55
        # self.rewards.reward_no_double_feet_air.weight = -1.0
        self.rewards.distance_feet.weight = 0.0 #-0.85 #
        # self.rewards.reward_feet_contact_forces.weight = 0.08 #F1_reward
        # self.rewards.reward_feet_contact_forces.weight = -0.01 #F2_reward
        # self.rewards.reward_feet_contact_forces.params["force_thresh"] = 880 

        #*******************************
        # robot_base
        # self.rewards.track_lin_vel_xy_exp.weight = 1.25 #1.75 #qxj
        # self.rewards.track_ang_vel_z_exp.weight = 1.0 #1.0
        # self.rewards.lin_vel_z_l2.weight = -0.2
        # self.rewards.ang_vel_xy_l2.weight = -0.5 #default -0.05
        # self.rewards.flat_orientation_l2.weight = -4.5 #
        # self.rewards.root_height.weight = -1.5
        # self.rewards.root_height.params["threshold"] = 0.9 #default 0.87
        # self.rewards.body_ang_vel_xy_l2.weight = -0.5

        # Joint deviation
        # self.rewards.joint_deviation_leg_yaw.weight = -0.5 # default -0.5  hip:yaw roll && ankle: roll
        # self.rewards.joint_deviation_leg_roll.weight = -1.0 # default -0.5 
        # self.rewards.joint_deviation_arms.weight = -0.2 # default -0.5
        # self.rewards.joint_deviation_torso.weight = -0.5 # waist yaw roll pitch

        # Joint coordination  存在奖励项
        # self.rewards.reward_joint_coordination_hip.weight = -0.5
        # self.rewards.reward_joint_coordination_lankle.weight = -0.25
        # self.rewards.reward_joint_coordination_rankle.weight = -0.25
        # Joint limits
        # self.rewards.dof_pos_limits.weight = -1.0
        # self.rewards.joint_hip_roll_torque_l2.weight = -5.0e-5 #-3.0e-5
        # self.rewards.joint_ankle_pitch_torque_l2.weight = -6.0e-5 #default -8.0e-6
        # self.rewards.joint_knee_pos_limit_l2.weight = -2
        # Gait related rewards
        # self.rewards.feet_slide.weight = -0.25 #qxj
        # self.rewards.feet_air_time.weight = 1.75 #1.75 #default 1.75
        # self.rewards.feet_air_time.params["threshold"] = 0.45 #default 0.45s 
        # self.rewards.feet_air_height.weight = 0.125
        # self.rewards.feet_air_height.params["threshold"] = 0.12 #default 0.12
        # self.rewards.step_distance.weight = 0.1
        # self.rewards.step_knee.weight = -8.5 #-4.5  #default -2.5
        # # self.rewards.step_knee = None
        # # self.rewards.joint_parallel_ankle_pitch.weight = 0.0 #-0.08
        # self.rewards.joint_parallel_ankle_pitch = None
        # self.rewards.feet_step_current_airtime.params["threshold"] = 0.6 #default 0.55
        # self.rewards.feet_swing_pos.weight = 0.45 #0.15 #
        # # self.rewards.reward_no_double_feet_air.weight = -1.0
        # self.rewards.distance_feet.weight = -0.85 #
        # # self.rewards.reward_feet_contact_forces.weight = 0.08 #F1_reward
        # # self.rewards.reward_feet_contact_forces.weight = -0.01 #F2_reward
        # # self.rewards.reward_feet_contact_forces.params["force_thresh"] = 880 
        # self.rewards.reward_feet_contact_vel.weight = -0.5 #F3_reward

        # Joint action
        # self.rewards.action_rate_l2.weight = -0.005 #-0.005
        # self.rewards.dof_acc_l2.weight = -5.0e-7 # default-1.0e-7
        # self.rewards.dof_torques_l2.weight = -2.0e-6 
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # )

        # Arm swing rewards
        # # self.rewards.reward_swing_arm_tracking.weight = 0.5 #0.5
        # self.rewards.joint_coordination_larm_leg.weight = -0.5 #0.5
        # self.rewards.joint_coordination_larm_leg.params["ratio"]=1.0
        # self.rewards.joint_coordination_rarm_leg.weight = -0.5 #0.5
        # self.rewards.joint_coordination_rarm_leg.params["ratio"]=1.0
        
        # lowvel
        # self.rewards.joint_deviation_zero_lowvel.weight = -0.5
        # self.rewards.feet_air_height_lowvel.weight = -50
        # self.rewards.feet_both_air.weight = -150
        # Commands
        # self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.8) # default (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5) # default (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5) # default (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 0.0) # default (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0) # default (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0) # default (-1.0, 1.0)

class CR1BSFlatEnvCfg_PLAY(CR1BSFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        #commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
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
