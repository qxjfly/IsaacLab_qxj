# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from typing import Tuple

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

DEG2RAD = 3.14159265358979323846 / 180.0
def feet_air_time(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.6 
    return reward


def feet_air_time_positive_biped(
        env, 
        command_name: str, 
        threshold: float, 
        sensor_cfg: SceneEntityCfg,
        vel_threshold: float = 0.1) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > vel_threshold
    return reward

def feet_both_contact_time(
        env, 
        command_name: str, 
        sensor_cfg: SceneEntityCfg,
        vel_threshold: float = 0.5) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    contact_time_double = contact_time.min(dim=1).values #.unsqueeze(1)
    both_contact_rew = torch.where(
        contact_time_double < 0.1, 
        0.8 * contact_time_double,  # 
        torch.where(
            contact_time_double <= 0.2, # 
            1.0 * contact_time_double,  # 
            -2.0 * contact_time_double  # 
        )
    )
    in_contact = contact_time > 0.0
    double_stance = torch.sum(in_contact.int(), dim=1) > 1.2

    reward = torch.where(
        double_stance,
        both_contact_rew,
        0.0
    )
    
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > vel_threshold
    return reward


def feet_slide(
        env, sensor_cfg: SceneEntityCfg, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])#quat_apply_inverse
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])#quat_apply_inverse
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    #qxj add
    # vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    vel_mask = vel_temp >= 0.1
    lin_vel_error = lin_vel_error * vel_mask
    #qxj add
    return torch.exp(-lin_vel_error / std**2) 


def track_ang_vel_z_world_exp(
    env, 
    command_name: str, 
    std: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])

    #qxj add
    # vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    vel_mask = vel_temp >= 0.1
    ang_vel_error = ang_vel_error * vel_mask
    #qxj add
    return torch.exp(-ang_vel_error / std**2)
    # return torch.exp(-ang_vel_error / std**2) * vel_mask

def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def feet_air_height_exp(
        env: ManagerBasedRLEnv, 
        sensor_cfg: SceneEntityCfg, 
        asset_cfg: SceneEntityCfg, 
        threshold: float , 
        std: float,
) -> torch.Tensor:
    
    """Reward swing feet height Z in world frame using exponential kernel."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts_false = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).min(dim=1)[0] < 10.0

    asset = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] #zpos
    error_feet_height =torch.clamp(feet_height - threshold, max=0.03) #feet_height - threshold

    reward = torch.exp(error_feet_height/std)

    reward = torch.sum(reward * contacts_false, dim=-1)
    return reward

def root_height_l1(
        env: ManagerBasedRLEnv, 
        threshold: float, 
        asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize root_link height Z position that deviate from the default position."""
    
    asset = env.scene[asset_cfg.name]
    
    pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    pos_z_ref = threshold
    # delta_z = torch.abs(pos_z - pos_z_ref)
    delta_z = pos_z - pos_z_ref
    clamped_delta_z = torch.clamp(delta_z, 
                                  min=-1.0, 
                                  max=0.05)
                                #   max=0.0) #站立姿态
    return clamped_delta_z.squeeze() 

def root_height_exp_zq(
        env: ManagerBasedRLEnv, 
        threshold: float, 
        asset_cfg: SceneEntityCfg) -> torch.Tensor:

    asset = env.scene[asset_cfg.name]
    base_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    reward = torch.exp(-torch.abs(base_height - threshold) * 100)
    return reward

def feet_air_height_lowvel(
        env: ManagerBasedRLEnv, 
        command_name: str, 
        asset_cfg: SceneEntityCfg, 
        threshold: float,
) -> torch.Tensor:
    # Penalize low vel ：feet 离开地面  ：Z的高度 
    asset = env.scene[asset_cfg.name]
    # vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    vel_mask = vel_temp < 0.1 #0.1
    
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] #zpos
    error_feet_height = feet_height - threshold
    error_feet_height = torch.clamp(error_feet_height, min=0)
    error_feet_height = torch.sum(error_feet_height, dim=-1)
    
    reward = error_feet_height
    return reward * vel_mask

def joint_pos_lowvel(
        env: ManagerBasedRLEnv, 
        command_name: str, 
        command_threshold: float,
        asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    # Penalize low vel ：ang  pitch 
    asset = env.scene[asset_cfg.name]
    vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    vel_mask = vel_temp < command_threshold
    
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    return torch.sum(torch.abs(diff_angle), dim=1) * vel_mask


def feet_both_air(
        env, 
        sensor_cfg: SceneEntityCfg, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """双脚腾空  惩罚
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    both_air_flag = (feet_forces_z.abs() < 10.0).all(dim=-1)
    reward = both_air_flag

    return reward

def world_to_local_transform_qxj(q_w: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
    """将世界坐标系向量转换到根链接局部坐标系
    
    Args:
        q_w: 根链接四元数 (w, x, y, z), shape (N, 4)
        v_w: 世界坐标系向量, shape (N, 3)
    
    Returns:
        v_l: 局部坐标系向量, shape (N, 3)
    """
    # Step 1: 四元数归一化
    q_w = torch.nn.functional.normalize(q_w, p=2, dim=-1)
    
    # Step 2: 计算四元数的逆 (q_w^{-1} = [w, -x, -y, -z])
    q_inv = q_w * torch.tensor([1, -1, -1, -1], device=q_w.device)
    
    # Step 3: 将向量转换为四元数形式 (实部为0)
    v_quat = torch.cat([torch.zeros_like(v_w[..., :1]), v_w], dim=-1)  # shape (N,4)
    
    # Step 4: 四元数旋转运算 q_inv * v_w * q_w
    v_l_quat = quaternion_multiply_qxj(
        quaternion_multiply_qxj(q_inv, v_quat),
        q_w
    )
    
    # Step 5: 提取虚部得到旋转后的向量
    v_l = v_l_quat[..., 1:]
    
    return v_l

def quaternion_multiply_qxj(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def feet_step_knee(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg, 
    command_name: str,
    threshold: float
) -> torch.Tensor:
    """
        惩罚 接触地面时 左右腿knee关节 与期望值的差值
        
    """
    #**add
    vel_temp = env.command_manager.get_command(command_name)[:, 0]
    vel_mask = vel_temp > 0.1
    # vel_mask = 1.0
    #*****
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    first_contact_flag = torch.logical_or(first_contact[:, 0], first_contact[:, 1])

    knee_angle = asset.data.joint_pos[:, asset_cfg.joint_ids]

    knee_angle_delta = torch.abs(knee_angle - threshold)

    reward = torch.sum(knee_angle_delta,dim=-1)

    return reward * first_contact_flag * vel_mask

def feet_step_ankle(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg, 
    command_name: str,
) -> torch.Tensor:
    #**add
    vel_temp = env.command_manager.get_command(command_name)[:, 0]
    vel_mask = vel_temp > 0.1
    # vel_mask = 1.0
    #*****
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]

    first_air_flag = torch.logical_or(first_air[:, 0], first_air[:, 1])

    ankle_pitch_angle = asset.data.joint_pos[:, asset_cfg.joint_ids]

    ankle_pitch_angle_air = torch.sum(ankle_pitch_angle * first_air, dim=-1)

    reward = torch.clamp(ankle_pitch_angle_air, max = 0.5)

    return reward * first_air_flag * vel_mask

def joint_torques_hip_roll_l2(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
def joint_torques_max(
        env: ManagerBasedRLEnv, 
        threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque_max = torch.clamp(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]) - threshold, min = 0)
    return torch.sum(torque_max, dim=1)

def joint_parallel_anklepitch_l1(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """保证ankle pitch在与身体处于平行状态 , 惩罚插值"""
    
    # hip_pitch : 0 , 1
    # knee : 6 , 7
    # ankle : 8, 9
    asset: Articulation = env.scene[asset_cfg.name]
    
    angle_total = asset.data.joint_pos[:, asset_cfg.joint_ids]
    angle_hip = angle_total[:, :2]
    angle_knee = angle_total[:,2:4]
    angle_ankle = angle_total[:,4:6]
    angle_ankle_ref = -(angle_knee + angle_hip)
    reward = angle_ankle - angle_ankle_ref

    return torch.sum(torch.abs(reward), dim=1)

def feet_step_current_airtime(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg, 
    threshold: float
) -> torch.Tensor:
    """
        
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    current_air = torch.sum(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids],dim=-1)
    current_contact = torch.sum(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids],dim=-1)
    
    
    reward = current_air > threshold # | (current_air < 0.45)

    return reward

def joint_knee_pos_limit(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        max: float = 1.5,
        min: float = -0.15,) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    knee_pos_flag = asset.data.joint_pos[:, asset_cfg.joint_ids] > max #1.0
    
    knee_pos_flag2 = asset.data.joint_pos[:, asset_cfg.joint_ids] < min #0.0

    reward1 = torch.sum(knee_pos_flag, dim=1)

    reward2 = torch.sum(knee_pos_flag2, dim=1)

    return reward1 + reward2

def reward_step_length_cache_ref(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_range: tuple[float, float] = (0.4, 0.5),
    reward_scale: float = 1.0
) -> torch.Tensor:
    """Reward step length within specified range."""
    # 获取资产和传感器数据
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # env.scene[sensor_cfg.name]
    
    # 修复：正确获取身体索引（使用find_bodies返回值的第一个元素）
    foot_indices = asset.find_bodies(sensor_cfg.body_names)[0]  # 修改点
    
    # 获取脚部位置（使用更稳定的索引方式）
    foot_pos = asset.data.body_pos_w[:, foot_indices, :]  # 现在应正常工作
    
    # 初始化/获取缓存
    if "prev_foot_pos" not in env.cache:
        env.cache["prev_foot_pos"] = foot_pos.clone()
    
    # 计算步长（保持形状一致性）
    step_distance = torch.abs(foot_pos[:, :, 0] - env.cache["prev_foot_pos"][:, :, 0])
    
    # 更新缓存（仅在接触时更新）
    contact_state = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0] > 0.1
    env.cache["prev_foot_pos"] = torch.where(
        contact_state.unsqueeze(-1),
        foot_pos,
        env.cache["prev_foot_pos"]
    )
    
    # 计算奖励（保持维度一致性）
    min_step, max_step = target_range
    in_range = (step_distance > min_step) & (step_distance < max_step)
    reward = reward_scale * in_range.float()  # 形状: (num_envs, num_feet)
    
    # 合并多脚奖励（使用mean代替sum防止形状不匹配）
    return reward.mean(dim=1)  # 形状: (num_envs,)
def leg_swing_pos(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg, 
) -> torch.Tensor:
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_force = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).min(dim=1)[0]

    in_swing = contact_force < 5.0  #swing leg

    allswing_flag = torch.all(in_swing, dim=1)  #all swing

    swing_hip_pitch_pos =asset.data.joint_pos[:, asset_cfg.joint_ids]  #hip_pitch_pos

    pos_error = torch.clamp(swing_hip_pitch_pos, min=0)  # 

    pos_reward = torch.exp(-pos_error / 0.05)

    swing_mask = in_swing.float()

    weighted_rewards = pos_reward * swing_mask
    
    return torch.mean(weighted_rewards, dim=-1) * (~allswing_flag).float()
def feet_stand_pos3(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg, 
    threshold: float
) -> torch.Tensor:
    """
        奖励 接触脚在前（可加速度方向逻辑判断.
        
    """
    temp = 0.25
    target_x = 1.0
    # if env.common_step_counter > 4500 * 24:
    #     temp_x = (env.common_step_counter-4500*24) / (3000 * 24)
    #     temp_x_clamped = max(0.0, min(1.0, temp_x))
    #     target_x = 1 + temp_x_clamped

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]


    stand_feet = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).min(dim=1)[0]

    in_stand = stand_feet < 5.0  #swing leg

    stand_flag = torch.all(in_stand, dim=1)  #all swing

    stand_feet_pos =asset.data.body_pos_w[:, asset_cfg.body_ids, :]  #feet pos

    base_link_pos = asset.data.root_link_pos_w  # base pos

    pos_error_w = stand_feet_pos - base_link_pos.unsqueeze(1) #feet pos - base pos

    body_quat_w = asset.data.root_link_quat_w  #四元数

    body_quat_wz = yaw_quat(body_quat_w)  #only-z

    pos_error_b = quat_apply_inverse(body_quat_wz.unsqueeze(1).repeat(1, 2, 1), pos_error_w) #相对位置

    pos_error = torch.clamp(threshold - pos_error_b[:,:,0], min=0)  # 未达目标时惩罚
    pos_reward = torch.exp(-(pos_error**2) / 0.05)

    stand_mask = in_stand.float()

    weighted_rewards = pos_reward * stand_mask
    
    return torch.mean(weighted_rewards, dim=-1) * (~stand_flag).float() * target_x

def feet_contact_vel(
        env, sensor_cfg: SceneEntityCfg, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet contact velocity.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2] #xyz
    reward = torch.sum(torch.abs(body_vel) * first_contact, dim=1)
    # reward = torch.sum(body_vel.norm(dim=-1) * first_contact, dim=1)
    return reward

def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg, # weight: -0.02
    force_thresh: float = 800.0,      # 最大允许冲击力(N)
) -> torch.Tensor:
    # -------------------------------- 数据准备 --------------------------------
    # asset = env.scene[sensor_cfg.asset_name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取接触力数据 (B, K, 3)
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :] #Fx Fy Fz
    force_norms = torch.norm(contact_forces, dim=-1)  # (B, K) 接触力合外力F

    return torch.sum((force_norms - force_thresh).clip(0, 350), dim=1)

def silent_single_leg_landing(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_thresh: float = 800.0,      # 最大允许冲击力(N)
    speed_adapt_ratio: float = 0.15   # 速度适应系数
) -> torch.Tensor:
    """单脚触地冲击力优化奖励函数"""
    # -------------------------------- 数据准备 --------------------------------
    # asset = env.scene[sensor_cfg.asset_name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取接触力数据 (B, K, 3)
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :] #Fx Fy Fz
    force_norms = torch.norm(contact_forces, dim=-1)  # (B, K) 接触力合外力F
    
    # 获取运动状态
    # base_lin_vel = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)  # 水平速度
    base_lin_vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) # 水平速度 vx vy
    # -------------------------------- 单脚触地检测 --------------------------------
    # 检测有效接触腿（力>5N视为触地）
    contact_mask = force_norms > 200.0  # (B, K) 可以保留一部分双脚支撑期
    single_contact = torch.sum(contact_mask.float(), dim=1) == 1  # (B,) 单脚触地标志
    
    # 提取触地腿的垂直力 (B,)
    vertical_forces = contact_forces[..., 2]  # 假设Z轴为垂直方向
    landing_forces = torch.sum(vertical_forces * contact_mask.float(), dim=1)  # (B,)
    
    # -------------------------------- 动态阈值调整 --------------------------------
    # 速度越快允许更大的冲击力
    dynamic_force_thresh = force_thresh * (1 + speed_adapt_ratio * base_lin_vel) #动态触地力
    
    # -------------------------------- 冲击力惩罚 --------------------------------
    # 1. 绝对力值惩罚
    force_ratio = torch.clamp(landing_forces / dynamic_force_thresh, 0.0, 2.0)
    force_penalty = torch.where(
        force_ratio < 0.9, 
        0.0,  # 安全区无惩罚
        torch.where(
            force_ratio <= 1.2, # 0.9< F <1.2
            0.5 * (force_ratio - 0.9),  # 缓冲区间线性惩罚
            0.2 + 2.0 * (force_ratio - 1.2)  # 超限区间陡峭惩罚
        )
    )
    
    
    # -------------------------------- 综合奖励 --------------------------------
    # 基础奖励曲线
    base_reward = torch.exp(-force_penalty) - 0.15
    
    # 应用单脚触地掩码
    return torch.where(single_contact, base_reward, 0.0)  # 非单脚触地时不惩罚

def reward_swing_knee_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    traj_amplitude: float = 0.8,    # 膝屈曲幅度（弧度）
    speed_compensation: float = 0.4, # 速度-幅度耦合系数
    temp: float = 0.08               # 跟踪精度系数
) -> torch.Tensor:
    """摆动腿膝关节轨迹跟踪奖励函数
    
    功能特性：
    1. 基于步态相位的余弦轨迹跟踪
    2. 速度自适应的轨迹幅度调整
    3. 关节运动平滑性约束
    """
    # -------------------------------- 数据准备 --------------------------------
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    is_swing = torch.norm(contact_forces, dim=-1) < 5.0  # 接触力<5N视为摆动  # (B,2)
    swing_flag = is_swing.any(dim=-1)
    # 获取膝关节状态 (B, 2)
    knee_pos = torch.sum(asset.data.joint_pos[:, asset_cfg.joint_ids]*is_swing,dim=-1) # (B,)
    knee_pos_clamp = torch.clamp(knee_pos, min=0, max=1.35)# (B,)

    # -------------------------------- 步态相位计算 --------------------------------
    
    current_swingtime = torch.sum(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] * is_swing, dim=-1)# (B,)
    current_swingtime_clamp = torch.clamp(current_swingtime, min=0.0, max=0.8)# (B,)

    gait_phase = current_swingtime_clamp / 0.3 #0.45
    # -------------------------------- 动态轨迹生成 --------------------------------
    # 基座速度影响屈曲幅度
    base_speed = env.command_manager.get_command(command_name)[:, 0]  # (B,)
    dynamic_amp = traj_amplitude * (1 + speed_compensation * base_speed) # (B,)
    
    # 理想膝角度轨迹（相位偏移π使触地时相位为0）
    target_angle = dynamic_amp * torch.sin(gait_phase * torch.pi) # (B,)
    target_angle_x = torch.clamp(target_angle,min=-0.25)
    # -------------------------------- 奖励计算 --------------------------------
    # 角度跟踪误差奖励
    angle_error = knee_pos_clamp - target_angle_x
    pos_reward = torch.exp(-(angle_error**2) / (2 * temp**2))  # (B,)
    
    # 关节运动平滑性惩罚
    # vel_penalty = 0.1 * torch.abs(knee_vel)  # (B, 2)
    
    # 综合奖励（仅作用于摆动腿）
    # swing_reward = (pos_reward - vel_penalty) * is_swing.float()
    return pos_reward * swing_flag  # 双取平均

def reward_swing_arm_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    traj_amplitude: float = 0.4,    # 肩关节幅度（弧度）
    speed_compensation: float = 0.2, # 速度-幅度耦合系数
    temp: float = 0.08               # 跟踪精度系数
) -> torch.Tensor:
    """摆动腿肩关节pitch轨迹跟踪奖励函数
    
    功能特性：
    1. 基于步态相位的余弦轨迹跟踪
    2. 速度自适应的轨迹幅度调整
    3. 关节运动平滑性约束
    """
    # -------------------------------- 数据准备 --------------------------------
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    is_swing = torch.norm(contact_forces, dim=-1) < 5.0  # 接触力<5N视为摆动  # (B,2)
    swing_flag = is_swing.any(dim=-1)

    is_contact = torch.norm(contact_forces, dim=-1) > 6.0  # 接触力>6N视为触地  # (B,2)
    contact_flag = is_contact.any(dim=-1)

    # 获取摆动腿侧的肩关节状态 (B, 2)
    shoulder_pos = torch.sum(asset.data.joint_pos[:, asset_cfg.joint_ids]*is_swing,dim=-1) # (B,)
    shoulder_pos_clamp = torch.clamp(shoulder_pos, min=-0.6, max=0.6)# (B,)

    # 获取支撑腿侧的肩关节状态 (B, 2)
    shoulder_pos2 = torch.sum(asset.data.joint_pos[:, asset_cfg.joint_ids]*is_contact,dim=-1) # (B,)
    shoulder_pos2_clamp = torch.clamp(shoulder_pos2, min=-0.6, max=0.6)# (B,)
    
    # -------------------------------- 步态相位计算 --------------------------------
    
    current_swingtime = torch.sum(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] * is_swing, dim=-1)# (B,)
    current_swingtime_clamp = torch.clamp(current_swingtime, min=0.0, max=0.4)# (B,)
    gait_phase = current_swingtime_clamp / 0.4

    current_contacttime = torch.sum(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] * is_contact, dim=-1)# (B,)
    current_contacttime_clamp = torch.clamp(current_contacttime, min=0.0, max=0.4)# (B,)
    gait_phase2 = current_contacttime_clamp / 0.4
    # -------------------------------- 动态轨迹生成 --------------------------------
    # 基座速度影响摆动幅度
    base_speed = env.command_manager.get_command(command_name)[:, 0]  # (B,)
    dynamic_amp = traj_amplitude * (1 + speed_compensation * base_speed) # (B,)
    
    # 理想摆动腿侧的肩关节轨迹
    target_angle = dynamic_amp * torch.sin(gait_phase * torch.pi - torch.pi/2) # (B,)[-1,1]
    target_angle_x = torch.clamp(target_angle,max=1.0)
    # 理想支撑腿侧的肩关节轨迹
    target_angle2 = dynamic_amp * torch.sin(gait_phase2 * torch.pi + torch.pi/2) # (B,)[1,-1]
    target_angle2_x = torch.clamp(target_angle2,max=1.0)
    # -------------------------------- 奖励计算 --------------------------------
    # 角度跟踪误差奖励
    angle_error = shoulder_pos_clamp - target_angle_x
    swing_pos_reward = torch.exp(-(angle_error**2) / (2 * temp**2))  # (B,)

    angle_error2 = shoulder_pos2_clamp - target_angle2_x
    contact_pos_reward = torch.exp(-(angle_error2**2) / (2 * temp**2))  # (B,)

    return swing_pos_reward * swing_flag + contact_pos_reward * contact_flag

def feet_step_distance_zq(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
        奖励 接触脚在前（可加速度方向逻辑判断.
        
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :2] # xy

    foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
    fd = 0.35 * vel #0.3
    max_df = 0.9 * vel#0.5
    d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
    d_max = torch.clamp(foot_dist - max_df, 0, 0.5)

    return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

def joint_coordination(
    env: ManagerBasedRLEnv, 
    joint1_cfg: SceneEntityCfg, 
    joint2_cfg: SceneEntityCfg,
    tolerance: float = 5.0,  # deg
    ratio: float = 1.0,
    coordination_reward: float = 0.05
) -> torch.Tensor:
    """Penalize lack of coordination between two joints.
    
    This function enforces a desired relationship between two joints,
    which is useful for enforcing natural movement patterns like 
    coordinated shoulder-hip motion during walking.
    
    Args:
        env: The environment instance
        joint1_cfg: Configuration for the first joint
        joint2_cfg: Configuration for the second joint
        tolerance: Angle tolerance in degrees before penalization
        ratio: Desired ratio between joint1 and joint2 angles
        coordination_reward: Reward value for staying within tolerance
        
    Returns:
        Reward for joint coordination
    """
    # extract the used quantities (to enable type-hinting)
    joint1: Articulation = env.scene[joint1_cfg.name]
    joint2: Articulation = env.scene[joint2_cfg.name]
    # compute devirations
    angle = torch.abs(joint1.data.joint_pos[:, joint1_cfg.joint_ids] - joint1.data.default_joint_pos[:, joint1_cfg.joint_ids] - ratio * (joint2.data.joint_pos[:, joint2_cfg.joint_ids] - joint2.data.default_joint_pos[:, joint2_cfg.joint_ids]))
    angle = torch.where(angle > tolerance * DEG2RAD, angle, torch.zeros_like(angle) - coordination_reward)
    reward = torch.max(angle, dim=1).values
    return reward

def biped_no_double_feet_air(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward having only one foot in the air at a time.
    
    This function penalizes having both feet in the air simultaneously,
    encouraging a stable walking gait where at least one foot is always
    in contact with the ground.
    
    Args:
        env: The environment instance
        sensor_cfg: Configuration for the foot contact sensors
        
    Returns:
        Reward for appropriate foot placement
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    double_air = torch.sum(in_contact.int(), dim=1) == 0
    reward = torch.min(torch.where(double_air.unsqueeze(-1), air_time, 0.0), dim=1)[0]
    return reward

def lateral_distance(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    min_dist: float, 
    max_dist: float, 
    constant_reward: float = 0.1
) -> torch.Tensor:
    """Penalize inappropriate lateral distance between feet.
    
    This function encourages the robot to maintain an appropriate stance width
    by penalizing when feet are too close together (unstable) or too far apart (unnatural).
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot
        min_dist: Minimum allowed distance between feet
        max_dist: Maximum allowed distance between feet
        constant_reward: Reward value when within distance bounds
        
    Returns:
        Reward for appropriate lateral foot distance
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w # [:,3]
    root_quat = asset.data.root_quat_w #[:,4]
    asset_pos_world = asset.data.body_pos_w[:, asset_cfg.body_ids, :] #[:,2,3]
    # asset_pos_body = quat_rotate_inverse(yaw_quat(root_quat.unsqueeze(1)), asset_pos_world - root_pos.unsqueeze(1))
    yaw_quat_only = yaw_quat(root_quat)#[:,4]
    yaw_quat_expanded = yaw_quat_only.unsqueeze(1).expand(-1, asset_pos_world.shape[1], -1) #[:,2,4]
    asset_pos_body = quat_apply_inverse(yaw_quat_expanded, asset_pos_world - root_pos.unsqueeze(1)) #[:,2,3]
    asset_dist = torch.abs(asset_pos_body[:, 0, 1] - asset_pos_body[:, 1, 1]).unsqueeze(1) #[:,1]

    dist = torch.where(
        asset_dist < min_dist, 
        torch.abs(asset_dist - min_dist), 
        torch.where(
            asset_dist > max_dist, 
            torch.abs(asset_dist - max_dist), 
            torch.zeros_like(asset_dist) - constant_reward
        )
    )
    reward = torch.min(dist, dim=1).values
    return reward