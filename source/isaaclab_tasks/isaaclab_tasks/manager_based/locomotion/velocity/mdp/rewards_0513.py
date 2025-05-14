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
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
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
    #qxj feet air time

    # reward = -torch.abs(reward)

    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1 #default 0.1  good 0.05
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
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
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    #qxj add
    vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    vel_mask = vel_temp >= 0.1
    lin_vel_error = lin_vel_error * vel_mask
    #qxj add
    return torch.exp(-lin_vel_error / std**2) 


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])

    #qxj add
    vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    vel_mask = vel_temp >= 0.1
    ang_vel_error = ang_vel_error * vel_mask
    #qxj add

    return torch.exp(-ang_vel_error / std**2) * vel_mask

def feet_air_height_exp(
        env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float , std: float,
) -> torch.Tensor:
    
    """Reward swing feet height Z in world frame using exponential kernel."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts_false = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).min(dim=1)[0] < 10.0

    asset = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] #zpos
    error_feet_height =torch.clamp(feet_height - threshold, max=0) #feet_height - threshold

    reward = torch.exp(error_feet_height/std)

    reward = torch.sum(reward * contacts_false, dim=-1)
    return reward

def root_height_l1(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize root_link height Z position that deviate from the default position."""
    
    asset = env.scene[asset_cfg.name]
    
    pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    pos_z_ref = threshold
    delta_z = torch.abs(pos_z - pos_z_ref)
    return delta_z.squeeze() 

def root_height_exp(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize root_link height Z position that deviate from the default position."""
    # 获取基座实际高度
    asset = env.scene[asset_cfg.name]
    pos_z = asset.data.root_link_pos_w[:, 2]  # (num_envs,)
    
    # 计算高度偏差（绝对值）
    delta_z = torch.abs(pos_z - threshold)
    
    # 动态调整温度系数（可选：根据训练阶段调整）
    # temp = max(0.1, 0.3 * (1 - env.common_step_counter / 1e6))  # 示例：线性衰减
    temp = 0.15
    # 高斯型奖励函数
    reward = torch.exp(-(delta_z**2) / (2 * temp**2))
    
    # 应用死区补偿（偏差在deadzone内时给予最大奖励）
    deadzone = 0.03
    in_deadzone = delta_z < deadzone
    reward = torch.where(in_deadzone, torch.ones_like(reward), reward)
    
    return reward

def step_hip_pitch_ang(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float ,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    """落地时, 落地脚比另一只脚在x方向上 前进 feet_air_time*v 距离*0.5= L*sin(theta) 
       奖励 hip_pitch_ang  迈步触地时的角度
    """
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids] #[n,2]
    first_contact_swapped = first_contact[:, [1, 0]]
    hip_pitch_pos = asset.data.joint_pos[:, asset_cfg.joint_ids] #[n,2]
    step_temp = threshold * env.command_manager.get_command(command_name)[:, 0] * 0.5/0.7 #[n,1]
    hip_pitch_pos_ref = -torch.asin( torch.clamp(step_temp, min=-1.0, max=1.0) ) - 0.1 #[n,1]

    reward = torch.sum((hip_pitch_pos - hip_pitch_pos_ref.unsqueeze(-1)) * first_contact, dim=1) #[n,]
    reward = -torch.abs(reward)

    #对调 两个body的位置 first_contact 
    stand_hip_pitch_pos_ref = torch.asin( torch.clamp(step_temp, min=-1.0, max=1.0) ) - 0.1 #[n,1]
    reward2 = torch.sum((hip_pitch_pos - stand_hip_pitch_pos_ref.unsqueeze(-1)) * first_contact_swapped, dim=1) #[n,]
    reward2 = -torch.abs(reward2)
    
    # reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05 #default 0.1
    return (reward+reward2)

def feet_air_height_lowvel(
        env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, threshold: float , std: float,
) -> torch.Tensor:
    # Penalize low vel ：feet 离开地面  ：Z的高度 
    asset = env.scene[asset_cfg.name]
    vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    vel_mask = vel_temp < 0.1
    
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] #zpos
    error_feet_height = feet_height - threshold
    error_feet_height = torch.clamp(error_feet_height, min=0)
    error_feet_height = torch.sum(error_feet_height, dim=-1)
    
    reward = error_feet_height
    return reward * vel_mask

def joint_pos_lowvel(
        env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    # Penalize low vel ：ang  pitch 
    asset = env.scene[asset_cfg.name]
    vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    vel_mask = vel_temp < 0.1
    
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    return torch.sum(torch.abs(angle), dim=1) * vel_mask


def feet_fly(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """feet fly time > threshold  get reward  Sparse Reward
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time > threshold) * first_contact, dim=1)

    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1 #default 0.1
    return reward
    
def feet_symmetric(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    
    """腾空时间  对称  奖励 
    """
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    leg_current = torch.sum((last_air_time) * first_contact, dim=1)
    leg_another = torch.sum((last_air_time) * first_contact[:, [1, 0]], dim=1)
    
    time_delta = torch.abs(leg_current - leg_another)
    
    reward = torch.max(torch.zeros_like(time_delta),1-time_delta/threshold)
    
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1 #default 0.1 
    return reward

def feet_both_air(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """双脚腾空  惩罚
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    both_air_flag = (feet_forces_z.abs() < 10.0).all(dim=-1)
    reward = both_air_flag

    return reward

def feet_step_distance(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        奖励 接触脚在前（可加速度方向逻辑判断.
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    first_contact_false = first_contact[:, [1, 0]]

    # body_quat_w = asset.data.root_quat_w[:, :]
    body_quat_w = asset.data.root_link_quat_w[:, :]

    first_contact_pos = torch.sum(asset.data.body_pos_w[:, asset_cfg.body_ids, :] * first_contact.unsqueeze(-1) ,dim=1)#zpos
    first_contact_false_pos = torch.sum(asset.data.body_pos_w[:, asset_cfg.body_ids, :] * first_contact_false.unsqueeze(-1) ,dim=1)#zpos

    step_temp = first_contact_pos - first_contact_false_pos
    
    step_ref_local = 0.85 * threshold * env.command_manager.get_command(command_name)[:, 0] #0501修改

    step_temp_local = world_to_local_transform_qxj(body_quat_w, step_temp)
    

    reward = torch.exp(torch.clamp(step_temp_local[:,0] - step_ref_local, max=0.15)) - 0.8 #0504修改  加了偏置0.8

    # reward = step_temp_local[:,0] #default
    return reward

def feet_step_distance2(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        奖励 接触脚在前（可加速度方向逻辑判断.
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    first_contact_false = first_contact[:, [1, 0]]

    contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).min(dim=1)[0] > 5.0
    contact_no = contact[:,[1, 0]]
    # body_quat_w = asset.data.root_quat_w[:, :]
    body_quat_w = asset.data.root_link_quat_w[:, :]

    first_contact_pos = torch.sum(asset.data.body_pos_w[:, asset_cfg.body_ids, :] * contact.unsqueeze(-1) ,dim=1)#zpos
    first_contact_false_pos = torch.sum(asset.data.body_pos_w[:, asset_cfg.body_ids, :] * contact_no.unsqueeze(-1) ,dim=1)#zpos

    step_temp = first_contact_false_pos - first_contact_pos
    
    # step_ref_local = 0.85 * threshold * env.command_manager.get_command(command_name)[:, 0] #0501修改

    step_temp_local = world_to_local_transform_qxj(body_quat_w, step_temp)
    

    reward = torch.exp(torch.clamp(step_temp_local[:,0], max=0.6)) #- 0.8 #0504修改  加了偏置0.8

    # reward = step_temp_local[:,0] #default
    return reward
def feet_step_distance_optimized(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg,
    target_range: Tuple[float, float] = (0.35, 0.45),  # 目标步长区间
    # base_speed: float = 1.0,  # 基准运动速度(m/s)
    temp: float = 0.08        # 奖励曲线陡峭度
) -> torch.Tensor:
    """动态步长优化奖励函数
    
    特性：
    1. 步长目标值随运动速度动态调整
    2. 高斯型奖励鼓励步长在目标区间
    3. 速度匹配奖励避免单纯追求大步长
    """
    # -------------------------------- 数据准备 --------------------------------
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取基座状态
    # base_lin_vel = asset.data.root_link_lin_vel_w
    base_quat = asset.data.root_quat_w
    
    # -------------------------------- 步长计算 --------------------------------
    # 计算双足相位差步长（前向分量）
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (B, 2, 3)
    foot_pos_local = quat_rotate_inverse(base_quat.unsqueeze(1), foot_pos_w - asset.data.root_pos_w.unsqueeze(1))
    step_diff = foot_pos_local[:, 0, 0] - foot_pos_local[:, 1, 0]  # (B,) 双足前向位置差
    
    # -------------------------------- 动态目标调整 --------------------------------
    # 根据实际速度调整目标步长
    # speed_ratio = torch.norm(base_lin_vel[:, :2], dim=1) / base_speed  # (B,)
    # dynamic_target = target_range[0] + (target_range[1]-target_range[0]) * torch.sigmoid(2*(speed_ratio-0.5))  # (B,)
    
    # -------------------------------- 奖励计算 --------------------------------
    # 步长误差（鼓励步长>=0.35）
    step_error = torch.where(
        step_diff < target_range[0], 
        target_range[0] - step_diff,  # 小于下限的负误差
        torch.clamp(step_diff - target_range[1], min=0)  # 超过上限的正误差
    )
    
    # 高斯型奖励函数
    step_reward = torch.exp(-(step_error**2)/(2*temp**2))
    
    # 速度匹配奖励
    # speed_reward = torch.exp(-(speed_ratio - 1.0)**2/0.5)
    
    # 综合奖励
    # return 0.7*step_reward + 0.3*speed_reward
    return step_reward

def feet_step_distance3(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        奖励 接触脚在前（可加速度方向逻辑判断.
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    first_contact_false = first_contact[:, [1, 0]]
   
    contact_flag = first_contact[:,0] | first_contact[:,1]

    contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).min(dim=1)[0] > 50.0
    contact_no = contact[:,[1, 0]]

    # body_quat_w = asset.data.root_quat_w[:, :]
    body_quat_w = asset.data.root_link_quat_w[:, :]

    first_contact_pos = torch.sum(asset.data.body_pos_w[:, asset_cfg.body_ids, :] * first_contact.unsqueeze(-1) ,dim=1)#zpos
    first_contact_false_pos = torch.sum(asset.data.body_pos_w[:, asset_cfg.body_ids, :] * first_contact_false.unsqueeze(-1) ,dim=1)#zpos

    step_temp = first_contact_pos - first_contact_false_pos
    
    # step_ref_local = 0.85 * threshold * env.command_manager.get_command(command_name)[:, 0] #0501修改

    step_temp_local = world_to_local_transform_qxj(body_quat_w, step_temp)
    

    reward = torch.exp(torch.clamp(step_temp_local[:,0], max=0.6)) #- 0.8 #0504修改  加了偏置0.8

    # reward = step_temp_local[:,0] #default
    return reward * contact_flag

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

def joint_deviation_knee2knee(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize Knee joint positions that deviate from the Another one. """
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    # angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    angleL = angle[:, 0]
    angleR = angle[:, 1]
    angle_delta=torch.abs(angleL - angleR)
    # angle_delta=torch.abs(angleL + angleR)
    
    return angle_delta

def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base angular velocity using L2 squared kernel."""
    
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, 2]), dim=1)

def feet_step_knee(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        惩罚 接触地面时 左右腿knee关节 与期望值的差值
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    first_contact_flag = torch.logical_or(first_contact[:, 0], first_contact[:, 1])

    knee_angle = asset.data.joint_pos[:, asset_cfg.joint_ids]

    knee_angle_delta = torch.abs(knee_angle - threshold)

    reward = torch.sum(knee_angle_delta,dim=-1)

    # reward = step_temp_local[:,0] #default
    return reward * first_contact_flag

def feet_step_knee_exp(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg, 
    target_angle: float = 0.0,  # 直膝目标角度（建议弧度制）
    sigma: float = 0.15         # 奖励曲线陡峭度控制
) -> torch.Tensor:
    """摆动腿落地膝关节伸直奖励（分腿独立处理）"""
    
    # -------------------------------- 数据准备 --------------------------------
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取双膝角度 (B, 2)
    knee_angles = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # -------------------------------- 接触检测 --------------------------------
    # 获取首次接触标志 (B, 2)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    
    # -------------------------------- 分腿奖励计算 --------------------------------
    # 计算每腿独立的角度误差 (B, 2)
    angle_deltas = torch.abs(knee_angles - target_angle)
    
    # 高斯型奖励函数 (B, 2)
    leg_rewards = torch.exp(-angle_deltas**2 / (2 * sigma**2))
    
    # -------------------------------- 接触掩码融合 --------------------------------
    # 仅当首次接触时应用奖励 (B, 2)
    masked_rewards = leg_rewards * first_contact.float()
    
    # 取双腿奖励的最大值 (B,)
    return torch.max(masked_rewards, dim=-1).values

def feet_step_knee2(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        接触地面时 左右腿knee关节 与期望值的差值
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    knee_angle = torch.sum(asset.data.joint_pos[:, asset_cfg.joint_ids] * first_contact ,dim=-1)

    knee_angle_delta = torch.abs(knee_angle - threshold)

    reward = torch.exp(-knee_angle_delta / 0.25) 

    return reward

def joint_torques_roll_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

def joint_deviation_anklepitch_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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

def feet_contact_force_exp(
        env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float , std: float,
) -> torch.Tensor:
    """奖励 足底接触力 越小越好"""

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    # first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    feet_contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]

    # reward = torch.exp(-torch.sum(feet_contact_force * first_contact,dim=-1))
    reward = torch.exp(-torch.sum(feet_contact_force,dim=-1)/std)
    return reward

def feet_step_symmetric_hip(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        全部时刻关节角度相对于 default 对称 hip_pitch  knee
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]


    
    hip_pitch_angle_delta = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    reward = torch.sum(hip_pitch_angle_delta,dim=-1)

    # hip_reward = torch.exp(-reward**2 / (2 * sigma**2))
    # return hip_reward
    return torch.abs(reward)

def feet_step_symmetric_knee(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        全部时刻关节角度相对于 default 对称 hip_pitch  knee
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    hip_pitch_angle_delta = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    reward = torch.sum(hip_pitch_angle_delta,dim=-1)

    # reward = step_temp_local[:,0] #default
    return torch.abs(reward)

def feet_step_current_airtime(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    current_air = torch.sum(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids],dim=-1)
    current_contact = torch.sum(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids],dim=-1)
    
    
    reward = current_air > threshold # | (current_air < 0.45)

    # reward = torch.where(
    #     current_air < 0.45,
    #     0.25 * torch.exp(-(current_air - 0.45)**2 / 0.1),
    #     torch.where(
    #         current_air > threshold,
    #         torch.tensor(-20.0),
    #         torch.tensor(0.25)
    #     )
    # )
    

    return reward

def feet_step_phase_balance(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    asset_cfg: SceneEntityCfg,
    target_duration: float = 0.5,  # 目标相位时长（秒）
    sigma: float = 0.1,            # 时间容差（标准差）
    balance_weight: float = 0.3   # 支撑/摆动平衡权重
) -> torch.Tensor:
    """步态相位时间平衡奖励函数
    
    特性：
    1. 独立计算支撑相/摆动相时间接近目标值
    2. 惩罚支撑与摆动时间不平衡
    3. 动态调整容差范围
    """
    # -------------------------------- 数据准备 --------------------------------
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取各腿相位时间 (B, K)
    swing_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    stance_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    
    # -------------------------------- 时间合规奖励 --------------------------------
    # 支撑相时间奖励 (B, K)
    stance_reward = torch.exp(-(stance_time - target_duration)**2 / (2 * sigma**2))
    
    # 摆动相时间奖励 (B, K)
    swing_reward = torch.exp(-(swing_time - target_duration)**2 / (2 * sigma**2))
    
    # -------------------------------- 时间平衡奖励 --------------------------------
    # 计算支撑/摆动时间差 (绝对值越大越不平衡)
    phase_diff = torch.abs(stance_time - swing_time)  # (B, K)
    balance_reward = torch.exp(-phase_diff**2 / (0.5 * sigma**2))  # (B, K)
    
    # -------------------------------- 综合奖励 --------------------------------
    # 各腿独立计算后取均值 (B,)
    phase_reward = 0.5*(stance_reward + swing_reward)  # 时间合规部分
    combined_reward = (1 - balance_weight)*phase_reward + balance_weight*balance_reward
    return torch.mean(combined_reward, dim=-1)  # (B,)

def step_phase_consistency(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    phase_threshold: float = 0.2
) -> torch.Tensor:
    """Reward consistent gait phase timing between legs."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取接触时间序列
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    
    # 计算相位差（0-1）
    cycle_time = contact_time + air_time
    phase = (contact_time / cycle_time).clamp(0, 1)
    phase_diff = torch.abs(phase[:, 0] - phase[:, 1])
    
    # 保持相位差在0.5±threshold范围内
    reward = 1.0 - torch.clamp(torch.abs(phase_diff - 0.5) / phase_threshold, 0, 1)
    return reward

def joint_knee_pos_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    knee_pos_flag = asset.data.joint_pos[:, asset_cfg.joint_ids] > 1.25 #1.745
    
    knee_pos_flag2 = asset.data.joint_pos[:, asset_cfg.joint_ids] < 0.0

    reward1 = torch.sum(knee_pos_flag, dim=1)

    reward2 = torch.sum(knee_pos_flag2, dim=1)

    return reward1 + reward2

def reward_step_length(
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

def feet_step_knee3(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        接触地面时 左右腿knee关节 与期望值的差值
        
    """
    # 关键参数
    target_min = torch.deg2rad(torch.tensor(2.0))
    target_max = torch.deg2rad(torch.tensor(8.0))
    # 获取环境信息
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]

    knee_angle = asset.data.joint_pos[:, asset_cfg.joint_ids]

    angle_error = torch.where(
        knee_angle < target_min,
        target_min - knee_angle,
        torch.where(
            knee_angle > target_max,
            knee_angle - target_max,
            torch.zeros_like(knee_angle)
        )
    )

    angle_rewards = torch.exp(-(angle_error**2) / (2 * threshold**2))
    angle_rewards = torch.where(first_contact, angle_rewards, 1.0)
    reward = torch.prod(angle_rewards, dim=-1)

    return reward


def feet_swing_pos(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        奖励 接触脚在前（可加速度方向逻辑判断.
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]


    swing_feet = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).min(dim=1)[0] < 50.0

    swing_feet_pos = torch.sum(asset.data.body_pos_w[:, asset_cfg.body_ids, :] * swing_feet.unsqueeze(-1), dim=1)

    base_link_pos = asset.data.root_link_pos_w[:, :]

    pos_error_w = swing_feet_pos - base_link_pos

    body_quat_w = asset.data.root_link_quat_w[:, :]

    pos_error_b = world_to_local_transform_qxj(body_quat_w, pos_error_w)

    pos_error = torch.clamp(threshold - pos_error_b[:,0], min=0)  # 未达目标时惩罚

    pos_reward = torch.exp(-(pos_error**2) / 0.25)
    
    reward = pos_reward

    return reward

def feet_stand_pos2(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """
        奖励 接触脚在前（可加速度方向逻辑判断.
        
    """
    temp = 0.3
    # if env.common_step_counter > 3000:
    #     threshold = 0.15
    #     temp = 0.1
    # if env.common_step_counter > 3000*600:
    #     threshold = 0.05
    #     temp = 0.1

    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]


    stand_feet = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).min(dim=1)[0]

    in_stand = stand_feet > 5.0

    stand_feet_pos =asset.data.body_pos_w[:, asset_cfg.body_ids, :]

    base_link_pos = asset.data.root_link_pos_w

    pos_error_w = stand_feet_pos - base_link_pos.unsqueeze(1)

    body_quat_w = asset.data.root_link_quat_w

    pos_error_b = quat_rotate_inverse(body_quat_w.unsqueeze(1), pos_error_w)

    pos_error = torch.clamp(pos_error_b[:,:,0]-threshold, min=0)  # 未达目标时惩罚

    # pos_reward = torch.tanh(pos_error / 0.25)  # (num_envs, 2)

    pos_reward = torch.tanh(pos_error / temp)  # (num_envs, 2)
    
    stand_mask = in_stand.float()

    weighted_rewards = pos_reward * stand_mask
    
    return torch.mean(weighted_rewards, dim=-1)

def encourage_large_strides(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    base_speed: float = 1.0,  # 基准运动速度(m/s)
    max_stride: float = 0.65,  # 最大允许步长(m)
    temp: float = 0.15          # 奖励敏感度
) -> torch.Tensor:
    """动态步长优化奖励函数（速度自适应版）
    
    特性：
    1. 动态目标步长：速度越快目标步长越大（0.35-0.65m可调）
    2. 高斯+线性复合奖励：步长达标时高奖励，超限时线性惩罚
    3. 摆动腿独立评估：取最优腿奖励
    4. 能量效率约束：惩罚高频小碎步
    """
    # -------------------------------- 数据准备 --------------------------------
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取基座状态
    base_vel = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)  # 水平速度
    base_quat = asset.data.root_quat_w
    
    # -------------------------------- 步长计算 --------------------------------
    # 计算各腿相对基座的前向位移（局部坐标系）
    feet_pos_w = asset.data.body_pos_w[:, sensor_cfg.body_ids, :]
    feet_pos_local = world_to_local_transform_qxj(base_quat.unsqueeze(1), 
                                                feet_pos_w - asset.data.root_pos_w.unsqueeze(1))
    stride_x = feet_pos_local[..., 0]  # (B, K) 前向位移
    
    # 接触状态掩码 (B, K)
    contact_mask = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).min(dim=1)[0] > 5.0
    
    # -------------------------------- 动态目标步长 --------------------------------
    # 速度标准化（0-2m/s映射到0.35-0.65m）
    speed_ratio = torch.clamp(base_vel / base_speed, 0.0, 1.0)
    dynamic_target = 0.25 + 0.3 * speed_ratio  # (B,)
    
    # -------------------------------- 复合奖励计算 --------------------------------
    stride_rewards = []
    for k in range(feet_pos_local.shape[1]):
        # 单腿步长奖励
        delta = stride_x[:, k] - dynamic_target
        reward = torch.where(
            delta >= 0,
            # 高斯奖励区（步长>=目标值）
            torch.exp(-(delta**2) / (2 * temp**2)),
            # 线性惩罚区（步长<目标值）
            torch.clamp(1.0 + delta / dynamic_target, 0.0, 1.0)
        )
        # 超限惩罚（步长>max_stride时指数衰减）
        # over_stride = stride_x[:, k] > max_stride
        # reward[over_stride] *= torch.exp(-(stride_x[over_stride, k] - max_stride) / 0.1)
        stride_rewards.append(reward)
    
    # 取最优腿奖励 (B,)
    stride_rewards = torch.stack(stride_rewards, dim=1)  # (B, K)
    best_reward = torch.max(stride_rewards * contact_mask.float(), dim=1).values
    
    # -------------------------------- 能量效率约束 --------------------------------
    # 惩罚高频小碎步（相位周期<0.3s）
    # 获取触地和摆动时间
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    swing_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    # 计算完整步态周期（触地+摆动）
    phase_period = contact_time + swing_time  # (B, K)

    # 找到各环境中最短的步态周期
    min_period = torch.min(phase_period, dim=1).values  # (B,)

    # 高频惩罚系数（周期<0.3秒时线性衰减）
    freq_penalty = torch.clamp(min_period / 0.3, 0.1, 1.0)  # (B,)
    
    return best_reward * freq_penalty

def silent_single_leg_landing(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_thresh: float = 300.0,      # 最大允许冲击力(N)
    speed_adapt_ratio: float = 0.1   # 速度适应系数
) -> torch.Tensor:
    """单脚触地冲击力优化奖励函数"""
    # -------------------------------- 数据准备 --------------------------------
    # asset = env.scene[sensor_cfg.asset_name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取接触力数据 (B, K, 3)
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norms = torch.norm(contact_forces, dim=-1)  # (B, K)
    
    # 获取运动状态
    # base_lin_vel = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)  # 水平速度
    base_lin_vel = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    # -------------------------------- 单脚触地检测 --------------------------------
    # 检测有效接触腿（力>5N视为触地）
    contact_mask = force_norms > 5.0  # (B, K)
    single_contact = torch.sum(contact_mask.float(), dim=1) == 1  # (B,)
    
    # 提取触地腿的垂直力 (B,)
    vertical_forces = contact_forces[..., 2]  # 假设Z轴为垂直方向
    landing_forces = torch.sum(vertical_forces * contact_mask.float(), dim=1)  # (B,)
    
    # -------------------------------- 动态阈值调整 --------------------------------
    # 速度越快允许更大的冲击力
    dynamic_force_thresh = force_thresh * (1 + speed_adapt_ratio * base_lin_vel)
    
    # -------------------------------- 冲击力惩罚 --------------------------------
    # 1. 绝对力值惩罚
    force_ratio = torch.clamp(landing_forces / dynamic_force_thresh, 0.0, 2.0)
    force_penalty = torch.where(
        force_ratio < 0.8, 
        0.0,  # 安全区无惩罚
        torch.where(
            force_ratio <= 1.2,
            0.5 * (force_ratio - 0.8),  # 缓冲区间线性惩罚
            0.2 + 2.0 * (force_ratio - 1.2)  # 超限区间陡峭惩罚
        )
    )
    
    
    # -------------------------------- 综合奖励 --------------------------------
    # 基础奖励曲线
    base_reward = torch.exp(-force_penalty)
    
    # 应用单脚触地掩码
    return torch.where(single_contact, base_reward, 0.0)  # 非单脚触地时不惩罚

def reward_step_length(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    temp: float = 0.25
) -> torch.Tensor:
    """Reward step length within specified range."""
    temp_s = 0.0
    if env.common_step_counter > 3000*600:
        temp_s = 1
    # 获取资产和传感器数据
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
   
    contact_flag = first_contact[:,0] | first_contact[:,1]
    # 获取脚部位置（使用更稳定的索引方式）
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # 现在应正常工作
    
    # 初始化/获取缓存
    if "prev_foot_pos" not in env.cache:
        env.cache["prev_foot_pos"] = foot_pos.clone()
    if "step_length_last" not in env.cache:
        env.cache["step_length_last"] = torch.zeros_like(foot_pos[:, 0, 0])
    
    # 计算步长（保持形状一致性）
    step_distance = torch.abs(foot_pos[:, :, 0] - env.cache["prev_foot_pos"][:, :, 0])
    # 计算last步长
    env.cache["step_length_last"] = torch.where(
        contact_flag,
        torch.max(step_distance, dim=1).values,
        env.cache["step_length_last"]
    )

    # 更新缓存（仅在接触时更新）
    contact_state = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1)[0] > 1
    env.cache["prev_foot_pos"] = torch.where(
        contact_state.unsqueeze(-1),
        foot_pos,
        env.cache["prev_foot_pos"]
    )

    swing_step = torch.max(step_distance, dim=1).values
    delta_step = swing_step - env.cache["step_length_last"]
    reward = torch.exp(-(delta_step)**2/temp)
    return reward * temp_s  # 形状: (num_envs,)