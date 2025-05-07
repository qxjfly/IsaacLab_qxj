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

    # reward = torch.sum(-torch.abs(error_feet_height * contacts_false), dim=-1)

    reward = torch.sum(reward * contacts_false, dim=-1)
    return reward

def root_height_l1(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize root_link height Z position that deviate from the default position."""
    
    asset = env.scene[asset_cfg.name]
    
    pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    pos_z_ref = threshold
    delta_z = torch.abs(pos_z - pos_z_ref)
    return delta_z.squeeze()

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

    contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).min(dim=1)[0] > 150.0
    contact_no = contact[:,[1, 0]]
    # body_quat_w = asset.data.root_quat_w[:, :]
    body_quat_w = asset.data.root_link_quat_w[:, :]

    first_contact_pos = torch.sum(asset.data.body_pos_w[:, asset_cfg.body_ids, :] * contact.unsqueeze(-1) ,dim=1)#zpos
    first_contact_false_pos = torch.sum(asset.data.body_pos_w[:, asset_cfg.body_ids, :] * contact_no.unsqueeze(-1) ,dim=1)#zpos

    step_temp = first_contact_pos - first_contact_false_pos
    
    # step_ref_local = 0.85 * threshold * env.command_manager.get_command(command_name)[:, 0] #0501修改

    step_temp_local = world_to_local_transform_qxj(body_quat_w, step_temp)
    

    reward = torch.exp(torch.clamp(step_temp_local[:,0], max=0.6)) #- 0.8 #0504修改  加了偏置0.8

    # reward = step_temp_local[:,0] #default
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

    # reward = step_temp_local[:,0] #default
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
        全部时刻关节角度相对于 default 对称 hip_pitch  knee
        
    """
    
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    current_air = torch.sum(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids],dim=-1)
    current_contact = torch.sum(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids],dim=-1)
    
    
    reward = current_air > threshold
    # reward = step_temp_local[:,0] #default
    return reward

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

    knee_pos_flag = asset.data.joint_pos[:, asset_cfg.joint_ids] > 2 #1.745

    reward = torch.sum(knee_pos_flag, dim=1)


    return reward

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