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
    # reward = torch.clamp(reward, max=threshold*0.5)
    reward = -torch.abs(reward)
    # reward = torch.exp(-reward)
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
    # Penalize feet air height
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts_false = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).min(dim=1)[0] < 1.0

    asset = env.scene[asset_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] #zpos
    error_feet_height = torch.abs(feet_height - threshold)
    # error_feet_height = torch.clamp(error_feet_height, max=0)
    reward = torch.sum(error_feet_height * contacts_false, dim=-1)
    return torch.exp(-reward / std**2)

def root_height_l1(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize root Z position that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute out of limits constraints
    pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    pos_z_ref = threshold
    delta_z = torch.abs(pos_z - pos_z_ref)
    return delta_z.squeeze()

def step_distance(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float ,asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """落地时, 落地脚比另一只脚在x方向上 前进 feet_air_time*v 距离*0.5= L*sin(theta) 目前只限制了摆动腿落地的位置
        未来可增加限制支撑腿离地的位置
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    # compute the reward
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
    # Penalize low vel ：feet 离开地面  ：Z的高度 + joint_vel 
    asset = env.scene[asset_cfg.name]
    vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    vel_mask = vel_temp < 0.1
    
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] #zpos
    error_feet_height = feet_height - threshold
    error_feet_height = torch.clamp(error_feet_height, min=0)
    error_feet_height = torch.sum(error_feet_height, dim=-1)
    # joint_vel_re = torch.norm(asset.data.joint_vel[:, asset_cfg.joint_ids], dim=1)

    # reward = error_feet_height + std * joint_vel 

    # reward = std * joint_vel_re
    reward = error_feet_height
    return reward * vel_mask
    
def joint_deviation_zero_l1(
        env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    # Penalize low vel ：ang  pitch 
    asset = env.scene[asset_cfg.name]
    vel_temp = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    vel_mask = vel_temp < 0.1
    
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    return torch.sum(torch.abs(angle), dim=1) * vel_mask

    