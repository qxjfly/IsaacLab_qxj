# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_rotate

from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg
from .motions import MotionLoader


class HumanoidAmpEnv(DirectRLEnv):
    cfg: HumanoidAmpEnvCfg

    def __init__(self, cfg: HumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits
        print("Action Offsets:\n", self.action_offset)
        print("Action Scales:\n", self.action_scale)
        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)
        # qxj_add_joint_names
        q_joint_names = ["left_hip_pitch_joint", 
                         "left_shoulder_pitch_joint",
                         "right_hip_pitch_joint",
                         "right_shoulder_pitch_joint",
                         "left_hip_roll_joint",
                         "left_shoulder_roll_joint",
                         "right_hip_roll_joint",
                         "right_shoulder_roll_joint",
                         "left_hip_yaw_joint",
                         "left_shoulder_yaw_joint",
                         "right_hip_yaw_joint",
                         "right_shoulder_yaw_joint",
                         "left_knee_joint",
                         "left_elbow_pitch_joint",
                         "right_knee_joint",
                         "right_elbow_pitch_joint",
                         "left_ankle_pitch_joint",
                         "right_ankle_pitch_joint",
                         "left_ankle_roll_joint",
                         "right_ankle_roll_joint"]
        q_motion_joint_names = ["left_hip_y", 
                         "left_shoulder_y",
                         "right_hip_y",
                         "right_shoulder_y",
                         "left_hip_x",
                         "left_shoulder_x",
                         "right_hip_x",
                         "right_shoulder_x",
                         "left_hip_z",
                         "left_shoulder_z",
                         "right_hip_z",
                         "right_shoulder_z",
                         "left_knee",
                         "left_elbow",
                         "right_knee",
                         "right_elbow",
                         "left_ankle_y",
                         "neck_x",
                         "right_ankle_y",
                         "neck_y",
                         "left_ankle_x",
                         "neck_z",
                         "right_ankle_x",
                         "abdomen_x",
                         "abdomen_y",
                         "abdomen_z"]
        # DOF and key body indexes
        # key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        # key_body_names = ["right_wrist_roll_link", "left_wrist_roll_link"]
        # key_body_names = ["right_wrist_roll_link", "left_wrist_roll_link","right_ankle_roll_link", "left_ankle_roll_link"]
        key_body_names = ["right_shoulder_pitch_link","left_shoulder_pitch_link","right_wrist_roll_link", "left_wrist_roll_link", "right_knee_link","left_knee_link","right_ankle_roll_link","left_ankle_roll_link"]
        key_body_names_motion = ["right_shoulder_pitch_link","left_shoulder_pitch_link","right_wrist_roll_link", "left_wrist_roll_link", "right_knee_link","left_knee_link","right_ankle_roll_link","left_ankle_roll_link"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        # self.motion_dof_indexes = self._motion_loader.get_dof_index(q_motion_joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_bodym])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names_motion)
        #qxj_add_feet_id
        feet_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.feet_body_indexes = [self.robot.data.body_names.index(name) for name in feet_body_names]
        #qxj_add_deviation_joint
        deviation_joint_names = ["waist_roll_joint","waist_pitch_joint","left_hip_roll_joint", "right_hip_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"]
        self.deviation_joint_indexes = [self.robot.data.joint_names.index(name) for name in deviation_joint_names]
        #qxj_add_undesired_contact_body_id
        undesired_contact_body_names = ["left_wrist_roll_link", "right_wrist_roll_link"]
        self._undesired_contact_body_ids = [self.robot.data.body_names.index(name) for name in undesired_contact_body_names]
        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )
        #qxj_add
        self._commands = torch.zeros(self.num_envs, 1, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.cfg.num_actions_cr1a, device=self.device)
        self.previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions_cr1a, device=self.device)
        #qxj_add_logging
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                # "track_ang_vel_z_exp",
                # "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                # "feet_air_time",
                # "undesired_contacts",
                "flat_orientation_l2",
                "termination_penalty",
                "deviation_penalty",
                "feet_slide",
            ]
        }
        

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # qxj_add_contact_sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # qxj_add_contact_sensor
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()
        # self.actions = torch.clamp(self.actions, min=-2.0, max=2.0)

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        #qxj_add_preaction
        self.previous_actions = self.actions.clone()
        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            # self.robot.data.joint_pos[:, self.q_joint_indexes],
            # self.robot.data.joint_vel[:, self.q_joint_indexes],
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        #cat obs
        obs_q = torch.cat(
                (self._commands,
                 obs,
                 self.previous_actions,#26 actions
                 ),
                dim=-1,
        )
        return {"policy": obs_q}
        # return {"policy": obs}

    #original get reward function
    # def _get_rewards(self) -> torch.Tensor:
    #     return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
    #anymalc的reward函数
    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        # lin_vel_error = torch.sum(torch.square(self._commands[:, 0] - self.robot.data.root_lin_vel_b[:, 0]), dim=1)
        lin_vel_error = torch.square(self._commands[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
        lin_vel_error_rew = torch.exp(-lin_vel_error / 0.25)
        # yaw angvel tracking
        # yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        # yaw_rate_error_rew = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        # z_vel_error = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error_rew = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques_rew = torch.sum(torch.square(self.robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_acc_rew = torch.sum(torch.square(self.robot.data.joint_acc), dim=1)
        # action rate
        action_rate_rew = torch.sum(torch.square(self.actions - self.previous_actions), dim=1)
        # feet air time
        # first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self.feet_body_indexes]
        # last_air_time = self._contact_sensor.data.last_air_time[:, self.feet_body_indexes]
        # air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
        #     torch.norm(self._commands[:, 0], dim=1) > 0.1
        # )
        # undesired contacts
        # net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # is_contact = (
        #     torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        # )
        # contacts = torch.sum(is_contact, dim=1)
        #feet_slide
        feet_contact_flag = self._contact_sensor.data.net_forces_w_history[:, :, self.feet_body_indexes, :].norm(dim=-1).max(dim=1)[0] > 10.0
        feet_vel_w = self.robot.data.body_lin_vel_w[:, self.feet_body_indexes, :2]
        feet_slide_rew = torch.sum(feet_vel_w.norm(dim=-1) * feet_contact_flag, dim=1)
        # flat orientation
        flat_orientation_rew = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        # flat_orientation_rew = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        #termination penalty
        termination_rew = self.reset_terminated.float()
        #deviation from default
        deviation_pos_err_rew = torch.sum(torch.abs(self.robot.data.joint_pos[:,self.deviation_joint_indexes]-self.robot.data.default_joint_pos[:,self.deviation_joint_indexes]), dim=1)
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_rew * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error_rew * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques_rew * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_acc_rew * self.cfg.joint_acc_reward_scale * self.step_dt,
            "action_rate_l2": action_rate_rew * self.cfg.action_rate_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation_rew * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "termination_penalty": termination_rew * self.cfg.termination_reward_scale * self.step_dt,
            "deviation_penalty": deviation_pos_err_rew * self.cfg.deviation_pos_scale * self.step_dt,
            "feet_slide": feet_slide_rew * self.cfg.feet_slide_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0) # 奖励函数按环境 进行加和（即每个环境的所有奖励加起来
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value #log 奖励为每个环境的对应key的奖励累加，直到reset时统一计算/刷新
        return reward

    def _get_rewards_qxj(self) -> torch.Tensor:
        total_reward, reward_log = compute_rewards(
            self.cfg.rew_termination,
            self.cfg.rew_action_l2,
            self.cfg.rew_joint_pos_limits,
            self.cfg.rew_joint_acc_l2,
            self.cfg.rew_joint_vel_l2,
            self.reset_terminated,
            self.actions,
            self.robot.data.joint_pos,
            self.robot.data.soft_joint_pos_limits,
            self.robot.data.joint_acc,
            self.robot.data.joint_vel,    
        )
        # self.extras["log"] = reward_log
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        #qxj_add_commands_reset
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(0.8, 1.5)
        #qxj_add_actions_reset
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        #qxj_add_Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        #qxj_add
        # self.extras["log"] = dict()
        self.extras = {"log": {}}
        self.extras["log"].update(extras)
        # extras = dict()
        # extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        # extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # self.extras["log"].update(extras)
        
    # reset strategies
    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # get root transforms (the humanoid torso)base_link  pelvis:run+walk_base  torso:init
        motion_torso_index = self._motion_loader.get_body_index(["base_link"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.02 #run:-0.15  -0.2 sideflip -0.05 # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_rotate(q, ref_tangent)
    normal = quat_rotate(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_rewards(
    rew_scale_termination: float,
    rew_scale_action_l2: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_joint_acc_l2: float,
    rew_scale_joint_vel_l2: float,
    reset_terminated: torch.Tensor,
    actions: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    joint_acc: torch.Tensor,
    joint_vel: torch.Tensor,
):
    rew_termination = rew_scale_termination * reset_terminated.float()
    rew_action_l2 = rew_scale_action_l2 * torch.sum(torch.square(actions), dim=1)
    
    out_of_limits = -(joint_pos - soft_joint_pos_limits[:,:,0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_joint_pos_limits[:,:,1]).clip(min=0.0)
    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(out_of_limits, dim=1)
    
    rew_joint_acc_l2 = rew_scale_joint_acc_l2 * torch.sum(torch.square(joint_acc), dim=1)
    rew_joint_vel_l2 = rew_scale_joint_vel_l2 * torch.sum(torch.square(joint_vel), dim=1)
    total_reward = rew_termination + rew_action_l2 + rew_joint_pos_limits + rew_joint_acc_l2 + rew_joint_vel_l2
    
    log = {
        "rew_termination": (rew_termination).mean(),
        "rew_action_l2": (rew_action_l2).mean(),
        "rew_joint_pos_limits": (rew_joint_pos_limits).mean(),
        "rew_joint_acc_l2": (rew_joint_acc_l2).mean(),
        "rew_joint_vel_l2": (rew_joint_vel_l2).mean(),
        }
    return total_reward, log