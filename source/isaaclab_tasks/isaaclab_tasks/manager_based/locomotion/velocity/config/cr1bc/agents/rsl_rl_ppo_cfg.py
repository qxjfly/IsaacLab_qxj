# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

@configclass
class CR1BCRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "cr1bc_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CR1BCFlatPPORunnerCfg(CR1BCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 35000
        self.experiment_name = "cr1bc_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]

        # symmetry
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=mirror_data_augmentation_func,
            mirror_loss_coeff=0.1,
        )


def mirror_data_augmentation_func(env, obs=None, actions=None, obs_type="policy", is_critic=False,):
    """Mirror data augmentation function for symmetry framework."""

    # Get mirror IDs
    mirror_cfg = getattr(getattr(env, "cfg", None), "mirror_cfg", None)
    policy_obvs_mirror_id_left  = getattr(mirror_cfg, "policy_obvs_mirror_id_left" , None)
    policy_obvs_mirror_id_right = getattr(mirror_cfg, "policy_obvs_mirror_id_right", None)
    policy_obvs_opposite_id     = getattr(mirror_cfg, "policy_obvs_opposite_id"    , None)
    action_mirror_id_left       = getattr(mirror_cfg, "action_mirror_id_left"      , None)
    action_mirror_id_right      = getattr(mirror_cfg, "action_mirror_id_right"     , None)
    action_opposite_id          = getattr(mirror_cfg, "action_opposite_id_a"         , None)

    if obs_type == "policy":
        # Mirror processing for policy observations
        if obs is not None:
            # batch_size = obs.shape[0]
            mirrored_obs = obs.clone()

            # Apply mirror transformations...
            if policy_obvs_opposite_id is not None:
                mirrored_obs[:, policy_obvs_opposite_id] = -mirrored_obs[:, policy_obvs_opposite_id]
            
            if policy_obvs_mirror_id_left is not None and policy_obvs_mirror_id_right is not None:
                temp = mirrored_obs[:, policy_obvs_mirror_id_left].clone()
                mirrored_obs[:, policy_obvs_mirror_id_left] = mirrored_obs[:, policy_obvs_mirror_id_right]
                mirrored_obs[:, policy_obvs_mirror_id_right] = temp
            
            combined_obs = torch.cat([obs, mirrored_obs], dim=0)
        else:
            combined_obs = None
    else:  # critic or other types
        # For critic, simply duplicate original data to match batch size
        if obs is not None:
            combined_obs = torch.cat([obs, obs.clone()], dim=0)
        else:
            combined_obs = None
    
    # Action processing (only when actions are provided)
    if actions is not None:
        # batch_size = actions.shape[0]
        mirrored_actions = actions.clone()
        
        # Apply mirror transformations...
        if action_opposite_id is not None:
            mirrored_actions[:, action_opposite_id] = -mirrored_actions[:, action_opposite_id]
        
        if action_mirror_id_left is not None and action_mirror_id_right is not None:
            temp = mirrored_actions[:, action_mirror_id_left].clone()
            mirrored_actions[:, action_mirror_id_left] = mirrored_actions[:, action_mirror_id_right]
            mirrored_actions[:, action_mirror_id_right] = temp
        
        combined_actions = torch.cat([actions, mirrored_actions], dim=0)
    else:
        combined_actions = None
    
    return combined_obs, combined_actions