# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg,ImuCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.modifiers import DigitalFilterCfg
import isaaclab.utils.modifiers as modifiers

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    #imu
    # imu_RF = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/LF_FOOT", debug_vis=True)
    # imu = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link",
    #     offset=ImuCfg.OffsetCfg(
    #     pos=(0.0, 0.0, 0.0),        # 相对于父框架的坐标 (x, y, z)
    #     rot=(1.0, 0.0, 0.0, 0.0)    # 四元数 (w, x, y, z) 表示无旋转
    #     ),
    #     debug_vis=False,
    #     gravity_bias=(0.0, 0.0, 9.81),
    # )
    
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*_hip_.*",".*_knee_joint",".*_ankle_.*",".*_shoulder_pitch_joint",".*_elbow_pitch_joint"], scale=0.5, use_default_offset=True)
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
                                           preserve_order = True,
                                           joint_names=["left_hip_pitch_joint",
                                                        "left_shoulder_pitch_joint",
                                                        "right_hip_pitch_joint",
                                                        "right_shoulder_pitch_joint",
                                                        "left_hip_roll_joint",
                                                        "right_hip_roll_joint",
                                                        "left_hip_yaw_joint",
                                                        "right_hip_yaw_joint",
                                                        "left_knee_joint",
                                                        # "left_elbow_pitch_joint",
                                                        "right_knee_joint",
                                                        # "right_elbow_pitch_joint",
                                                        "left_ankle_pitch_joint",
                                                        "right_ankle_pitch_joint",
                                                        "left_ankle_roll_joint",
                                                        "right_ankle_roll_joint",
                                                        "waist_yaw_joint",#17
                                                        # "left_shoulder_roll_joint",#18
                                                        # "right_shoulder_roll_joint",#19
                                                        # "left_shoulder_yaw_joint",#20
                                                        # "right_shoulder_yaw_joint",#21
                                                        ], 
                                           scale=0.5, 
                                           use_default_offset=True)
    # 站立姿态
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
    #                                        preserve_order = True,
    #                                        joint_names=["left_hip_pitch_joint",
    #                                                     "right_hip_pitch_joint",
    #                                                     "left_hip_roll_joint",
    #                                                     "right_hip_roll_joint",
    #                                                     "left_hip_yaw_joint",
    #                                                     "right_hip_yaw_joint",
    #                                                     "left_knee_joint",
    #                                                     "right_knee_joint",
    #                                                     "left_ankle_pitch_joint",
    #                                                     "right_ankle_pitch_joint",
    #                                                     "left_ankle_roll_joint",
    #                                                     "right_ankle_roll_joint",
    #                                                     "waist_yaw_joint",
    #                                                     "waist_roll_joint",
    #                                                     "waist_pitch_joint",
    #                                                     ], 
    #                                        scale=0.5, 
    #                                        use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))#3
        # base_lin_acc = ObsTerm(func=mdp.imu_lin_acc, noise=Unoise(n_min=-0.1, n_max=0.1))#3

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))#3
        # base_ang_vel = ObsTerm(
        #     func=mdp.base_ang_vel, 
        #     noise=Unoise(n_min=-0.2, n_max=0.2),
        #     modifiers=[DigitalFilterCfg(
        #                         A=[0.0],               # 不需要输出反馈（递归部分）
        #                         B=[0.0, 0.0, 1.0]   # 延迟2个周期的系数
        #                     )],
        # )#3
        
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )#3

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})#3

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)) #26
        # joint_pos = ObsTerm(
        #     func=mdp.joint_pos_rel, 
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     modifiers=[DigitalFilterCfg(
        #                         A=[0.0],               # 不需要输出反馈（递归部分）
        #                         B=[0.0, 0.0, 1.0]   # 延迟2个周期的系数
        #                     )],
        # ) #26
        # 站立姿态不观测上身的关节角度
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
        #                     noise=Unoise(n_min=-0.01, n_max=0.01),
        #                     params={
        #                         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*",
        #                                                                           ".*_knee_.*",
        #                                                                           ".*_ankle_.*",
        #                                                                           "waist_.*"]),
        #                     },
        #                     ) #26
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)) #26
        # 站立姿态不观测上身的关节角度
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, 
        #                     noise=Unoise(n_min=-1.5, n_max=1.5),
        #                     params={
        #                         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*",
        #                                                                           ".*_knee_.*",
        #                                                                           ".*_ankle_.*",
        #                                                                           "waist_.*"]),
        #                     },
        #                     )
        actions = ObsTerm(func=mdp.last_action) #26  // 3 + 3 + 3 + 3 + 26 + 26 + 26=90

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )#187

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),#0.8 0.8
            "dynamic_friction_range": (0.6, 0.8),#0.6 0.6
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # scale_all_link_masses = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.9, 1.1),
    #             "operation": "scale"},
    # )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "mass_distribution_params": (0.0, 4.0),
            # "mass_distribution_params": (-1.5, 4.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.02, 0.02), "z": (-0.01, 0.05)},
            # "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # scale_all_joint_armature = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "armature_distribution_params": (1.0, 1.05),
    #             "operation": "scale"},
    # )

    # add_all_joint_default_pos = EventTerm(
    #     func=mdp.randomize_joint_default_pos,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "pos_distribution_params": (-0.05, 0.05),
    #             "operation": "add"},
    # )

    # scale_all_joint_friction_model = EventTerm(
    #     func=mdp.randomize_joint_friction_model,
    #     mode="startup",
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "friction_distribution_params": (0.9, 1.1),
    #             "operation": "scale"},
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
            # "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*",".*_elbow_.*",".*_wrist_.*"]),
        },
    )
    joint_pd_randomization = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",  # 在每次环境重置时触发
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),  # 刚度随机范围 (N·m/rad)
            "damping_distribution_params": (0.9, 1.1),  # 阻尼临界阻尼系数比例
            "operation":"scale",
            "distribution":"log_uniform",
        }
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )#线速度xy的跟踪奖励
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )#角速度z的跟踪奖励
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0) # 线速度z的惩罚
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05) # 角速度xy的惩罚
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5) # 特定关节力矩的惩罚
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) # 特定关节加速度的惩罚
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01) # 动作改变率的惩罚
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )# 足底腾空时间的奖励
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )# 碰撞的惩罚
    
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0) # 姿态角度平稳
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0) # 关节位置限制


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    arm_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    elbow_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    base_height = DoneTerm(
        func=mdp.illegal_height,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "command_name": "base_velocity",
                 "threshold": 0.84},
    )
    # knee_torque = DoneTerm(
    #     func=mdp.illegal_torque,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee_joint"), "threshold": 100},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # push force follows curriculum
    # push_force_levels = CurrTerm(func=mdp.modify_push_force,
    #                              params={"term_name": "push_robot", "max_velocity": [3.0, 3.0], "interval": 200 * 24,
    #                                      "starting_step": 1500 * 24})
    # command vel follows curriculum
    command_vel = CurrTerm(func=mdp.modify_command_velocity,
                           params={"term_name": "track_lin_vel_xy_exp", "max_velocity": [-0.8, 1.5],
                                   "interval": 1000 * 24, "starting_step": 2000 * 24})
    
    # modify_reward_weight = CurrTerm(func=mdp.modify_reward_weight,
    #                        params={"term_name": "step_knee2",
    #                                "weight": 0.1, 
    #                                "num_steps": 20*50*10})


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005  #200hz
        self.sim.render_interval = self.decimation #50hz
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
