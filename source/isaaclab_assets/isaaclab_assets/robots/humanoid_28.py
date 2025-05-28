# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 28-DOFs Mujoco Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

HUMANOID_28_CFG = ArticulationCfg(
    # prim_path="{ENV_REGEX_NS}/Robot",
    # spawn=sim_utils.UsdFileCfg(
    #     usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Humanoid28/humanoid_28.usd",
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=None,
    #         max_depenetration_velocity=10.0,
    #         enable_gyroscopic_forces=True,
    #     ),
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=True,
    #         solver_position_iteration_count=4,
    #         solver_velocity_iteration_count=0,
    #         sleep_threshold=0.005,
    #         stabilization_threshold=0.001,
    #     ),
    #     copy_from_source=False,
    # ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.8),
    #     joint_pos={".*": 0.0},
    # ),
    # actuators={
    #     "body": ImplicitActuatorCfg(
    #         joint_names_expr=[".*"],
    #         stiffness=None,
    #         damping=None,
    #     ),
    # },
    #*****************************************************************************************
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/K188_usd.usd",
        # usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/usd_sideflip/K188qxj.usd",#sideflip
        usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/usd_dance/K188dance.usd",#dance
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    },
    #********************************************************************************************
    # prim_path="{ENV_REGEX_NS}/Robot",
    # spawn=sim_utils.UsdFileCfg(
    #     usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/CR01A_noarm/CR01A_noarm.usd",#noarm walk
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=None,
    #         max_depenetration_velocity=10.0,
    #         enable_gyroscopic_forces=True,
    #     ),
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=False,
    #         solver_position_iteration_count=4,
    #         solver_velocity_iteration_count=0,
    #         sleep_threshold=0.005,
    #         stabilization_threshold=0.001,
    #     ),
    #     copy_from_source=False,
    # ),
    # init_state=ArticulationCfg.InitialStateCfg(
    #     pos=(0.0, 0.0, 0.9),
    #     joint_pos={".*": 0.0},
    # ),
    # actuators={
    #     "body": ImplicitActuatorCfg(
    #         joint_names_expr=[".*"],
    #         stiffness=None,
    #         damping=None,
    #     ),
    # },
)
"""Configuration for the 28-DOFs Mujoco Humanoid robot."""
CR01A_noarm_amp_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/usd_amp_cra/cra1_noarm_amp.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),#enabled_self_collisions=False  #default is False
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.88),
        joint_pos={
            ".*_hip_y": -0.2,
            ".*_knee": 0.4,
            ".*_ankle_y": -0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_z",
                ".*_hip_x",
                ".*_hip_y",
                ".*_knee",
            ],
            effort_limit={
                ".*_hip_z": 120.0,
                ".*_hip_x": 80.0,
                ".*_hip_y": 400.0,
                ".*_knee": 400.0,
            },
            velocity_limit=10.0,
            stiffness={
                ".*_hip_z": 150.0,
                ".*_hip_x": 150.0,
                ".*_hip_y": 200.0,
                ".*_knee": 200.0,
            },
            damping={
                ".*_hip_z": 3.0,
                ".*_hip_x": 3.0,
                ".*_hip_y": 4.0,
                ".*_knee": 4.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_y", ".*_ankle_x"],
            effort_limit={
                ".*_ankle_y": 120,
                ".*_ankle_x": 40,
            },
            velocity_limit=10.0,
            stiffness={
                ".*_ankle_y": 150.0, #30 #100
                ".*_ankle_x": 80.0, #20 #50
            },
            # stiffness = 20,
            damping={
                ".*_ankle_y": 3.0, #1 #2
                ".*_ankle_x": 1.0, #0.8 #1
            },
            armature=0.01,
        ),
    },
)