# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with DC motor model for the legs
* :obj:`UNITREE_GO1_CFG`: Unitree Go1 robot with actuator net model for the legs
* :obj:`UNITREE_GO2_CFG`: Unitree Go2 robot with DC motor model for the legs
* :obj:`H1_CFG`: H1 humanoid robot
* :obj:`H1_MINIMAL_CFG`: H1 humanoid robot with minimal collision bodies
* :obj:`G1_CFG`: G1 humanoid robot
* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdentifiedActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration - Actuators.
##


Wukong4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/NavAlpha_usd/NavAlpha.usd",
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
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            ".*_hip_pitch_joint": -0.15,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.15,
            ".*_elbow_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.6,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.6,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit=120,
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "torso_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=30,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit=30,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
            },
        ),
    },
)
"""Configuration for the NavAlpha Humanoid robot."""
Wukong4_MINIMAL_CFG = Wukong4_CFG.copy()
# Wukong4_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd"

CR01A_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/CR01A_usd/CR01A.usd",
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
            ".*_hip_pitch_joint": -0.2,
            ".*_knee_joint": 0.4,
            ".*_ankle_pitch_joint": -0.2,
            ".*_elbow_pitch_joint": -1.50,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.4,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 400.0,
                ".*_knee_joint": 400.0,
            },
            velocity_limit=20.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit={
                ".*_ankle_pitch_joint": 120,
                ".*_ankle_roll_joint": 40,
            },
            velocity_limit=20.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 80.0,
                ".*_shoulder_roll_joint": 80.0,
                ".*_shoulder_yaw_joint": 80.0,
                ".*_elbow_pitch_joint": 80.0,
                ".*_wrist_pitch_joint": 30.0,
                ".*_wrist_yaw_joint": 30.0,
                ".*_wrist_roll_joint": 30.0,
            },
            velocity_limit=20.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01,
            },
        ),
    },
)
"""Configuration for the NavAlpha Humanoid robot."""
CR01A_MINIMAL_CFG = CR01A_CFG.copy()

# CR01A_noarm_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/CR01A_noarm/CR01A_noarm.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
#         ),#enabled_self_collisions=False  #default is False
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.85),
#         joint_pos={
#             ".*_hip_pitch_joint": -0.35,
#             ".*_knee_joint": 0.7,
#             ".*_ankle_pitch_joint": -0.35,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "legs": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_hip_yaw_joint",
#                 ".*_hip_roll_joint",
#                 ".*_hip_pitch_joint",
#                 ".*_knee_joint",
#             ],
#             effort_limit={
#                 ".*_hip_yaw_joint": 120.0,
#                 ".*_hip_roll_joint": 120.0,
#                 ".*_hip_pitch_joint": 400.0,
#                 ".*_knee_joint": 400.0,
#             },
#             velocity_limit=20.0,
#             stiffness={
#                 ".*_hip_yaw_joint": 150.0,
#                 ".*_hip_roll_joint": 150.0,
#                 ".*_hip_pitch_joint": 200.0,
#                 ".*_knee_joint": 200.0,
#             },
#             damping={
#                 ".*_hip_yaw_joint": 5.0,
#                 ".*_hip_roll_joint": 5.0,
#                 ".*_hip_pitch_joint": 5.0,
#                 ".*_knee_joint": 5.0,
#             },
#             armature={
#                 ".*_hip_.*": 0.01,
#                 ".*_knee_joint": 0.01,
#             },
#         ),
#         "feet": ImplicitActuatorCfg(
#             joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
#             effort_limit={
#                 ".*_ankle_pitch_joint": 120,
#                 ".*_ankle_roll_joint": 40,
#             },
#             velocity_limit=20.0,
#             stiffness=20.0,
#             damping=2.0,
#             armature=0.01,
#         ),
#         # "arms": ImplicitActuatorCfg(
#         #     joint_names_expr=[
#         #         ".*_shoulder_pitch_joint",
#         #         ".*_shoulder_roll_joint",
#         #         ".*_shoulder_yaw_joint",
#         #         ".*_elbow_pitch_joint",
#         #         ".*_wrist_pitch_joint",
#         #         ".*_wrist_yaw_joint",
#         #         ".*_wrist_roll_joint",
#         #     ],
#         #     effort_limit={
#         #         ".*_shoulder_pitch_joint": 80.0,
#         #         ".*_shoulder_roll_joint": 80.0,
#         #         ".*_shoulder_yaw_joint": 80.0,
#         #         ".*_elbow_pitch_joint": 80.0,
#         #         ".*_wrist_pitch_joint": 30.0,
#         #         ".*_wrist_yaw_joint": 30.0,
#         #         ".*_wrist_roll_joint": 30.0,
#         #     },
#         #     velocity_limit=20.0,
#         #     stiffness=40.0,
#         #     damping=10.0,
#         #     armature={
#         #         ".*_shoulder_.*": 0.01,
#         #         ".*_elbow_.*": 0.01,
#         #         ".*_wrist_.*": 0.01,
#         #     },
#         # ),
#     },
# )
# """Configuration for the NavAlpha Humanoid robot."""
# CR01A_noarm_MINIMAL_CFG = CR01A_noarm_CFG.copy()

CR01ADC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/CR01A_usd/CR01A.usd",
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
            ".*_hip_pitch_joint": -0.2,
            ".*_knee_joint": 0.4,
            ".*_ankle_pitch_joint": -0.2,
            ".*_elbow_pitch_joint": -1.50,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.4,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs1": IdentifiedActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
            },
            velocity_limit=20.0,
            saturation_effort = 120.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
            },
            friction_static=0.0,
            activation_vel=0.0,
            friction_dynamic=0.0,
        ),
        "legs2": IdentifiedActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_pitch_joint": 400.0,
                ".*_knee_joint": 400.0,
            },
            velocity_limit=20.0,
            saturation_effort = 400.0,
            stiffness={
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
            friction_static=0.0,
            activation_vel=0.1,
            friction_dynamic=0.0,
        ),
        "feet1": IdentifiedActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit={
                ".*_ankle_pitch_joint": 120,
            },
            velocity_limit=20.0,
            saturation_effort = 120.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
            friction_static=0.0,
            activation_vel=0.1,
            friction_dynamic=0.0,
        ),
        "feet2": IdentifiedActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit={
                ".*_ankle_roll_joint": 40,
            },
            velocity_limit=20.0,
            saturation_effort = 40.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
            friction_static=0.0,
            activation_vel=0.1,
            friction_dynamic=0.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 80.0,
                ".*_shoulder_roll_joint": 80.0,
                ".*_shoulder_yaw_joint": 80.0,
                ".*_elbow_pitch_joint": 80.0,
                ".*_wrist_pitch_joint": 30.0,
                ".*_wrist_yaw_joint": 30.0,
                ".*_wrist_roll_joint": 30.0,
            },
            velocity_limit=20.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01,
            },
        ),
    },
)
"""Configuration for the NavAlpha Humanoid robot."""
CR01ADC_MINIMAL_CFG = CR01ADC_CFG.copy()

CR01ADC_noarm_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/CR01A_usd/CR01A.usd",
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
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*_hip_pitch_joint": -0.35,
            ".*_knee_joint": 0.7,
            ".*_ankle_pitch_joint": -0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs1": IdentifiedActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
            },
            velocity_limit=20.0,
            saturation_effort = 120.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
            },
            friction_static=0.0,
            activation_vel=0.0,
            friction_dynamic=0.0,
        ),
        "legs2": IdentifiedActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_pitch_joint": 400.0,
                ".*_knee_joint": 400.0,
            },
            velocity_limit=20.0,
            saturation_effort = 400.0,
            stiffness={
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
            friction_static=0.0,
            activation_vel=0.1,
            friction_dynamic=0.0,
        ),
        "feet1": IdentifiedActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit={
                ".*_ankle_pitch_joint": 120,
            },
            velocity_limit=20.0,
            saturation_effort = 120.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
            friction_static=0.0,
            activation_vel=0.1,
            friction_dynamic=0.0,
        ),
        "feet2": IdentifiedActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit={
                ".*_ankle_roll_joint": 40,
            },
            velocity_limit=20.0,
            saturation_effort = 40.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
            friction_static=0.0,
            activation_vel=0.1,
            friction_dynamic=0.0,
        ),
    },
)
"""Configuration for the NavAlpha Humanoid robot."""
CR01ADC_noarm_MINIMAL_CFG = CR01ADC_noarm_CFG.copy()



################################
CR01B_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/ma/Learning/IsaacLab/IsaacLab_qxj/Robots_usd/CRB0_usd/CRB0.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),#enabled_self_collisions=False  #default is False
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*_hip_pitch_joint": -0.2,
            ".*_knee_joint": 0.4,
            ".*_ankle_pitch_joint": -0.2,
            ".*_elbow_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 80.0,
                ".*_hip_pitch_joint": 400.0,
                ".*_knee_joint": 400.0,
            },
            velocity_limit_sim=20.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 3.0,
                ".*_hip_roll_joint": 3.0,
                ".*_hip_pitch_joint": 4.0,
                ".*_knee_joint": 4.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 120,
                ".*_ankle_roll_joint": 40,
            },
            velocity_limit_sim=20.0,
            stiffness={
                ".*_ankle_pitch_joint": 100.0, #30 #150
                ".*_ankle_roll_joint": 50.0, #20 #80
            },
            damping={
                ".*_ankle_pitch_joint": 2.0, #1 #2
                ".*_ankle_roll_joint": 1.0, #0.8 #1
            },
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 120.0,
                ".*_shoulder_roll_joint": 120.0,
                ".*_shoulder_yaw_joint": 120.0,
                ".*_elbow_pitch_joint": 120.0,
                ".*_wrist_pitch_joint": 30.0,
                ".*_wrist_yaw_joint": 30.0,
                ".*_wrist_roll_joint": 30.0,
            },
            velocity_limit_sim=20.0,
            stiffness={
                ".*_shoulder_pitch_joint": 150.0,
                ".*_shoulder_roll_joint": 150.0,
                ".*_shoulder_yaw_joint": 150.0,
                ".*_elbow_pitch_joint": 150.0,
                ".*_wrist_pitch_joint": 50.0,
                ".*_wrist_yaw_joint": 50.0,
                ".*_wrist_roll_joint": 50.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 3.0,
                ".*_shoulder_roll_joint": 3.0,
                ".*_shoulder_yaw_joint": 3.0,
                ".*_elbow_pitch_joint": 3.0,
                ".*_wrist_pitch_joint": 1.0,
                ".*_wrist_yaw_joint": 1.0,
                ".*_wrist_roll_joint": 1.0,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01,
            },
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_.*"],
            effort_limit_sim={
                "waist_yaw_joint": 120,
                "waist_roll_joint": 120,
                "waist_pitch_joint": 400,
            },
            velocity_limit_sim=20.0,
            stiffness={
                "waist_yaw_joint": 150,
                "waist_roll_joint": 150,
                "waist_pitch_joint": 200,
            },
            damping={
                "waist_yaw_joint": 3,
                "waist_roll_joint": 3,
                "waist_pitch_joint": 4,
            },
            armature=0.01,
        ),
    },
)
"""Configuration for the NavAlpha Humanoid robot."""
CR01B_RL_CFG = CR01B_CFG.copy()
CR01B_amp_CFG = CR01B_CFG.copy()