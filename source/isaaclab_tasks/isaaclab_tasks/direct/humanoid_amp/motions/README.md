# Motion files

The motion files are in NumPy-file format that contains data from the skeleton DOFs and bodies that perform the motion.

The data (accessed by key) is described in the following table, where:

* `N` is the number of motion frames recorded
* `D` is the number of skeleton DOFs
* `B` is the number of skeleton bodies

| Key | Dtype | Shape | Description |
| --- | ---- | ----- | ----------- |
| `fps` | int64 | () | FPS at which motion was sampled |
| `dof_names` | unicode string | (D,) | Skeleton DOF names |
| `body_names` | unicode string | (B,) | Skeleton body names |
| `dof_positions` | float32 | (N, D) | Skeleton DOF positions |
| `dof_velocities` | float32 | (N, D) | Skeleton DOF velocities |
| `body_positions` | float32 | (N, B, 3) | Skeleton body positions |
| `body_rotations` | float32 | (N, B, 4) | Skeleton body rotations (as `wxyz` quaternion) |
| `body_linear_velocities` | float32 | (N, B, 3) | Skeleton body linear velocities |
| `body_angular_velocities` | float32 | (N, B, 3) | Skeleton body angular velocities |

## Motion visualization

The `motion_viewer.py` file allows to visualize the skeleton motion recorded in a motion file.

Open an terminal in the `motions` folder and run the following command.

```bash
python motion_viewer.py --file MOTION_FILE_NAME.npz
```

See `python motion_viewer.py --help` for available arguments.

## joint_names
[
'abdomen_x' 
'abdomen_y' 
'abdomen_z' 
'neck_x' 
'neck_y' 
'neck_z'
 'right_shoulder_x' 
 'right_shoulder_y' 
 'right_shoulder_z' 
 'right_elbow'
 'left_shoulder_x' 
 'left_shoulder_y' 
 'left_shoulder_z' 
 'left_elbow'
 'right_hip_x' 
 'right_hip_y' 
 'right_hip_z' 
 'right_knee' 
 'right_ankle_x'
 'right_ankle_y' 
 'right_ankle_z' 
 'left_hip_x' 
 'left_hip_y' 
 'left_hip_z'
 'left_knee' 
 'left_ankle_x' 
 'left_ankle_y' 
 'left_ankle_z'
 ]
 [
 'pelvis' 
 'torso' 
 'head' 
 'right_upper_arm' 
 'right_lower_arm' 
 'right_hand'
 'left_upper_arm' 
 'left_lower_arm' 
 'left_hand' 
 'right_thigh' 
 'right_shin'
 'right_foot' 
 'left_thigh' 
 'left_shin' 
 'left_foot'
 ]

 CR1A
left_hip_pitch_joint
left_shoulder_pitch_joint
right_hip_pitch_joint
right_shoulder_pitch_joint
left_hip_roll_joint
left_shoulder_roll_joint
right_hip_roll_joint
right_shoulder_roll_joint
left_hip_yaw_joint
left_shoulder_yaw_joint
right_hip_yaw_joint
right_shoulder_yaw_joint
left_knee_joint
left_elbow_pitch_joint
right_knee_joint
right_elbow_pitch_joint
left_ankle_pitch_joint
left_wrist_yaw_joint
right_ankle_pitch_joint
right_wrist_yaw_joint
left_ankle_roll_joint
left_wrist_pitch_joint
right_ankle_roll_joint
right_wrist_pitch_joint
left_wrist_roll_joint
right_wrist_roll_joint
