# Go2 standing mesh

`standing.npz` contains the triangle coordinates, face material indices and RGBA palette used by the Go2 mapping display. Coordinates are in meters in the body frame. The standing joint angles are 0, +0.8 and −1.55 radians for hip, thigh and calf. This fixed pose is for visualization.

The geometry comes from Unitree Robotics' Go2 CAD. The mapping display used 16 reduced material submeshes derived from [MuJoCo Menagerie's Go2 assets](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_go2). Material colors were adjusted for visibility.

The transforms for all four legs and feet come from the [RoboVerse Go2 SDK URDF](https://github.com/abizovnuralem/go2_ros2_sdk/blob/b440609591a249e7bdd4bbc88e056a3660575447/go2_robot_sdk/urdf/go2.urdf).

The corresponding notices are in `LICENSE-Unitree.txt` and `LICENSE-RoboVerse.txt`.
