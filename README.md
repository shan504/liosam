# MID-360S point cloud adapter for LIO-SAM

ROS 2 Foxy package for converting Livox Driver 2 CustomMsg to the PointCloud2 format consumed by the Livox path of [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM/tree/ros2). The upstream algorithm is pinned as a submodule. Hardware calibration, sensor addresses and site-specific configuration belong in the deployment workspace.

## Check the inputs

With the Livox driver running, source ROS 2 and the workspace that provides livox_ros_driver2 in another terminal. Keep the sensor stationary for the IMU check.

```bash
python3 tools/check_inputs.py
```

The probe reports point counts per line, time offsets and the point cloud frame. It also reports the stationary acceleration magnitude, quaternion and orientation covariance. Use the readings to verify acceleration units, orientation availability and scan parameters.

## Build and run the cloud adapter

Clone this repository into a colcon workspace's src directory. From the workspace root, with the Livox driver overlay sourced:

```bash
colcon build --packages-select go2_mid360s_bridge --parallel-workers 1
source install/setup.bash
ros2 run go2_mid360s_bridge cloud_adapter
```

Defaults: input /livox/lidar, output /mid360s/points. Use ROS remapping to change either topic. The adapter preserves the frame and stamp. It emits float32 x/y/z/intensity/time and uint16 ring, converts per-point time from nanoseconds to seconds and orders points by time if necessary. Empty frames and nonfinite positions are discarded. Only a point cloud is published.

## Upstream integration

From the repository directory, run `git submodule update --init upstream/LIO-SAM` to acquire the pinned ROS 2 code. For the Livox projection path, select sensor=livox, pointCloudTopic=/mid360s/points and size the projection from observed scan lines and their point counts. Provide an IMU stream with verified acceleration units, valid orientation and aligned timestamps. Validate LiDAR-to-IMU extrinsics independently of any camera calibration. Record both original topics for offline replay. When integrating with another robot description or localization stack, configure a single TF publisher per edge; the upstream example launch includes a sample robot description and a static map-to-odom transform.
