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

## Validate both adapted inputs

Install the Foxy package `ros-foxy-imu-filter-madgwick`, rebuild the bridge, then run
`ros2 launch go2_mid360s_bridge inputs.launch.py` with the Livox driver already running.
This launches the cloud adapter, a g-to-m/s² converter and a six-axis Madgwick filter.
Do not also run a standalone cloud adapter. The SI converter expects the Livox protocol's
acceleration in g and angular velocity in rad/s. Timestamps and gyro units are preserved.
The filter keeps gravity and has relative yaw without a magnetometer.

From another sourced terminal run `python3 tools/check_inputs.py --adapted`.
This checks the outgoing PointCloud2 fields, scans one frame, and samples filtered IMU
acceleration and orientation. An acceleration magnitude near 9.8 while stationary and
a unit quaternion are basic input checks. Also verify roll/pitch direction under a
known tilt before relying on the attitude.

`config/mid360s_inputs.yaml` in the installed bridge share directory is a sensor-only
LIO-SAM overlay for four lines with a starting column budget of 6000 per line.
Review the maximum per-line count over a longer bag before final tuning.
The overlay is not a full mapping configuration; supply the verified internal
LiDAR/IMU transforms and robot mounting TF separately.

## Standalone mapping on Foxy

Use the pinned checkout and run `bash tools/prepare_foxy.sh` from this repository. This applies a small local patch to the submodule (Foxy rosidl and TF2 APIs, an empty-queue guard and a timestamped sensor-frame TF). A modified submodule working tree after this step is expected. See the pinned upstream README for GTSAM 4.1 and ROS dependencies. Build the bridge and `lio_sam` from the workspace root with the Livox overlay sourced.

With exactly one set of adapted input publishers running, launch `ros2 launch go2_mid360s_bridge mapping.launch.py`. For bag playback, record both adapted topics, then stop the live adapters before playing back their recording. Mapping needs several scans and IMU measurements spanning each scan. Watch `/lio_sam/deskew/cloud_deskewed`, `/lio_sam/mapping/odometry` and `/lio_sam/mapping/cloud_registered` for data, not just discovery. The world frame for this isolated test is `lio_odom`; TF traffic goes to `/lio_sam/tf` and `/lio_sam/tf_static`. The regular Go2 robot TF and velocity interface are separate. The orientation estimate uses relative yaw. Use the manufacturer's internal IMU lever arm only when the data is still expressed in the unmodified sensor coordinates. No camera or robot-body extrinsic is required for this bench profile.

Ground truth, loop acceptance and map quality must be assessed using motion data. Initial stationary odometry only checks that the pipeline is alive.
