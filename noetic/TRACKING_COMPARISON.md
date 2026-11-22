# Go2 baseline comparison

Inspected `shan504/go2` main: `patches/dds/go2_edu_dds_bridge.py`,
`go2_3d/mapper.py`, `go2_3d/registration.py` and `go2_3d/localization_hold.py`.

The old system republishes `/utlidar/robot_odom` as `/odom` with validated timestamp
conversion and odom-to-body TF. The mapper aligns clouds with that time-indexed
body odometry, uses GICP acceptance criteria, and rejects inconsistent corrections
when independent body odometry indicates stationarity. Mapping additionally holds
its established correction anchor when that independent motion is sufficiently
small, while rechecking actual cloud overlap and residual. Localization filters
accepted map corrections only, without filtering raw body odometry.

This is not the current LIO-SAM predictor. Its output can change on optimization
feedback, so dividing a predictor pose step by its 5 ms source interval does not
measure real robot speed. It cannot independently certify that the robot is still.
The user's post-walk stationary report confirms residual mapping drift; the earlier
attitude feedback fix has not resolved it completely.

The previous model display also sampled poses at 2 Hz and retransmitted a 32436-
vertex marker. `model_tf.py` reuses its actual mesh, sole height, mounting transform,
calibrated ground and white reference plane, but latches the mesh once and publishes
each new mapping pose as a small stamped TF. A 20 Hz poll consumes new mapping
poses (currently 5 Hz). No extrapolation, smoothing or pose clamping is added.
`/go2_visual/live/model_pose` carries its actual ground-frame pose for the monitor.
The old `check-still` marker section describes the mesh's local frame after this
change; use the motion monitor's model_pose stream for actual model movement.

This display change does NOT repair drifting mapping coordinates. Before adopting
the old independent-motion constraint, `check_native.sh` reads available ROS2
Odometry topics, publishers, stationary displacement and unconverted clock offset.
It sends no robot commands and does not record a bag. Absence of discovered data
may be a DDS environment issue. Do not assume a native topic is independent based
only on its name, do not substitute a drifting LIO estimate, and do not guess a
ROS2-to-ROS1 time offset or sensor-to-body transform for fusion.

Run the check with the robot stationary while the current mapping monitor runs.
Only after checking the publisher, freshness and frame/time relationship should
native odometry constrain registration. Original maps remain untouched.

Local checks cover the actual mesh sole height, unchanged rigid placement, per-new-stamp TF emission, duplicate-stamp rejection and single mesh publication. ROS2 availability and robot motion accuracy remain unverified here.
