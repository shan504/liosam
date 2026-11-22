# Motion-triggered drift: targeted feedback correction

The online stationary checks were stable but did not reproduce the user's failure
after walking. They cannot establish moving accuracy or prove this fix sufficient.

The installed MID360 fork mixes a hard-coded 0.1 IMU attitude weight into its
incremental mapping odometry even when `imuRPYWeight` is zero. That odometry is
fed back into `IMUPreintegration::odometryHandler`. The current input is a
Madgwick orientation synthesized from six-axis Livox measurements, not an
independent nine-axis attitude measurement.

References inspected:

- https://github.com/TixiaoShan/LIO-SAM (IMU data, synchronization and extrinsics)
- https://github.com/YJZLuckyBoy/liorf (six-axis IMU support)
- https://github.com/YJZLuckyBoy/liorf/blob/main/src/mapOptmization.cpp
  (incremental attitude correction is guarded by `imuType`)

This patch adds the six/nine-axis guard and makes the incremental correction obey
`imuRPYWeight`, which startup explicitly sets to zero. It retains SI conversion,
initial tilt, gyroscope deskew, preintegration and the previous reset repair.
It does not freeze, smooth, clamp or fabricate the robot's pose. It does not
claim the hard-coded feedback is the only cause of the reported drift.

After stopping the original startup session, run `bash noetic/repair_mapping.sh`.
It backs up the installed source, applies the patch, builds the actual catkin
workspace and starts a fresh map. Prior archives remain. Build failure restores
source and prevents startup; a full successful rebuild is required before use.

`I_motion.log` and the startup terminal report source-stamped step sizes for map,
incremental map, IMU predictor and model every four seconds, including predictor
reset transitions and LIO-SAM warnings. It observes only and does not record bags
or send movement commands. Step sizes during walking include real movement;
they are not automatically classified as drift. No new model messages while
stationary can be normal for the thresholded model publisher.

Local validation: patch upgrade from pristine and previously patched source,
idempotency, extracted C++ feedback branch with test quaternion stand-ins, shell
syntax and telemetry tests for jumps, resets and quaternion sign. Full ROS/PCL/
GTSAM build and moving accuracy require the robot; neither is claimed tested here.
