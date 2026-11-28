# 农场录包与离线导航回归

此流程复用每一张所选地图和公共定位、点云过滤、Nav2 参数。无需重新编译 C++。

## 农场录一次

正常启动导航；在另一个机器狗终端执行：

```bash
bash ~/go2-detail/navigation/record.sh farm_test
```

开始时静止约 10 秒，让定位有效，然后操作机器人经过需排查的路段，复现问题。录包窗口按 `Ctrl+C` 正常结束，保留导航窗口。脚本打印目录，例如 `~/go2_bags/farm_test_20261128_170000_abcd`；`~/go2_bags/LATEST` 指向最近完成的包。

每次记录携带当时所选地图的完整源存档、标定、轨迹、配置快照、代码版本和 ROS2 bag。录制：

- `/point_cloud2`：已按本体里程计对齐时间、去畸变的完整导航输入点云，机身坐标系。
- `/odom` 和 `/utlidar/robot_odom`：导航使用的归一化本体里程计及原始本体消息。
- 相机、TF、定位状态、障碍物输出、激光扫描、目标、规划轨迹和速度诊断。

`.db3` 每 2 GiB 分段，不限制总包大小。结束时检查点云和里程计是否确实录到消息。需要保留整个目录，不能只复制某一个 `.db3`。空间消耗取决于点云和相机频率，录制时留意磁盘余量。

这个包是**导航输入回归包**，不含 Noetic 原始 Livox CustomMsg/IMU，不能用它重新跑 LIO-SAM 原始建图或检查更早的雷达时钟转换。若将来要排查建图，需另外录制 ROS1 原始雷达和 IMU，不能把导航包当作原始建图包。

## 回家回放

先结束正在运行的建图或导航，释放 Foxglove 8766 端口。运行：

```bash
bash ~/go2-detail/navigation/replay.sh latest
```

指定某次包，或用半速降低计算负载：

```bash
bash ~/go2-detail/navigation/replay.sh ~/go2_bags/farm_test_20261128_170000_abcd 0.5
```

同一台机器狗可在家离线回放；不需要雷达、相机在线，也不需要连接农场。其他 Linux 电脑需要可运行的 `go2-3d:edu` 镜像、完整代码和录包目录，ARM64 镜像不能假定在 x86 电脑原生运行。Windows Foxglove 仍连接运行回放机器的 `ws://IP:8766`，使用同一个 `Go2.json`。

回放使用独立 Docker 网络、DDS domain 77 和 localhost-only。只播放点云、归一化里程计和相机，重新计算 TF、GICP、障碍物、代价地图。历史 TF、定位结论、目标、速度和运动授权只用于取证，**不重新发布**。启动链不包含 Unitree 控制适配器、本体 SDK 或在线传感器；Foxglove 只允许发布初始位姿和规划目标，运动按钮在离线模式不可用。

自动提取录包前 30 秒内首个有效定位，仅作为起始位姿提示；仍须重新通过 GICP。若没有有效定位，在 Foxglove 左侧设置实际初始位姿。每次更新代码后重跑同一个命令，会为相同源地图生成当前公共逻辑的缓存，不会覆盖录包源地图。

定位有效后，在 `map` 坐标系点选目标，会调用 Nav2 `ComputePathToPose` 并显示 `/plan`，`/operator/status` 显示规划成功或失败。回放轨迹由录包决定，**新路径不会改变机器人运动**；此流程能验证定位、障碍物与路径规划，不能证明新路线的闭环控制、多点到达、步态或实际避障已成功。最终仍需一次实地验证。

所有处理节点使用 `/clock`。默认原速，可用 0.5～1.0 倍速；不提供循环、倒退或任意跳转，避免旧时间戳污染 TF 和定位状态。暂停可能触发定位超时，恢复后须重新通过定位；重复试验请 `Ctrl+C` 后重新执行命令。

ROS2 参考：[录制与播放](https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html)、[QoS 覆盖](https://docs.ros.org/en/humble/How-To-Guides/Overriding-QoS-Policies-For-Recording-And-Playback.html)。
