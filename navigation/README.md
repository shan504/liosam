# Go2 多点导航

先保存建图结果，等保存窗口打印地图路径后，在原建图窗口按 Ctrl+C。Go2 站稳后运行 `bash navigation/start.sh 地图名`。地图名对应 `~/go2_maps/` 下由 `noetic/save_map.sh` 保存的地图；默认读取该名称下最近完成的版本。也可以传入版本目录的绝对路径。启动命令复用实机已安装的 Noetic MID-360S 驱动、Humble 镜像 `go2-3d:edu` 和本体 DDS 环境。

首次启动会从逐帧索引、优化关键帧和保存的点云生成 5 cm 二维占用图和 10 cm GICP 配准目标。地图使用保存时的地面坐标系；用于静态地图的地面、障碍高度带和安装外参可查本版本派生目录的 `metadata.json`。未观测区域保留为未知。源图需要配套帧索引、轨迹和标定。派生结果在 `~/.cache/go2_lio_navigation/`，更新地图内容、标定或地图处理代码会得到新的缓存。

## 首次多点测试

1. 如果从建图起点同朝向出发，等待自动定位。在其他位置出发时，在 Foxglove 左侧的“估计位姿”工具指定机器狗在图上的位置和朝向。GICP 是局部配准，初始估计要接近实际机身。
2. 使用终端打印的 Wi-Fi WebSocket 地址，导入本目录的 `foxglove.json`。等定位有效且 Nav2 就绪，先在右侧导航视图选一个空闲位置并指定到达朝向；再添加其余航点，编号按加入顺序排列。
3. 点击“启用运动”开始或恢复队列。Nav2 按顺序执行，每次只接受一个目标。“停止运动＋取消导航”停下机器狗，撤销本次 Nav2 任务，保留当前及后续航点。“取消当前目标”停下并移除活动航点，剩余航点等手动启用。需要放弃全部航点时点“清空整条队列”。操作结果显示在 `/operator/status` 和 `/control/status`。Ctrl+C 结束导航。
4. 需要数值输入时，可在另一个终端使用下面的命令；将 `X,Y,YAW` 换为这张地图的实际目标，单位为米、米、度。该命令只排队。
```bash
sudo docker exec go2-lio-nav bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && python3 /repo/navigation/waypoints.py -- "$@"' bash X1,Y1,YAW1 X2,Y2,YAW2
```

## 共用参数

`nav2.yaml` 供所有地图使用。NavFn A* 使用全局静态栅格与膨胀代价；局部 DWB 在 4 × 4 m 滚动窗口内使用静态层、MID‑360S 扫描障碍层和膨胀层。机身轮廓默认 0.80 × 0.44 m，另设 0.01 m padding，两个膨胀层的半径均为 0.225 m。调节转向可修改 `FollowPath.angular_dist_threshold`、`angular_disengage_threshold`、`rotate_to_heading_angular_vel` 和终点 `yaw_goal_tolerance`。默认约 20° 进入路径朝向校正，约 11.5° 退出；前进上限 0.30 m/s，原地转向上限 0.50 rad/s；到点容差 0.30 m 和约 11.5°。这些是初始参数，现场精度取决于定位及地面条件。

Go2 原生里程计发布 odom→base_link。GICP 按扫描时间配准，当连续三帧通过检查后发布 map→odom。定位有效且 TF 就绪后才激活 Nav2；运动桥还检查实时扫描、里程计及当前队列动作。建图与导航是两个独立运行会话。导航界面用白色表示已观测的可通行栅格、黑色表示静态障碍、灰色表示未知。全局与局部代价图可以在 3D 面板中单独勾选。两个视图通过 `/go2_visual/live/model_mesh` 显示建图所用的 Go2 模型。

在两个 3D 视图中，已存障碍点云 `/map_cloud` 和实时障碍点云 `/navigation/cloud_filtered` 都按距标定地面的高度着色。离地达到 0.10 m 的有效回波保留到原始数据的最高处，墙面上部、横梁和顶棚都可以显示。地图准备时从原始地图筛出这些点，默认以 4 cm 体素生成细节显示缓存，大地图会自动调粗以控制传输量。实时点使用扫描时间的已确认 GICP 位姿及雷达安装外参换算至同一地面坐标系，定位失效时清除上次显示的实时点。

GICP 使用完整点云，栅格使用地面回波判断已观测的通行区域。实时障碍层使用 `/navigation/obstacles` 中具有实际雷达坐标与时间戳的障碍回波。离线的跨地图回归可用 `python3 navigation/tests/test_map_scope.py` 执行。
