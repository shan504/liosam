# Go2 MID-360S 实机建图（ROS 1 Noetic）

本目录是一个独立启动入口，用于已完成雷达和 Noetic 环境配置的 Go2。使用雷达内置 IMU、Madgwick 姿态与原有的 `lio_sam module_loam.launch`。需已有 `~/mid360s_nk_noetic_ws`、`~/mid360s_imu_noetic_ws`、`~/go2_farm_ground/start_ground.sh` 和 `~/go2_model_live.pyz`。从雷达驱动的 MID360s 配置读取有线收发地址。

## 开始一轮

先在上一轮的启动窗口按 Ctrl+C 并等它退出。若另开过贴地终端，也在那个窗口退出。保持机器狗在平地静止，从此目录运行：

```bash
bash start.sh
```

地面拟合采用雷达中心垂直离地 0.50 m 作为先验，由这一轮点云确定地面方向和原点。看到“地面已锁定”、贴地模型已发布及稠密扫描数增长后开始走动。结束时在启动窗口按 Ctrl+C。每轮从新的 LIO-SAM 进程开始，点云与日志写到终端打印的 `~/go2_mapping_runs/run_*/`；`dense/scans.sqlite` 保存本轮选中的配准扫描和关键帧位姿。地图数据默认留在本机。

## Foxglove 布局

在已连接的 Foxglove 中导入一次同目录的 `farm_live_detail.json`。WebSocket 端口为 8766。左窗显示本轮累计地图；右窗在固定的地面参考系中保留最近 0.7 秒的配准扫描。两个面板都用 `/go2_visual/live/model_mesh` 显示脚底贴地的模型，白色网格来自 `/go2_visual/live/surface`。回环约束、路径和里程计的调试标记在这两个 3D 面板隐藏。相机与导航控件沿用原布局，运行相应节点后才有数据。

## 点云设置及检验

启动脚本中的 F 段设置稠密显示参数。输入为 `/lio_sam/mapping/cloud_registered_raw`，与本轮 `/lio_sam/mapping/path` 按消息头时间戳配对。每次捕获最短间隔 0.09 秒，显示体素边长 0.015 m，每格选取最接近格心的实测点并保留其反射强度。全局显示最多 45 万点、最多约每 3.5 秒刷新一次。显示达到点数上限后仍归档成功配对和选中的扫描。位姿优化后，以归档重新生成显示地图。

原地看同一处门框，再缓慢往返经过它。留意门框边缘是否保持单层，并查看 `Dense: … scans saved, … displayed, … unmatched/queue losses`。静止清楚而运动后出现双边时，优先用同一段数据核对时间戳、配准与位姿。显示体素的尺寸不代表测距精度。

- [Livox Mid-360S 官方参数](https://www.livoxtech.com/cn/mid-360s/specs)
- [LIO-SAM-MID360 的高分辨率扫描发布代码](https://github.com/nkymzsy/LIO-SAM-MID360/blob/21ef7bb108fe3c7218c102bf7cb3a06582a2fed9/src/mapOptmization.cpp#L1775-L1789)
- [Foxglove 3D 点云显示设置](https://docs.foxglove.dev/docs/visualization/panels/3d)
