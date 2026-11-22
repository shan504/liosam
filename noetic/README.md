# Go2 MID-360S 实机建图（ROS 1 Noetic）

本目录是一个独立启动入口，用于已完成雷达和 Noetic 环境配置的 Go2。使用雷达内置 IMU、Madgwick 姿态与原有的 `lio_sam module_loam.launch`。需已有 `~/mid360s_nk_noetic_ws`、`~/mid360s_imu_noetic_ws`、`~/go2_farm_ground/start_ground.sh` 和 `~/go2_model_live.pyz`。从雷达驱动的 MID360s 配置读取有线收发地址。

## 开始一轮

先在上一轮的启动窗口按 Ctrl+C 并等它退出。若另开过贴地终端，也在那个窗口退出。保持机器狗在平地静止，从此目录运行：

```bash
bash start.sh
```

地面拟合采用雷达中心垂直离地 0.50 m 作为先验，由这一轮点云确定地面方向和原点。看到“地面已锁定”、贴地模型已发布及稠密扫描数增长后开始走动。结束时在启动窗口按 Ctrl+C。每轮从新的 LIO-SAM 进程开始，点云与日志写到终端打印的 `~/go2_mapping_runs/run_*/`；`dense/scans.sqlite` 保存本轮选中的配准扫描和关键帧位姿。地图数据默认留在本机。

## Foxglove 布局

左侧“三维建图”的主题列表中开启 `/go2_visual/detail/local`，点大小选 1 像素。在同一处门框来回切换这层的眼睛图标，可以用原视角比较轮廓。也可一次导入本目录的 `farm_point_detail.json`；该布局已在旧版左窗的累计地图上叠加这层，统一两层高度色标，保留相机姿态、白地板、单个 Go2 模型、右窗实时扫描和导航面板。

已经在建图时，更新此分支后从仓库顶层另开终端执行 `bash noetic/live_detail.sh`。这条命令接入当前 ROS master 11321。本轮建图继续运行，下次正常执行 `bash noetic/start.sh` 将自动启动细节层。WebSocket 端口是 8766。

## 轮廓细节与验证

- 局部层从在线稠密节点的 SQLite 归档读取已有配准的逐帧实测回波，按当前关键帧位姿转换到已标定的地面坐标系。每次使用最近 6 秒内至多 30 帧、机器狗周围 7 m 内的回波（优先使用同期里程计，缺少同期位姿时以最近锚点为中心），按 5 mm 格选实测点。超过 12 万个格时先保留空间覆盖，再选剩余细节点。每次重新检查位姿，避免长期累积旧轮廓。默认只显示高于地面 4.5 cm 的局部点，已配置 `/go2_ground_view/ground_band` 时采用该值。建图输入和归档保留完整地面回波。归档暂不可用时终端会标记单帧回退；看到“局部实测”及大于一的帧数才表示已经叠加本轮实测数据。
- 左窗全局累计图仍来自 `/lio_sam/mapping/cloud_registered_raw` 与关键帧位姿。新启动时，7.5 mm 子格可保留原来因落在同一 15 mm 格里而合并的边缘点。细格将超过 40.5 万时，全局切回同期维护的 15 mm 地图，总显示上限 45 万。局部层独立保留 5 mm 选择尺度。格子边长是显示采样设置，最终能分开的物体间距取决于传感器、距离、入射角和配准误差。
- 用门框边沿或桌腿测试。先静止看窄缝，再从另一位置回看同一条边。静止时也缺点，可以在建图时运行 `bash noetic/live_detail.sh --check-returns`。8 秒统计按时间戳配对的原始与配准帧，并给出量程、线号、每线列数限制分别影响多少回波。若只有多帧叠加后出现平行双边，可在结束本轮后选取一块平墙的横向端点和高度，比较单帧厚度与合并厚度。端点与高度在归档的 `odom` 坐标系中读取，单位为米。

```bash
python3 noetic/check_wall.py "$RUN/dense/scans.sqlite" --wall X1 Y1 X2 Y2 --z ZMIN ZMAX
```

`RUN` 取启动时打印的本轮目录。输出包括每帧 P90−P10 厚度的中位数、合并厚度及各时段法向位置中位数。选墙时避开窗洞、墙角和遮挡物。

- [Livox Mid-360S 官方参数](https://www.livoxtech.com/cn/mid-360s/specs)
- [LIO-SAM-MID360 的高分辨率扫描发布代码](https://github.com/nkymzsy/LIO-SAM-MID360/blob/21ef7bb108fe3c7218c102bf7cb3a06582a2fed9/src/mapOptmization.cpp#L1775-L1789)
- [Foxglove 3D 点云显示设置](https://docs.foxglove.dev/docs/visualization/panels/3d)
