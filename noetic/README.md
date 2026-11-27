# Go2 MID-360S 实机建图（ROS 1 Noetic）

本目录是一个独立启动入口，用于已完成雷达和 Noetic 环境配置的 Go2。使用雷达内置 IMU、Madgwick 姿态与原有的 `lio_sam module_loam.launch`。需已有 `~/mid360s_nk_noetic_ws`、`~/mid360s_imu_noetic_ws`、`~/go2_farm_ground/start_ground.sh` 和 `~/go2_model_live.pyz`。从雷达驱动的 MID360s 配置读取有线收发地址。

## Go2 本体里程计建图、存图和导航

以下四个入口直接运行机器狗 `~/go2-detail` 里的代码。雷达、本体 DDS 和 Foxglove 使用现场局域网。示例中的 `inroom` 可换成任意地图名。

| 顺序 | 在机器狗上执行 | 使用时机 |
| --- | --- | --- |
| 1. 建图 | `bash ~/go2-detail/noetic/start.sh --go2` | 保留建图终端，等地面锁定后开始走动。 |
| 2. 三个状态面板 | `bash ~/go2-detail/noetic/start.sh --status` | 建图中可另开终端接入。新版建图会自动启动状态节点，提示面板已由现有节点供应即可。 |
| 3. 保存地图 | `bash ~/go2-detail/noetic/save_map.sh inroom` | 停稳后另开终端保存。等到“地图保存完成”再回建图窗口按 Ctrl+C。每次保存会创建新的版本目录。 |
| 4. 定位和导航 | `bash ~/go2-detail/navigation/start.sh inroom` | 建图退出后执行。程序选取同名地图中最近保存完成的版本，一次启动 GICP 和 Nav2。 |

有代码更新时，在两次实验之间联网执行一次：

```bash
git -C ~/go2-detail -c http.version=HTTP/1.1 pull --ff-only origin noetic-mid360s-detail
```

## 雷达里程计建图模式

先在上一轮的启动窗口按 Ctrl+C 并等它退出。若另开过贴地终端，也在那个窗口退出。保持机器狗在平地静止，从此目录运行：

```bash
bash start.sh
```

地面拟合采用雷达中心垂直离地 0.50 m 作为先验，由这一轮点云确定地面方向和原点。看到“地面已锁定”、贴地模型已发布及稠密扫描数增长后开始走动。结束时在启动窗口按 Ctrl+C。每轮从新的 LIO-SAM 进程开始，点云与日志写到终端打印的 `~/go2_mapping_runs/run_*/`；`dense/scans.sqlite` 保存本轮选中的配准扫描和关键帧位姿。地图数据默认留在本机。

## Foxglove 布局

左侧“三维建图”的主题列表中开启 `/go2_visual/detail/local`，点大小选 1 像素。在同一处门框来回切换这层的眼睛图标，可以用原视角比较轮廓。也可一次导入本目录的 `farm_point_detail.json`；该布局已在旧版左窗的累计地图上叠加这层，统一两层高度色标，保留相机姿态、白地板、单个 Go2 模型、右窗实时扫描和导航面板。

已经在建图时，更新此分支后从仓库顶层另开终端执行 `bash noetic/live_detail.sh`。这条命令接入当前 ROS master 11321。本轮建图继续运行，下次正常执行 `bash noetic/start.sh` 将自动启动细节层。WebSocket 端口是 8766。

## 实机静止时，检查模型跳动与轮廓展宽

保持 Go2 四脚静止和现有建图窗口运行，在仓库顶层执行以下命令：

```bash
bash noetic/live_detail.sh --check-still
```

检查使用 ROS master 11321，在同一段时间采集建图位姿、贴地坐标、模型和扫描。观察同一发布者的相邻位姿变化及整段漂移；旧锁存模型可能只出现一次。注册扫描与贴地扫描按消息头时间戳比较。若视觉跳动仍在而 ROS 位姿及贴地模型在采样期间稳定，可结合显示时的桥接警告、扫描间隔和 Foxglove 问题提示进一步排查。若传感器坐标下的墙面稳定、配准后叠帧明显变厚，应先排查建图位姿与同步。报告中缺少有效样本的项目需要重新采样。

### 用同一段 Go2 回波对比 LIORF

实机保持静止，尽量让扫描同时覆盖墙角和门框。当前建图和 Foxglove 照常运行。在仓库顶层执行一次：

```bash
bash noetic/live_detail.sh --capture-still
```

该命令同步运行上面的 12 秒测量，并在 `~/go2_alignment_runs/still_*/` 新建一份样本。终端会打印确切路径。目录含原始 Livox 与滤波 IMU、当前 LIO-SAM 位姿、去畸变和配准扫描组成的 `baseline.bag`，以及 `lio_sam.yaml`、`alignment.txt`。有足够的同戳扫描时还会输出 `contours.png`。它在同一片约 1.2 米的区域按相同尺度并排展示单帧和多帧的实际回波，并给出可在 Windows 直接查看的命令。报告先比较传感器坐标中的表面厚度和配准后同一处表面的厚度，再对照模型、里程计和时间戳。两窗使用的布局可继续用。

在已经编译了 ROS 1 Noetic LIORF 的 Linux 机器上，将整份样本目录传入回放命令。以下两个路径分别替换为上一步打印的目录和 LIORF 的 catkin 工作空间：

```bash
bash noetic/liorf_replay.sh "$RUN" "$LIORF_WS"
```

回放器使用独立的 ROS master，将 `livox_ros_driver2/CustomMsg` 转成 LIORF 读取的 `PointCloud2`。它保留回波位置、反射率、线号和原时间戳，将 `offset_time` 从纳秒转换为相对秒，从录包读取已滤波的 SI 单位 IMU，并采用采集时的外参、量程与线数。LIORF 的 `point_filter_num` 取 1，供配准用的点仍按上游算法做体素下采样。回放输出写到单独目录，不向在线 11321 发布。上游版本见 [LIORF ROS1](https://github.com/YJZLuckyBoy/liorf/tree/1d777a6409836fc743a770c22dc2897814f14fcb)。

比较程序只选两套算法都有的源时间戳，在首帧选同一批真实平面区域，然后分别计算静止位姿的最大变化及多帧表面厚度（毫米）。如果一侧缺帧或共同观测不足，报告会标明。当前建图可能已有历史关键帧，LIORF 从录包开头初始化；该结果衡量这段现场数据，若要评价移动、回环或长期精度，还需要相同初始条件和动态轨迹。

## 轮廓细节与验证

- 局部层从在线稠密节点的 SQLite 归档读取已有配准的逐帧实测回波，按当前关键帧位姿转换到已标定的地面坐标系。每次使用最近 6 秒内至多 30 帧、机器狗周围 7 m 内的回波（优先使用同期里程计，缺少同期位姿时以最近锚点为中心），按 5 mm 格选实测点。超过 12 万个格时先保留空间覆盖，再选剩余细节点。每次重新检查位姿，避免长期累积旧轮廓。在上述取样范围内，离地至少 10 cm 的回波均可进入局部显示层。设置较高的 `/go2_detail_local/local_ground_band`（米）可进一步提高局部层的最低显示高度。建图输入和归档保留完整地面回波。归档暂不可用时终端会标记单帧回退；看到“局部实测”及大于一的帧数才表示已经叠加本轮实测数据。
- 左窗全局累计图仍来自 `/lio_sam/mapping/cloud_registered_raw` 与关键帧位姿。新启动时，7.5 mm 子格可保留原来因落在同一 15 mm 格里而合并的边缘点。细格将超过 40.5 万时，全局切回同期维护的 15 mm 地图，总显示上限 45 万。局部层独立保留 5 mm 选择尺度。格子边长是显示采样设置，最终能分开的物体间距取决于传感器、距离、入射角和配准误差。
- 用门框边沿或桌腿测试。先静止看窄缝，再从另一位置回看同一条边。静止时也缺点，可以在建图时运行 `bash noetic/live_detail.sh --check-returns`。8 秒统计按时间戳配对的原始与配准帧，并给出量程、线号、每线列数限制分别影响多少回波。若只有多帧叠加后出现平行双边，可在结束本轮后选取一块平墙的横向端点和高度，比较单帧厚度与合并厚度。端点与高度在归档的 `odom` 坐标系中读取，单位为米。

```bash
python3 noetic/check_wall.py "$RUN/dense/scans.sqlite" --wall X1 Y1 X2 Y2 --z ZMIN ZMAX
```

`RUN` 取启动时打印的本轮目录。输出包括每帧 P90−P10 厚度的中位数、合并厚度及各时段法向位置中位数。选墙时避开窗洞、墙角和遮挡物。

- [Livox Mid-360S 官方参数](https://www.livoxtech.com/cn/mid-360s/specs)
- [LIO-SAM-MID360 的高分辨率扫描发布代码](https://github.com/nkymzsy/LIO-SAM-MID360/blob/21ef7bb108fe3c7218c102bf7cb3a06582a2fed9/src/mapOptmization.cpp#L1775-L1789)
- [Foxglove 3D 点云显示设置](https://docs.foxglove.dev/docs/visualization/panels/3d)
