# Go2 原版 Nav2 巡检

在机器狗上运行以下命令。参数是已保存的地图名，也可以传入具体的版本目录。建图和保存可继续使用原有的 Noetic 命令。

```bash
bash ~/go2-detail/navigation/start.sh inroom1
```

脚本读取 `~/go2_maps/` 下对应地图最新完成的版本，复用本机的 `go2-3d:edu` 镜像、MID‑360S 驱动和本体 DDS 配置。先结束其他正在使用雷达或控制接口的实验窗口。首次启动会在 `~/.cache/go2_lio_navigation/` 中准备导航用地图，之后复用缓存。

## 定位和行走

1. 从该地图建图起点、原朝向启动，或在 Foxglove 左侧窗口用“估计位姿”指定实际位置和朝向。建图和导航共用仓库根目录的 [Go2.json](../Go2.json)。首次连接终端打印的 Wi-Fi WebSocket 地址后导入一次，切换模式时继续使用同一布局。两个视图默认三维透视，按住右键拖动旋转，左键拖动平移，滚轮缩放。
2. 等待 GICP 报告定位有效、Nav2 就绪。在右侧导航视图依次设置目标，核对航点编号。初次实验可以从附近一个实际畅通的目标开始。
3. 点击“启用运动”。“停止运动＋取消导航”和“取消路线”按旧 Go2 的语义取消整条路线。“重试暂停路线”用于恢复保留了待执行航点的暂停任务。
4. 在机器狗终端按 Ctrl+C 结束实验。状态话题包括 `/localization/valid`、`/navigation/startup_status`、`/operator/status` 和 `/control/status`。原版故障记录器产生的事件在 `~/.local/state/go2-navigation/diagnostics/` 下。

也可在机器狗的第二个终端按地图坐标依次提交目标。每个参数为 x,y,航向角，单位为米、米、度。

```bash
sudo docker exec go2-lio-nav bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && python3 /repo/navigation/waypoints.py -- "$@"' bash X1,Y1,YAW1 X2,Y2,YAW2
```

## 道路显示和目标坐标（所有地图共用）

统一 `Go2.json` 默认关闭完整 `/map_cloud`、`/registered_cloud` 的显示，
改用 `/navigation/display/map_obstacles` 和 `/navigation/display/scan_obstacles`。
显示节点读取当前所选地图 `metadata.yaml` 的地面标定，过滤离地不足 0.10 m
的点，保留其余全部高度；不改保存的 PCD、GICP 输入、障碍层或运动参数。
右侧用浅色显示已观测通行格、深灰显示未知区域，叠加全局/局部代价地图，
全局路径为绿色、局部路径为橙色。该画面表达栅格通行信息，不识别相机中的
道路、草地或农田语义。两个视图仍可三维旋转。

**拉取代码不会替换 Foxglove 已导入的布局。** 更新后重新导入仓库根目录
`Go2.json`，或在现有两个 3D 面板里将固定和显示参考系都设为 `map`；关闭
`/map_cloud`、`/registered_cloud`，打开上述两路显示点云、`/map`、
`/global_costmap/costmap`、`/local_costmap/costmap`、`/plan` 和 `/local_plan`。
Windows 可从机器狗复制布局（把地址替换为机器狗当前 Wi-Fi 地址）：

```powershell
scp unitree@172.20.10.2:/home/unitree/go2-detail/Go2.json "$env:USERPROFILE\Downloads\Go2.json"
```

旧布局若仍发布 `odom` 目标，公共操作节点会在定位有效时按真实的
`map <- odom` TF 转换位置和朝向。非零时间戳使用对应时刻的变换，暂缺时
短暂异步等待；零时间戳使用新鲜的最新变换。变换缺失、过期或定位无效时
明确拒绝，不把坐标直接改名成 `map`。多个点击按输入顺序处理；手动停止、
取消或重新初始化会撤销待处理点击。地图检查、队列和运动授权检查继续有效。

## 保存和复用航点

在导航运行期间按顺序点选目标，确认航点已经进入队列后，另开终端保存。
保存完整点位顺序和朝向，包括本次已完成的点；不会改变正在执行的任务。

```bash
bash ~/go2-detail/navigation/routes.sh save farm_route1
bash ~/go2-detail/navigation/routes.sh list
```

下次先启动同一版本地图的导航，再加载路线。加载只排队，等待定位有效后
点击“启用运动”执行：

```bash
bash ~/go2-detail/navigation/start.sh farm
# 另一个终端：
bash ~/go2-detail/navigation/routes.sh load farm_route1
```

文件保存在机器狗 `~/.local/share/go2-navigation/routes/`，重启容器或机器狗
不会丢失。路线校验所用地图内容和坐标指纹，不能将一张地图的坐标直接用于
另一张地图或重新建图后的版本；通用保存/加载功能适用于所有地图。
已有路线不为空时需先完成或明确取消，加载不会静默覆盖活动队列。

## 移植范围

### 大范围地图栅格（所有地图共用）

导航地图准备不再按固定格数或外接矩形面积拒绝地图。已移除公共
`ObservedGrid`、通行区域转移、Go2 存档适配及历史地图转换入口的固定
容量检查；不是把 400 万改成另一个上限。保留公共配置的 0.05 m 分辨率、
完整地图范围、实测通行证据、障碍优先级及原来的去噪规则。

当前 Go2 导航入口使用磁盘映射写入完整 PGM，分块导出通行格坐标，
并按带邻域的图块去噪；默认孤立单点/双点规则避免建立全图障碍 Python
集合。分块尺寸只限制一次处理的工作量，不限制地图大小，也不自动降低
分辨率。原始存档和完整定位点云保留。实际可处理规模仍取决于设备可用
磁盘、内存，以及 Nav2 加载完整代价地图的资源需求。

首次转换会显示源点数/帧数、扫描回放、体素聚合、栅格尺寸、定位点云
写入及去噪阶段进度。公共转换代码变化后自动为任意地图重建缓存；只有
全部输出完成并校验后才发布缓存，再次启动复用完成的版本。无需重新
建图、手动删除缓存或编译 C++，日常启动仍不访问 GitHub。

### GICP 定位超时修复（所有地图共用）

`Registration result too old` 表示结果未通过扫描源戳或本机接收时间的 1 秒时效检查，并不等于对应点搜索距离不够。定位模式使用静态地图空间索引、带法向与正则化协方差的局部目标缓存，以及有点数预算的粗、精 GICP；相同位置不再重复对整张地图筛选、计算目标法向。默认定位源点不超过 6000、精目标不超过 20000，粗目标不超过 8000；仅求解输入降采样，保存的完整地图和建图密度不变。

初始化对应点距离由 0.70 m 扩至 1.20 m，初始位姿修正上限由 2 m / 约 46° 扩至 3 m / 90°，最终仍用 0.70 m 对应点距离、原有重叠率和 0.20 m RMSE 上限验证。3 m / 90° 是允许修正上限，不保证在任意这样大的初值误差下都能收敛，也不是全图位置识别。仍从该地图自己的起点/朝向启动，或发送实际 `/initialpose`。正常跟踪的创新、静止漂移、原生里程计、恢复确认和运动授权检查保留。

加载地图时预热起点局部目标；定位时完整静态 `/map_cloud` 仅发布一次，通过原有可靠 `TRANSIENT_LOCAL` QoS 向后接入订阅者提供完整地图，避免每两秒重新打包和传输。活动建图继续发布更新。超时日志现在分别给出 `processing_time`（本轮计算时间）、`source_age`（实际贡献扫描接收后时间）、`stamp_age`（源戳距现在时间）和参与求解点数；未接受的结果不会刷新定位有效期。

方法参考 Open3D 官方 [粗到细配准](https://www.open3d.org/docs/release/tutorial/t_pipelines/t_icp_registration.html#multi-scale-icp-example) 和 [收敛条件](https://www.open3d.org/docs/0.14.1/python_api/open3d.pipelines.registration.ICPConvergenceCriteria.html)。使用现有 Python/Open3D，无需编译 C++。机器人 ARM64 的实时耗时和实际场景收敛需复测；桌面合成几何验证不替代实机结果。

原代码来自 [shan504/go2 的 40a836d](https://github.com/shan504/go2/tree/40a836d513f5ce4c8938a3bde4713dc83388f20c)。导航加载 `original_go2/go2_3d/stack.launch.py`，用原 `prepare.py` 和 `motion_profile.py` 从镜像的 SDK 配置生成实际运行参数。GICP、NavFn A*、RotationShim 和 DWB、行为树、去噪和膨胀、航点队列和重试均由原实现运行。全局和局部膨胀半径均为原版的 0.05 m，使用原机身足迹核对碰撞。

输入采用 Go2 本体 `/utlidar/robot_odom`。外置 MID‑360S 根据建图时保存的安装外参和带时间戳的本体位姿完成去畸变，在 `base_link` 发布 `/point_cloud2`。运动桥订阅这两路已对时的输入，复用旧桥的定位门控、速度限制和停车流程。Foxglove 端口使用 8766，模型话题为 `/go2_visual/live/model_mesh`，导航画面使用原 Go2 前置相机；建图画面继续使用 D435i。

公共运动配置保留原 Go2 的两档 X 速度候选：0（原地转向）和 0.30 m/s（前进）。保留原版 0.40 m/s² 加减速和 20 Hz OPEN_LOOP 平滑；持续前进指令从零约 0.75 秒升至 0.30 m/s，起停期间仍会经过低速，停车由控制器、平滑器和运动桥处理。

扩大转向实验：DWB 跟踪转向与路径对准上限统一为 ±1.20 rad/s（约 ±69°/s），平滑器和运动桥同步，角加减速度为 1.60 rad/s²。角速度采样参数从 31 改为 7；静止附近的主要候选为 `-1.2、-0.8、-0.4、0、0.4、0.8、1.2`，高实际转速时仍受加减速可达范围约束。用联合最小速度条件过滤几乎不动的原地转向候选（包括浮点近零值），不阻止原地有效转向、直行和停车。全部候选由 DWB 或 RotationShim 检查碰撞，不在桥接端强行放大指令。

保留 Oscillation critic，方向锁定增加 2 秒超时释放，原 0.05 m 位移和 0.20 rad 转角释放条件也保留。这个参数只允许重新评估转向方向，不清除障碍、不会自动启用运动。依据是 Nav2 Humble 官方 [角速度采样](https://api.nav2.org/nav2-humble/html/xy__theta__iterator_8cpp_source.html) 和 [振荡检查](https://api.nav2.org/nav2-humble/html/oscillation_8cpp_source.html)。这些是角速度和候选范围，不是总转角限制；能否转到侧后方仍取决于足迹扫掠空间。

若控制器输出 `No valid trajectories`，同时打印拒绝候选的 critic 和原因。最新现场日志已确认 0.30 m/s 下发和约 1.79 m 前进，之后出现全部候选碰撞，以及碰撞与 Oscillation 联合拒绝。因此本次扩大转向是实验，不能认定已解决所有停顿。保留完整足迹检查；全碰撞时仍停车。

地图修正 TF 使用独立的回调组，最多 50 Hz，由新鲜本体里程计触发，继续采用实际源时间戳。GICP 定位无效、数据过期或里程计推算预算耗尽时仍停止更新。最新日志仍有 RotationShim 查询当前 TF 超时；Humble 的这次调用默认等待 0.1 秒，DWB 的 `transform_tolerance` 不能覆盖它。本次不宣称该延迟已经解决。这些配置在每次启动时统一生成，适用于所有地图，无需重新建图、删缓存或编译 C++。仍需通过实机转弯到点测试确认。

`prepare_go2_map.py` 使用每个存档自己的地面标定、安装外参、轨迹和扫描索引。导航先验以原 dense 模式的 2 cm 体素跨全部高度聚合，二维地图使用归档回波、可确定射线起点的关键帧及原 Go2 的地面、占用、去噪规则。每张地图保留自己的建图起点和原始 PCD。

导入的源码中，`stack.launch.py` 选择传感器适配后的运动桥并传递本项目的 Foxglove 设置，`control_bridge.py` 和 `patches/dds/go2_edu_dds_bridge.py` 提供注入该传感器输入的入口。规划器与控制器的参数在导入的 `go2_3d/prepare.py` 与 `go2_3d/motion_profile.py`。完整先验点云和配准点云话题保留，可手动开启检查；导航默认显示去地面后的障碍点云。

