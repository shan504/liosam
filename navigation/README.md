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

## 移植范围

原代码来自 [shan504/go2 的 40a836d](https://github.com/shan504/go2/tree/40a836d513f5ce4c8938a3bde4713dc83388f20c)。导航加载 `original_go2/go2_3d/stack.launch.py`，用原 `prepare.py` 和 `motion_profile.py` 从镜像的 SDK 配置生成实际运行参数。GICP、NavFn A*、RotationShim 和 DWB、行为树、去噪和膨胀、航点队列和重试均由原实现运行。全局和局部膨胀半径均为原版的 0.05 m，使用原机身足迹核对碰撞。

输入采用 Go2 本体 `/utlidar/robot_odom`。外置 MID‑360S 根据建图时保存的安装外参和带时间戳的本体位姿完成去畸变，在 `base_link` 发布 `/point_cloud2`。运动桥订阅这两路已对时的输入，复用旧桥的定位门控、速度限制和停车流程。Foxglove 端口使用 8766，模型话题为 `/go2_visual/live/model_mesh`，导航画面使用原 Go2 前置相机；建图画面继续使用 D435i。

公共运动配置保留原 Go2 的两档 X 速度候选：0（原地转向）和 0.30 m/s（前进）。保留原版 0.40 m/s² 加减速和 20 Hz OPEN_LOOP 平滑；持续前进指令从零约 0.75 秒升至 0.30 m/s，起停期间仍会经过低速，停车由控制器、平滑器和运动桥处理。

扩大转向实验：DWB 跟踪转向与路径对准上限统一为 ±1.20 rad/s（约 ±69°/s），平滑器和运动桥同步，角加减速度为 1.60 rad/s²。角速度采样参数从 31 改为 7；静止附近的主要候选为 `-1.2、-0.8、-0.4、0、0.4、0.8、1.2`，高实际转速时仍受加减速可达范围约束。用联合最小速度条件过滤几乎不动的原地转向候选（包括浮点近零值），不阻止原地有效转向、直行和停车。全部候选由 DWB 或 RotationShim 检查碰撞，不在桥接端强行放大指令。

保留 Oscillation critic，方向锁定增加 2 秒超时释放，原 0.05 m 位移和 0.20 rad 转角释放条件也保留。这个参数只允许重新评估转向方向，不清除障碍、不会自动启用运动。依据是 Nav2 Humble 官方 [角速度采样](https://api.nav2.org/nav2-humble/html/xy__theta__iterator_8cpp_source.html) 和 [振荡检查](https://api.nav2.org/nav2-humble/html/oscillation_8cpp_source.html)。这些是角速度和候选范围，不是总转角限制；能否转到侧后方仍取决于足迹扫掠空间。

若控制器输出 `No valid trajectories`，同时打印拒绝候选的 critic 和原因。最新现场日志已确认 0.30 m/s 下发和约 1.79 m 前进，之后出现全部候选碰撞，以及碰撞与 Oscillation 联合拒绝。因此本次扩大转向是实验，不能认定已解决所有停顿。保留完整足迹检查；全碰撞时仍停车。

地图修正 TF 使用独立的回调组，最多 50 Hz，由新鲜本体里程计触发，继续采用实际源时间戳。GICP 定位无效、数据过期或里程计推算预算耗尽时仍停止更新。最新日志仍有 RotationShim 查询当前 TF 超时；Humble 的这次调用默认等待 0.1 秒，DWB 的 `transform_tolerance` 不能覆盖它。本次不宣称该延迟已经解决。这些配置在每次启动时统一生成，适用于所有地图，无需重新建图、删缓存或编译 C++。仍需通过实机转弯到点测试确认。

`prepare_go2_map.py` 使用每个存档自己的地面标定、安装外参、轨迹和扫描索引。导航先验以原 dense 模式的 2 cm 体素跨全部高度聚合，二维地图使用归档回波、可确定射线起点的关键帧及原 Go2 的地面、占用、去噪规则。每张地图保留自己的建图起点和原始 PCD。

导入的源码中，`stack.launch.py` 选择传感器适配后的运动桥并传递本项目的 Foxglove 设置，`control_bridge.py` 和 `patches/dds/go2_edu_dds_bridge.py` 提供注入该传感器输入的入口。规划器与控制器的参数在导入的 `go2_3d/prepare.py` 与 `go2_3d/motion_profile.py`。导航端使用原仓库的导航图层，左、右视图可显示完整先验点云和配准点云。
