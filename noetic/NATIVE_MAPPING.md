# 用 Go2 本体里程计建图

在机器人上执行 `bash noetic/native_odom.sh --inspect`，可在旧建图运行时检查 Noetic 的 11321、11311 和原 Go2 Humble DDS 环境。输出包含每路里程计的发布者、实际收到的消息与坐标系，以及当前静态 TF 能解析的机身到相机、雷达的安装变换。DDS 使用本地已有的 `go2-3d:edu` 镜像、domain 0 和 `~/cyclonedds_ws/cyclonedds.xml`。自定义 DDS XML 可通过 `GO2_DDS_FILE` 指定。

结束原建图进程后，让 Go2 在平地站稳，执行 `bash noetic/start.sh --native`。终端打印本机完成 Foxglove 协议握手的 Wi-Fi 地址，在电脑 Foxglove 的 WebSocket 连接中使用该地址。每次启动建立独立的稠密点云归档，Ctrl+C 结束本轮。

模型和点云的机身运动都来自同一路原生里程计。程序读取 `/utlidar/robot_odom`，也识别原项目节点 `/go2_edu_dds_bridge` 发布的 `/odom`。有效输入为 `nav_msgs/Odometry` 的 `odom → base_link`。首次接收通过时间与坐标校验的来源后输出“已接通 Go2 原生里程计”。原生数据中断时停止更新。雷达回波按采集时间在本体位姿之间插值，显示点云按初始站姿的水平地面去除地面附近的点。此模式按原生轨迹累积回波，不执行 LIO-SAM 回环优化。

模型可在收到本体里程计后单独显示。要将回波放到图中，还需要一条物理 `base_link ← 雷达消息 header.frame_id` 的固定安装变换。程序从同一 ROS 图中的完整静态链读取，也可使用 `~/.config/go2/body_from_lidar.json` 或 `GO2_BODY_FROM_LIDAR` 指定的 JSON 文件。文件字段为 `parent`（base_link）、`child`（实际雷达帧名）、`xyz`（雷达原点在机身坐标系的位置，米）、`xyzw`（雷达坐标旋转到机身坐标的单位四元数）。雷达至相机的标定还需要相机与机身之间的固定关系，组合前应确认厂商定义的相机帧、矩阵方向与平移单位。
