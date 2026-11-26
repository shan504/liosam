"""Activate the motion stack after GICP and the robot TF are available."""
import time

import rclpy
from nav2_msgs.srv import ManageLifecycleNodes
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener


class Nav2Ready(Node):
    def __init__(self):
        super().__init__('go2_nav2_ready')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.manager = self.create_client(ManageLifecycleNodes, '/lifecycle_manager_navigation/manage_nodes')
        self.status = self.create_publisher(String, '/navigation/startup_status',
                                            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.last_valid = None
        self.ready_since = None
        self.last_status = None
        self.create_subscription(Bool, '/localization/valid', self.on_valid, qos_profile_sensor_data)
        self.timer = self.create_timer(0.1, self.tick)
        self.report('地图服务已启动，等待 GICP 定位和机身坐标变换')

    def report(self, text):
        if text != self.last_status:
            self.status.publish(String(data=text))
            self.get_logger().info(text)
            self.last_status = text

    def on_valid(self, msg):
        self.last_valid = time.monotonic() if msg.data else None
        if not msg.data:
            self.ready_since = None

    def tick(self):
        now = time.monotonic()
        ready = self.last_valid is not None and now - self.last_valid < 0.5
        if ready:
            try:
                tf = self.buffer.lookup_transform('map', 'base_footprint', Time())
                age = (self.get_clock().now() - Time.from_msg(tf.header.stamp)).nanoseconds * 1e-9
                ready = -0.10 <= age < 0.35
            except TransformException:
                ready = False
        if not ready:
            self.ready_since = None
            self.report('等待有效的 GICP 定位和 map 到 base_footprint 的实时 TF')
            return
        if self.ready_since is None:
            self.ready_since = now
        if now - self.ready_since < 0.5 or not self.manager.service_is_ready():
            return
        self.timer.cancel()
        req = ManageLifecycleNodes.Request()
        req.command = req.STARTUP
        self.report('GICP 和 TF 已就绪，启动规划器、控制器和行为树')
        self.manager.call_async(req).add_done_callback(self.completed)

    def completed(self, future):
        if not future.result().success:
            self.report('Nav2 生命周期启动失败，请查看规划器和控制器日志')
            raise RuntimeError('Nav2 lifecycle did not activate')
        self.report('Nav2 已就绪，等待导航任务')


if __name__ == '__main__':
    rclpy.init()
    node = Nav2Ready()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
