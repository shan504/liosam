#!/usr/bin/env python3
"""Rebuild measured odom TF and plan-only goals. Never imports a robot actuator API."""
import json
from pathlib import Path
import numpy as np


def pose_matrix(position, quaternion):
    p=np.asarray(position,dtype=float);q=np.asarray(quaternion,dtype=float)
    if not np.isfinite(p).all() or not np.isfinite(q).all() or not .9<=np.linalg.norm(q)<=1.1:
        raise ValueError('Invalid recorded pose')
    q=q/np.linalg.norm(q);x,y,z,w=q
    m=np.eye(4);m[:3,3]=p
    m[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    return m


def extract_seed(directory):
    """Use a fresh, explicitly valid recorded correction only as the initial hint."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(directory),storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('',''))
    types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types()
           if t.name in ('/odom','/tf','/localization/valid')}
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(types)))
    correction=None; valid_at=None; first=None
    def ns(stamp):return stamp.sec*10**9+stamp.nanosec
    while reader.has_next():
        topic,data,recorded=reader.read_next()
        if first is None:first=recorded
        if recorded-first>30*10**9:break
        msg=deserialize_message(data,types[topic])
        if topic=='/localization/valid':valid_at=recorded if msg.data else None
        elif topic=='/tf':
            for t in msg.transforms:
                if t.header.frame_id.lstrip('/')=='map' and t.child_frame_id.lstrip('/')=='odom':
                    p,q=t.transform.translation,t.transform.rotation
                    try:
                        correction=(ns(t.header.stamp),pose_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w]))
                    except ValueError:
                        correction=None
        elif topic=='/odom' and valid_at is not None and correction is not None:
            stamp=ns(msg.header.stamp)
            if (msg.header.frame_id.lstrip('/')!='odom' or msg.child_frame_id.lstrip('/')!='base_link'
                    or not 0<=recorded-valid_at<500000000 or not 0<=stamp-correction[0]<300000000):continue
            p,q=msg.pose.pose.position,msg.pose.pose.orientation
            try:
                pose=correction[1]@pose_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
            except ValueError:
                continue
            return {'stamp_ns':stamp,'matrix':pose.tolist()}
    return None


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PoseStamped,PoseWithCovarianceStamped,TransformStamped
    from nav_msgs.msg import Odometry,Path as RosPath
    from nav2_msgs.action import ComputePathToPose
    from std_msgs.msg import Bool,String
    from tf2_ros import TransformBroadcaster
    from scipy.spatial.transform import Rotation

    class Inputs(Node):
        def __init__(self):
            super().__init__('go2_replay_inputs')
            self.tf=TransformBroadcaster(self)
            self.seed=json.loads(Path('/runtime/replay_seed.json').read_text())
            self.initial=self.create_publisher(PoseWithCovarianceStamped,'/initialpose',1)
            self.status=self.create_publisher(String,'/operator/status',1)
            self.plan=self.create_publisher(RosPath,'/plan',1)
            self.valid=False;self.valid_received=0;self.busy=False
            self.client=ActionClient(self,ComputePathToPose,'/compute_path_to_pose')
            self.create_subscription(Odometry,'/odom',self.odom,qos_profile_sensor_data)
            self.create_subscription(Bool,'/localization/valid',self.validity,1)
            self.create_subscription(PoseStamped,'/goal_pose',self.goal,1)
            self.create_timer(5.,lambda:self.report('离线回放：点选目标仅验证路径规划；实机运动输出未启动'))
        def report(self,text):
            self.status.publish(String(data=text));self.get_logger().info(text)
        def validity(self,msg):
            self.valid=msg.data;self.valid_received=self.get_clock().now().nanoseconds
        def odom(self,msg):
            if msg.header.frame_id.lstrip('/')!='odom' or msg.child_frame_id.lstrip('/')!='base_link':return
            p,q=msg.pose.pose.position,msg.pose.pose.orientation
            try:pose_matrix([p.x,p.y,p.z],[q.x,q.y,q.z,q.w])
            except ValueError:return
            tf=TransformStamped();tf.header=msg.header;tf.child_frame_id='base_link'
            tf.transform.translation.x=p.x;tf.transform.translation.y=p.y;tf.transform.translation.z=p.z
            tf.transform.rotation=q;self.tf.sendTransform(tf)
            stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
            if self.seed and stamp>=self.seed['stamp_ns']:
                pose=np.array(self.seed['matrix']);initial=PoseWithCovarianceStamped()
                initial.header.frame_id='map';initial.header.stamp=msg.header.stamp
                initial.pose.pose.position.x,initial.pose.pose.position.y,initial.pose.pose.position.z=map(float,pose[:3,3])
                rotation=Rotation.from_matrix(pose[:3,:3]).as_quat()
                initial.pose.pose.orientation.x,initial.pose.pose.orientation.y,initial.pose.pose.orientation.z,initial.pose.pose.orientation.w=map(float,rotation)
                initial.pose.covariance[0]=initial.pose.covariance[7]=.04;initial.pose.covariance[35]=.03
                self.initial.publish(initial);self.seed=None
        def goal(self,msg):
            if self.busy:return self.report('上一条规划请求仍在处理')
            if not self.valid or self.get_clock().now().nanoseconds-self.valid_received>500000000:
                return self.report('规划未执行：回放定位尚未有效')
            if msg.header.frame_id.lstrip('/')!='map':return self.report('请使用 map 坐标系点选规划目标')
            if not self.client.server_is_ready():return self.report('规划未执行：Nav2 planner 尚未就绪')
            request=ComputePathToPose.Goal();request.goal=msg;request.goal.header.stamp=self.get_clock().now().to_msg()
            request.planner_id='GridBased';request.use_start=False;self.busy=True
            self.client.send_goal_async(request).add_done_callback(self.accepted)
        def accepted(self,future):
            try:
                handle=future.result()
                if not handle.accepted:self.busy=False;return self.report('离线路径规划被拒绝')
                handle.get_result_async().add_done_callback(self.completed)
            except Exception as e:self.busy=False;self.report('规划失败：'+str(e))
        def completed(self,future):
            self.busy=False
            try:
                result=future.result()
                if result.status==4 and result.result.path.poses:
                    self.plan.publish(result.result.path);self.report('离线路径规划成功；机器人继续按照录包轨迹回放')
                else:self.report(f'离线路径规划失败：action status={result.status}')
            except Exception as e:self.report('规划失败：'+str(e))
    rclpy.init();node=Inputs()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
