"""Real goal input/queue logic with a simulated TF clock, without ROS hardware."""
import ast
import copy
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from uuid import UUID, uuid4

import numpy as np

GO2 = Path(__file__).resolve().parents[1] / 'original_go2/go2_3d'
sys.path.insert(0, str(GO2))
from geometry import pose_matrix, grid_cell, quaternion_from_matrix
from goal_frames import (GOAL_TF_WAIT_SECONDS, goal_frame, odom_goal_to_map,
                         validate_odom_goal_time)


def stamp(ns):
    return NS(sec=ns//10**9, nanosec=ns%10**9)


def quaternion(yaw):
    return NS(x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2))


def goal(x=2., y=1., z=.4, yaw=.3, frame='odom', ns=100_000_000_000):
    return NS(header=NS(frame_id=frame, stamp=stamp(ns)),
              pose=NS(position=NS(x=x,y=y,z=z), orientation=quaternion(yaw)))


def transform(x=10., y=-2., z=.2, yaw=math.pi/2, ns=100_000_000_000):
    return NS(header=NS(frame_id='map', stamp=stamp(ns)), child_frame_id='odom',
              transform=NS(translation=NS(x=x,y=y,z=z), rotation=quaternion(yaw)))


class GoalGeometry(unittest.TestCase):
    def test_two_map_origins_rotate_position_and_heading(self):
        for xyz, yaw in [((10.,-2.,.2), math.pi/2), ((-12.,17.,-.6), -.7)]:
            with self.subTest(origin=xyz, yaw=yaw):
                source = goal(frame='/odom')
                before = copy.deepcopy(source)
                result = odom_goal_to_map(source, transform(*xyz,yaw), 100_050_000_000)
                c,s = math.cos(yaw),math.sin(yaw)
                np.testing.assert_allclose(list(vars(result.pose.position).values()),
                                           [xyz[0]+2*c-s,xyz[1]+2*s+c,xyz[2]+.4],atol=1e-12)
                q = result.pose.orientation
                np.testing.assert_allclose([q.x,q.y,q.z,q.w],
                                           [0,0,math.sin((yaw+.3)/2),math.cos((yaw+.3)/2)],atol=1e-12)
                self.assertEqual(result.header.frame_id,'map')
                self.assertEqual(source,before)

    def test_zero_stamp_uses_fresh_latest_and_stamped_requires_exact_time(self):
        self.assertEqual(odom_goal_to_map(goal(ns=0), transform(ns=99_900_000_000),
                                         100_000_000_000).header.stamp.sec,99)
        with self.assertRaisesRegex(ValueError,'目标时刻'):
            odom_goal_to_map(goal(),transform(ns=99_900_000_000),100_000_000_000)
        for ns in (0,99_499_999_999,100_100_000_001):
            with self.subTest(ns=ns), self.assertRaises(ValueError):
                odom_goal_to_map(goal(ns=0),transform(ns=ns),100_000_000_000)

    def test_stale_future_and_malformed_inputs_fail(self):
        for ns in (98_999_999_999,100_100_000_001):
            with self.subTest(ns=ns), self.assertRaisesRegex(ValueError,'时间戳'):
                odom_goal_to_map(goal(ns=ns),transform(ns=ns),100_000_000_000)
        mutations = [lambda g,t:setattr(g.pose.orientation,'w',2.),
                     lambda g,t:setattr(t.transform.rotation,'w',2.),
                     lambda g,t:setattr(g.pose.position,'x',float('nan')),
                     lambda g,t:setattr(t.transform.translation,'x',float('inf')),
                     lambda g,t:setattr(t,'child_frame_id','base_link'),
                     lambda g,t:setattr(t.header,'frame_id','odom'),
                     lambda g,t:setattr(g.header.stamp,'nanosec',-1)]
        for mutate in mutations:
            g,t=goal(),transform()
            mutate(g,t)
            with self.assertRaises(ValueError):
                odom_goal_to_map(g,t,100_000_000_000)
        for frame in ('','base_link','odom/map',None):
            with self.assertRaises(ValueError):
                goal_frame(frame)
        self.assertEqual(goal_frame('/map'),'map')


class FakeTFError(Exception):
    pass


class FakeTime:
    def __init__(self, ns=0):
        self.nanoseconds=ns

    @classmethod
    def from_msg(cls, value):
        return cls(value.sec*10**9+value.nanosec)


class GoalQueue(unittest.TestCase):
    def setUp(self):
        # Compile the unchanged production Operator class body. Only ROS IO
        # is replaced, so the tests exercise real methods rather than copies.
        self.clock=NS(mono=10.,ns=100_000_000_000)
        namespace=dict(Node=object,copy=copy,deque=deque,math=math,
            time=NS(monotonic=lambda:self.clock.mono),Time=FakeTime,
            TransformException=FakeTFError,pose_matrix=pose_matrix,grid_cell=grid_cell,
            GOAL_TF_WAIT_SECONDS=GOAL_TF_WAIT_SECONDS,goal_frame=goal_frame,
            odom_goal_to_map=odom_goal_to_map,validate_odom_goal_time=validate_odom_goal_time,
            json=json,hashlib=hashlib,UUID=UUID,String=lambda **kw:NS(**kw),Empty=NS)
        path=GO2/'operator_bridge.py'
        body=next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n,ast.ClassDef) and n.name=='Operator')
        exec(compile(ast.Module(body=[body],type_ignores=[]),str(path),'exec'),namespace)
        self.op=namespace['Operator'].__new__(namespace['Operator'])
        op=self.op
        op.action=object()
        op.mode='navigation'
        op.auto_resume=None
        op.goal_inputs=deque()
        op.waypoint_edit=None
        op.motion_requests=[]
        op.waypoints=deque()
        op.inflight_waypoint=op.goal_handle=op.enable_query=None
        op.goal_pending=op.cancel_requested=op.cancelling=False
        op.route_started=op.route_paused=False
        op.goal_generation=op.waypoint_number=op.completed_waypoints=0
        op.planned_route=[]
        op.planned_route_overflow=False
        op.valid=True
        op.valid_received=self.clock.mono
        self.reports=[]
        op.report=self.reports.append
        op.get_clock=lambda:NS(now=lambda:NS(nanoseconds=self.clock.ns))
        op.publish_waypoints=lambda:None
        self.dispatch_checks=[]
        op.check_motion_state=lambda:self.dispatch_checks.append(True)
        op.pause_for_localization=lambda:setattr(op,'route_paused',True)
        self.tf=transform()
        self.tf_available=True
        self.queries=[]
        def lookup(parent,child,query):
            self.queries.append((parent,child,query.nanoseconds))
            if not self.tf_available:
                raise FakeTFError('requested timestamp not yet available')
            return self.tf
        op.buffer=NS(lookup_transform=lookup)
        grid=NS(header=NS(frame_id='map'),info=NS(resolution=1.,width=100,height=100,
            origin=NS(position=NS(x=-50.,y=-50.,z=0.),orientation=quaternion(0.))),data=[0]*10000)
        op.grids={'/map':(grid,self.clock.mono)}

    def test_odom_queues_map_pose_and_never_enables_new_route(self):
        self.op.on_goal(goal())
        self.assertEqual(len(self.op.waypoints),1)
        queued=self.op.waiting_goal
        self.assertEqual(queued.header.frame_id,'map')
        self.assertAlmostEqual(queued.pose.position.x,9.)
        self.assertAlmostEqual(queued.pose.position.y,0.)
        np.testing.assert_allclose(self.op.planned_route[0]['position'],[9.,0.,.6],atol=1e-12)
        self.assertEqual(self.queries,[('map','odom',100_000_000_000)])
        self.assertFalse(self.op.route_started)
        self.assertEqual(self.dispatch_checks,[])

    def test_missing_then_ready_tf_preserves_click_order_and_exact_stamp(self):
        self.tf_available=False
        self.op.on_goal(goal())
        self.op.on_goal(goal(x=3.,frame='map'))
        self.assertFalse(self.op.waypoints)
        self.assertEqual(len(self.op.goal_inputs),2)
        self.op.on_enable(NS(data=True))
        self.assertIn('Enable deferred',self.reports[-1])
        self.assertFalse(self.op.route_started)
        self.clock.mono+=.1
        self.tf_available=True
        self.op.process_goal_inputs()
        self.assertEqual([n for n,_ in self.op.waypoints],[1,2])
        self.assertEqual([p.pose.position.x for _,p in self.op.waypoints],[9.,3.])
        self.assertTrue(all(q[2]==100_000_000_000 for q in self.queries))
        self.assertFalse(self.op.route_started)
        self.assertEqual(self.dispatch_checks,[])

    def test_missing_tf_times_out_and_does_not_queue_even_if_it_arrives_late(self):
        self.tf_available=False
        self.op.on_goal(goal())
        self.clock.mono+=.51
        self.tf_available=True
        self.op.process_goal_inputs()
        self.assertFalse(self.op.waypoints)
        self.assertFalse(self.op.goal_inputs)
        self.assertIn('超时',self.reports[-1])

    def test_stale_tf_and_invalid_localization_never_queue(self):
        self.tf=transform(ns=99_000_000_000)
        self.op.on_goal(goal(ns=0))
        self.assertIn('已过期',self.reports[-1])
        self.assertFalse(self.op.waypoints)
        self.op.valid=False
        self.op.on_goal(goal())
        self.assertIn('新鲜有效定位',self.reports[-1])
        self.assertFalse(self.op.goal_inputs)

    def test_localization_loss_while_waiting_discards_input(self):
        self.tf_available=False
        self.op.on_goal(goal())
        self.op.valid=False
        self.op.process_goal_inputs()
        self.assertFalse(self.op.goal_inputs)
        self.assertFalse(self.op.waypoints)
        self.assertIn('定位无效',self.reports[-1])

    def test_zero_stamp_lookup_and_map_old_stamp_behavior_preserved(self):
        self.op.on_goal(goal(ns=0))
        self.assertEqual(self.queries[-1],('map','odom',0))
        self.op.on_goal(goal(frame='/map',ns=2_000_000_000))
        self.assertEqual(len(self.op.waypoints),2)
        self.assertEqual(self.op.waypoints[-1][1].header.stamp.sec,2)
        self.assertEqual(len(self.queries),1)

    def test_transformed_goal_still_checks_map_bounds_and_collision(self):
        self.tf=transform(x=1000.)
        self.op.on_goal(goal())
        self.assertFalse(self.op.waypoints)
        self.assertIn('outside grid',self.reports[-1])
        self.tf=transform()
        grid=self.op.grids['/map'][0]
        grid.data[50*100+59]=100  # transformed target (9,0)
        self.op.on_goal(goal())
        self.assertEqual(len(self.op.waypoints),1)
        self.assertIn('execution requires a free goal cell',' '.join(self.reports))
        with self.assertRaisesRegex(ValueError,'not free'):
            self.op.validate_map_goal(self.op.waiting_goal)
        self.assertFalse(self.op.route_started)

    def test_cancel_and_initial_pose_clear_delayed_goals(self):
        for action in ('on_cancel','on_initial_pose'):
            with self.subTest(action=action):
                self.tf_available=False
                self.op.on_goal(goal())
                self.assertEqual(len(self.op.goal_inputs),1)
                getattr(self.op,action)(NS())
                self.tf_available=True
                self.op.process_goal_inputs()
                self.assertFalse(self.op.goal_inputs)
                self.assertFalse(self.op.waypoints)
                self.assertFalse(self.op.route_started)

    def test_disable_clears_pending_even_if_motion_service_is_down(self):
        self.tf_available=False
        self.op.on_goal(goal())
        self.op.motion_generation=0
        self.op.parameters=NS(service_is_ready=lambda:False)
        self.op.on_enable(NS(data=False))
        self.tf_available=True
        self.op.process_goal_inputs()
        self.assertFalse(self.op.goal_inputs)
        self.assertFalse(self.op.waypoints)
        self.assertFalse(self.op.route_started)

    def test_pending_counts_towards_limit_and_blocks_route_file_commands(self):
        self.tf_available=False
        self.op.on_goal(goal())
        self.op.waypoints=deque((i,goal(frame='map')) for i in range(99))
        self.op.on_goal(goal())
        self.assertEqual(len(self.op.goal_inputs),1)
        self.assertIn('limit is 100',self.reports[-1])
        self.op.waypoints.clear()
        self.op.route_map={'test':'map identity not used before pending check'}
        self.op.route_requests={}
        results=[]
        self.op.route_results=NS(publish=lambda msg:results.append(json.loads(msg.data)))
        for command in ('snapshot','load','run'):
            request={'request_id':str(uuid4()),'command':command}
            if command!='snapshot':
                request['route']={}
            self.op.on_route_command(NS(data=json.dumps(request)))
            self.assertFalse(results[-1]['success'])
            self.assertEqual(len(self.op.goal_inputs),1)


if __name__ == '__main__':
    unittest.main()
