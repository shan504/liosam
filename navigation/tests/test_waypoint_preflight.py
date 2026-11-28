"""Run the real operator queue/action lifecycle with deterministic ROS IO doubles."""
import copy
import json
from collections import deque
from types import SimpleNamespace as NS
import unittest
from uuid import uuid4

from navigation.tests import test_goal_frames as goal_tests
from navigation.tests.test_goal_frames import FakeTime, goal, stamp, transform
from route_store import validate_route
from geometry import planar_pose, grid_cell, quaternion_from_matrix


class ClockTime(FakeTime):
    def to_msg(self):
        return stamp(self.nanoseconds)

    def __sub__(self, other):
        return NS(nanoseconds=self.nanoseconds-other.nanoseconds)


class Future:
    def __init__(self, result=None):
        self.value, self.ready, self.callbacks = result, result is not None, []

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callbacks.append(callback)
        if self.ready:
            callback(self)

    def finish(self, result):
        self.value, self.ready = result, True
        for callback in list(self.callbacks):
            callback(self)


class Parameter:
    Type = NS(BOOL=NS(value=1))

    def __init__(self, name, value):
        self.name,self.value=name,value

    def to_parameter_msg(self):
        return self


class Marker:
    DELETEALL,ADD,SPHERE,TEXT_VIEW_FACING = 3,0,2,9

    def __init__(self, **kwargs):
        self.header=NS(frame_id='')
        self.scale=NS(x=0.,y=0.,z=0.)
        self.color=NS(r=0.,g=0.,b=0.,a=0.)
        self.__dict__.update(kwargs)


class WaypointPreflight(unittest.TestCase):
    def setUp(self):
        goal_tests.GoalQueue.setUp(self)
        op=self.op
        ns=op.on_goal.__globals__
        ns.update(Time=ClockTime,Parameter=Parameter,
                  SetParameters=NS(Request=lambda:NS()),GetParameters=NS(Request=lambda **kw:NS(**kw)),
                  Bool=lambda **kw:NS(**kw),Empty=lambda:NS(),PoseStamped=lambda:goal(frame='map'),
                  validate_route=validate_route,planar_pose=planar_pose,quaternion_from_matrix=quaternion_from_matrix,ARRIVAL_RADIUS=.3,
                  Marker=Marker,MarkerArray=lambda **kw:NS(**kw),DispatchNotReady=type('DispatchNotReady',(ValueError,),{}))
        op.get_clock=lambda:NS(now=lambda:ClockTime(self.clock.ns))
        op.motion_generation=0
        op.pause_disable_future=None
        op.active_dispatch=None
        op.route_pause_reason=None
        op.last_goal_status=None
        op.last_auto_resume_revocation=''
        op.startup_status='ready'
        op.dispatch_retry_at=0.
        op.dispatch_wait_reason=''
        op.route_map=dict(frame_id='map',fingerprint='a'*64,label='synthetic_map')
        op.route_map_error=''
        op.route_requests={}
        self.results=[]
        op.route_results=NS(publish=lambda msg:self.results.append(json.loads(msg.data)))
        self.markers=[]
        op.waypoint_markers=NS(publish=self.markers.append)
        grid=op.grids['/map'][0]
        for topic in ('/global_costmap/costmap','/local_costmap/costmap'):
            op.grids[topic]=(copy.deepcopy(grid),self.clock.mono)
        op.buffer=NS(lookup_transform=lambda *args:transform(ns=self.clock.ns))
        self.enabled=False
        self.parameter_calls=[]
        self.delay_enable=False
        self.enable_futures=[]
        def set_parameters(request):
            self.parameter_calls.append(request.parameters[0].value)
            value=request.parameters[0].value
            if not self.delay_enable:
                self.enabled=value
            future=Future(None if self.delay_enable else NS(results=[NS(successful=True,reason='')]))
            self.enable_futures.append(future)
            return future
        op.parameters=NS(service_is_ready=lambda:True,call_async=set_parameters)
        op.motion_state=NS(service_is_ready=lambda:True,remove_pending_request=lambda f:None,
            call_async=lambda req:Future(NS(values=[NS(type=1,bool_value=self.enabled)])))
        op.goal_type=NS(Goal=lambda:NS())
        self.sent=[]
        self.action_results=[]
        self.action_response_futures=[]
        self.delay_action=False
        def send_goal(request):
            self.sent.append(copy.deepcopy(request.pose))
            result=Future()
            self.action_results.append(result)
            handle=NS(accepted=True,get_result_async=lambda:result,
                      cancel_goal_async=lambda:Future(NS()))
            response=Future(None if self.delay_action else handle)
            response.handle=handle
            self.action_response_futures.append(response)
            return response
        op.action=NS(server_is_ready=lambda:True,send_goal_async=send_goal)
        del op.check_motion_state

    def add_route(self,count=12,offset=0.):
        for number in range(1,count+1):
            self.op.on_goal(goal(x=offset+number,y=1.,z=0.,frame='map'))
        self.assertEqual(len(self.op.waypoints),count)

    def block(self,number,value=100):
        pose=next(p for n,p in self.op.waypoints if n==number)
        grid=self.op.grids['/map'][0]
        x,y=grid_cell(grid,[pose.pose.position.x,pose.pose.position.y])
        grid.data[y*grid.info.width+x]=value

    def command(self,command,**kwargs):
        payload=dict(request_id=str(uuid4()),command=command,**kwargs)
        self.op.on_route_command(NS(data=json.dumps(payload)))
        return self.results[-1]

    def test_twelve_real_action_results_advance_once_in_order(self):
        self.add_route()
        self.op.on_enable(NS(data=True))
        self.assertEqual(len(self.sent),1)
        for number in range(1,13):
            self.assertEqual(self.op.inflight_waypoint[0],number)
            result=self.action_results[-1]
            result.finish(NS(status=4))
            self.op.check_motion_state()
            # A repeated callback cannot clear/complete the next dispatch.
            result.finish(NS(status=4))
            self.op.check_motion_state()
            self.assertEqual(self.op.completed_waypoints,number)
        self.assertEqual([p.pose.position.x for p in self.sent],list(range(1,13)))
        self.assertFalse(self.op.waypoints)
        self.assertFalse(self.op.route_started)

    def test_bad_tail_prevents_departure_on_two_maps(self):
        for offset,label in ((0.,'yard'),(20.,'orchard')):
            self.setUp()
            self.op.route_map['label']=label
            self.add_route(offset=offset)
            self.block(12)
            self.op.on_enable(NS(data=True))
            self.assertFalse(self.sent)
            self.assertFalse(self.parameter_calls)
            self.assertEqual(len(self.op.waypoints),12)
            self.assertIn('#12',self.reports[-1])
            self.assertIn('value=100',self.reports[-1])

    def test_blocked_second_after_first_preserves_tail_and_replace_then_resume(self):
        self.add_route()
        history=copy.deepcopy(self.op.planned_route)
        self.op.on_enable(NS(data=True))
        self.block(2)
        self.action_results[0].finish(NS(status=4))
        self.op.check_motion_state()
        self.assertEqual(self.op.completed_waypoints,1)
        self.assertEqual([n for n,_ in self.op.waypoints],list(range(2,13)))
        self.assertTrue(self.op.route_paused)
        result=self.command('replace',number=2)
        self.assertTrue(result['success'])
        self.op.on_resume(NS())
        self.op.on_enable(NS(data=True))
        self.assertEqual(len(self.sent),1)
        self.op.on_goal(goal(x=2.,y=3.,z=0.,frame='map'))
        self.assertEqual(self.op.planned_route[0],history[0])
        self.assertEqual(self.op.planned_route[2:],history[2:])
        self.assertEqual(self.op.planned_route[1]['position'],[2.,3.,0.])
        self.assertEqual(len(self.sent),1)
        self.assertEqual(len(self.op.waypoints),11)
        self.assertTrue(self.op.route_paused)
        self.op.on_resume(NS())
        for number in range(2,13):
            self.assertEqual(self.op.inflight_waypoint[0],number)
            self.action_results[-1].finish(NS(status=4))
            self.op.check_motion_state()
        self.assertEqual(len(self.sent),12)
        self.assertEqual(self.op.completed_waypoints,12)

    def test_edit_invalid_click_preserves_original(self):
        self.add_route(12)
        original=copy.deepcopy(self.op.planned_route)
        self.block(2)
        self.assertTrue(self.command('replace',number=2)['success'])
        self.op.on_goal(goal(x=2.,y=1.,frame='map'))
        self.assertEqual(self.op.planned_route,original)
        self.assertEqual(self.op.waypoint_edit,2)
        self.op.on_goal(goal(x=2.,y=3.,frame='map'))
        self.assertIsNone(self.op.waypoint_edit)
        self.assertEqual(len(self.op.waypoints),12)
        self.assertFalse(self.sent)

    def test_pending_enable_prevents_edit_until_ack(self):
        self.add_route()
        self.delay_enable=True
        self.op.on_enable(NS(data=True))
        self.assertEqual(len(self.op.motion_requests),1)
        self.assertFalse(self.command('replace',number=2)['success'])
        self.assertIsNone(self.op.waypoint_edit)
        # A changed map while enable is pending is checked again at ack.
        self.block(2)
        self.enable_futures[-1].finish(NS(results=[NS(successful=True,reason='')]))
        self.assertFalse(self.sent)
        self.assertTrue(self.op.route_paused)
        self.assertFalse(self.op.motion_requests)
        self.assertTrue(self.command('replace',number=2)['success'])

    def test_pending_edit_tf_cancel_never_becomes_append(self):
        self.add_route()
        self.assertTrue(self.command('replace',number=2)['success'])
        from navigation.tests.test_goal_frames import FakeTFError
        self.op.buffer=NS(lookup_transform=lambda *args:(_ for _ in ()).throw(FakeTFError('waiting')))
        self.op.on_goal(goal(frame='odom'))
        self.assertEqual(self.op.goal_inputs[0]['replace_number'],2)
        self.op.on_enable(NS(data=False))
        self.op.buffer=NS(lookup_transform=lambda *args:transform(ns=self.clock.ns))
        self.op.process_goal_inputs()
        self.assertFalse(self.op.goal_inputs)
        self.assertFalse(self.op.waypoints)
        self.assertIsNone(self.op.waypoint_edit)
        self.assertFalse(self.sent)

    def test_check_is_read_only_without_map_identity_and_marker_colors_match(self):
        self.add_route(3)
        self.block(2)
        self.block(3,-1)
        self.op.route_paused=True
        self.op.route_map=None
        before=copy.deepcopy(self.op.waypoints)
        result=self.command('check')
        from navigation.routes import queue_report
        rendered=queue_report(result)
        self.assertIn('#2',rendered)
        self.assertIn('障碍格',rendered)
        self.assertTrue(result['success'])
        self.assertEqual([p['status'] for p in result['queue']['entries']],['free','occupied','unknown'])
        self.assertEqual(self.op.waypoints,before)
        self.assertFalse(self.parameter_calls)
        del self.op.publish_waypoints
        self.op.publish_waypoints()
        spheres=[m for m in self.markers[-1].markers if m.__dict__.get('ns')=='waypoints']
        self.assertEqual([(m.color.r,m.color.g,m.color.b) for m in spheres],
                         [(.1,.9,.1),(.9,.1,.1),(.55,.55,.55)])

    def test_replace_protocol_retry_does_not_rearm_after_consumed_click(self):
        from navigation.route_client import validate_request
        self.add_route(3)
        request=dict(request_id=str(uuid4()),command='replace',number=2)
        validate_request(request)
        message=NS(data=json.dumps(request))
        self.op.on_route_command(message)
        answer=copy.deepcopy(self.results[-1])
        self.assertTrue(answer['success'])
        self.assertEqual(answer['editing'],2)
        self.op.on_goal(goal(x=2.,y=3.,frame='map'))
        self.assertIsNone(self.op.waypoint_edit)
        self.op.on_route_command(message)
        self.assertEqual(self.results[-1],answer)
        self.assertIsNone(self.op.waypoint_edit)
        self.assertEqual(len(self.op.waypoints),3)
        self.assertFalse(self.parameter_calls)

    def test_run_preflight_is_atomic_and_load_remains_nonmoving(self):
        self.add_route(3)
        self.block(2)
        document=dict(schema_version=1,map=self.op.route_map,waypoints=copy.deepcopy(self.op.planned_route))
        self.op.on_cancel(NS())
        result=self.command('run',route=document)
        self.assertFalse(result['success'])
        self.assertIn('#2',result['message'])
        self.assertFalse(self.op.waypoints)
        self.assertFalse(self.parameter_calls)
        self.assertTrue(self.command('load',route=document)['success'])
        self.assertEqual(len(self.op.waypoints),3)
        self.assertFalse(self.sent)

    def test_manual_resume_checks_all_remaining_points(self):
        self.add_route()
        self.op.route_paused=True
        self.op.route_pause_reason='validation'
        self.block(11)
        self.op.on_resume(NS())
        self.assertTrue(self.op.route_paused)
        self.assertFalse(self.op.route_started)
        self.assertIn('#11',self.reports[-1])
        self.assertFalse(self.sent)

    def test_late_action_result_after_cancel_cannot_reopen_cleared_route(self):
        self.add_route()
        self.delay_action=True
        self.op.on_enable(NS(data=True))
        response=self.action_response_futures[-1]
        self.op.on_cancel(NS())
        response.finish(response.handle)
        self.action_results[-1].finish(NS(status=5))
        self.op.check_motion_state()
        self.assertFalse(self.op.waypoints)
        self.assertFalse(self.op.route_started)
        self.assertEqual(len(self.sent),1)


if __name__=='__main__':
    unittest.main()
