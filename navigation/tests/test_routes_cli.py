"""Host persistence, acknowledged commands and retry behavior without a robot."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
from uuid import uuid4

NAV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NAV))
import route_client
import routes
from route_store import RouteStore, validate_route


def document(label, fingerprint, x):
    return dict(schema_version=1,
                map=dict(frame_id='map', fingerprint=fingerprint*64, label=label),
                waypoints=[dict(position=[x, 2., 0.], orientation=[0., 0., 0., 1.]),
                           dict(position=[x+1., 3., 0.], orientation=[0., 0., .6, .8])])


class RouteCommands(unittest.TestCase):
    def queue_result(self, offset=0.):
        return dict(success=True, queue=dict(current=None, completed=1, queued=2,
                    started=False, paused=True, editing=None, entries=[
            dict(number=2,x=offset+7.1,y=30.76,z=0.,active=False,status='occupied',value=100,
                 cell=[1,2],message='/map goal occupied'),
            dict(number=3,x=offset+8.,y=33.,z=0.,active=False,status='free',value=0,
                 cell=[4,5],message='')]))

    def test_live_check_reports_exact_blocked_number_and_all_pending_points(self):
        for offset in (0., 120.):
            with self.subTest(map_offset=offset):
                response = self.queue_result(offset)
                with patch.object(routes,'request_route',return_value=response) as request:
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(routes.main(['check']),0)
                    request.assert_called_once_with('check')
                    text = output.getvalue()
                    self.assertIn('待执行：2；已完成：1',text)
                    self.assertIn(f'#2 待执行  x={offset+7.1:.3f}',text)
                    self.assertIn('障碍格  value=100',text)
                    self.assertIn('#3 待执行',text)
                    self.assertIn('需要检查的点：#2',text)
                    self.assertNotIn('均空闲',text)
                response['queue']['entries'][0]['status'] = 'unavailable'
                response['queue']['entries'][0]['value'] = None
                self.assertIn('地图未收到',routes.queue_report(response))
                response['queue']['entries'][0]['status'] = 'free'
                self.assertIn('不代表整条路径可达',routes.queue_report(response))
                response['queue']['entries'][0].update(status='invalid',x=None,y=None,z=None)
                self.assertIn('#2 待执行  坐标无效',routes.queue_report(response))

    def test_replace_arms_one_number_without_loading_or_enabling(self):
        response = dict(success=True,editing=2,message='next click replaces waypoint #2')
        with patch.object(routes,'request_route',return_value=response) as request:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(routes.main(['replace','2']),0)
            request.assert_called_once_with('replace',number=2)
            self.assertIn('其余队列保留',output.getvalue())
            self.assertIn('没有启用运动',output.getvalue())
        with patch.object(routes,'request_route',return_value=dict(success=True,editing=3)):
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                self.assertEqual(routes.main(['replace','2']),1)
            self.assertIn('编号不匹配',output.getvalue())

    def test_queue_commands_transport_exact_schema_and_rejection(self):
        seen = []
        def docker(command, **kwargs):
            request = json.loads(kwargs['input'])
            seen.append(request)
            answer = dict(request_id=request['request_id'],success=request['command']=='check',
                          message='Active navigation cannot be edited')
            return NS(returncode=0,stdout=routes.RESULT_PREFIX+json.dumps(answer),stderr='')
        with patch.object(routes,'docker_prefix',return_value=['docker']), \
                patch.object(routes.subprocess,'run',side_effect=docker):
            routes.request_route('check')
            with self.assertRaisesRegex(ValueError,'cannot be edited'):
                routes.request_route('replace',number=2)
        self.assertEqual(set(seen[0]),{'request_id','command'})
        self.assertEqual(set(seen[1]),{'request_id','command','number'})
        self.assertEqual(seen[1]['number'],2)
        for response in ({}, {'queue':dict(entries=[])}):
            with self.assertRaises(RuntimeError):
                routes.queue_report(response)
        with patch.object(routes,'request_route') as request, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                routes.main(['replace','0'])
            request.assert_not_called()

    def test_two_maps_save_restart_load_keep_order_and_dont_enable(self):
        # A simulated container accepts only matching map identity, as the real
        # operator does. File IO and transport parsing are the actual CLI code.
        with tempfile.TemporaryDirectory() as temp:
            store = Path(temp)/'persistent_routes'
            calls = []
            current = [None]

            def docker(command, **kwargs):
                if command == ['docker', 'info']:
                    return NS(returncode=0)
                self.assertEqual(command[:5], ['docker', 'exec', '-i', 'go2-lio-nav', 'bash'])
                request = json.loads(kwargs['input'])
                calls.append(request)
                if request['command'] == 'snapshot':
                    answer = dict(request_id=request['request_id'], success=True, route=current[0])
                else:
                    try:
                        route = validate_route(request['route'], expected_map=current[0]['map'])
                        answer = dict(request_id=request['request_id'], success=True, points=len(route['waypoints']))
                    except ValueError as error:
                        answer = dict(request_id=request['request_id'], success=False, message=str(error))
                return NS(returncode=0, stdout='ROS diagnostic\n'+routes.RESULT_PREFIX+json.dumps(answer)+'\n', stderr='')

            with patch.object(routes.subprocess, 'run', side_effect=docker):
                for label, fingerprint, x in [('yard', 'a', 1.), ('orchard', 'b', 100.)]:
                    current[0] = document(label, fingerprint, x)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(routes.main(['--directory', str(store), 'save', label+'_route']), 0)
                    saved = RouteStore(store).load(label+'_route')
                    self.assertEqual(saved, current[0])
                    # Fresh host CLI/store instance models restarting either
                    # the navigation container or the computer.
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(routes.main(['--directory', str(store), 'load', label+'_route']), 0)
                    self.assertEqual(calls[-1]['command'], 'load')
                    self.assertEqual(calls[-1]['route']['waypoints'], saved['waypoints'])
                self.assertEqual([c['command'] for c in calls], ['snapshot', 'load', 'snapshot', 'load'])
                # Switching maps does not silently reinterpret saved positions.
                before = (store/'yard_route.json').read_bytes()
                output = io.StringIO()
                with contextlib.redirect_stderr(output):
                    self.assertEqual(routes.main(['--directory', str(store), 'load', 'yard_route']), 1)
                self.assertIn('map mismatch', output.getvalue())
                self.assertEqual((store/'yard_route.json').read_bytes(), before)

    def test_no_silent_overwrite_and_list_show_need_no_ros(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, updated = document('yard', 'a', 1.), document('yard', 'a', 5.)
            RouteStore(root).save('route1', original)
            with patch.object(routes, 'request_route', return_value={'route':updated}):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(routes.main(['--directory', temp, 'save', 'route1']), 1)
                self.assertEqual(RouteStore(root).load('route1'), original)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(routes.main(['--directory', temp, 'save', 'route1', '--overwrite']), 0)
                self.assertEqual(RouteStore(root).load('route1'), updated)
            with patch.object(routes.subprocess, 'run', side_effect=AssertionError('No container needed')):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(routes.main(['--directory', temp, 'list']), 0)
                    self.assertEqual(routes.main(['--directory', temp, 'show', 'route1']), 0)
                self.assertIn('route1', output.getvalue())
                self.assertIn('yard', output.getvalue())

    def test_wrong_or_missing_reply_does_not_save(self):
        with tempfile.TemporaryDirectory() as temp:
            wrong = routes.RESULT_PREFIX+json.dumps(dict(request_id=str(uuid4()), success=True,
                                                         route=document('yard', 'a', 1.)))
            with patch.object(routes, 'docker_prefix', return_value=['docker']), \
                 patch.object(routes.subprocess, 'run', return_value=NS(returncode=0, stdout=wrong, stderr='')):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(routes.main(['--directory', temp, 'save', 'never_saved']), 1)
            self.assertEqual(list(Path(temp).glob('*.json')), [])

    def test_run_is_explicit_and_timeout_does_not_claim_success(self):
        with tempfile.TemporaryDirectory() as temp:
            route = document('yard', 'a', 1.)
            RouteStore(temp).save('r', route)
            with patch.object(routes, 'request_route', return_value={'message':'start requested'}) as request:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(routes.main(['--directory', temp, 'run', 'r']), 0)
                request.assert_called_once_with('run', route)
                self.assertIn('不是到达确认', output.getvalue())
            with patch.object(routes, 'docker_prefix', return_value=['docker']), \
                 patch.object(routes.subprocess, 'run', side_effect=subprocess.TimeoutExpired('docker', 20)):
                output = io.StringIO()
                with contextlib.redirect_stderr(output):
                    self.assertEqual(routes.main(['--directory', temp, 'run', 'r']), 1)
                self.assertIn('结果未确认', output.getvalue())

    def test_local_file_validation_atomicity_and_copy_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            store = RouteStore(temp)
            route = document('yard', 'a', 1.)
            store.save('r', route)
            invalid = copy.deepcopy(route)
            invalid['waypoints'][1]['orientation'] = [0, 0, 0, 0]
            with self.assertRaises(ValueError):
                store.save('r', invalid, overwrite=True)
            self.assertEqual(store.load('r'), route)
            with self.assertRaises(ValueError):
                store.load('../r')
            copied_identity = dict(route['map'], label='copied_directory')
            self.assertEqual(validate_route(route, copied_identity), route)


class ROSClientRetry(unittest.TestCase):
    def test_queue_protocol_validation_and_identical_replace_retry(self):
        for command in ('check','replace'):
            request = dict(request_id=str(uuid4()),command=command)
            if command == 'replace':
                request['number'] = 2
            self.assertEqual(route_client.validate_request(request),request['request_id'])
            bad = dict(request, route=document('yard','a',1.))
            with self.assertRaisesRegex(ValueError,'Unexpected'):
                route_client.validate_request(bad)
        for number in (0,-1,True,2.5,'2'):
            with self.subTest(number=number), self.assertRaises(ValueError):
                route_client.validate_request(dict(request_id=str(uuid4()),command='replace',number=number))
        clock,published,modules = self.fake_runtime(2)
        request = dict(request_id=str(uuid4()),command='replace',number=2)
        with patch.dict(sys.modules,modules), patch.object(route_client.time,'monotonic',side_effect=lambda:clock[0]):
            result = route_client.exchange(request)
        self.assertTrue(result['success'])
        self.assertEqual(published,[request,request])

    def fake_runtime(self, answer_after):
        clock = [0.0]
        published = []
        callbacks = []
        node = NS(destroy_node=lambda:None)
        def publish(message):
            request = json.loads(message.data)
            published.append(request)
            if len(published) == answer_after:
                # Unrelated clients cannot consume one another's result.
                callbacks[0](NS(data=json.dumps(dict(request_id=str(uuid4()),success=True))))
                callbacks[0](NS(data=json.dumps(dict(request_id=request['request_id'],success=True))))
        node.create_publisher = lambda *args: NS(get_subscription_count=lambda:1, publish=publish)
        node.create_subscription = lambda cls, topic, cb, depth: callbacks.append(cb)
        def spin_once(node, timeout_sec):
            clock[0] += timeout_sec
        ros = ModuleType('rclpy')
        ros.init = lambda: None
        ros.shutdown = lambda: None
        ros.ok = lambda: True
        ros.create_node = lambda name: node
        ros.spin_once = spin_once
        package, messages = ModuleType('std_msgs'), ModuleType('std_msgs.msg')
        messages.String = lambda **kwargs: NS(**kwargs)
        package.msg = messages
        return clock, published, {'rclpy':ros, 'std_msgs':package, 'std_msgs.msg':messages}

    def test_lost_first_response_retries_identical_request_not_duplicate_route(self):
        clock, published, modules = self.fake_runtime(2)
        request = dict(request_id=str(uuid4()), command='load', route=document('yard','a',1.))
        with patch.dict(sys.modules, modules), patch.object(route_client.time, 'monotonic', side_effect=lambda:clock[0]):
            answer = route_client.exchange(request)
        self.assertEqual(answer['request_id'], request['request_id'])
        self.assertEqual(published, [request, request])

    def test_no_reply_times_out_without_false_success(self):
        clock, published, modules = self.fake_runtime(100)
        request = dict(request_id=str(uuid4()), command='snapshot')
        with patch.dict(sys.modules, modules), patch.object(route_client.time, 'monotonic', side_effect=lambda:clock[0]):
            with self.assertRaisesRegex(TimeoutError, '结果未确认'):
                route_client.exchange(request, timeout=1.0)
        self.assertGreaterEqual(len(published), 1)
        self.assertEqual(len({p['request_id'] for p in published}), 1)


if __name__ == '__main__':
    unittest.main()
