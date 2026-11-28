"""Read-only isolated map verification, durable new-name output, no robot requests."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import routes
import route_revalidate
from route_store import RouteStore


def route(fingerprint='a'):
    return dict(schema_version=1,map=dict(frame_id='map',fingerprint=fingerprint*64,label='orchard'),
                waypoints=[dict(position=[7.,30.,0.],orientation=[0.,0.,0.,1.]),
                           dict(position=[9.,35.,0.],orientation=[0.,0.,.6,.8])])


class RevalidateCLI(unittest.TestCase):
    def test_isolated_current_image_and_readonly_mounts_save_new_file_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current = root/'cache/original-go2-current'
            current.mkdir(parents=True)
            store = RouteStore(root/'routes')
            store.save('old',route())
            before = (store.directory/'old.json').read_bytes()
            commands=[]
            def docker(command, **kwargs):
                commands.append(command)
                if command[1]=='inspect':
                    return NS(returncode=0,stdout=json.dumps([dict(Image='sha256:'+'f'*64,
                        State=dict(Running=True),Mounts=[dict(Type='bind',Source=str(current),
                            Destination='/runtime/navigation_map')])]))
                self.assertEqual(command[1:6],['run','--rm','-i','--network','none'])
                self.assertIn('sha256:'+'f'*64,command)
                mounts=[command[i+1] for i,p in enumerate(command) if p=='-v']
                self.assertEqual(len(mounts),3)
                self.assertTrue(all(mount.endswith(':ro') for mount in mounts))
                self.assertNotIn('--privileged',command)
                self.assertNotIn('--device',command)
                self.assertEqual(json.loads(kwargs['input']),route())
                return NS(returncode=0,stdout=routes.REVALIDATE_PREFIX+json.dumps(route('b'))+'\n')
            with patch.object(routes,'docker_prefix',return_value=['docker']), \
                    patch.object(routes.subprocess,'run',side_effect=docker), \
                    patch.object(routes,'request_route',side_effect=AssertionError('No ROS requests')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(routes.main(['--directory',str(store.directory),'revalidate','old','--as','new']),0)
            self.assertEqual(store.load('new'),route('b'))
            self.assertEqual((store.directory/'old.json').read_bytes(),before)
            self.assertEqual(len(commands),2)

    def test_no_overwrite_and_failure_does_not_publish_new_route(self):
        with tempfile.TemporaryDirectory() as temp:
            store=RouteStore(temp)
            store.save('old',route())
            before=(Path(temp)/'old.json').read_bytes()
            with patch.object(routes,'revalidate_saved_route') as helper, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(routes.main(['--directory',temp,'revalidate','old','--as','old']),1)
                helper.assert_not_called()
            with patch.object(routes,'revalidate_saved_route',side_effect=subprocess.CalledProcessError(1,'readonly check')), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(routes.main(['--directory',temp,'revalidate','old','--as','new']),1)
            self.assertFalse((Path(temp)/'new.json').exists())
            self.assertEqual((Path(temp)/'old.json').read_bytes(),before)

    def test_only_matching_old_fingerprint_is_used_no_name_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            bad,old=root/'original-go2-a',root/'original-go2-b'
            bad.mkdir();old.mkdir()
            identities=[ValueError('corrupt cache'),route()['map']]
            with patch.object(route_revalidate,'map_identity',side_effect=identities), \
                    patch.object(route_revalidate,'revalidate_route',return_value=route('b')) as verify, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(route_revalidate.revalidate_from_cache(route(),root,'/current'),route('b'))
                verify.assert_called_once_with(route(),old,'/current')
            with patch.object(route_revalidate,'map_identity',return_value=route('c')['map']), \
                    patch.object(route_revalidate,'revalidate_route') as verify, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'指纹完全一致'):
                    route_revalidate.revalidate_from_cache(route(),root,'/current')
                verify.assert_not_called()

    def test_stopped_container_or_modified_waypoints_are_not_accepted(self):
        with patch.object(routes,'docker_prefix',return_value=['docker']), \
                patch.object(routes.subprocess,'run',return_value=NS(stdout=json.dumps([dict(State=dict(Running=False))]))):
            with self.assertRaisesRegex(RuntimeError,'未运行'):
                routes.revalidate_saved_route(route())
        with tempfile.TemporaryDirectory() as temp:
            current=Path(temp)/'original-go2-a';current.mkdir()
            changed=copy.deepcopy(route('b'));changed['waypoints'].reverse()
            info=[dict(State=dict(Running=True),Image='sha256:'+'f'*64,Mounts=[
                dict(Type='bind',Source=str(current),Destination='/runtime/navigation_map')])]
            with patch.object(routes,'docker_prefix',return_value=['docker']), \
                    patch.object(routes.subprocess,'run',side_effect=[NS(stdout=json.dumps(info)),
                        NS(stdout=routes.REVALIDATE_PREFIX+json.dumps(changed))]), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'坐标或顺序'):
                    routes.revalidate_saved_route(route())


if __name__=='__main__':
    unittest.main()
