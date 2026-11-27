"""Check the mapping TF alias used by the shared layout."""
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from display_map_frame import connect_display_map, main


class MappingFrame(unittest.TestCase):
    def test_all_modes_use_their_ground_calibrator_parent(self):
        for mode, world in [('go2', 'odom'), ('lio', 'odom'), ('native', 'go2_native/odom')]:
            with self.subTest(mode=mode):
                ros = MagicMock()
                ros.get_param.return_value = mode
                with patch.dict(sys.modules, {'rospy': ros}), patch('display_map_frame.connect_display_map') as connect:
                    main()
                connect.assert_called_once_with(world)
                ros.spin.assert_called_once()

    def test_unified_map_frame_respects_ground_calibration_and_existing_tf(self):
        for existing, connected in [('', False), ('', True), ('map: {parent: world}', False),
                                    ('odom2: {parent: map}', False)]:
            with self.subTest(frames=existing, connected=connected):
                ros, tf, geometry = MagicMock(), MagicMock(), MagicMock()
                ros.is_shutdown.return_value = False
                buffer = tf.Buffer.return_value
                # The Go2 ground calibrator starts after the mapping display node.
                buffer.can_transform.side_effect = [False, True, connected]
                buffer.all_frames_as_yaml.return_value = existing
                with patch.dict(sys.modules, {'rospy': ros, 'tf2_ros': tf, 'geometry_msgs.msg': geometry}):
                    pub = connect_display_map('odom')
                self.assertEqual(buffer.can_transform.call_args_list[0].args[:2], ('go2_ground', 'odom'))
                tf.TransformListener.return_value.unregister.assert_called_once()
                if existing or connected:
                    self.assertIsNone(pub)
                    tf.StaticTransformBroadcaster.assert_not_called()
                else:
                    alias, = pub.sendTransform.call_args.args
                    self.assertEqual((alias.header.frame_id, alias.child_frame_id), ('go2_ground', 'map'))
                    self.assertEqual(alias.transform.rotation.w, 1.)
