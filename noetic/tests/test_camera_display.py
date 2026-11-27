"""Exercise display pixels independently of ROS hardware."""
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

NOETIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NOETIC))
from d435_camera import upright_image


class Message(SimpleNamespace):
    def __init__(self, **values):
        super().__init__(header=SimpleNamespace())
        self.__dict__.update(values)


def asymmetric_image(fmt):
    pixels = Image.new('RGB', (32, 24), (0, 0, 220))
    pixels.paste((220, 0, 0), (0, 0, 16, 12))
    output = io.BytesIO()
    pixels.save(output, format=fmt)
    return output.getvalue()


class CameraDisplay(unittest.TestCase):
    def test_mapping_rotates_jpeg_and_png_display_only(self):
        for fmt in ('JPEG', 'PNG'):
            with self.subTest(fmt=fmt):
                raw = asymmetric_image(fmt)
                source = Message(header=SimpleNamespace(stamp=123, frame_id='camera_color_optical_frame'),
                                 format=fmt.lower(), data=raw)
                result = upright_image(source)
                self.assertIs(result.header, source.header)
                self.assertEqual(result.format, fmt.lower())
                self.assertEqual(source.data, raw)
                with Image.open(io.BytesIO(result.data)) as image:
                    self.assertEqual(image.size, (32, 24))
                    self.assertEqual(image.format, fmt)
                    self.assertGreater(image.getpixel((3, 3))[2], 180)
                    self.assertGreater(image.getpixel((28, 20))[0], 180)

    def test_navigation_publishes_unaltered_camera_sample_to_both_consumers(self):
        sensors = NOETIC.parent / 'navigation/original_go2/patches/sensors'
        substitutes = {'rclpy': Mock(), 'rclpy.node': SimpleNamespace(Node=object),
                       'rclpy.qos': Mock(), 'rclpy.executors': Mock(),
                       'sensor_msgs.msg': SimpleNamespace(CompressedImage=Message), 'unitree_api.msg': Mock()}
        with patch.dict(sys.modules, substitutes), patch.object(sys, 'path', [str(sensors)] + sys.path):
            spec = importlib.util.spec_from_file_location('go2_camera_sample', sensors/'go2_camera_sample.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            node = Mock(request_id=42, received_frames=0)
            raw = asymmetric_image('JPEG')
            header = SimpleNamespace(identity=SimpleNamespace(id=42, api_id=1001),
                                     status=SimpleNamespace(code=0))
            node.display.get_subscription_count.return_value = 1
            module.CameraSample.on_response(node, SimpleNamespace(header=header, binary=raw))
            original = node.images.publish.call_args.args[0]
            node.display.publish.assert_called_once_with(original)
            self.assertEqual(original.header.frame_id, 'go2_front_camera')
            self.assertEqual(original.data, raw)


if __name__ == '__main__':
    unittest.main()
