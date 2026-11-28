"""Check the layout an operator actually imports, including its visible panel tree."""
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAYOUT = ROOT / "Go2.json"


def visible_panel_ids(layout):
    if isinstance(layout, str):
        yield layout
    else:
        yield from visible_panel_ids(layout["first"])
        yield from visible_panel_ids(layout["second"])


class SharedFoxgloveLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        export = json.loads(LAYOUT.read_text(encoding="utf-8"))
        cls.panels = {}
        for panel_id in visible_panel_ids(export["layout"]):
            panel_kind = panel_id.split("!", 1)[0]
            cls.panels.setdefault(panel_kind, []).append(export["configById"][panel_id])

    def test_previous_import_paths_lead_to_one_layout(self):
        for name in ("navigation/foxglove.json", "noetic/farm_live_detail.json",
                     "noetic/farm_point_detail.json"):
            with self.subTest(name=name):
                self.assertEqual((ROOT / name).read_bytes(), LAYOUT.read_bytes())

    def test_3d_views_show_mapping_and_navigation_in_the_same_frame(self):
        views = self.panels["3D"]
        self.assertEqual(len(views), 2)
        mapping_layers = {
            "pose_estimate": ("/go2_visual/ground/dense_map", "poseEstimateTopic", "/initialpose"),
            "pose": ("/go2_visual/ground/scan", "poseTopic", "/goal_pose"),
        }
        self.assertCountEqual([view["publish"]["type"] for view in views], mapping_layers)
        for view in views:
            tool = view["publish"]
            cloud_topic, tool_key, tool_topic = mapping_layers[tool["type"]]
            with self.subTest(tool=tool_topic):
                self.assertEqual(tool[tool_key], tool_topic)
                self.assertEqual(view["fixedFrame"], "map")
                self.assertEqual(view["followTf"], "map")
                self.assertTrue(view["cameraState"]["perspective"])
                self.assertGreater(view["cameraState"]["phi"], 0)
                self.assertLess(view["cameraState"]["phi"], 90)
                for topic in (cloud_topic, "/map", "/navigation/display/scan_obstacles",
                              "/go2_visual/live/model_mesh"):
                    self.assertTrue(view["topics"][topic]["visible"], topic)
                self.assertFalse(view["topics"]["/registered_cloud"]["visible"])

    def test_environment_view_keeps_full_3d_geometry_above_subtle_map(self):
        view, = [view for view in self.panels["3D"] if view["publish"]["type"] == "pose_estimate"]
        topics = view["topics"]
        self.assertTrue(topics["/map_cloud"]["visible"])
        self.assertEqual(topics["/map_cloud"]["colorMode"], "colormap")
        self.assertLessEqual(topics["/map"]["alpha"], .2)
        for topic in ("/navigation/display/map_obstacles", "/global_costmap/costmap",
                      "/local_costmap/costmap"):
            self.assertFalse(topics[topic]["visible"], topic)

    def test_goal_view_exposes_roads_costmaps_and_both_paths(self):
        view, = [view for view in self.panels["3D"] if view["publish"]["type"] == "pose"]
        topics = view["topics"]
        self.assertGreaterEqual(topics["/map"]["alpha"], .6)
        self.assertLess(topics["/map"]["alpha"], 1.)
        self.assertTrue(topics["/navigation/display/map_obstacles"]["visible"])
        self.assertEqual(topics["/navigation/display/map_obstacles"]["colorMode"], "colormap")
        self.assertFalse(topics["/map_cloud"]["visible"])
        self.assertNotEqual(topics["/map"]["minColor"], topics["/map"]["unknownColor"])
        for topic in ("/global_costmap/costmap", "/local_costmap/costmap", "/plan", "/local_plan"):
            self.assertTrue(topics[topic]["visible"], topic)
        self.assertNotEqual(topics["/plan"]["gradient"], topics["/local_plan"]["gradient"])
        for topic in ("/global_costmap/costmap", "/local_costmap/costmap"):
            self.assertLessEqual(topics[topic]["alpha"], .2)
        self.assertGreater(topics["/navigation/display/scan_obstacles"]["decayTime"], 0.)

    def test_camera_and_status_panels(self):
        image, = self.panels["Image"]
        self.assertEqual(image["imageMode"]["imageTopic"], "/camera/display/compressed")
        self.assertEqual(image["imageMode"].get("rotation", 0), 0)
        topics = {panel["topicPath"] for panel in self.panels["RawMessages"]}
        self.assertTrue({"/localization/valid", "/localization/status", "/operator/status"} <= topics)

    def test_navigation_buttons_still_send_the_expected_commands(self):
        messages = [(button["topicName"], button["datatype"], json.loads(button["value"]))
                    for button in self.panels["Publish"]]
        for enable in (True, False):
            self.assertIn(("/control/enable", "std_msgs/msg/Bool", {"data": enable}), messages)
        for command in ("/navigation/cancel", "/navigation/resume"):
            self.assertIn((command, "std_msgs/msg/Empty", {}), messages)


if __name__ == "__main__":
    unittest.main()
