"""Measured terrain must remove slopes, not erase real front obstacles."""
from pathlib import Path
import sys
import unittest

import numpy as np

GO2 = Path(__file__).resolve().parents[1]/'original_go2/go2_3d'
sys.path.insert(0,str(GO2))
from live_ground import measured_ground,relative_heights
from terrain import TerrainModel


class LiveGroundTests(unittest.TestCase):
    def scene(self,shift=(0,0,0),slope=(.12,-.04)):
        xx,yy = np.meshgrid(np.arange(-2.37,2.4,.06),np.arange(-2.37,2.4,.06))
        xy = np.column_stack((xx.ravel(),yy.ravel()))
        z = xy@np.asarray(slope)
        return np.column_stack((xy,z))+shift

    def test_two_maps_slope_outside_cleared_road_and_real_front_obstacle(self):
        for name,shift,slope in [('yard',np.array([0.,0.,0.]),[0.,0.]),
                                 ('orchard',np.array([28.,-14.,1.8]),[.12,-.04])]:
            with self.subTest(map=name):
                floor = self.scene(shift,slope)
                body = shift+[0,0,.327]
                local = measured_ground(floor,body,.327)
                self.assertIsNotNone(local)
                # Saved corridor only knows a few narrow cells. Actual ground
                # outside it still has measured support, independent of name.
                saved = TerrainModel([[0,0]],[[0,0,0]])
                floor_h,support,live = relative_heights(floor,{'plane':[0,0,1,0]},saved,local)
                self.assertGreater(live.mean(),.90)
                self.assertLess(np.max(abs(floor_h[live])),1e-8)
                query = np.array([[.7,.1,.7*slope[0]+.1*slope[1]+.28],
                                  [.7,.1,.7*slope[0]+.1*slope[1]-.2],
                                  [8,0,0]],dtype=float)+shift
                h,support,live = relative_heights(query,{'plane':[0,0,1,0]},saved,local)
                self.assertTrue(support[0])
                self.assertAlmostEqual(h[0],.28)
                self.assertAlmostEqual(h[1],-.2)
                self.assertFalse(support[2])
                keep = ~support | ((h>=.1)&(h<=1.5)) | (h<=-.1)
                self.assertTrue(keep.all())

    def test_step_and_table_are_not_reclassified_as_ground(self):
        floor = self.scene(slope=(0,0))
        step = floor.copy()
        step[step[:,0]>.9,2] += .3
        local = measured_ground(step,[0,0,.327],.327)
        self.assertIsNotNone(local)
        heights,supported = local.evaluate([[1.6,0,.3],[-1.6,0,0.]])
        self.assertFalse(supported[0])
        self.assertTrue(supported[1])
        self.assertIsNone(measured_ground(floor+[0,0,.8],[0,0,.327],.327))

    def test_missing_sparse_or_steep_ground_does_not_assert_free(self):
        self.assertIsNone(measured_ground(np.zeros((0,3)),[0,0,.327],.327))
        self.assertIsNone(measured_ground([[1,0,0],[0,1,0]],[0,0,.327],.327))
        self.assertIsNone(measured_ground(self.scene(slope=(1.,0)),[0,0,.327],.327))
        # A wall (one XY dimension has no support) is never a ground seed.
        yy,zz = np.meshgrid(np.linspace(-2,2,80),np.linspace(-.1,1.5,60))
        wall = np.column_stack((np.full(yy.size,.5),yy.ravel(),zz.ravel()))
        self.assertIsNone(measured_ground(wall,[0,0,.327],.327))

    def test_ground_fit_does_not_mutate_points_or_radius_truncate_output(self):
        floor = self.scene()
        original=floor.copy()
        model=measured_ground(floor,[0,0,.327],.327,radius=1.)
        np.testing.assert_array_equal(floor,original)
        _,supported=model.evaluate([[2.,0,.24]])
        self.assertFalse(supported[0])

if __name__=='__main__':
    unittest.main()
