"""Synthetic uneven-ground classification; these are not robot traversal tests."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

GO2 = Path(__file__).resolve().parents[1] / 'original_go2/go2_3d'
sys.path.insert(0, str(GO2))
from terrain import TerrainModel, build_terrain, load_terrain


def grid(x0, x1, y0, y1, surface, spacing=.05):
    x, y = np.meshgrid(np.arange(x0, x1, spacing)+spacing/2,
                       np.arange(y0, y1, spacing)+spacing/2)
    return np.column_stack((x.ravel(), y.ravel(), surface(x.ravel(), y.ravel())))


def flat(x, y):
    return np.zeros_like(x)


def transformed(points, yaw, offset):
    result = points.copy()
    c, s = math.cos(yaw), math.sin(yaw)
    result[:, :2] = points[:, :2] @ np.array([[c, s], [-s, c]]) + offset[:2]
    result[:, 2] += offset[2]
    return result


class LocalTerrain(unittest.TestCase):
    def build(self, cloud, anchors, chunk=4000):
        return build_terrain((cloud[i:i+chunk] for i in range(0,len(cloud),chunk)), anchors)

    def test_two_maps_rising_crest_downhill_and_roundtrip(self):
        def surface(x,y):
            return .09*x + .2*np.sin(x*.6) + .015*y
        cloud = grid(0, 18, -1.5, 1.5, surface)
        anchor_x = np.arange(.2, 17.9, .5)
        anchors = np.column_stack((anchor_x, np.zeros_like(anchor_x), surface(anchor_x,anchor_x*0)))
        self.assertGreater(cloud[:,2].max(),1.)
        with tempfile.TemporaryDirectory() as temp:
            for name, yaw, offset in [('yard',0.,(0.,0.,0.)),('farm',.47,(137.,-82.,1.8))]:
                with self.subTest(map=name):
                    points = transformed(cloud,yaw,offset)
                    model = self.build(points,transformed(anchors,yaw,offset))
                    h, valid = model.evaluate(points)
                    self.assertGreater(valid.mean(), .90)
                    self.assertLess(np.quantile(np.abs(h[valid]), .99), .035)
                    # Height is local terrain-relative, not global starting Z.
                    tall = points[::100].copy()
                    tall[:,2] += 9.
                    height, support = model.evaluate(tall)
                    self.assertTrue((height[support] > 8.96).all())
                    target = Path(temp)/name
                    target.mkdir()
                    metadata = {'terrain_model':model.save(target/'terrain.npz')}
                    again = load_terrain(target,metadata)
                    new_h,new_valid = again.evaluate(points)
                    np.testing.assert_array_equal(valid,new_valid)
                    np.testing.assert_allclose(h,new_h,equal_nan=True)
                    self.assertIsNone(load_terrain(target,{}))

    def test_obstacles_and_unsupported_tabletops_remain(self):
        cloud = grid(0,12,-2,2,flat)
        # No ground observation exists below this elevated tabletop or block.
        table = (cloud[:,0]>3)&(cloud[:,0]<5)&(cloud[:,1]>.7)&(cloud[:,1]<1.7)
        block = (cloud[:,0]>7)&(cloud[:,0]<8)&(np.abs(cloud[:,1])<.5)
        obstacles = cloud[table|block].copy()
        obstacles[:,2] = np.where(obstacles[:,0]<5,.8,.15)
        ground = cloud[~(table|block)]
        stems = np.column_stack((np.full(80,9.3),np.full(80,-.6),np.linspace(.1,4.,80)))
        anchors = np.column_stack((np.arange(.2,12,.4),np.zeros(30),np.zeros(30)))
        model = self.build(np.vstack((ground,obstacles,stems)),anchors)
        h,valid = model.evaluate(obstacles)
        self.assertTrue((~valid | (h>=.1)).all())
        h,valid = model.evaluate(stems)
        self.assertTrue((~valid | (h>=.099)).all())
        h,valid = model.evaluate(ground)
        self.assertGreater(valid.mean(),.75)
        self.assertLess(np.max(np.abs(h[valid])),.001)

    def test_steps_dropoffs_and_empty_gaps_do_not_become_free(self):
        low = grid(0,4,-1,1,flat)
        high = grid(4,8,-1,1,lambda x,y:np.full_like(x,.2))
        anchors = np.array([[.3,0,0],[2.,0,0],[4.4,0,.2],[6.,0,.2],[7.5,0,.2]])
        model = self.build(np.vstack((low,high)),anchors)
        _, support = model.evaluate(np.array([[3.9,0,0],[4.1,0,.2]]))
        self.assertFalse(support.any(), 'both sides of a sharp step stay unsupported')
        # The upper plateau can be observed ground away from its blocked edge.
        h,support = model.evaluate(np.array([[6.1,.1,.2]]))
        self.assertTrue(support[0])
        self.assertAlmostEqual(h[0],0.)
        hole = grid(0,8,-1,1,flat)
        missing = (hole[:,0]>3.2)&(hole[:,0]<4.8)
        model = self.build(hole[~missing],np.array([[1.,0,0],[6.,0,0]]))
        self.assertFalse(model.height_at(np.array([[4.,0]]))[1][0])
        # No anchor or connected support can turn a sunken low surface free.
        depression = hole[missing].copy()
        depression[:,2] = -.4
        model = self.build(np.vstack((hole[~missing],depression)),np.array([[1.,0,0],[6.,0,0]]))
        self.assertFalse(model.evaluate(np.array([[4.,0,-.4]]))[1][0])

    def test_sparse_vertical_and_isolated_low_noise_are_not_ground(self):
        ground = grid(0,3,-1,1,flat)
        noise = np.array([[.13,.13,-2.],[1.1,.2,-1.],[2.1,-.3,-3.]])
        noise = np.repeat(noise,20,axis=0)  # Repeated identical hits lack spatial support.
        wall = np.column_stack((np.full(80,6.2),np.full(80,0.),np.linspace(0,4,80)))
        cloud = np.vstack((ground,noise,wall,[[12.,0,0]]))
        model = self.build(cloud,np.array([[.2,0,0],[1.5,0,0],[2.8,0,0],[6.2,0,0],[12.,0,0]]))
        h,support = model.evaluate(ground)
        self.assertGreater(support.mean(),.9)
        self.assertLess(np.max(np.abs(h[support])),.001)
        self.assertFalse(model.height_at(np.array([[6.2,0],[12.,0]]))[1].any())
        h,support = model.evaluate(noise)
        self.assertTrue((~support | (h<-.1)).all())

    def test_large_extent_no_global_size_or_height_cutoff_and_chunk_invariance(self):
        a=grid(0,2,-1,1,flat)
        b=a+np.array([15000.,-22000.,120.])
        cloud=np.vstack((a,b))
        anchors=np.array([[.5,0,0],[1.5,0,0],[15000.5,-22000.,120.],[15001.5,-22000.,120.]])
        first=self.build(cloud,anchors,chunk=137)
        second=self.build(cloud[::-1],anchors[::-1],chunk=509)
        np.testing.assert_array_equal(first.cells,second.cells)
        np.testing.assert_allclose(first.planes,second.planes,atol=1e-7)
        h,support=first.evaluate(cloud)
        self.assertGreater(support.mean(),.95)
        self.assertLess(np.max(np.abs(h[support])),1e-5)

    def test_invalid_models_fail_instead_of_using_global_plane(self):
        with self.assertRaisesRegex(ValueError,'foot-height'):
            self.build(grid(0,1,0,1,flat),np.empty((0,3)))
        with self.assertRaisesRegex(ValueError,'No observed local terrain'):
            self.build(grid(0,1,0,1,flat)+[0,0,2],np.array([[.5,.5,0]]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_terrain(tmp,{'terrain_model':{'version':1,'file':'../terrain.npz'}})
            with self.assertRaises(FileNotFoundError):
                load_terrain(tmp,{'terrain_model':{'version':1,'file':'terrain.npz'}})
        model=TerrainModel([[0,0]],[[0,0,0]])
        h,support=model.evaluate(np.array([[np.nan,0,0],[.1,.1,np.inf],[100,0,0]]))
        self.assertFalse(support.any())
        self.assertTrue(np.isnan(h).all())


if __name__ == '__main__':
    unittest.main()
