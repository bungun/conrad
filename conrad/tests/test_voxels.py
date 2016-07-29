from conrad.compat import *
from conrad.physics.units import cm, mm
from conrad.physics.voxels import *
from conrad.tests.base import *

class VoxelsTestCase(ConradTestCase):
	def test_voxel_grid(self):
		mx = 10
		my = 8
		mz = 7
		m = mx * my * mz

		dx = 1 * cm
		dy = 5 * mm
		dz = 2 * mm

		g = VoxelGrid(mx, my, mz)
		self.assertEqual(g.voxels, m)
		self.assertTrue(g.total_volume.value is nan)

		g.set_scale(dx, dy, dz)
		self.assertTrue(g.total_volume.value is not nan)
		vol = g.voxels * dx.to_cm.value * dy.to_cm.value * dz.to_cm.value
		self.assertEqual(g.total_volume.value, vol)

		idx = int(m * rand())
		self.assertEqual(g.position2index(*g.index2position(idx)), idx)
		g.set_order('zyx')
		self.assertEqual(g.position2index(*g.index2position(idx)), idx)
		g.set_order('yxz')
		self.assertEqual(g.position2index(*g.index2position(idx)), idx)

		indices = (m * rand(10)).astype(int)
		idx_recovered = listmap(
				lambda pos : g.position2index(*pos),
				listmap(g.index2position, indices))
		self.assertTrue(
				all(listmap(lambda a, b : a == b, indices, idx_recovered)))
