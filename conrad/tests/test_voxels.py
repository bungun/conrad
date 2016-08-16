"""
Copyright 2016 Baris Ungun, Anqi Fu

This file is part of CONRAD.

CONRAD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CONRAD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CONRAD.  If not, see <http://www.gnu.org/licenses/>.
"""
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
