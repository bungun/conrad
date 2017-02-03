"""
Unit tests for :mod:`conrad.physics.beams`.
"""
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
from conrad.physics.beams import *
from conrad.tests.base import *

class BeamsTestCase(ConradTestCase):
	def test_bixel_grid(self):
		nx = 12
		ny = 9
		n = nx * ny

		dx = 1 * cm
		dy = 0.5 * cm

		g = BixelGrid(nx, ny)
		self.assertEqual( g.bixels, n )

		idx = int(n * rand())
		g.set_order('xy')
		self.assertEqual( g.position2index(*g.index2position(idx)), idx )
		g.set_order('yx')
		self.assertEqual( g.position2index(*g.index2position(idx)), idx )

		indices = (n * rand(10)).astype(int)
		idx_recovered = listmap(
				lambda pos : g.position2index(*pos),
				listmap(g.index2position, indices))
		self.assertTrue(
				all(listmap(lambda a, b : a == b, indices, idx_recovered)) )

	def test_beam_set(self):
		pass