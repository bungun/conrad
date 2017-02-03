"""
Unit tests for :mod:`conrad.physics.grid`.
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

from conrad.physics.grid import *
from conrad.tests.base import *

class TestGrid(AbstractGrid):
	@property
	def shape(self):
		raise NotImplementedError

class AbstractGridTestCase(ConradTestCase):
	def test_grid_init(self):
		grid = TestGrid()
		self.assertEqual( grid._AbstractGrid__x, 0 )
		self.assertEqual( grid._AbstractGrid__y, 0 )
		self.assertEqual( grid._AbstractGrid__z, 0 )
		self.assertIsInstance( grid.x_unit_length, Length )
		self.assertIsInstance( grid.y_unit_length, Length )
		self.assertIsInstance( grid._AbstractGrid__z_unit_length, Length )
		self.assertIsNone( grid.z_unit_length )
		self.assertEqual( grid.order, '' )
		self.assertIsInstance( grid.dims, list )
		self.assertEqual( len(grid.dims), 0 )

		self.assertIsInstance( grid._AbstractGrid__pos, dict )
		self.assertEqual( len(grid._AbstractGrid__pos), 0 )

	def test_grid_helpers(self):
		grid = TestGrid()
		x = 1
		grid.validate_nonnegative_int(x, 'x')
		grid.validate_positive_int(x, 'x')
		x = 1.
		with self.assertRaises(TypeError):
			grid.validate_nonnegative_int(x, 'x')
		with self.assertRaises(TypeError):
			grid.validate_positive_int(x, 'x')
		x = '1'
		with self.assertRaises(TypeError):
			grid.validate_nonnegative_int(x, 'x')
		with self.assertRaises(TypeError):
			grid.validate_positive_int(x, 'x')
		x = -1
		with self.assertRaises(ValueError):
			grid.validate_nonnegative_int(x, 'x')
		x = 0
		grid.validate_nonnegative_int(x, 'x')
		with self.assertRaises(ValueError):
			grid.validate_positive_int(x, 'x')

		grid.validate_length(1 * mm, 'l')
		with self.assertRaises(TypeError):
			grid.validate_length(1, 'l')

		grid = TestGrid()
		with self.assertRaises(ValueError):
			grid.calculate_strides()

class Grid2DTestCase(ConradTestCase):
	def test_grid2D(self):
		g2 = Grid2D(4, 5)
		self.assertTupleEqual( g2.dims, ('x', 'y') )
		self.assertTupleEqual( g2.shape, (4, 5) )
		self.assertEqual( g2.order, 'xy' )
		self.assertDictEqual( g2.strides, {'x': 1, 'y': 4} )
		g2.set_order('yx')
		self.assertDictEqual( g2.strides, {'x': 5, 'y': 1} )


		with self.assertRaises(AttributeError):
			g2.z_unit_length
		with self.assertRaises(AttributeError):
			g2.z_length

		# TODO: test index2position
		# TODO: test position2index

class Grid3DTestCase(ConradTestCase):
	def test_grid3D(self):
		g3 = Grid3D(4, 5, 6)
		self.assertTupleEqual( g3.dims, ('x', 'y', 'z') )
		self.assertTupleEqual( g3.shape, (4, 5, 6) )
		self.assertEqual( g3.order, 'xyz' )
		self.assertDictEqual( g3.strides, {'x': 1, 'y': 4, 'z': 20})
		# TODO: test index2position
		# TODO: test position2index

