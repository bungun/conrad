"""
Unit tests for :mod:`conrad.abstract.matrix`.
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

from conrad.abstract.vector import *
from conrad.tests.base import *

class SliceCachingVectorTestCase(ConradTestCase):
	def test_sc_vec_init_attr(self):
		v_ = np.random.rand(10)
		v = SliceCachingVector(v_)
		self.assertEqual(v.size, v_.size)
		self.assert_vector_equal(v_, v.data)

		v_ = [3, 5, 9]
		v = SliceCachingVector(v_)
		self.assertEqual(v.size, len(v_))
		self.assert_vector_equal(v_, v.data)

		v_ = { 0: np.random.rand(3), 1: np.random.rand(4),
			   2: np.random.rand(5)}
		v = SliceCachingVector(v_)
		self.assertEqual( len(v._SliceCachingVector__slices), 3 )
		self.assertIsNone( v.data )
		self.assertEqual( v.size, 3 + 4 + 5 )
		self.assertTrue( all([label in v for label in [0, 1, 2]]) )

	def test_sc_vec_slice(self):
		v_ = np.random.rand(10)
		v = SliceCachingVector(v_)

		idx = {}
		idx[0] = [0, 3, 5]
		idx[1] = [1, 2, 4, 6]
		idx[2] = xrange(7, 10)

		v_sub = {}
		for i in xrange(3):
			v_sub[i] = v_[idx[i]]
			v.slice(i, idx[i])
			self.assertIn( i, v )
			self.assert_vector_equal( v._SliceCachingVector__slices[i], v_sub[i] )
			self.assert_vector_equal( v.slice(i), v_sub[i] )

	def test_sc_vec_assemble(self):
		v_ = np.random.rand(10)
		v = SliceCachingVector(v_)

		with self.assertRaises(AttributeError):
			v.assemble()

		v_ = {i: np.random.rand(i) for i in [4, 7, 9]}
		v = SliceCachingVector(v_)
		v.assemble()
		self.assertEqual( v.data.size, 4 + 7 + 9 )