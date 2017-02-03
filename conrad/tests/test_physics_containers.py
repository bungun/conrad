"""
Unit tests for :mod:`conrad.physics.containers`.
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

import numpy as np

from conrad.physics.containers import *
from conrad.tests.base import *

class WeightVectorTestCases(ConradTestCase):
	def test_weight_vec_init_attr(self):
		v = WeightVector(np.random.rand(10))
		with self.assertRaises(ValueError):
			v = WeightVector(-np.random.rand(10))

		v_ = { 0: np.random.rand(3), 1: np.random.rand(4),
			   2: np.random.rand(5)}
		v = WeightVector(v_)
		with self.assertRaises(ValueError):
			v_[0] = -v_[0]
			v = WeightVector(v_)

class DoseMatrixTestCase(ConradTestCase):
	def test_dose_mat_init_attr(self):
		m, n = 20, 10
		A_ = np.random.rand(m, n)
		A = DoseMatrix(A_)

		self.assertEqual( A.voxel_dim, m )
		self.assertEqual( A.beam_dim, n )
		self.assert_vector_equal( A.data, A_ )

		data = {i: np.random.rand(m, n) for i in xrange(4)}
		A = DoseMatrix(data)
		self.assertEqual( A.voxel_dim, 4 * m )
		self.assertEqual( A.beam_dim, n )

		self.assertTrue( all(i in A for i in xrange(4)) )
		self.assertTrue( all(('voxel', i) in A for i in xrange(4)) )
		self.assertFalse( any(('beam', i) in A for i in xrange(4)) )
		self.assertFalse( any(('both', (i, i)) in A for i in xrange(4)) )

		data['labeled_by'] = 'voxels'
		A = DoseMatrix(data)
		self.assertEqual( A.voxel_dim, 4 * m )
		self.assertEqual( A.beam_dim, n )
		self.assertTrue( all(i in A for i in xrange(4)) )
		self.assertTrue( all(('voxel', i) in A for i in xrange(4)) )
		self.assertFalse( any(('beam', i) in A for i in xrange(4)) )
		self.assertFalse( any(('both', (i, i)) in A for i in xrange(4)) )

		data['labeled_by'] = 'beams'
		A = DoseMatrix(data)
		self.assertEqual( A.voxel_dim, m )
		self.assertEqual( A.beam_dim, 4 * n )
		self.assertFalse( any(i in A for i in xrange(4)) )
		self.assertTrue( all(('beam', i) in A for i in xrange(4)) )
		self.assertFalse( any(('voxel', i) in A for i in xrange(4)) )
		self.assertFalse( any(('both', (i, i)) in A for i in xrange(4)) )

		with self.assertRaises(ValueError):
			data['labeled_by'] = 'invalid specification'
			A = DoseMatrix(data)

	def test_dose_mat_voxel_slice(self):
		m, n = 30, 40
		indices = [1, 5, 8, 15, 20, 22]

		A = np.random.rand(m, n)
		A_sub_check = A[indices, :]

		D = DoseMatrix(A)
		D.voxel_slice(0, indices)
		self.assertIn( 0, D )
		self.assertIn( ('voxel', 0), D )
		self.assert_vector_equal(D.voxel_slice(0, None), A_sub_check)

	def test_dose_mat_beam_slice(self):
		m, n = 30, 40
		indices = [1, 5, 8, 15, 20, 22, 33, 35, 37]

		A = np.random.rand(m, n)
		A_sub_check = A[:, indices]

		D = DoseMatrix(A)
		D.beam_slice(0, indices)
		self.assertIn( ('beam', 0), D )
		self.assert_vector_equal(D.beam_slice(0, None), A_sub_check)

	def test_dose_mat_slice(self):
		m, n = 30, 40
		v_indices = [1, 5, 8, 15, 20, 22]
		b_indices = [1, 5, 8, 15, 20, 22, 33, 35, 37]

		A = np.random.rand(m, n)
		A_sub_v = A[v_indices, :]
		A_sub_b = A[:, b_indices]
		A_sub_vb = A[v_indices, :][:, b_indices]

		D = DoseMatrix(A)
		self.assertNotIn( 'labeled_by', D.manifest )

		with self.assertRaises(ValueError):
			D.slice()

		with self.assertRaises(ValueError):
			D.slice(voxel_label=0)

		D.slice(voxel_label=0, voxel_indices=v_indices)
		self.assertIn( 0, D )
		self.assertIn( ('voxel', 0), D )
		self.assertNotIn( ('beam', 0), D )
		self.assertNotIn( ('both', (0, 0)), D )
		self.assert_vector_equal( D.slice(voxel_label=0), A_sub_v )

		self.assertEqual( len(D.cached_slices['voxel']), 1 )
		self.assertEqual( len(D.cached_slices['beam']), 0 )
		self.assertEqual( len(D.cached_slices['both']), 0 )

		self.assertIn( 'labeled_by', D.manifest )
		self.assertEqual( D.manifest['labeled_by'], 'voxels' )

		with self.assertRaises(ValueError):
			D.slice(beam_label=0)

		D.slice(beam_label=0, beam_indices=b_indices)
		self.assertIn( 0, D )
		self.assertIn( ('voxel', 0), D )
		self.assertIn( ('beam', 0), D )
		self.assertNotIn( ('both', (0, 0)), D )
		self.assert_vector_equal( D.slice(beam_label=0), A_sub_b )

		self.assertEqual( len(D.cached_slices['voxel']), 1 )
		self.assertEqual( len(D.cached_slices['beam']), 1 )
		self.assertEqual( len(D.cached_slices['both']), 0 )

		self.assertIn( 'labeled_by', D.manifest )
		self.assertEqual( D.manifest['labeled_by'], 'voxels' )