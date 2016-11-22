"""
Unit tests for :mod:`conrad.physics.matrix`.
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

from conrad.physics.matrix import *
from conrad.tests.base import *

class SparseMatrixSlicingTestCase(ConradTestCase):
	def test_csx_slice_compressed(self):
		m = 50
		n = 40
		A_csr = sp.rand(m, n, 0.5).tocsr()
		A_csc = sp.rand(m, n, 0.5).tocsc()

		# 10 rows or columns
		indices = [1, 4, 7, 12, 19, 22, 25, 34, 37, 38]

		A_csr_sub = csx_slice_compressed(A_csr, indices)
		A_csr_sub_check = A_csr[indices, :]
		self.assertTrue( ( A_csr_sub - A_csr_sub_check).nnz == 0 )

		A_csc_sub = csx_slice_compressed(A_csc, indices)
		A_csc_sub_check = A_csc[:, indices]
		self.assertTrue( ( A_csc_sub - A_csc_sub_check).nnz == 0 )

	def test_csx_slice_uncompressed(self):
		m = 50
		n = 40
		A_csr = sp.rand(m, n, 0.5).tocsr()
		A_csc = sp.rand(m, n, 0.5).tocsc()

		# 10 rows or columns
		indices = [1, 4, 7, 12, 19, 22, 25, 34, 37, 38]

		# A_csr_sub = csx_slice_uncompressed(A_csr, indices)
		# A_csr_sub_check = A_csr[:, indices]
		# self.assertTrue( (A_csr_sub - A_csr_sub_check).nnz == 0 )

		A_csc_sub = csx_slice_uncompressed(A_csc, indices)
		A_csc_sub_check = A_csc[indices, :]
		self.assertTrue( (A_csc_sub - A_csc_sub_check).nnz == 0 )

class WeightVectorTestCase(ConradTestCase):
	def test_weight_vec_init_attr(self):
		w_ = np.random.rand(10)
		w = WeightVector(w_)
		self.assertTrue(w.size == w_.size)
		self.assert_vector_equal(w_, w.data)

		with self.assertRaises(ValueError):
			w = WeightVector(-np.random.rand(10))

		w_ = [3, 5, 9]
		w = WeightVector(w_)
		self.assertTrue(w.size == len(w_))
		self.assert_vector_equal(w_, w.data)

		w_ = { 0: np.random.rand(3), 1: np.random.rand(4),
			   2: np.random.rand(5)}
		w = WeightVector(w_)
		self.assertTrue( len(w._WeightVector__slices) == 3 )
		self.assertTrue( w.data is None )
		self.assertTrue( w.size == 3 + 4 + 5 )
		self.assertTrue( all([label in w for label in [0, 1, 2]]) )

		with self.assertRaises(ValueError):
			w_[0] = -w_[0]
			w = WeightVector(w_)

	def test_weight_vec_slice(self):
		w_ = np.random.rand(10)
		w = WeightVector(w_)

		idx = {}
		idx[0] = [0, 3, 5]
		idx[1] = [1, 2, 4, 6]
		idx[2] = xrange(7, 10)

		w_sub = {}
		for i in xrange(3):
			w_sub[i] = w_[idx[i]]
			w.slice(i, idx[i])
			self.assertTrue( i in w )
			self.assert_vector_equal( w._WeightVector__slices[i], w_sub[i] )
			self.assert_vector_equal( w.slice(i), w_sub[i] )

	def test_weight_vec_assemble(self):
		w_ = np.random.rand(10)
		w = WeightVector(w_)

		with self.assertRaises(AttributeError):
			w.assemble()

		w_ = {i: np.random.rand(i) for i in [4, 7, 9]}
		w = WeightVector(w_)
		w.assemble()
		self.assertTrue( w.data.size == 4 + 7 + 9 )

class DoseMatrixTestCase(ConradTestCase):
	def test_dose_mat_init_attr(self):
		m, n = 20, 10
		A_ = np.random.rand(m, n)
		A = DoseMatrix(A_)

		self.assertTrue( A.voxel_dim == m )
		self.assertTrue( A.beam_dim == n )
		self.assert_vector_equal( A.data, A_ )
		self.assertTrue( len(A._DoseMatrix__voxel_slices) == 0 )
		self.assertTrue( len(A._DoseMatrix__beam_slices) == 0 )
		self.assertTrue( len(A._DoseMatrix__double_slices) == 0 )

		with self.assertRaises(TypeError):
			# 1-D array not accepted
			A = DoseMatrix(np.random.rand(m))

		with self.assertRaises(TypeError):
			# COO sparse not accepted
			A = DoseMatrix('not a matrix')

		with self.assertRaises(TypeError):
			# COO sparse not accepted
			A = DoseMatrix(sp.rand(m, n))

		with self.assertRaises(TypeError):
			# dict with entries other than dense/CSR/CSC sparse not accepted
			A = DoseMatrix({1: np.random.rand(m, n), 2: 'not a matrix'})

		data = {i: np.random.rand(m, n) for i in xrange(4)}
		A = DoseMatrix(data)
		self.assertTrue( A.voxel_dim == 4 * m )
		self.assertTrue( A.beam_dim == n )
		self.assertTrue( len(A._DoseMatrix__voxel_slices) == 4 )
		self.assertTrue( len(A._DoseMatrix__beam_slices) == 0 )
		self.assertTrue( len(A._DoseMatrix__double_slices) == 0 )

		self.assertTrue( all(i in A for i in xrange(4)) )
		self.assertTrue( all(('voxel', i) in A for i in xrange(4)) )
		self.assertFalse( any(('beam', i) in A for i in xrange(4)) )
		self.assertFalse( any(('both', (i, i)) in A for i in xrange(4)) )

		data['labeled_by'] = 'voxels'
		A = DoseMatrix(data)
		self.assertTrue( A.voxel_dim == 4 * m )
		self.assertTrue( A.beam_dim == n )
		self.assertTrue( len(A._DoseMatrix__voxel_slices) == 4 )
		self.assertTrue( len(A._DoseMatrix__beam_slices) == 0 )
		self.assertTrue( len(A._DoseMatrix__double_slices) == 0 )

		self.assertTrue( all(i in A for i in xrange(4)) )
		self.assertTrue( all(('voxel', i) in A for i in xrange(4)) )
		self.assertFalse( any(('beam', i) in A for i in xrange(4)) )
		self.assertFalse( any(('both', (i, i)) in A for i in xrange(4)) )

		data['labeled_by'] = 'beams'
		A = DoseMatrix(data)
		self.assertTrue( A.voxel_dim == m )
		self.assertTrue( A.beam_dim == 4 * n )
		self.assertTrue( len(A._DoseMatrix__voxel_slices) == 0 )
		self.assertTrue( len(A._DoseMatrix__beam_slices) == 4 )
		self.assertTrue( len(A._DoseMatrix__double_slices) == 0 )

		self.assertFalse( any(i in A for i in xrange(4)) )
		self.assertTrue( all(('beam', i) in A for i in xrange(4)) )
		self.assertFalse( any(('voxel', i) in A for i in xrange(4)) )
		self.assertFalse( any(('both', (i, i)) in A for i in xrange(4)) )

		with self.assertRaises(ValueError):
			data['labeled_by'] = 'invalid specification'
			A = DoseMatrix(data)

	def test_dose_mat_voxel_slice(self):
		m, n = 30, 40
		DM = DoseMatrix(np.random.rand(2, 2))
		A_list = [np.random.rand(m, n), sp.rand(m, n).tocsr(),
				  sp.rand(m, n).tocsc()]
		indices = [1, 5, 8, 15, 20, 22]
		for A in A_list:
			with self.assertRaises(ValueError):
				DM._DoseMatrix__beam_slice_generic(A, None)

			A_sub = DM._DoseMatrix__voxel_slice_generic(A, indices)
			A_sub_check = A[indices, :]
			if isinstance(A, np.ndarray):
				self.assert_vector_equal(A_sub, A_sub_check)
			else:
				self.assertTrue( (A_sub - A_sub_check).nnz == 0 )

			D = DoseMatrix(A)
			D.voxel_slice(0, indices)
			self.assertTrue( 0 in D )
			self.assertTrue( ('voxel', 0) in D )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal(D.voxel_slice(0, None), A_sub_check)
			else:
				self.assertTrue( (D.voxel_slice(0, None) - A_sub_check).nnz
								 == 0 )

	def test_dose_mat_beam_slice(self):
		m, n = 30, 40
		DM = DoseMatrix(np.random.rand(2, 2))
		A_list = [np.random.rand(m, n), sp.rand(m, n).tocsr(),
				  sp.rand(m, n).tocsc()]
		indices = [1, 5, 8, 15, 20, 22, 33, 35, 37]
		for A in A_list:
			with self.assertRaises(ValueError):
				DM._DoseMatrix__beam_slice_generic(A, None)

			A_sub = DM._DoseMatrix__beam_slice_generic(A, indices)
			A_sub_check = A[:, indices]
			if isinstance(A, np.ndarray):
				self.assert_vector_equal(A_sub, A_sub_check)
			else:
				self.assertTrue( (A_sub - A_sub_check).nnz == 0 )

			D = DoseMatrix(A)
			D.beam_slice(0, indices)
			self.assertTrue( ('beam', 0) in D )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal(D.beam_slice(0, None), A_sub_check)
			else:
				self.assertTrue( (D.beam_slice(0, None) - A_sub_check).nnz
								 == 0 )
	def test_dose_mat_slice(self):
		m, n = 30, 40
		v_indices = [1, 5, 8, 15, 20, 22]
		b_indices = [1, 5, 8, 15, 20, 22, 33, 35, 37]

		A_list = [np.random.rand(m, n), sp.rand(m, n).tocsr(),
				  sp.rand(m, n).tocsc()]
		for A in A_list:
			A_sub_v = A[v_indices, :]
			A_sub_b = A[:, b_indices]
			A_sub_vb = A[v_indices, :][:, b_indices]

			D = DoseMatrix(A)

			with self.assertRaises(ValueError):
				D.slice()

			with self.assertRaises(ValueError):
				D.slice(voxel_label=0)

			D.slice(voxel_label=0, voxel_indices=v_indices)
			self.assertTrue( 0 in D )
			self.assertTrue( ('voxel', 0) in D )
			self.assertTrue( ('beam', 0) not in D )
			self.assertTrue( ('both', (0, 0)) not in D )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal( D.slice(voxel_label=0), A_sub_v )
			else:
				self.assertTrue( (D.slice(voxel_label=0) - A_sub_v).nnz == 0 )

			with self.assertRaises(ValueError):
				D.slice(beam_label=0)

			D.slice(beam_label=0, beam_indices=b_indices)
			self.assertTrue( 0 in D )
			self.assertTrue( ('voxel', 0) in D )
			self.assertTrue( ('beam', 0) in D )
			self.assertTrue( ('both', (0, 0)) not in D )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal( D.slice(beam_label=0), A_sub_b )
			else:
				self.assertTrue( (D.slice(beam_label=0) - A_sub_b).nnz == 0 )

			E = DoseMatrix(A)
			# when slicing in both dimensions, make sure easy path taken
			# w.r.t. compression
			E.slice(voxel_label=0, voxel_indices=v_indices, beam_label=0,
					beam_indices=b_indices)
			if isinstance(A, (np.ndarray, sp.csr_matrix)):
				self.assertTrue( ('voxel', 0) in E )
				self.assertTrue( ('beam', 0) not in E )
				if isinstance(A, np.ndarray):
					self.assert_vector_equal( E.slice(voxel_label=0), A_sub_v )
				else:
					self.assertTrue( (E.slice(voxel_label=0) - A_sub_v).nnz
									 == 0 )

			else:
				self.assertTrue( ('voxel', 0) not in E )
				self.assertTrue( ('beam', 0) in E )
				self.assertTrue( (E.slice(beam_label=0) - A_sub_b).nnz
								 == 0 )
			self.assertTrue( ('both', (0, 0)) in E )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal( E.slice(voxel_label=0, beam_label=0),
										  A_sub_vb)
			else:
				self.assertTrue(
						(E.slice(voxel_label=0, beam_label=0) - A_sub_vb).nnz
						== 0 )
			# if partial slice already cached, ignore easy path
			if isinstance(A, (np.ndarray, sp.csr_matrix)):
				F = DoseMatrix(A)
				F.beam_slice(0, b_indices)
				F.slice(voxel_label=0, voxel_indices=v_indices, beam_label=0,
						beam_indices=b_indices)
				self.assertTrue( ('voxel', 0) not in F )
				self.assertTrue( ('beam', 0) in F )
				if isinstance(A, np.ndarray):
					self.assert_vector_equal(F.slice(beam_label=0), A_sub_b)
				else:
					self.assertTrue(
						(F.slice(beam_label=0) != A_sub_b).nnz
						== 0 )
			else:
				F = DoseMatrix(A)
				F.voxel_slice(0, v_indices)
				F.slice(voxel_label=0, voxel_indices=v_indices, beam_label=0,
						beam_indices=b_indices)
				self.assertTrue( ('voxel', 0) in F )
				self.assertTrue( ('beam', 0) not in F )
				self.assertTrue( (F.slice(voxel_label=0) - A_sub_v).nnz
								 == 0 )
			self.assertTrue( ('both', (0, 0)) in F )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal(
						F.slice(voxel_label=0, beam_label=0), A_sub_vb)
			else:
				self.assertTrue(
					(F.slice(voxel_label=0, beam_label=0) - A_sub_vb).nnz
					== 0 )
