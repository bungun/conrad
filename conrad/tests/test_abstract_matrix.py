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

from conrad.abstract.matrix import *
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
		self.assertEqual( (A_csr_sub - A_csr_sub_check).nnz, 0 )

		A_csc_sub = csx_slice_compressed(A_csc, indices)
		A_csc_sub_check = A_csc[:, indices]
		self.assertEqual( (A_csc_sub - A_csc_sub_check).nnz, 0 )

	def test_csx_slice_uncompressed(self):
		m = 50
		n = 40
		A_csr = sp.rand(m, n, 0.5).tocsr()
		A_csc = sp.rand(m, n, 0.5).tocsc()

		# 10 rows or columns
		indices = [1, 4, 7, 12, 19, 22, 25, 34, 37, 38]

		# A_csr_sub = csx_slice_uncompressed(A_csr, indices)
		# A_csr_sub_check = A_csr[:, indices]
		# self.assertEqual( (A_csr_sub - A_csr_sub_check).nnz, 0 )

		A_csc_sub = csx_slice_uncompressed(A_csc, indices)
		A_csc_sub_check = A_csc[indices, :]
		self.assertEqual( (A_csc_sub - A_csc_sub_check).nnz, 0 )

class SliceCachingMatrixTestCase(ConradTestCase):
	def test_sc_mat_init_attr(self):
		m, n = 20, 10
		A_ = np.random.rand(m, n)
		A = SliceCachingMatrix(A_)

		self.assertEqual( A.row_dim, m )
		self.assertEqual( A.column_dim, n )
		self.assert_vector_equal( A.data, A_ )
		self.assertEqual( len(A._SliceCachingMatrix__row_slices), 0 )
		self.assertEqual( len(A._SliceCachingMatrix__column_slices), 0 )
		self.assertEqual( len(A._SliceCachingMatrix__double_slices), 0 )

		with self.assertRaises(TypeError):
			# 1-D array not accepted
			A = SliceCachingMatrix(np.random.rand(m))

		with self.assertRaises(TypeError):
			# COO sparse not accepted
			A = SliceCachingMatrix('not a matrix')

		with self.assertRaises(TypeError):
			# COO sparse not accepted
			A = SliceCachingMatrix(sp.rand(m, n))

		with self.assertRaises(TypeError):
			# dict with entries other than dense/CSR/CSC sparse not accepted
			A = SliceCachingMatrix({1: np.random.rand(m, n), 2: 'not a matrix'})

		data = {i: np.random.rand(m, n) for i in xrange(4)}
		A = SliceCachingMatrix(data)
		self.assertEqual( A.row_dim, 4 * m )
		self.assertEqual( A.column_dim, n )
		self.assertEqual( len(A._SliceCachingMatrix__row_slices), 4 )
		self.assertEqual( len(A._SliceCachingMatrix__column_slices), 0 )
		self.assertEqual( len(A._SliceCachingMatrix__double_slices), 0 )

		self.assertTrue( all(i in A for i in xrange(4)) )
		self.assertTrue( all(('row', i) in A for i in xrange(4)) )
		self.assertFalse( any(('column', i) in A for i in xrange(4)) )
		self.assertFalse( any(('both', (i, i)) in A for i in xrange(4)) )

		data['labeled_by'] = 'rows'
		A = SliceCachingMatrix(data)
		self.assertEqual( A.row_dim, 4 * m )
		self.assertEqual( A.column_dim, n )
		self.assertEqual( len(A._SliceCachingMatrix__row_slices), 4 )
		self.assertEqual( len(A._SliceCachingMatrix__column_slices), 0 )
		self.assertEqual( len(A._SliceCachingMatrix__double_slices), 0 )

		self.assertTrue( all(i in A for i in xrange(4)) )
		self.assertTrue( all(('row', i) in A for i in xrange(4)) )
		self.assertFalse( any(('column', i) in A for i in xrange(4)) )
		self.assertFalse( any(('both', (i, i)) in A for i in xrange(4)) )

		data['labeled_by'] = 'columns'
		A = SliceCachingMatrix(data)
		self.assertEqual( A.row_dim, m )
		self.assertEqual( A.column_dim, 4 * n )
		self.assertEqual( len(A._SliceCachingMatrix__row_slices), 0 )
		self.assertEqual( len(A._SliceCachingMatrix__column_slices), 4 )
		self.assertEqual( len(A._SliceCachingMatrix__double_slices), 0 )

		self.assertFalse( any(i in A for i in xrange(4)) )
		self.assertTrue( all(('column', i) in A for i in xrange(4)) )
		self.assertFalse( any(('row', i) in A for i in xrange(4)) )
		self.assertFalse( any(('both', (i, i)) in A for i in xrange(4)) )

		with self.assertRaises(ValueError):
			data['labeled_by'] = 'invalid specification'
			A = SliceCachingMatrix(data)

	def test_sc_mat_row_slice(self):
		m, n = 30, 40
		SCM = SliceCachingMatrix(np.random.rand(2, 2))
		A_list = [np.random.rand(m, n), sp.rand(m, n).tocsr(),
				  sp.rand(m, n).tocsc()]
		indices = [1, 5, 8, 15, 20, 22]
		for A in A_list:
			with self.assertRaises(ValueError):
				SCM._SliceCachingMatrix__row_slice_generic(A, None)

			A_sub = SCM._SliceCachingMatrix__row_slice_generic(A, indices)
			A_sub_check = A[indices, :]
			if isinstance(A, np.ndarray):
				self.assert_vector_equal(A_sub, A_sub_check)
			else:
				self.assertEqual( (A_sub - A_sub_check).nnz, 0 )

			D = SliceCachingMatrix(A)
			D.row_slice(0, indices)
			self.assertIn( 0, D )
			self.assertIn( ('row', 0), D )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal(D.row_slice(0, None), A_sub_check)
			else:
				self.assertEqual( (D.row_slice(0, None) - A_sub_check).nnz, 0 )

	def test_sc_mat_column_slice(self):
		m, n = 30, 40
		SCM = SliceCachingMatrix(np.random.rand(2, 2))
		A_list = [np.random.rand(m, n), sp.rand(m, n).tocsr(),
				  sp.rand(m, n).tocsc()]
		indices = [1, 5, 8, 15, 20, 22, 33, 35, 37]
		for A in A_list:
			with self.assertRaises(ValueError):
				SCM._SliceCachingMatrix__column_slice_generic(A, None)

			A_sub = SCM._SliceCachingMatrix__column_slice_generic(A, indices)
			A_sub_check = A[:, indices]
			if isinstance(A, np.ndarray):
				self.assert_vector_equal(A_sub, A_sub_check)
			else:
				self.assertEqual( (A_sub - A_sub_check).nnz, 0 )

			D = SliceCachingMatrix(A)
			D.column_slice(0, indices)
			self.assertIn( ('column', 0), D )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal(D.column_slice(0, None), A_sub_check)
			else:
				self.assertEqual(
						(D.column_slice(0, None) - A_sub_check).nnz, 0 )
	def test_sc_mat_slice(self):
		m, n = 30, 40
		v_indices = [1, 5, 8, 15, 20, 22]
		b_indices = [1, 5, 8, 15, 20, 22, 33, 35, 37]

		A_list = [np.random.rand(m, n), sp.rand(m, n).tocsr(),
				  sp.rand(m, n).tocsc()]
		for A in A_list:
			A_sub_v = A[v_indices, :]
			A_sub_b = A[:, b_indices]
			A_sub_vb = A[v_indices, :][:, b_indices]

			D = SliceCachingMatrix(A)

			with self.assertRaises(ValueError):
				D.slice()

			with self.assertRaises(ValueError):
				D.slice(row_label=0)

			D.slice(row_label=0, row_indices=v_indices)
			self.assertIn( 0, D )
			self.assertIn( ('row', 0), D )
			self.assertNotIn( ('column', 0), D )
			self.assertNotIn( ('both', (0, 0)), D )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal( D.slice(row_label=0), A_sub_v )
			else:
				self.assertEqual( (D.slice(row_label=0) - A_sub_v).nnz, 0 )

			with self.assertRaises(ValueError):
				D.slice(column_label=0)

			D.slice(column_label=0, column_indices=b_indices)
			self.assertIn( 0, D )
			self.assertIn( ('row', 0), D )
			self.assertIn( ('column', 0), D )
			self.assertNotIn( ('both', (0, 0)), D )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal( D.slice(column_label=0), A_sub_b )
			else:
				self.assertEqual( (D.slice(column_label=0) - A_sub_b).nnz, 0 )

			E = SliceCachingMatrix(A)
			# when slicing in both dimensions, make sure easy path taken
			# w.r.t. compression
			E.slice(row_label=0, row_indices=v_indices, column_label=0,
					column_indices=b_indices)
			if isinstance(A, (np.ndarray, sp.csr_matrix)):
				self.assertIn( ('row', 0), E )
				self.assertNotIn( ('column', 0), E )
				if isinstance(A, np.ndarray):
					self.assert_vector_equal( E.slice(row_label=0), A_sub_v )
				else:
					self.assertEqual( (E.slice(row_label=0) - A_sub_v).nnz, 0 )

			else:
				self.assertNotIn( ('row', 0), E )
				self.assertIn( ('column', 0), E )
				self.assertEqual( (E.slice(column_label=0) - A_sub_b).nnz, 0 )
			self.assertIn( ('both', (0, 0)), E )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal( E.slice(row_label=0, column_label=0),
										  A_sub_vb)
			else:
				self.assertTrue(
						(E.slice(row_label=0, column_label=0) - A_sub_vb).nnz
						== 0 )
			# if partial slice already cached, ignore easy path
			if isinstance(A, (np.ndarray, sp.csr_matrix)):
				F = SliceCachingMatrix(A)
				F.column_slice(0, b_indices)
				F.slice(row_label=0, row_indices=v_indices, column_label=0,
						column_indices=b_indices)
				self.assertNotIn( ('row', 0), F )
				self.assertIn( ('column', 0), F )
				if isinstance(A, np.ndarray):
					self.assert_vector_equal(F.slice(column_label=0), A_sub_b)
				else:
					self.assertTrue(
						(F.slice(column_label=0) != A_sub_b).nnz
						== 0 )
			else:
				F = SliceCachingMatrix(A)
				F.row_slice(0, v_indices)
				F.slice(row_label=0, row_indices=v_indices, column_label=0,
						column_indices=b_indices)
				self.assertIn( ('row', 0), F )
				self.assertNotIn( ('column', 0), F )
				self.assertEqual( (F.slice(row_label=0) - A_sub_v).nnz, 0 )

			self.assertIn( ('both', (0, 0)), F )
			if isinstance(A, np.ndarray):
				self.assert_vector_equal(
						F.slice(row_label=0, column_label=0), A_sub_vb)
			else:
				self.assertEqual(
					(F.slice(row_label=0, column_label=0) - A_sub_vb).nnz, 0 )
