from scipy.sparse import rand as sprand, isspmatrix
from numpy import ndarray, zeros

from conrad.compat import *
from conrad.physics.units import cm, mm
from conrad.physics.voxels import VoxelGrid
from conrad.physics.beams import BixelGrid
from conrad.physics.physics import *
from conrad.tests.base import *

class DoseFrameTestCase(ConradTestCase):
	def test_doseframe_init_basic(self):
		m, n = 100, 50

		# dose frame initialization needs a VOXELS x BEAMS size
		self.assert_exception(call=DoseFrame, args=(None, None, None))

		d = DoseFrame(m, n, None)
		self.assertTrue( d.voxels == m )
		self.assertTrue( d.beams == n )

		b = BeamSet(n)
		d = DoseFrame(m, b, None)
		self.assertTrue( d.voxels == m )
		self.assertTrue( d.beams == n )

		A = rand(m, n)
		d = DoseFrame(None, None, A)
		self.assertTrue( d.voxels == m )
		self.assertTrue( d.beams == n )
		# size mismatches
		self.assert_exception( call=DoseFrame, args=(None, n + 1, A) )
		self.assert_exception( call=DoseFrame, args=(m + 1, None, A) )
		self.assert_exception( call=DoseFrame, args=(m, n + 1, A) )
		self.assert_exception( call=DoseFrame, args=(m + 1, n, A) )

		A = sprand(m, n, 0.2, 'csr')
		self.assertTrue( d.voxels == m )
		self.assertTrue( d.beams == n )

		A = sprand(m, n, 0.2, 'csc')
		self.assertTrue( d.voxels == m )
		self.assertTrue( d.beams == n )

		A = sprand(m, n, 0.2) # COO sparse storage, not supported
		self.assert_exception( call=DoseFrame, args=(None, None, A) )

	def test_doseframe_properties(self):
		m, n = 100, 50
		d = DoseFrame(m, n, None)

		# frame dimensions immutable once set
		self.assert_exception( call=d.voxels, args=(m) )
		self.assert_exception( call=d.beams, args=(m) )

		d.data = rand(m, n)
		self.assertTrue( isinstance(d.data, ndarray) )

		# matrix with wrong size fails
		self.assert_exception( call=d.data, args=(rand(m + 1, n)) )

		d.data = sprand(m, n, 0.2, 'csr')
		self.assertTrue( isspmatrix(d.data) )
		d.data = sprand(m, n, 0.2, 'csc')

		# coo matrix fails
		self.assert_exception( call=d.data, args=(sprand(m, n, 0.2)) )

		# voxel labels
		vl = (10 * rand(m)).astype(int)
		d.voxel_labels = vl
		self.assertTrue( sum(vl - d.voxel_labels) == 0 )

		vl_missized = (10 * rand(m + 1)).astype(int)
		self.assert_exception( call=d.voxel_labels, args=(vl_missized) )

		# beam labels
		bl = (4 * rand(n)).astype(int)
		d.beam_labels = bl
		self.assertTrue( sum(bl - d.beam_labels) == 0 )

		bl_missized = (10 * rand(n + 1)).astype(int)
		self.assert_exception( call=d.beam_labels, args=(bl_missized) )

		# voxel weights
		self.assertTrue( isinstance(d.voxel_weights, ndarray) )
		self.assertTrue( d.voxel_weights.size == m )
		self.assertTrue( sum(d.voxel_weights != 1) == 0 )
		vw = (5 * rand(m)).astype(int).astype(float)
		d.voxel_weights = vw
		self.assert_vector_equal( vw, d.voxel_weights )

		vw_missized = (5 * rand(m + 1)).astype(int).astype(float)
		self.assert_exception( call=d.voxel_weights, args=(vw_missized) )

		# beam weights
		self.assertTrue( isinstance(d.beam_weights, ndarray) )
		self.assertTrue( d.beam_weights.size == n )
		self.assertTrue( sum(d.beam_weights != 1) == 0 )
		bw = (5 * rand(n)).astype(int).astype(float)
		d.beam_weights = bw
		self.assert_vector_equal( bw, d.beam_weights )

		bw_missized = (5 * rand(n + 1)).astype(int).astype(float)
		self.assert_exception( call=d.beam_weights, args=(bw_missized) )

	def test_doseframe_init_options(self):
		m, n = 100, 50
		A = rand(m, n)
		vl = (10 * rand(m)).astype(int)
		vw = (5 * rand(m)).astype(int).astype(float)
		bl = (3 * rand(n)).astype(int)
		bw = (5 * rand(n)).astype(int).astype(float)

		d = DoseFrame(m, n, A, voxel_labels=vl, beam_labels=bl,
					  voxel_weights=vw, beam_weights=bw)

		self.assert_vector_equal( A, d.data )
		self.assert_vector_equal( vl, d.voxel_labels )
		self.assert_vector_equal( bl, d.beam_labels )
		self.assert_vector_equal( vw, d.voxel_weights )
		self.assert_vector_equal( bw, d.beam_weights )

	def test_indices_by_label(self):
		maxlabel = 10
		x = (maxlabel * rand(100)).astype(int)

		for i in xrange(maxlabel):
			if sum(x == i) > 0:
				label = i
				break

		n_labeled = sum(x == label)
		indices = (x == label).argsort()[-n_labeled:]

		d = DoseFrame(50, 50, None)
		indices_calculated = d.indices_by_label(x, label, 'test')
		indices.sort()
		indices_calculated.sort()
		self.assert_vector_equal( indices, indices_calculated )

		self.assert_exception( d.indices_by_label, args=(None, label, 'test') )

		self.assert_exception(
					d.indices_by_label, args=(None, maxlabel + 2, 'test') )

	def test_lookup_by_label(self):
		m, n = 100, 50
		vl = (10 * rand(m)).astype(int)
		bl = (3 * rand(n)).astype(int)

		v_label = 3
		b_label = 1

		for i in xrange(10):
			if sum(vl == i) > 0:
				v_label = i
				break

		for i in xrange(3):
			if sum(bl == i) > 0:
				b_label = i
				break

		v_idx = (vl == v_label).argsort()[-sum(vl == v_label):]
		b_idx = (bl == b_label).argsort()[-sum(bl == b_label):]

		d = DoseFrame(None, None, None, voxel_labels=vl, beam_labels=bl)
		v_idx_lookup = d.voxel_lookup_by_label(v_label)
		b_idx_lookup = d.beam_lookup_by_label(b_label)

		v_idx.sort()
		b_idx.sort()
		v_idx_lookup.sort()
		b_idx_lookup.sort()

		self.assert_vector_equal( v_idx, v_idx_lookup )
		self.assert_vector_equal( b_idx, b_idx_lookup )

class PhysicsTestCase(ConradTestCase):
	def test_physics_init(self):
		self.assert_exception( call=Physics, args=(None) )

		m, n = 100, 50
		A = rand(m, n)

		p = Physics(m, n)
		self.assertTrue( 0 in p._Physics__frames )
		self.assertTrue( 'full' in p._Physics__frames )
		self.assertFalse( 'geometric' in p._Physics__frames )
		self.assertTrue( p.dose_matrix is None )
		self.assertTrue( p.dose_grid is None )
		self.assertTrue( p.voxel_labels is None )
		self.assertTrue( p._Physics__beams.count == n )
		self.assertFalse( p.plannable )

		p = Physics(dose_matrix=A)
		self.assert_vector_equal( p.dose_matrix, A )
		self.assertTrue( p.dose_grid is None )
		self.assertTrue( p.voxel_labels is None )
		self.assertTrue( p._Physics__beams.count == n )
		self.assertFalse( p.plannable )

		b = BeamSet(n)
		p = Physics(m, b)
		self.assertTrue( p.dose_matrix is None )
		self.assertTrue( p.dose_grid is None )
		self.assertTrue( p.voxel_labels is None )
		self.assertTrue( p._Physics__beams.count == n )
		self.assertFalse( p.plannable )

		vl = (10 * rand(m)).astype(int)
		p = Physics(beams=n, voxel_labels=vl)
		self.assertTrue( p.dose_matrix is None )
		self.assertTrue( p.dose_grid is None )
		self.assert_vector_equal( p.voxel_labels, vl )
		self.assertTrue( p._Physics__beams.count == n )
		self.assertFalse( p.plannable )

		vg = VoxelGrid(5, 5, 4)
		p = Physics(beams=n, dose_grid=vg)
		self.assertTrue( p.dose_matrix is None )
		self.assertFalse( p.dose_grid is None )
		self.assertTrue( p.voxel_labels is None )
		self.assertTrue( p._Physics__beams.count == n )
		self.assertFalse( p.plannable )

		self.assertTrue( vg.voxels == m )
		self.assertTrue( 'geometric' in p._Physics__frames )

		vw = rand(m)
		bw = rand(n)
		bl = (10 * rand(n)).astype(int)
		p = Physics(m, n, dose_matrix=A, dose_grid=vg, voxel_labels=vl,
					voxel_weights=vw, beam_weights=bw, beam_labels=bl)
		self.assert_vector_equal( p.dose_matrix, A )
		self.assertFalse( p.dose_grid is None )
		self.assert_vector_equal( p.voxel_labels, vl )
		self.assertTrue( p._Physics__beams.count == n )
		self.assert_vector_equal( p.frame.beam_labels, bl )
		self.assert_vector_equal( p.frame.voxel_weights, vw )
		self.assert_vector_equal( p.frame.beam_weights, bw )
		self.assertTrue( p.plannable )

		# test copy constructor
		p2 = Physics(p)
		self.assert_vector_equal( p.dose_matrix, A )
		self.assertFalse( p.dose_grid is None )
		self.assert_vector_equal( p.voxel_labels, vl )
		self.assertTrue( p._Physics__beams.count == n )
		self.assert_vector_equal( p.frame.beam_labels, bl )
		self.assert_vector_equal( p.frame.voxel_weights, vw )
		self.assert_vector_equal( p.frame.beam_weights, bw )
		self.assertTrue( p.plannable )

	def test_physics_planning_requirements(self):
		m, n = 100, 50
		A = rand(m, n)
		vl = (10 * rand(m)).astype(int)

		p = Physics(m, n)
		self.assertFalse(p.plannable)
		p.dose_matrix = A
		self.assertFalse(p.plannable)
		p.voxel_labels = vl
		self.assertTrue(p.plannable)

		p = Physics(dose_matrix=A, voxel_labels=vl)
		self.assertTrue(p.plannable)

	def test_physics_frames(self):
		pass

	def test_data_retrieval(self):
		LABEL = 0

		m, n = 100, 50
		A = rand(m, n)
		voxel_labels = (2 * rand(m)).astype(int)
		voxel_weights = rand(m)

		beam_labels = (2 * rand(n)).astype(int)
		beam_weights = rand(n)

		m0 = sum(voxel_labels == LABEL)
		A0_rows = zeros((m0, n))
		vw0 = zeros(m0)
		ptr = 0
		for i, w in enumerate(voxel_labels):
			if w == LABEL:
				A0_rows[ptr, :] = A[i, :]
				vw0[ptr] = voxel_weights[i]
				ptr += 1

		n0 = sum(beam_labels == LABEL)
		A0_cols = zeros((m, n0))
		A0 = zeros((m0, n0))
		bw0 = zeros(n0)
		ptr = 0
		for j, w in enumerate(beam_labels):
			if w == LABEL:
				A0_cols[:, ptr] = A[:, j]
				A0[:, ptr] = A0_rows[:, j]
				bw0[ptr] = beam_weights[j]
				ptr += 1

		p = Physics(dose_matrix=A, voxel_labels=voxel_labels,
					voxel_weights=voxel_weights, beam_labels=beam_labels,
					beam_weights=beam_weights)
		vw_retrieved = p.voxel_weights_by_label(LABEL)
		self.assert_vector_equal( vw0, vw_retrieved )

		bw_retrieved = p.beam_weights_by_label(LABEL)
		self.assert_vector_equal( bw0, bw_retrieved )


		A0_row_retrieved_default_arg = p.dose_matrix_by_label(LABEL)
		self.assert_vector_equal( A0_rows, A0_row_retrieved_default_arg )

		A0_row_retrieved = p.dose_matrix_by_label(voxel_label=LABEL)
		self.assert_vector_equal( A0_rows, A0_row_retrieved )

		A0_col_retrieved = p.dose_matrix_by_label(beam_label=LABEL)
		self.assert_vector_equal( A0_cols, A0_col_retrieved )

		A0_retrieved = p.dose_matrix_by_label(
				voxel_label=LABEL, beam_label=LABEL)
		self.assert_vector_equal( A0, A0_retrieved )