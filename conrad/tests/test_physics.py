"""
Unit tests for :mod:`conrad.physics.physics`.
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

import scipy.sparse as sp
import numpy as np

from conrad.abstract.mapping import PermutationMapping
from conrad.physics.units import cm, mm
from conrad.physics.voxels import VoxelGrid
from conrad.physics.beams import BixelGrid
from conrad.physics.physics import *
from conrad.tests.base import *

class DoseFrameTestCase(ConradTestCase):
	def test_doseframe_init_basic(self):
		m, n = 100, 50

		d = DoseFrame()
		self.assert_nan( d.voxels )
		self.assert_nan( d.beams )

		d = DoseFrame(m, n, None)
		self.assertTrue( d.voxels == m )
		self.assertTrue( d.beams == n )

		b = BeamSet(n)
		d = DoseFrame(m, b, None)
		self.assertTrue( d.voxels == m )
		self.assertTrue( d.beams == n )

		A = np.random.rand(m, n)
		d = DoseFrame(None, None, A)
		self.assertTrue( d.voxels == m )
		self.assertTrue( d.beams == n )
		# size mismatches
		with self.assertRaises(ValueError):
			DoseFrame(None, n + 1, A)
		with self.assertRaises(ValueError):
			DoseFrame(m + 1, None, A)
		with self.assertRaises(ValueError):
			DoseFrame(m, n + 1, A)
		with self.assertRaises(ValueError):
			DoseFrame(m + 1, n, A)

		A = sp.rand(m, n, 0.2, 'csr')
		self.assertTrue( d.voxels == m )
		self.assertTrue( d.beams == n )

		A = sp.rand(m, n, 0.2, 'csc')
		self.assertTrue( d.voxels == m )
		self.assertTrue( d.beams == n )

		A = sp.rand(m, n, 0.2) # COO sparse storage, not supported
		with self.assertRaises(TypeError):
			DoseFrame(None, None, A)

	def test_doseframe_properties(self):
		m, n = 100, 50
		d = DoseFrame(m, n, None)

		# frame dimensions immutable once set
		with self.assertRaises(ValueError):
			d.voxels = m
		with self.assertRaises(ValueError):
			d.beams = m

		d.dose_matrix = np.random.rand(m, n)
		self.assertTrue( isinstance(d.dose_matrix, DoseMatrix) )

		# matrix with wrong size fails
		with self.assertRaises(ValueError):
			d.dose_matrix = np.random.rand(m + 1, n)

		d.dose_matrix = sp.rand(m, n, 0.2, 'csr')
		self.assertTrue( sp.isspmatrix(d.dose_matrix.data) )
		d.dose_matrix = sp.rand(m, n, 0.2, 'csc')
		self.assertTrue( sp.isspmatrix(d.dose_matrix.data) )

		# coo matrix fails
		with self.assertRaises(TypeError):
			d.dose_matrix = sp.rand(m, n, 0.2)

		# voxel labels
		vl = (10 * np.random.rand(m)).astype(int)
		d.voxel_labels = vl
		self.assertTrue( sum(vl - d.voxel_labels) == 0 )

		vl_missized = (10 * np.random.rand(m + 1)).astype(int)
		with self.assertRaises(ValueError):
			d.voxel_labels = vl_missized

		# beam labels
		bl = (4 * np.random.rand(n)).astype(int)
		d.beam_labels = bl
		self.assertTrue( sum(bl - d.beam_labels) == 0 )

		bl_missized = (10 * np.random.rand(n + 1)).astype(int)
		with self.assertRaises(ValueError):
			d.beam_labels = bl_missized

		# voxel weights
		self.assertTrue( isinstance(d.voxel_weights, WeightVector) )
		self.assertTrue( isinstance(d.voxel_weights.data, np.ndarray) )
		self.assertTrue( d.voxel_weights.size == m )
		self.assertTrue( sum(d.voxel_weights.data != 1) == 0 )
		vw = (5 * np.random.rand(m)).astype(int).astype(float)
		d.voxel_weights = vw
		self.assert_vector_equal( vw, d.voxel_weights.data )

		vw_missized = (5 * np.random.rand(m + 1)).astype(int).astype(float)
		with self.assertRaises(ValueError):
			d.voxel_weights = vw_missized

		# beam weights
		self.assertTrue( isinstance(d.beam_weights, WeightVector) )
		self.assertTrue( isinstance(d.beam_weights.data, np.ndarray) )
		self.assertTrue( d.beam_weights.size == n )
		self.assertTrue( sum(d.beam_weights.data != 1) == 0 )
		bw = (5 * np.random.rand(n)).astype(int).astype(float)
		d.beam_weights = bw
		self.assert_vector_equal( bw, d.beam_weights.data )

		bw_missized = (5 * np.random.rand(n + 1)).astype(int).astype(float)
		with self.assertRaises(ValueError):
			d.beam_weights = bw_missized

	def test_doseframe_init_options(self):
		m, n = 100, 50
		A = np.random.rand(m, n)
		vl = (10 * np.random.rand(m)).astype(int)
		vw = (5 * np.random.rand(m)).astype(int).astype(float)
		bl = (3 * np.random.rand(n)).astype(int)
		bw = (5 * np.random.rand(n)).astype(int).astype(float)

		d = DoseFrame(m, n, A, voxel_labels=vl, beam_labels=bl,
					  voxel_weights=vw, beam_weights=bw)

		self.assert_vector_equal( A, d.dose_matrix.data )
		self.assert_vector_equal( vl, d.voxel_labels )
		self.assert_vector_equal( bl, d.beam_labels )
		self.assert_vector_equal( vw, d.voxel_weights.data )
		self.assert_vector_equal( bw, d.beam_weights.data )

	def test_indices_by_label(self):
		maxlabel = 10
		x = (maxlabel * np.random.rand(100)).astype(int)

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

		with self.assertRaises(ValueError):
			d.indices_by_label(None, label, 'test')

		with self.assertRaises(KeyError):
			d.indices_by_label(x, maxlabel + 2, 'test')

	def test_lookup_by_label(self):
		m, n = 100, 50
		vl = (10 * np.random.rand(m)).astype(int)
		bl = (3 * np.random.rand(n)).astype(int)

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

		d = DoseFrame(voxel_labels=vl, beam_labels=bl)
		v_idx_lookup = d.voxel_lookup_by_label(v_label)
		b_idx_lookup = d.beam_lookup_by_label(b_label)

		v_idx.sort()
		b_idx.sort()
		v_idx_lookup.sort()
		b_idx_lookup.sort()

		self.assert_vector_equal( v_idx, v_idx_lookup )
		self.assert_vector_equal( b_idx, b_idx_lookup )

	def test_submatrix(self):
		m, n = 100, 50
		A = np.random.rand(m, n)
		vl = (10 * np.random.rand(m)).astype(int)
		bl = (3 * np.random.rand(n)).astype(int)

		for i in xrange(10):
			if sum(vl == i) > 0:
				v_label = i
				break

		for i in xrange(3):
			if sum(bl == i) > 0:
				b_label = i
				break

		d = DoseFrame(data=A, voxel_labels=vl, beam_labels=bl)
		v_idx = d.indices_by_label(vl, v_label, 'voxel labels')
		b_idx = d.indices_by_label(bl, b_label, 'beam labels')

		A_sub_v = A[v_idx, :]
		A_sub_b = A[:, b_idx]
		A_sub_bv = A[v_idx, :][:, b_idx]
		self.assert_vector_equal( d.submatrix(v_label), A_sub_v )
		self.assert_vector_equal( d.submatrix(voxel_label=v_label), A_sub_v )
		self.assert_vector_equal( d.submatrix(beam_label=b_label), A_sub_b )
		self.assert_vector_equal( d.submatrix(v_label, b_label), A_sub_bv )

class DoseFrameMappingTestCase(ConradTestCase):
	def test_dose_frame_mapping(self):
		dfm = DoseFrameMapping('source', 'target')
		self.assertTrue( dfm.source == 'source' )
		self.assertTrue( dfm.target == 'target' )
		self.assertTrue( dfm.voxel_map is None )
		self.assertTrue( dfm.beam_map is None )
		self.assertTrue( dfm.voxel_map_type is None )
		self.assertTrue( dfm.beam_map_type is None )

		dfm.voxel_map = DiscreteMapping([1, 1, 3, 4])
		self.assertTrue( isinstance(dfm.voxel_map, DiscreteMapping) )
		self.assertTrue( dfm.voxel_map_type == 'discrete' )

		dfm.beam_map = PermutationMapping([1, 3, 2, 4, 0])
		self.assertTrue( isinstance(dfm.beam_map, PermutationMapping) )
		self.assertTrue( isinstance(dfm.beam_map, DiscreteMapping) )
		self.assertTrue( dfm.beam_map_type == 'permutation' )

class PhysicsTestCase(ConradTestCase):
	def test_physics_init(self):
		# with self.assertRaises(TypeError):
		Physics()

		m, n = 100, 50
		A = np.random.rand(m, n)

		p = Physics(m, n)
		self.assertTrue( DEFAULT_FRAME0_NAME in p._Physics__frames )
		self.assertFalse( 'geometric' in p._Physics__frames )
		self.assertTrue( p.dose_matrix is None )
		self.assertTrue( p.dose_grid is None )
		self.assertTrue( p.voxel_labels is None )
		self.assertTrue( p._Physics__beams.count == n )
		self.assertFalse( p.plannable )

		p = Physics(dose_matrix=A)
		self.assert_vector_equal( p.dose_matrix.data, A )
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

		vl = (10 * np.random.rand(m)).astype(int)
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

		vw = np.random.rand(m)
		bw = np.random.rand(n)
		bl = (10 * np.random.rand(n)).astype(int)
		p = Physics(m, n, dose_matrix=A, dose_grid=vg, voxel_labels=vl,
					voxel_weights=vw, beam_weights=bw, beam_labels=bl)
		self.assert_vector_equal( p.dose_matrix.data, A )
		self.assertFalse( p.dose_grid is None )
		self.assert_vector_equal( p.voxel_labels, vl )
		self.assertTrue( p._Physics__beams.count == n )
		self.assert_vector_equal( p.frame.beam_labels, bl )
		self.assert_vector_equal( p.frame.voxel_weights.data, vw )
		self.assert_vector_equal( p.frame.beam_weights.data, bw )
		self.assertTrue( p.plannable )

		# test copy constructor
		p2 = Physics(p)
		self.assert_vector_equal( p.dose_matrix.data, A )
		self.assertFalse( p.dose_grid is None )
		self.assert_vector_equal( p.voxel_labels, vl )
		self.assertTrue( p._Physics__beams.count == n )
		self.assert_vector_equal( p.frame.beam_labels, bl )
		self.assert_vector_equal( p.frame.voxel_weights.data, vw )
		self.assert_vector_equal( p.frame.beam_weights.data, bw )
		self.assertTrue( p.plannable )

	def test_physics_planning_requirements(self):
		m, n = 100, 50
		A = np.random.rand(m, n)
		vl = (10 * np.random.rand(m)).astype(int)

		p = Physics(m, n)
		self.assertFalse(p.plannable)
		p.dose_matrix = A
		self.assertFalse(p.plannable)
		p.voxel_labels = vl
		self.assertTrue(p.plannable)

		p = Physics(dose_matrix=A, voxel_labels=vl)
		self.assertTrue(p.plannable)

	def test_physics_frames(self):
		m, n = 100, 50

		p = Physics(m, n)
		self.assertTrue( DEFAULT_FRAME0_NAME in p.available_frames )
		self.assertTrue( len(p.available_frames) == 1 )
		self.assertTrue( len(p.unique_frames) == 1 )
		self.assertTrue( isinstance(p.unique_frames[0], DoseFrame) )
		self.assertTrue( p.frame.name == DEFAULT_FRAME0_NAME )

		# add
		p.add_dose_frame('another frame', voxels=2*m, beams=2*n)

		# available frames
		self.assertTrue( 'another frame' in p.available_frames )
		self.assertTrue( len(p.available_frames) == 2 )

		# unique frames
		self.assertTrue( len(p.unique_frames) == 2 )
		self.assertTrue(
				all([isinstance(f, DoseFrame) for f in p.unique_frames]))
		self.assertTrue(
				p.unique_frames[1].name == 'another frame' or
				p.unique_frames[0].name == 'another frame')

		# change
		self.assertTrue( p.frame in p.unique_frames )
		p.change_dose_frame('another frame')
		self.assertTrue( p.frame in p.unique_frames )

		with self.assertRaises(KeyError):
			p.change_dose_frame('bad key')

	def test_physics_frame_mappings(self):
		p = Physics()
		# available frame mappings
		self.assertTrue( len(p.available_frame_mappings) == 0 )

		# add
		with self.assertRaises(TypeError):
			p.add_frame_mapping(DoseFrame())

		p.add_dose_frame('frame1', voxels=100, beams=50)
		p.add_frame_mapping(DoseFrameMapping('frame0', 'frame1'))

		# available
		self.assertTrue( len(p.available_frame_mappings) == 1 )

		# retrieve
		with self.assertRaises(ValueError):
			p.retrieve_frame_mapping('frame0', 'frame0')
		with self.assertRaises(ValueError):
			p.retrieve_frame_mapping('frame0', 'frame2')
		with self.assertRaises(ValueError):
			p.retrieve_frame_mapping('frame1', 'frame0')
		fm = p.retrieve_frame_mapping('frame0', 'frame1')
		self.assertTrue( isinstance(fm, DoseFrameMapping) )

	def test_data_retrieval(self):
		LABEL = 0

		m, n = 100, 50
		A = np.random.rand(m, n)
		voxel_labels = (2 * np.random.rand(m)).astype(int)
		voxel_weights = np.random.rand(m)

		beam_labels = (2 * np.random.rand(n)).astype(int)
		beam_weights = np.random.rand(n)

		m0 = sum(voxel_labels == LABEL)
		A0_rows = np.zeros((m0, n))
		vw0 = np.zeros(m0)
		ptr = 0
		for i, w in enumerate(voxel_labels):
			if w == LABEL:
				A0_rows[ptr, :] = A[i, :]
				vw0[ptr] = voxel_weights[i]
				ptr += 1

		n0 = sum(beam_labels == LABEL)
		A0_cols = np.zeros((m, n0))
		A0 = np.zeros((m0, n0))
		bw0 = np.zeros(n0)
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