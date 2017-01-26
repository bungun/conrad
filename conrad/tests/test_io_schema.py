"""
Unit tests for :mod:`conrad.io.schema`.
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

from conrad.io.schema import *
from conrad.tests.base import *

class ConradDatabaseEntryTestCase(ConradTestCase):
	def test_cdbe_init(self):
		cdbe = ConradDatabaseEntry()
		self.assertFalse( cdbe.complete )
		self.assertTrue( cdbe.nested_dictionary is NotImplemented )
		self.assertTrue( cdbe.flat_dictionary is NotImplemented )

class ConradDBSchemaUtilitesTestCase(ConradTestCase):
	def test_cdb_util_methods(self):
		db_types = CONRAD_DB_ENTRY_PREFIXES.keys()
		db_strings = CONRAD_DB_ENTRY_PREFIXES.values()
		str_type_pairs = zip(db_strings, db_types)
		for s, t in str_type_pairs:
			self.assertTrue(
					cdb_util.is_database_pointer(s + 'qfbkefklqjbfwl', t) )
			self.assertFalse(
					cdb_util.is_database_pointer('qfbkefklqjbfwl' + s, t) )
			instance = t()
			self.assertTrue(
					cdb_util.isinstance_or_db_pointer(
						instance, t))
			self.assertTrue(
					cdb_util.isinstance_or_db_pointer(
						s + 'asldksjdasjl', t))
			self.assertFalse(
					cdb_util.isinstance_or_db_pointer(
						'asldksjdasjl' + s, t))

		test_dict = {'a': 1, 'b': 2, 'c': ConradDatabaseEntry()}
		class TestEntry(ConradDatabaseEntry):
			nested_dictionary = {'somefield': 3}

		self.assertTrue( cdb_util.expand_if_db_entry(test_dict['a']) == 1 )
		self.assertTrue( cdb_util.expand_if_db_entry(test_dict['b']) == 2 )
		self.assertTrue(
				cdb_util.expand_if_db_entry(test_dict['c']) is NotImplemented )

		test_dict['d'] = TestEntry()
		self.assertTrue(
				cdb_util.expand_if_db_entry(test_dict['d'], 'somefield') == 3 )


		entry_list = [1, ConradDatabaseEntry(), TestEntry()]
		expanded_list = cdb_util.expand_list_if_db_entries(
				entry_list, field='somefield')
		self.assertTrue(expanded_list[0] == entry_list[0])
		self.assertTrue(expanded_list[1] is NotImplemented)
		self.assertTrue(expanded_list[2] == 3)

		self.assertTrue( cdb_util.check_flat(['first', 'second', 'third']) )
		self.assertFalse(
				cdb_util.check_flat(['first', 'second', TestEntry()]) )

class DataFragmentEntryTestCase(ConradTestCase):
	def test_data_fragment_entry(self):
		dfe = DataFragmentEntry()
		self.assertFalse( dfe._DataFragmentEntry__check_npyz_file_key(
				'test.npx', None) )
		self.assertTrue( dfe._DataFragmentEntry__check_npyz_file_key(
				'test.npy', None) )
		self.assertTrue( dfe._DataFragmentEntry__check_npyz_file_key(
				'test.npz', 'key') )
		self.assertFalse( dfe._DataFragmentEntry__check_npyz_file_key(
				'test.npz', None) )

		self.assertTrue( dfe.type is None )
		self.assertTrue( dfe.nested_dictionary is NotImplemented )
		self.assertTrue( dfe.flat_dictionary is NotImplemented )
		self.assertTrue( isinstance(dfe.fragment_flat_dictionary, dict) )
		self.assertTrue( dfe.ingest_dictionary(**{}) is NotImplemented )

class DataDictionaryEntryTestCase(ConradTestCase):
	def test_data_dictionary_entry(self):
		dde = DataDictionaryEntry()
		self.assertTrue( isinstance(dde, DataFragmentEntry) )
		self.assertTrue( dde.type == 'dictionary' )
		self.assertTrue( isinstance(dde.entries, dict) )
		self.assertTrue( len(dde.entries) == 0 )
		self.assertFalse( dde.complete )
		dde.entries = {'1': 2}
		self.assertTrue( dde.entries['1'] == 2 )
		self.assertTrue( dde.complete )

		dde.entries = '{"3": 4}'
		self.assertTrue( dde.entries['3'] == 4 )

		dde.ingest_dictionary(entries={'5': 6})
		self.assertTrue( dde.entries['5'] == 6 )

		dde2 = DataDictionaryEntry(**dde.nested_dictionary)
		dde3 = DataDictionaryEntry(**dde.flat_dictionary)

class DenseArrayEntryTestCase(ConradTestCase):
	def test_dense_array_entry(self):
		dae = DenseArrayEntry()
		self.assertTrue( isinstance(dae, DataFragmentEntry) )
		self.assertTrue( dae.type is None )
		self.assertTrue( dae.data_file is None )
		self.assertTrue( dae.data_key is None )

		self.assertFalse( dae._DenseArrayEntry__complete )
		dae.data_file = 'test.npy'
		self.assertTrue( dae._DenseArrayEntry__complete )
		dae.data_file = 'test.npz'
		self.assertFalse( dae._DenseArrayEntry__complete )
		dae.data_key = 'key'
		self.assertTrue( dae._DenseArrayEntry__complete )

class VectorEntryTestCase(ConradTestCase):
	def test_vector_entry(self):
		ve = VectorEntry()
		self.assertTrue( isinstance(ve, DenseArrayEntry) )

		ve2 = VectorEntry(**ve.nested_dictionary)
		ve3 = VectorEntry(**ve.flat_dictionary)

class DenseMatrixEntryTestCase(ConradTestCase):
	def test_dense_matrix_entry(self):
		dme = DenseMatrixEntry()
		self.assertTrue( isinstance(dme, DenseArrayEntry) )
		self.assertTrue( dme.layout_rowmajor is None )

		dme.data_file = 'test.npy'
		self.assertFalse( dme.complete )
		dme.layout_rowmajor = False
		self.assertTrue( dme.complete )

		dme2 = DenseMatrixEntry(**dme.nested_dictionary)
		dme3 = DenseMatrixEntry(**dme.flat_dictionary)

class SparseMatrixEntryTestCase(ConradTestCase):
	def test_sparse_matrix_entry(self):
		sme = SparseMatrixEntry()
		self.assertTrue( isinstance(sme, DataFragmentEntry) )
		self.assertTrue( sme.layout_CSR is None )
		self.assertTrue( sme.layout_fortran_indexing is None )
		self.assertTrue( sme.shape is None )

		self.assertFalse( sme.complete )
		sme.layout_CSR = True
		self.assertFalse( sme.complete )
		sme.layout_fortran_indexing = False
		self.assertFalse( sme.complete )
		sme.shape = (100, 20)
		self.assertFalse( sme.complete )
		sme.data_pointers_file = 'ptrs.npy'
		self.assertFalse( sme.complete )
		sme.data_indices_file = 'inds.npy'
		self.assertFalse( sme.complete )
		sme.data_values_file = 'vals.npy'
		self.assertTrue( sme.complete )

		sme2 = SparseMatrixEntry(**sme.nested_dictionary)
		sme3 = SparseMatrixEntry(**sme.flat_dictionary)

class DoseFrameEntryTestCase(ConradTestCase):
	def test_dose_frame_entry(self):
		dfe = DoseFrameEntry()
		self.assertTrue( isinstance(dfe, ConradDatabaseEntry) )
		self.assertTrue( dfe.name is None )
		self.assertTrue( dfe.n_voxels is None )
		self.assertTrue( dfe.n_beams is None )
		self.assertTrue( dfe.dose_matrix is None )
		self.assertTrue( dfe.voxel_labels is None )
		self.assertTrue( dfe.voxel_weights is None )
		self.assertTrue( dfe.beam_labels is None )
		self.assertTrue( dfe.beam_weights is None )

		self.assertFalse( dfe.complete )
		dfe.name = 'frame name'
		self.assertFalse( dfe.complete )
		dfe.n_voxels = 12
		self.assertFalse( dfe.complete )
		dfe.n_beams = 20
		self.assertFalse( dfe.complete )
		dfe.dose_matrix = 'data_fragment.<INT>'
		self.assertTrue( dfe.complete )

		dfe2 = DoseFrameEntry(**dfe.nested_dictionary)
		dfe3 = DoseFrameEntry(**dfe.flat_dictionary)

		dfe.voxel_weights = VectorEntry()
		dfe4 = DoseFrameEntry(**dfe.nested_dictionary)
		with self.assertRaises(ValueError):
			dfe5 = DoseFrameEntry(**dfe.flat_dictionary)

		dfe.voxel_weights = 'data_fragment.<INT>'
		dfe5 = DoseFrameEntry(**dfe.flat_dictionary)

class DoseFrameMappingEntryTestCase(ConradTestCase):
	def test_frame_mapping_entry(self):
		dfie = DoseFrameMappingEntry()
		self.assertTrue( isinstance(dfie, ConradDatabaseEntry) )

		# TODO: remainder of unit test

class PhysicsEntryTestCase(ConradTestCase):
	def test_physics_entry(self):
		pe = PhysicsEntry()
		self.assertTrue( isinstance(pe, ConradDatabaseEntry) )
		self.assertTrue( isinstance(pe.voxel_grid, dict) )
		self.assertTrue( pe.voxel_bitmask is None )
		self.assertTrue( isinstance(pe.beam_set, dict) )
		self.assertTrue( isinstance(pe.beam_set['control_points'], dict) )
		self.assertTrue( isinstance(pe.frames, list) )
		self.assertTrue( isinstance(pe.frame_mappings, list) )

		self.assertFalse( pe.complete )

		with self.assertRaises(ValueError):
			pe.add_frames('vector.<INT>')
		pe.add_frames('frame.<INT>')
		pe.add_frames(DoseFrameEntry())
		pe.add_frames('frame.<INT>', DoseFrameEntry())

		with self.assertRaises(ValueError):
			pe.add_frame_mappings('vector.<INT>')
		pe.add_frame_mappings('frame_mapping.<INT>')
		pe.add_frame_mappings(DoseFrameMappingEntry())
		pe.add_frame_mappings(
				'frame_mapping.<INT>', DoseFrameMappingEntry())


		pe2 = PhysicsEntry(**pe.nested_dictionary)
		with self.assertRaises(ValueError):
			pe3 = PhysicsEntry(**pe.flat_dictionary)

		pe.frames = ['frame.<INT>', 'frame.<INT>']
		pe.frame_mappings = ['frame_mapping.<INT>', 'frame_mapping.<INT>']

		pe3 = PhysicsEntry(**pe.flat_dictionary)

class SolutionEntryTestCase(ConradTestCase):
	def test_solution_entry(self):
		se = SolutionEntry()
		self.assertTrue( isinstance(se, ConradDatabaseEntry) )
		self.assertTrue( se.name is None )
		self.assertTrue( se.frame is None )
		self.assertTrue( se.x is None )
		self.assertTrue( se.y is None )
		self.assertTrue( se.x_dual is None )
		self.assertTrue( se.y_dual is None )

		self.assertFalse( se.complete )
		se.name = 'sol'
		self.assertFalse( se.complete )
		se.frame = 'frame'
		self.assertFalse( se.complete )
		se.x = 'data_fragment.<INT>'
		self.assertTrue( se.complete )

		se2 = SolutionEntry()
		self.assertFalse( se2.complete )
		se2.name = 'sol'
		self.assertFalse( se2.complete )
		se2.frame = 'frame'
		self.assertFalse( se2.complete )
		se2.y = 'data_fragment.<INT>'
		self.assertTrue( se2.complete )

		se2.y = VectorEntry()
		self.assertTrue( se2.complete )

		se3 = SolutionEntry(**se.nested_dictionary)
		with self.assertRaises(ValueError):
			se4 = SolutionEntry(**se2.flat_dictionary)

class HistoryEntryTestCase(ConradTestCase):
	def test_history_entry(self):
		he = HistoryEntry()
		self.assertTrue( isinstance(he, ConradDatabaseEntry) )
		self.assertTrue( isinstance(he.solutions, list) )
		self.assertTrue( len(he.solutions) == 0 )

		self.assertFalse( he.complete )
		he.add_solutions(SolutionEntry())
		self.assertTrue( he.complete )

		he.add_solutions('solution.<INT>')
		he.add_solutions(SolutionEntry(), 'solution.<INT>')

		he2 = HistoryEntry(**he.nested_dictionary)
		with self.assertRaises(ValueError):
			he3 = HistoryEntry(**he.flat_dictionary)
		he.solutions = ['solution.<INT>'] * 3
		he3 = HistoryEntry(**he.flat_dictionary)

class SolverCacheEntryTestCase(ConradTestCase):
	def test_solver_cache_entry(self):
		sce = SolverCacheEntry()
		self.assertTrue( isinstance(sce, ConradDatabaseEntry) )
		self.assertTrue( sce.name is None )
		self.assertTrue( sce.frame is None )
		self.assertTrue( sce.solver is None )
		self.assertTrue( sce.left_preconditioner is None )
		self.assertTrue( sce.matrix is None )
		self.assertTrue( sce.right_preconditioner is None )
		self.assertTrue( sce.projector_type is None )
		self.assertTrue( sce.projector_matrix is None )

		self.assertFalse( sce.complete )
		sce.name = 'my cache'
		self.assertFalse( sce.complete )
		sce.frame = 'default'
		self.assertFalse( sce.complete )
		sce.solver = 'POGS'
		self.assertFalse( sce.complete )
		sce.left_preconditioner = 'data_fragment.<INT>'
		self.assertFalse( sce.complete )
		sce.matrix = 'data_fragment.<INT>'
		self.assertFalse( sce.complete )
		sce.right_preconditioner = 'data_fragment.<INT>'
		self.assertFalse( sce.complete )
		sce.projector_type = 'projector type description, e.g., indirect'
		self.assertTrue( sce.complete )

		sce.projector_matrix = 'data_fragment.<INT>'
		self.assertTrue( sce.complete )

		sce2 = SolverCacheEntry(**sce.nested_dictionary)
		sce3 = SolverCacheEntry(**sce.flat_dictionary)

		sce.projector_matrix = DenseMatrixEntry()
		sce4 = SolverCacheEntry(**sce.nested_dictionary)
		with self.assertRaises(ValueError):
			sce5 = SolverCacheEntry(**sce.flat_dictionary)

class StructureEntryTestCase(ConradTestCase):
	def test_structure_entry(self):
		se = StructureEntry()
		self.assertTrue( isinstance(se, ConradDatabaseEntry) )
		self.assertTrue( se.label is None )
		self.assertTrue( se.name is None )
		self.assertTrue( se.target is None )
		self.assertTrue( se.rx is None )
		self.assertTrue( se.size is None )
		self.assertTrue( isinstance(se.constraints, list) )
		self.assertTrue( len(se.constraints) == 0 )
		self.assertTrue( isinstance(se.objective, dict) )

		self.assertFalse( se.complete )
		se.label = 0
		self.assertFalse( se.complete )
		se.name = 'structure'
		self.assertFalse( se.complete )
		se.target = False
		self.assertTrue( se.complete )

		se.target = True
		self.assertFalse( se.complete )
		se.rx = 34
		self.assertTrue( se.complete )

		se2 = StructureEntry(**se.nested_dictionary)
		se3 = StructureEntry(**se.flat_dictionary)

class AnatomyEntryTestCase(ConradTestCase):
	def test_anatomy_entry(self):
		ae = AnatomyEntry()
		self.assertTrue( isinstance(ae, ConradDatabaseEntry) )
		self.assertTrue( isinstance(ae.structures, list) )
		self.assertTrue( len(ae.structures) == 0 )

		self.assertFalse( ae.complete )
		ae.add_structures(StructureEntry())
		self.assertTrue( ae.complete )

		ae.add_structures('structure.<INT>')
		ae.add_structures(StructureEntry(), 'structure.<INT>')

		ae2 = AnatomyEntry(**ae.nested_dictionary)
		with self.assertRaises(ValueError):
			ae3 = AnatomyEntry(**ae.flat_dictionary)
		ae.structures = ['structure.<INT>'] * 3
		ae3 = AnatomyEntry(**ae.flat_dictionary)

class CaseEntryTestCase(ConradTestCase):
	def test_case_entry(self):
		ce = CaseEntry()
		self.assertTrue( isinstance(ce, ConradDatabaseEntry) )
		self.assertTrue( ce.name is None )
		self.assertTrue( ce.prescription is None )
		self.assertTrue( ce.anatomy is None )
		self.assertTrue( ce.physics is None )
		self.assertTrue( ce.history is None )
		self.assertTrue( isinstance(ce.solver_caches, list) )
		self.assertTrue( len(ce.solver_caches) == 0 )

		self.assertFalse( ce.complete )
		ce.name = 'case name'
		self.assertFalse( ce.complete )
		ce.physics = PhysicsEntry()
		self.assertFalse( ce.complete )
		ce.anatomy = AnatomyEntry()
		self.assertTrue( ce.complete )

		ce2 = CaseEntry(**ce.nested_dictionary)
		with self.assertRaises(ValueError):
			ce3 = CaseEntry(**ce.flat_dictionary)

		ce.physics = 'physics.<INT>'
		ce.anatomy = 'anatomy.<INT>'
		self.assertTrue( ce.complete )

		ce4 = CaseEntry(**ce.nested_dictionary)
		ce5 = CaseEntry(**ce.flat_dictionary)