"""
Unit tests for :mod:`conrad.io.accessors`.
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

import os
import re
import numpy as np
import operator as op
import scipy.sparse as sp

from conrad.abstract.mapping import DiscreteMapping
from conrad.medicine import Structure
from conrad.optimization.solvers.solver_cvxpy import SolverCVXPY
from conrad import Gy
from conrad.io.accessors.base_accessor import *
from conrad.io.accessors.anatomy_accessor import *
from conrad.io.accessors.physics_accessor import *
from conrad.io.accessors.solver_accessor import *
from conrad.io.accessors.history_accessor import *
from conrad.io.accessors.case_accessor import *
from conrad.tests.base import *
from conrad.tests.test_io_filesystem import FilesystemTestCaching

SKIP_POGS_CACHING_TESTS = os.getenv('CONRAD_SKIP_POGS_CACHING_TESTS', False)

class ConradDBAccessorTestCase(ConradTestCase):
	def test_cdba_init(self):
		cdba = ConradDBAccessor()
		self.assertIsInstance( cdba.FS, ConradFilesystemBase )
		self.assertIsInstance( cdba.DB, ConradDatabaseBase )
		self.assertEqual( len(cdba._ConradDBAccessor__subaccessors), 0 )

	def test_cdba_set(self):
		cdba = ConradDBAccessor()
		cdba.set_filesystem()
		self.assertIsInstance( cdba.FS, LocalFilesystem )

		cdba.set_database()
		self.assertIsInstance( cdba.DB, LocalPythonDatabase )

	def test_cdba_record_entry(self):
		cdba = ConradDBAccessor(filesystem=FilesystemTestCaching())

		# unrecorded basic types
		for val in [None, 1, 1.0, '1', {'1': 2, 3: 4.0, '5': '6'}]:
			res = cdba.record_entry('dir', 'name', val)
			self.assertEqual( res, val )

		# unhandled basic types
		for val in [set(), list()]:
			with self.assertRaises(TypeError):
				cdba.record_entry('dir', 'name', val)

		# recorded data types
		for val in [
				np.random.rand(30), np.random.rand(30, 20),
				sp.rand(30, 20, 0.2, format='csr'),
				{
						1: np.random.rand(30),
						2: np.random.rand(30)
				}]:
			res = cdba.record_entry('dir', 'name', val)
			self.assertTrue( cdba.DB.has_key(res) )

		# unhandled custom types
		for val in [Structure(0, 'name', False), StructureEntry()]:
			with self.assertRaises(TypeError):
				cdba.record_entry('dir', 'name', val)

	def test_cdba_pop_record(self):
		cdba = ConradDBAccessor(filesystem=FilesystemTestCaching())

		input_dict = {}

		result = cdba.pop_and_record(input_dict, 'key', 'dir')
		self.assertIsNone( result )

		for val in [2, 2.0, '2', {'some data': 'value'}]:
			# test unrecorded types
			input_dict['key'] = val
			result = cdba.pop_and_record(input_dict, 'key', 'dir')
			self.assertEqual( result, val )
			self.assertEqual( len(cdba.FS.files), 0 )

			# test alternate keys
			input_dict.pop('key')
			input_dict['alternate_key'] = val
			result = cdba.pop_and_record(
					input_dict, 'key', 'dir', alternate_keys=['alternate_key'])
			self.assertEqual( result, val )
			self.assertEqual( len(cdba.FS.files), 0 )

			# test nested keys
			input_dict.pop('alternate_key')
			input_dict['key2'] = {'subkey': val}
			result = cdba.pop_and_record(
					input_dict, 'key', 'dir',
					alternate_keys=['alternate_key', ['key2', 'subkey']])
			self.assertEqual( result, val )
			self.assertEqual( len(cdba.FS.files), 0 )

		# test rejected types
		for val in [
				list(), set(), Structure(0, 'name', False), StructureEntry()]:
			with self.assertRaises(TypeError):
				cdba.pop_and_record({'key': val}, 'key', 'dir')

		# test recorded types
		for val in [
				np.random.rand(30), np.random.rand(30, 20),
				sp.rand(30, 20, 0.2, format='csr')]:
			result = cdba.pop_and_record({'key': val}, 'key', 'dir')
			self.assertIsInstance( result, str )
			self.assertTrue( cdba.DB.has_key(result) )
			rex = r'dir/key.*.npy'
			entry_vals = cdba.DB.get(result).flat_dictionary.values()
			self.assertTrue( any([
					re.match(rex, str(s)) != None for s in entry_vals]) )

			result = cdba.pop_and_record(
					{'key': val}, 'key', 'dir', name_base='base_name')
			self.assertIsInstance( result, str )
			self.assertTrue( cdba.DB.has_key(result) )
			entry_vals = cdba.DB.get(result).flat_dictionary.values()
			rex = r'dir/base_name_key.*.npy'
			entry_vals = cdba.DB.get(result).flat_dictionary.values()
			self.assertTrue( any([
					re.match(rex, str(s)) != None for s in entry_vals]) )

	def test_cdba_load_entry(self):
		cdba = ConradDBAccessor(filesystem=FilesystemTestCaching())

		input_ = np.random.rand(30)
		db_key = cdba.record_entry('dir', 'name', input_)
		output = cdba.load_entry(db_key)
		self.assert_vector_equal( input_, output )

class StructureAccessorTestCase(ConradTestCase):
	def test_structure_accessor_save_load(self):
		sa = StructureAccessor()
		ptr = sa.save_structure(Structure(0, 'name', True, w_under=2.75))
		self.assertTrue( sa.DB.has_key(ptr) )
		s = sa.load_structure(ptr)
		self.assertIsInstance( s, Structure )
		self.assertEqual( s.label, 0 )
		self.assertEqual( s.name, 'name' )
		self.assertTrue( s.is_target )
		self.assertEqual( s.objective.weight_underdose_raw, 2.75 )

class AnatomyAccessorTestCase(ConradTestCase):
	def test_anatomy_accessor_init(self):
		aa = AnatomyAccessor()
		self.assertTrue( aa.FS is aa.structure_accessor.FS )
		self.assertTrue( aa.DB is aa.structure_accessor.DB )

	def test_anatomy_accessor_save_load(self):
		aa = AnatomyAccessor()

		anatomy_in = Anatomy()
		anatomy_in += Structure(0, 'name0', False)
		anatomy_in += Structure(1, 'name1', False)

		ptr = aa.save_anatomy(anatomy_in)
		self.assertTrue( aa.DB.has_key(ptr) )
		a = aa.load_anatomy(ptr)
		self.assertIsInstance( a, Anatomy )
		self.assertEqual( len(a.list), 2 )

		self.assertEqual( a['name0'].label, 0 )
		self.assertFalse( a['name0'].is_target )

		self.assertEqual( a[1].name, 'name1' )
		self.assertFalse( a[1].is_target )

class DoseFrameAccessorTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.mat = np.random.rand(30, 20)
		self.vw = np.random.rand(30)
		self.bw = np.random.rand(20)
		self.frame  = DoseFrame(
				data=self.mat, voxel_weights=self.vw, beam_weights=self.bw)

	def test_dose_frame_accessor_save_load(self):
		dfa = DoseFrameAccessor(filesystem=FilesystemTestCaching())

		with self.assertRaises(TypeError):
			dfa.save_frame(DoseFrameMapping('1', '2'))

		ptr = dfa.save_frame(self.frame, 'dir')
		self.assertTrue( dfa.DB.has_key(ptr) )
		df = dfa.load_frame(ptr)
		self.assertIsInstance( df, DoseFrame )
		self.assert_vector_equal( df.dose_matrix.data, self.mat )
		self.assert_vector_equal( df.voxel_weights.data, self.vw )
		self.assert_vector_equal( df.beam_weights.data, self.bw )

	def test_dose_frame_accessor_select(self):
		dfa = DoseFrameAccessor(filesystem=FilesystemTestCaching())

		frame_list = []
		frame_list.append(DoseFrameEntry(name='first'))
		frame_list.append(DoseFrameEntry(name='second'))

		with self.assertRaises(ValueError):
			dfa.select_frame_entry(frame_list, 'third')

		frame_list.append(DoseFrameEntry(name='third'))
		fe = dfa.select_frame_entry(frame_list, 'third')
		self.assertIsInstance( fe, DoseFrameEntry )

		frame_list.append(dfa.save_frame(self.frame, 'dir'))
		fe = dfa.select_frame_entry(frame_list, self.frame.name)
		self.assertIsInstance( fe, DoseFrameEntry )

class FrameMappingAccessorTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.fmap = [1, 2, 0, 1]
		self.frame_mapping = DoseFrameMapping(
				's', 't', DiscreteMapping(self.fmap))

	def test_dose_frame_mapping_accessor_save_load(self):
		fma = FrameMappingAccessor(filesystem=FilesystemTestCaching())

		with self.assertRaises(TypeError):
			ptr = fma.save_frame_mapping(DoseFrame(), 'dir')

		ptr = fma.save_frame_mapping(self.frame_mapping, 'dir')
		self.assertTrue( fma.DB.has_key(ptr) )
		dfm = fma.load_frame_mapping(ptr)
		self.assertIsInstance( dfm, DoseFrameMapping )
		self.assert_vector_equal( dfm.voxel_map.vec, self.fmap )

	def test_dose_frame_mapping_accessor_select(self):
		fma = FrameMappingAccessor(filesystem=FilesystemTestCaching())

		frame_mapping_list = []
		frame_mapping_list.append(DoseFrameMappingEntry(
				source_frame='0', target_frame='1'))
		frame_mapping_list.append(DoseFrameMappingEntry(
				source_frame='1', target_frame='2'))

		with self.assertRaises(ValueError):
			fma.select_frame_mapping_entry(frame_mapping_list, '0', '0')
		with self.assertRaises(ValueError):
			fma.select_frame_mapping_entry(frame_mapping_list, '0', '2')
		fme = fma.select_frame_mapping_entry(frame_mapping_list, '0', '1')
		self.assertIsInstance( fme, DoseFrameMappingEntry )

class PhysicsAccessorTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.mat = np.random.rand(30, 20)
		self.vw = np.random.rand(30)
		self.bw = np.random.rand(20)
		self.frame0 = DoseFrame(
				data=self.mat, voxel_weights=self.vw, beam_weights=self.bw)
		self.frame0.name = '0'
		self.frame1 = DoseFrame(
				data=self.mat, voxel_weights=self.vw, beam_weights=self.bw)
		self.frame1.name = '1'
		self.fmap = [1, 2, 0, 1]
		self.frame_mapping = DoseFrameMapping(
				'0', '1', DiscreteMapping(self.fmap))
		self.physics = Physics(dose_frame=self.frame0)
		self.physics.add_dose_frame('1', dose_frame=self.frame1)
		self.physics.add_frame_mapping(self.frame_mapping)

	def test_physics_accessor_init(self):
		pa = PhysicsAccessor(filesystem=FilesystemTestCaching())
		self.assertIs( pa.FS, pa.frame_accessor.FS )
		self.assertIs( pa.DB, pa.frame_accessor.DB )
		self.assertIs( pa.FS, pa.frame_mapping_accessor.FS )
		self.assertIs( pa.DB, pa.frame_mapping_accessor.DB )
		self.assertEqual( len(pa.available_frames), 0 )
		self.assertEqual( len(pa.available_frame_mappings), 0 )

	def test_physics_accessor_save_load(self):
		pa = PhysicsAccessor(filesystem=FilesystemTestCaching())

		with self.assertRaises(TypeError):
			pa.save_physics('bad_type', 'dir')

		ptr = pa.save_physics(self.physics, 'dir')
		self.assertTrue( pa.DB.has_key(ptr) )

		# load physics
		p = pa.load_physics(ptr)
		self.assertIsInstance( p, Physics )

		self.assertIn( p.frame.name, ('0', '1') )
		self.assert_vector_equal( p.frame.dose_matrix.data, self.mat )
		self.assert_vector_equal( p.frame.voxel_weights.data, self.vw )
		self.assert_vector_equal( p.frame.beam_weights.data, self.bw )

		self.assertEqual( len(p.unique_frames), 1 )
		self.assertEqual( len(p.available_frame_mappings), 0 )

		# load frame
		next_name = '1' if p.frame.name == '0' else '0'
		next_frame = pa.load_frame(next_name)
		self.assertIsInstance( next_frame, DoseFrame )

		# load frame mapping
		fm = pa.load_frame_mapping('0', '1')
		self.assertIsInstance( fm, DoseFrameMapping )

class SolutionAccessorTestCase(ConradTestCase):
	def test_solution_accessor_save_load(self):
		sa = SolutionAccessor(filesystem=FilesystemTestCaching())

		sol_x = np.random.rand(10)
		ptr = sa.save_solution('dir', 'solution0', 'frame0', x=sol_x)
		self.assertTrue( sa.DB.has_key(ptr) )
		s = sa.load_solution(ptr)
		self.assertIsInstance( s, dict )
		for key in ['x', 'y', 'x_dual', 'y_dual']:
			self.assertTrue( key in s)
		self.assert_vector_equal( s['x'], sol_x )
		for key in ['y', 'x_dual', 'y_dual']:
			self.assertIsNone( s[key] )

	def test_solution_accessor_select(self):
		sa = SolutionAccessor(filesystem=FilesystemTestCaching())

		solution_list = [
			SolutionEntry(name='sol1', frame='f1'),
			SolutionEntry(name='sol2', frame='f1'),
			SolutionEntry(name='sol3', frame='f1'),
			SolutionEntry(name='sol4', frame='f2'),
		]

		with self.assertRaises(ValueError):
			sa.select_solution_entry(solution_list, 'bad frame', 'sol1')

		with self.assertRaises(ValueError):
			sa.select_solution_entry(solution_list, 'f1', 'bad name')

		se = sa.select_solution_entry(solution_list, 'f1', 'sol1')
		self.assertIsInstance( se, SolutionEntry )

class HistoryAccessorTestCase(ConradTestCase):
	def test_history_accessor_init(self):
		ha = HistoryAccessor(filesystem=FilesystemTestCaching())
		self.assertIs( ha.FS, ha.solution_accessor.FS )
		self.assertIs( ha.DB, ha.solution_accessor.DB )
		self.assertIsInstance( ha._HistoryAccessor__solution_cache, dict )
		self.assertEqual( len(ha._HistoryAccessor__solution_cache), 0 )

	def test_history_accessor_save_load(self):
		ha = HistoryAccessor(filesystem=FilesystemTestCaching())

		# load solution fails---empty cache
		with self.assertRaises(ValueError):
			ha.load_solution('f1', 'sol1')

		# save/load history
		h_dict = {
				'sol1': {'x': np.random.rand(30), 'frame': 'f1'},
				'sol2': {'x': np.random.rand(30), 'frame': 'f1'},
				'sol3': {'x': np.random.rand(30), 'frame': 'f1'},
				'sol4': {'x': np.random.rand(30), 'frame': 'f2'},
		}
		ptr = ha.save_history(h_dict, 'dir')
		self.assertTrue( ha.DB.has_key(ptr) )
		he = ha.load_history(ptr)
		self.assertIsInstance( he, HistoryEntry )

		# load solution
		s = ha.load_solution('f1', 'sol1')
		self.assert_vector_equal( s['x'], h_dict['sol1']['x'] )

		s = ha.load_solution('f2', 'sol4')
		self.assert_vector_equal( s['x'], h_dict['sol4']['x'] )

class SolverCacheAccessorTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.structures = [
			Structure(0, 'target', True, A=5 * np.random.rand(30, 20)),
			Structure(0, 'avoid', False, A=np.random.rand(100, 20)),
		]

	def test_solver_cache_accessor_save_load_pogs(self):
		if SKIP_POGS_CACHING_TESTS:
			return

		sca = SolverCacheAccessor(filesystem=FilesystemTestCaching())

		solver = SolverOptkit()
		solver.build(self.structures)
		ptr = sca._SolverCacheAccessor__save_pogs_solver_cache(
				solver, 'cache0', 'frame0', 'dir')
		self.assertTrue( sca.DB.has_key(ptr) )
		cache = sca._SolverCacheAccessor__load_pogs_solver_cache(ptr)

		self.assertIsInstance( cache, dict )
		for key in ['matrix', 'left_preconditioner', 'right_preconditioner']:
			self.assertIn( key, cache )
			self.assertIsInstance( cache[key], np.ndarray )
		self.assertTupleEqual( cache['matrix'].shape, (31, 20) )
		self.assertEqual( cache['left_preconditioner'].size, 31 )
		self.assertEqual( cache['right_preconditioner'].size, 20 )
		projector_matrix = cache.pop('projector_matrix', None)
		if projector_matrix is not None:
			self.assertIsInstance( projector_matrix, np.ndarray )
			self.assertTupleEqual( projector_matrix.shape, (20, 20) )

	def test_solver_cache_accessor_save_load(self):
		if SKIP_POGS_CACHING_TESTS:
			return

		sca = SolverCacheAccessor(filesystem=FilesystemTestCaching())

		solver = SolverOptkit()
		solver.build(self.structures)
		ptr = sca.save_solver_cache(solver, 'cache0', 'frame0', 'dir')
		self.assertTrue( sca.DB.has_key(ptr) )
		cache = sca.load_solver_cache(ptr)

		self.assertIsInstance( cache, dict )
		for key in ['matrix', 'left_preconditioner', 'right_preconditioner']:
			self.assertTrue( key in cache )
			self.assertIsInstance( cache[key], np.ndarray )
		self.assertTupleEqual( cache['matrix'].shape, (31, 20) )
		self.assertEqual( cache['left_preconditioner'].size, 31 )
		self.assertEqual( cache['right_preconditioner'].size, 20 )
		projector_matrix = cache.pop('projector_matrix', None)
		if projector_matrix is not None:
			self.assertIsInstance( projector_matrix, np.ndarray )
			self.assertTupleEqual( projector_matrix.shape, (20, 20) )

		# caching only for SolverOptkit
		with self.assertRaises(TypeError):
			sca.save_solver_cache(SolverCVXPY(), 'cache0', 'frame0', 'dir')

		# incomplete: cannot load
		with self.assertRaises(ValueError):
			sca.load_solver_cache(SolverCacheEntry())

		sc_entry = sca.DB.get(ptr)
		sc_entry.solver = 'bad solver'
		with self.assertRaises(ValueError):
			sca.load_solver_cache(sc_entry)

	def test_solver_cache_accessor_select(self):
		sca = SolverCacheAccessor(filesystem=FilesystemTestCaching())

		cache_list = [
			SolverCacheEntry(name='first', frame='f1'),
			SolverCacheEntry(name='second', frame='f2'),
		]

		entry = sca.select_solver_cache_entry(cache_list, 'first', 'f1')
		self.assertIs( entry, cache_list[0] )

		with self.assertRaises(ValueError):
			sca.select_solver_cache_entry(cache_list, 'bad name', 'f1')

		with self.assertRaises(ValueError):
			sca.select_solver_cache_entry(cache_list, 'first', 'bad frame')

class CaseAccessorTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.__files = []

	def setUp(self):
		self.A =np.random.rand(30, 20)
		self.case = Case(
				physics=Physics(dose_matrix=self.A),
				anatomy=Anatomy([Structure(0, 'ptv', True)]))

	@classmethod
	def tearDownClass(self):
		for f in self.__files:
			if os.path.exists(f):
				os.remove(f)

	def test_case_accessor_init(self):
		ca = CaseAccessor(filesystem=FilesystemTestCaching())
		self.assertIs( ca.DB, ca.anatomy_accessor.DB )
		self.assertIs( ca.DB, ca.physics_accessor.DB )
		self.assertIs( ca.DB, ca.solver_cache_accessor.DB )
		self.assertIs( ca.DB, ca.history_accessor.DB )

	def test_case_accessor_save_load(self):
		ca = CaseAccessor(filesystem=FilesystemTestCaching())

		ptr = ca.save_case(self.case, 'case0', 'dir')
		self.assertTrue( ca.DB.has_key(ptr) )

		case = ca.load_case(ptr)
		self.assertIsInstance( case, Case )
		self.assertTrue( case.anatomy[0].is_target )
		self.assertEqual( case.anatomy[0].name, 'ptv' )

	def test_case_accessor_update(self):
		ca = CaseAccessor(filesystem=FilesystemTestCaching())

		ptr = ca.save_case(self.case, 'case0', 'dir')
		case_entry = ca.DB.get(ptr)

		self.case.anatomy += Structure(1, 'oar', False)

		ptr2 = ca.update_case_entry(case_entry, self.case, 'dir', case_ID=ptr)
		self.assertEqual( ptr2, ptr )

		self.case.anatomy += Structure(2, 'oar2', False)
		ptr3 = ca.update_case_entry(case_entry, self.case, 'dir')
		self.assertNotEqual( ptr3, ptr )

	def test_case_accessor_save_load_components(self):
		ca = CaseAccessor(filesystem=FilesystemTestCaching())

		self.case.physics.add_dose_frame('frame1', data=np.random.rand(10, 15))
		self.case.physics.add_frame_mapping(
				DoseFrameMapping(
						'frame0', 'frame1',
						DiscreteMapping(10 * np.random.rand(20))))

		ptr = ca.save_case(self.case, 'case0', 'dir')
		case = ca.load_case(ptr)
		case_entry = ca.DB.get(ptr)

		# load frame
		frame1 = ca.load_frame(case_entry, 'frame1')
		self.assertIsInstance( frame1, DoseFrame )
		self.assertEqual( frame1.voxels, 10 )
		self.assertEqual( frame1.beams, 15 )

		# load frame mapping
		fmap01 = ca.load_frame_mapping(case_entry, 'frame0', 'frame1')
		self.assertIsInstance( fmap01, DoseFrameMapping )
		self.assertEqual( fmap01.source, 'frame0' )
		self.assertEqual( fmap01.target, 'frame1' )

		# save solver cache
		case.physics.voxel_labels = np.zeros(case.physics.frame.voxels)
		_, run = case.plan(verbose=0)
		if not (case.problem.solver_pogs is None or SKIP_POGS_CACHING_TESTS):
			ptr = ca.save_solver_cache(
					case_entry, case.problem.solver, 'cache0', 'frame0', 'dir')
			self.assertTrue( ca.DB.has_key(ptr) )
		else:
			ptr = None
			with self.assertRaises(TypeError):
				ca.save_solver_cache(
						case_entry, case.problem.solver, 'cache0', 'frame0',
						'dir')

		# load solver cache
		if not (case.problem.solver_pogs is None or SKIP_POGS_CACHING_TESTS):
			cache = ca.load_solver_cache(case_entry, 'cache0', 'frame0')
			self.assertIsInstance( cache, dict )
		else:
			with self.assertRaises(ValueError):
				ca.load_solver_cache(case_entry, 'cache0', 'frame0')

		# save solution
		history = {}
		ptr = ca.save_solution(
				case_entry, 'frame0', 'solution0', 'dir', x=run.x)
		history['solution0'] = ptr
		self.assertTrue( ca.DB.has_key(ptr) )

		# load solution
		sol = ca.load_solution(case_entry.history, 'frame0', 'solution0')
		self.assert_vector_equal( sol['x'], run.x )

	def test_case_accessor_yaml(self):
		ca = CaseAccessor(filesystem=FilesystemTestCaching())

		# primary mode: dump case-related parts of database into YAML
		# file containing multiple documents
		y = ca.write_case_yaml(self.case, 'my_case', '.')
		self.__files.append(y)
		self.assertIn( '.yaml', y )
		self.assertTrue( os.path.exists(y) )

		case = ca.load_case_yaml(y)
		self.assertIsInstance( case, Case )

		# secondary mode: dump case dictionary into YAML file
		y2 = ca.write_case_yaml(
				self.case, 'my_case2', '.', single_document=True)
		self.__files.append(y2)
		self.assertIn( '.yaml', y2 )
		self.assertTrue( os.path.exists(y2) )

		case2 = ca.load_case_yaml(y2)
		self.assertIsInstance( case2, Case )

		# dose matrix dims in `case_example.yml` = (30, 20)
		# database pointer is `data_fragment.1000`
		ca.DB.set(1000, ca.FS.write_data('dir', 'A', np.random.rand(30, 20)))
		case3 = ca.load_case_yaml(os.path.join(
				os.path.dirname(__file__), 'case_example.yml'))
		self.assertIsInstance( case3, Case )
