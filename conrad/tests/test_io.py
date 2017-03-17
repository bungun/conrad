"""
Unit tests for :mod:`conrad.io.io`.
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
import numpy as np

from conrad.case import *
from conrad.io.io import *
from conrad.medicine import Structure
from conrad.physics.physics import DoseFrameMapping
from conrad.tests.base import *
from conrad.tests.test_io_filesystem import FilesystemTestCaching

SKIP_POGS_CACHING_TESTS = os.getenv('CONRAD_SKIP_POGS_CACHING_TESTS', False)

class CaseIOTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.__files = []
		self.rx_test_file = os.path.join(
				os.path.dirname(__file__), 'yaml_rx.yml')

	@classmethod
	def tearDownClass(self):
		for f in self.__files:
			if os.path.exists(f):
				os.remove(f)

	def setUp(self):
		self.case = Case(
				physics={'dose_matrix': np.random.rand(30, 20)},
				prescription=self.rx_test_file)

		# build voxel labels to be even split across voxels
		m = self.case.physics.voxels
		vl = np.ones(m)
		N = self.case.anatomy.n_structures
		for i in xrange(N):
			label = self.case.anatomy.list[i].label
			vl[i * int(m/N):(i + 1) * int(m/N)] = label
		self.voxel_labels = vl

	def test_caseio_init(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)
		self.assertIs( caseio.FS, caseio.accessor.FS )
		self.assertIs( caseio.DB, caseio.accessor.DB )
		self.assertIsNone( caseio.working_directory )
		self.assertIsNone( caseio.active_case )
		self.assertIsNone( caseio.active_meta )
		with self.assertRaises(ValueError):
			caseio.active_frame_name

	def test_caseio_working_directory(self):
		caseio = CaseIO()
		with self.assertRaises(OSError):
			caseio.working_directory = '/BAD_CONRAD_FILEPATH'

		caseio.working_directory = '.'

	def test_caseio_available(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)
		self.assertEqual( len(caseio.available_cases), 0 )

		caseio.accessor.save_case(
				Case(physics={'dose_matrix': np.random.rand(30, 20)}),
				'test_case', 'dir')
		self.assertIn( 'test_case', caseio.available_cases )

	def test_caseio_select(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)

		ptr1 = caseio.accessor.save_case(
				Case(physics={'dose_matrix': np.random.rand(30, 20)}),
				'test_case', 'dir')
		ptr2 = caseio.accessor.save_case(
				Case(physics={'dose_matrix': np.random.rand(30, 20)}),
				'test_case2', 'dir')

		ce = caseio.select_case_entry('test_case2')
		self.assertIsInstance( ce, CaseEntry )
		self.assertEqual( ce.name, 'test_case2' )

		ce = caseio.select_case_entry('bad name', ptr1)
		self.assertIsInstance( ce, CaseEntry )
		self.assertEqual( ce.name, 'test_case' )

		with self.assertRaises(ValueError):
			caseio.select_case_entry('bad name', 'bad ID')

	def test_caseio_load_active(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)

		caseio.accessor.save_case(self.case, 'test_case', 'dir')
		case = caseio.load_case('test_case')
		self.assertIs( case, caseio.active_case )
		self.assertEqual( caseio.active_meta.name, 'test_case' )
		self.assertEqual( caseio.active_frame_name, 'frame0' )

	def test_caseio_save_close_active(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)

		caseio.accessor.save_case(self.case, 'test_case', 'dir')

		# no active case
		with self.assertRaises(ValueError):
			caseio.save_active_case('dir')

		case = caseio.load_case('test_case')

		# no directory specified
		with self.assertRaises(ValueError):
			caseio.save_active_case()

		self.assertTrue( case.physics.frame.voxel_labels is None)
		label = 1
		case.physics.frame.voxel_labels = label * np.ones(case.physics.voxels)

		# update
		caseio.working_directory = 'dir'
		caseio.save_active_case()

		# verify update
		case2 = caseio.accessor.load_case(caseio._CaseIO__active_case_ID)
		self.assertIsNotNone( case2.physics.frame.voxel_labels )
		self.assertTrue( all(case2.physics.frame.voxel_labels == label) )

		# close
		caseio.close_active_case()
		self.assertIsNone( caseio.active_case )
		self.assertIsNone( caseio.active_meta )

	def test_caseio_save_new(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)
		caseio.save_new_case(self.case, 'test case', 'dir')

		self.assertIs( caseio.active_case, self.case )

		c = Case(
				prescription=self.rx_test_file,
				physics={'dose_matrix': np.random.rand(30, 20)})

		with self.assertRaises(ValueError):
			# name used
			caseio.save_new_case(c, 'test case', 'dir')

		# attempt to save new case should close active case:
		self.assertIs( caseio.active_case, None )

		with self.assertRaises(ValueError):
			# no directory
			caseio.save_new_case(c, 'test case')
		self.assertIs( caseio.active_case, None )

		caseio.save_new_case(c, 'test case 2', 'dir')
		self.assertIsNot( caseio.active_case, self.case )
		self.assertIs( caseio.active_case, c )

	def test_caseio_rename(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)
		caseio.save_new_case(self.case, 'test case', 'dir')
		self.assertIn( 'test case', caseio.available_cases )
		self.assertEqual( caseio.active_meta.name, 'test case' )

		caseio.rename_active_case('test case 2')
		self.assertNotIn( 'test case', caseio.available_cases )
		self.assertEqual( caseio.active_meta.name, 'test case 2' )

	def test_caseio_frame_load(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)

		self.case.physics.add_dose_frame(
				'frame1', data=np.random.rand(40, 40),
				voxel_labels=np.ones(40))
		self.case.physics.add_dose_frame(
				'frame2', data=np.random.rand(50, 30),
				voxel_labels=np.ones(50))

		caseio.save_new_case(self.case, 'test case', 'dir')
		caseio.close_active_case()
		case = caseio.load_case('test case')
		case.physics.add_dose_frame('frame3', voxels=100, beams=20)

		for i in [0, 3]:
			self.assertIn( 'frame%i' %i, case.physics.available_frames )
		for i in [1, 2]:
			self.assertNotIn( 'frame%i' %i, case.physics.available_frames )

		for i in xrange(4):
			caseio.load_frame('frame%i' %i)
			self.assertEqual( case.physics.frame.name, 'frame%i' %i )

	def test_caseio_frame_mapping_load(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)
		self.case.physics.add_dose_frame(
				'frame1', data=np.random.rand(40, 40),
				voxel_labels=np.ones(40))
		self.case.physics.add_dose_frame(
				'frame2', data=np.random.rand(50, 30),
				voxel_labels=np.ones(50))

		self.case.physics.add_frame_mapping(
				DoseFrameMapping('frame1', 'frame0', 30 * np.random.rand(40)))
		self.case.physics.add_frame_mapping(
				DoseFrameMapping('frame2', 'frame0', 30 * np.random.rand(50)))
		self.assertEqual( len(self.case.physics.available_frame_mappings), 2 )

		caseio.save_new_case(self.case, 'test case', 'dir')
		caseio.close_active_case()
		case = caseio.load_case('test case')
		self.assertEqual( len(case.physics.available_frame_mappings), 0 )

		caseio.load_frame_mapping('frame1', 'frame0')
		self.assertEqual( len(case.physics.available_frame_mappings), 1 )

		caseio.load_frame_mapping('frame2', 'frame0')
		self.assertEqual( len(case.physics.available_frame_mappings), 2 )

		case.physics.add_frame_mapping(
				DoseFrameMapping('frame2', 'frame1', 40 * np.random.rand(50)))

		self.assertEqual( len(case.physics.available_frame_mappings), 3 )
		caseio.load_frame_mapping('frame2', 'frame1')
		self.assertEqual( len(case.physics.available_frame_mappings), 3 )

	def test_caseio_solver_cache_save_load(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)
		self.case.physics.frame.voxel_labels = self.voxel_labels
		caseio.save_new_case(self.case, 'test case', 'dir')

		with self.assertRaises(TypeError):
			# no solver built
			caseio.save_solver_cache('my cache', 'my_dir')

		# clear dose constraints from anatomy to enable POGS solver
		# build if possible, since POGS solver is only cached solver
		# at this time
		for s in self.case.anatomy:
			s.constraints.clear()

		self.case.plan(verbose=False)

		if self.case.problem.solver is self.case.problem.solver_pogs:
			if SKIP_POGS_CACHING_TESTS:
				return
			# get setup time from first run
			t0 = caseio.active_case.problem.solver_pogs.pogs_solver.info.setup_time

			with self.assertRaises(ValueError):
				# no directory specified
				caseio.save_solver_cache('my cache')

			caseio.save_solver_cache('my cache', 'my_dir')

			# load solver cache
			caseio.close_active_case()

			caseio.load_case('test case')
			cache = caseio.load_solver_cache('my cache')
			self.assertIsInstance( cache, dict )
			self.assertIn( 'matrix', cache )
			self.assertIn( 'left_preconditioner', cache )
			self.assertIn( 'right_preconditioner', cache )
			self.assertIn( 'projector_matrix', cache )
			output = caseio.active_case.plan(solver_cache=cache, verbose=False)

			# get setup time from second run
			t1 = caseio.active_case.problem.solver_pogs.pogs_solver.info.setup_time
			# self.assertTrue( t1 <= t0 )
		else:
			with self.assertRaises(TypeError):
				# bad solver type
				caseio.save_solver_cache('my cache', 'my_dir')

	def test_caseio_solution_save_load(self):
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)
		self.case.physics.frame.voxel_labels = self.voxel_labels
		caseio.save_new_case(self.case, 'test case', 'dir')

		# solve a case, save solution
		status, run = self.case.plan(verbose=False)
		caseio.save_solution('my solution', 'dir', x=run.x)

		# load the solution
		caseio.close_active_case()
		caseio.load_case('test case')
		sol = caseio.load_solution('my solution')

		self.assertIsInstance( sol, dict )
		self.assert_vector_equal( sol['x'], run.x )

	def test_caseio_case_YAML_transfers(self):
		# build a case, dump to YAML
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)

		directory = os.path.dirname(__file__)
		case_name = 'yaml_testing_case'

		yaml_store = caseio.case_to_YAML(self.case, case_name, directory)
		self.__files.append(yaml_store)
		self.assertTrue( os.path.exists(yaml_store) )

		# take example YAML, build a case
		caseio2 = CaseIO()

		case = caseio.YAML_to_case(yaml_store)
		self.assertIsInstance( case, Case )
		for s in self.case.anatomy:
			self.assertIn( s.label, case.anatomy )
		self.assert_vector_equal( case.A, self.case.A )

		# test anatomy->physics (on save),
		# physics->anatomy transfer (on load)
		m1, m2, n = 30, 30, 20
		A1 = np.random.rand(m1, n)
		A2 = np.random.rand(m2, n)
		case2 = Case()
		case2.anatomy += Structure(0, 'target', True, A=A1)
		case2.anatomy += Structure(1, 'OAR', False, A=A2)
		self.assertTrue( case2.plannable )

		yaml_store2 = caseio.case_to_YAML(case2, 'yaml_testing_2', directory)
		self.__files.append(yaml_store2)
		self.assertTrue( os.path.exists(yaml_store2) )

		# take example YAML, build a case
		caseio2 = CaseIO()
		case2 = caseio.YAML_to_case(yaml_store2)
		self.assertTrue( case2.plannable )

	def test_caseio_transfer(self):
		# make CaseIO with some database entries
		caseio = CaseIO(FS_constructor=FilesystemTestCaching)
		caseio.save_new_case(self.case, 'test case', 'dir')

		# dump database to yaml or dictionary
		db_dictionary = caseio.DB.dump_to_dictionary()

		# recreate a CaseIO with the dumped database
		caseio2 = CaseIO(filesystem=caseio.FS, DB_dict=db_dictionary)

		self.assertIn( 'test case', caseio2.available_cases )
		case = caseio2.load_case('test case')
		self.assert_vector_equal( case.A, self.case.A )
