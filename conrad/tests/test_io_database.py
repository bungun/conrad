"""
Unit tests for :mod:`conrad.io.database`.
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

from conrad.io.database import *
from conrad.tests.base import *

class DatabaseTest(ConradDatabaseBase):
	def logged_entries(self):
		pass
	def next_available_key(self, data_type):
		raise NotImplementedError
	def get(self, key):
		raise NotImplementedError
	def get_keys(self, entry_type=None):
		raise NotImplementedError
	def set(self, key, value, overwrite=False):
		raise NotImplementedError
	def set_next(self, value):
		raise NotImplementedError
	def has_key(self, key):
		raise NotImplementedError
	def ingest_dictionary(self, dictionary):
		raise NotImplementedError
	def dump_to_dictionary(self, logged_entries_only=False):
		raise NotImplementedError
	def clear_log(self, filename):
		raise NotImplementedError

class ConradDatabaseBaseTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.entry_types = [
				VectorEntry, DenseMatrixEntry, SparseMatrixEntry,
				DataDictionaryEntry, DoseFrameEntry, DoseFrameMappingEntry,
				PhysicsEntry, SolutionEntry, HistoryEntry, SolverCacheEntry,
				StructureEntry, AnatomyEntry, CaseEntry]

	def test_raw_data_to_entry(self):
		dbt = DatabaseTest()

		test_input = []
		with self.assertRaises(TypeError):
			dbt.raw_data_to_entry(test_input)

		test_input = {}
		with self.assertRaises(ValueError):
			dbt.raw_data_to_entry(test_input)


		test_input['conrad_db_type'] = 'not a type'
		with self.assertRaises(ValueError):
			dbt.raw_data_to_entry(test_input)

		test_input['conrad_db_type'] = 'data_fragment: not a type'
		with self.assertRaises(ValueError):
			dbt.raw_data_to_entry(test_input)

		for constructor in self.entry_types:
			obj_in = constructor()
			obj_out = dbt.raw_data_to_entry(obj_in.nested_dictionary)
			self.assertEqual( type(obj_out), constructor )

	def test_key_methods(self):
		dbt = DatabaseTest()
		key_in = 28

		for t in self.entry_types:
			key_expect = CONRAD_DB_ENTRY_PREFIXES[t] + str(key_in)
			key_out = dbt.join_key(t, key_in)
			self.assertEqual( key_out, key_expect )

			key_out = dbt.join_key(
					t, CONRAD_DB_ENTRY_PREFIXES[t] + str(key_in))
			self.assertEqual( key_out, key_expect )

			key_out = dbt.join_key(CONRAD_DB_ENTRY_TYPES[t], key_in)
			self.assertEqual( key_out, key_expect )


			type_expect = CONRAD_DB_ENTRY_TYPES[t]
			type_out = dbt.type_from_key(key_out)
			self.assertEqual( type_out, type_expect )

		with self.assertRaises(TypeError):
			dbt.join_key('invalid_type', key_in)

		with self.assertRaises(TypeError):
			dbt.join_key(str, key_in)

class LocalPythonDatabaseTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.basic_entry_types = [
				VectorEntry, DenseMatrixEntry, SparseMatrixEntry,
				DataDictionaryEntry]
		self.custom_entry_types = [
				DoseFrameEntry, DoseFrameMappingEntry, PhysicsEntry,
				SolutionEntry, HistoryEntry, SolverCacheEntry, StructureEntry,
				AnatomyEntry, CaseEntry]
		self.entry_types = self.basic_entry_types + self.custom_entry_types
		self.__test_files = []

	def tearDown(self):
		for f in self.__test_files:
			if os.path.exists(f):
				os.remove(f)

	def test_lpdb_next_available(self):
		lpdb = LocalPythonDatabase()

		idx_frag = 0
		for et in self.entry_types:
			if et in self.basic_entry_types:
				key = idx_frag
				idx_frag += 2
			else:
				key = 0

			key_expect = CONRAD_DB_ENTRY_PREFIXES[et] + str(key)
			key_out = lpdb.next_available_key(et)
			self.assertEqual( key_out, key_expect )

			lpdb._LocalPythonDatabase__tables[
					CONRAD_DB_ENTRY_TYPES[et]][key_out] = 'taken'

			key_expect = CONRAD_DB_ENTRY_PREFIXES[et] + str(key + 1)
			key_out = lpdb.next_available_key(CONRAD_DB_ENTRY_TYPES[et])
			self.assertEqual( key_out, key_expect )

			lpdb._LocalPythonDatabase__tables[
					CONRAD_DB_ENTRY_TYPES[et]][key_out] = 'taken'

		with self.assertRaises(ValueError):
			lpdb.next_available_key(str)

	def test_lpdb_set(self):
		lpdb = LocalPythonDatabase()

		self.assertIsNone( lpdb.set('garbage key', None) )

		idx_frag = 0
		for entry_type in self.entry_types:
			if entry_type in self.basic_entry_types:
				idx = idx_frag
				idx_frag += 2
			else:
				idx = 0

			prefix = CONRAD_DB_ENTRY_PREFIXES[entry_type]
			key = prefix + str(idx)

			with self.assertRaises(TypeError):
				lpdb.set(key, {})

			key_out = lpdb.set(key, entry_type())
			self.assertEqual( key_out, key )
			with self.assertRaises(KeyError):
				lpdb.set(key, entry_type())
			lpdb.set(key, entry_type(), overwrite=True)
			self.assertEqual( key_out, key )

			key_expect = prefix + str(idx + 1)
			key_out = lpdb.set(idx + 1, entry_type())
			self.assertEqual( key_out, key_expect )

	def test_lpdb_set_next(self):
		lpdb = LocalPythonDatabase()

		self.assertIsNone( lpdb.set_next(None) )
		with self.assertRaises(TypeError):
			lpdb.set_next({})

		idx_frag = 0
		for entry_type in self.entry_types:
			prefix = CONRAD_DB_ENTRY_PREFIXES[entry_type]

			offset = idx_frag if entry_type in self.basic_entry_types else 0
			for i in xrange(10):
				key_expect = prefix + str(i + offset)
				key_out = lpdb.set_next(entry_type())
				self.assertEqual( key_out, key_expect )
			idx_frag += 10

	def test_lpdb_get(self):
		lpdb = LocalPythonDatabase()

		self.assertIsNone( lpdb.get(None) )

		idx_frag = 0
		for entry_type in self.entry_types:
			et_instance = entry_type()
			self.assertIsInstance( lpdb.get(et_instance), entry_type )
			self.assertIsInstance(
					lpdb.get(et_instance.nested_dictionary), entry_type )

			offset = idx_frag if entry_type in self.basic_entry_types else 0
			for i in xrange(10):
				key = lpdb.set_next(entry_type())
				self.assertIsInstance( lpdb.get(key), entry_type )
				with self.assertRaises(KeyError):
					key = CONRAD_DB_ENTRY_PREFIXES[entry_type] + str(i + offset + 1)
					lpdb.get(key)
			idx_frag += 10

	def test_lpdb_clear_log(self):
		lpdb = LocalPythonDatabase()

		self.assertEqual( len(lpdb.logged_entries), 0 )

		for entry_type in self.entry_types:
			lpdb.set_next(entry_type())

		self.assertGreater( len(lpdb.logged_entries), 0 )
		lpdb.clear_log()
		self.assertEqual( len(lpdb.logged_entries), 0 )

	def test_lpdb_entry_flattening(self):
		lpdb = LocalPythonDatabase()
		n_logged = 0

		ve = VectorEntry()
		ve.nested_dictionary
		ve.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )

		dme = DenseMatrixEntry()
		dme.nested_dictionary
		dme.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )

		sme = SparseMatrixEntry()
		sme.nested_dictionary
		sme.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )

		dde = DataDictionaryEntry(entries={1: VectorEntry(), 2: VectorEntry()})
		dde.nested_dictionary
		with self.assertRaises(ValueError):
			dde.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		dde.flatten(lpdb)
		n_logged += 2
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		dde.flat_dictionary

		dfe = DoseFrameEntry(
				dose_matrix=DenseMatrixEntry(), voxel_weights=VectorEntry())
		dfe.nested_dictionary
		with self.assertRaises(ValueError):
			dfe.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		dfe.flatten(lpdb)
		n_logged += 2
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		dfe.flat_dictionary

		dfme = DoseFrameMappingEntry(voxel_map=VectorEntry())
		dfme.nested_dictionary
		with self.assertRaises(ValueError):
			dfme.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		dfme.flatten(lpdb)
		n_logged += 1
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		dfme.flat_dictionary

		pe = PhysicsEntry(frames=[DoseFrameEntry(
				dose_matrix=DenseMatrixEntry(), beam_labels=VectorEntry())])
		pe.nested_dictionary
		with self.assertRaises(ValueError):
			pe.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		pe.flatten(lpdb)
		n_logged += 3
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		pe.flat_dictionary

		se = SolutionEntry(x=VectorEntry(), y_dual=VectorEntry())
		se.nested_dictionary
		with self.assertRaises(ValueError):
			se.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		se.flatten(lpdb)
		n_logged += 2
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		se.flat_dictionary

		he = HistoryEntry(
				solutions=[SolutionEntry(), SolutionEntry(
						x=VectorEntry(), y_dual=VectorEntry())])
		he.nested_dictionary
		with self.assertRaises(ValueError):
			he.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		he.flatten(lpdb)
		n_logged += 4
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		he.flat_dictionary

		sce = SolverCacheEntry(
				matrix=DenseMatrixEntry(), left_preconditioner=VectorEntry())
		sce.nested_dictionary
		with self.assertRaises(ValueError):
			sce.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		sce.flatten(lpdb)
		n_logged += 2
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		sce.flat_dictionary

		se = StructureEntry()
		se.nested_dictionary
		se.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )

		ae = AnatomyEntry(structures=[StructureEntry(), StructureEntry()])
		ae.nested_dictionary
		with self.assertRaises(ValueError):
			ae.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		ae.flatten(lpdb)
		n_logged += 2
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		ae.flat_dictionary

		ce = CaseEntry(
				anatomy=AnatomyEntry(),
				physics=PhysicsEntry(
						frames=[DoseFrameEntry()],
						frame_mappings=[DoseFrameMappingEntry()]),
				solver_caches=[SolverCacheEntry(), SolverCacheEntry()])
		ce.nested_dictionary
		with self.assertRaises(ValueError):
			ce.flat_dictionary
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		ce.flatten(lpdb)
		n_logged += 6
		self.assertEqual( len(lpdb.logged_entries), n_logged )
		ce.flat_dictionary

	def test_lpdb_dump_dictionary(self):
		lpdb = LocalPythonDatabase()

		lpdb.set_next(VectorEntry())
		db_dict = lpdb.dump_to_dictionary()

		self.assertIsInstance( db_dict, dict )
		self.assertIn( 'data_fragments', db_dict )
		self.assertGreater( len(db_dict['data_fragments']), 0 )
		key_expect = CONRAD_DB_ENTRY_PREFIXES[VectorEntry] + str(0)
		self.assertIn( key_expect, db_dict['data_fragments'] )
		self.assertIsInstance( db_dict['data_fragments'][key_expect], dict )

		self.assertIn( 'solutions', db_dict )
		self.assertEqual( len(db_dict['solutions']), 0 )

		lpdb.set_next(SolutionEntry(x=VectorEntry()))
		db_dict = lpdb.dump_to_dictionary()

		self.assertEqual( len(db_dict['solutions']), 1 )
		key_expect = CONRAD_DB_ENTRY_PREFIXES[SolutionEntry] + str(0)
		self.assertIn( key_expect, db_dict['solutions'] )
		self.assertIsInstance( db_dict['solutions'][key_expect], dict )
		sol_entry = db_dict['solutions'][key_expect]

		self.assertEqual( len(db_dict['data_fragments']), 2 )
		key_expect = CONRAD_DB_ENTRY_PREFIXES[VectorEntry] + str(1)
		self.assertIn( 'x', sol_entry )
		self.assertEqual( sol_entry['x'], key_expect )

	def test_lpdb_ingest_dictionary(self):
		lpdb = LocalPythonDatabase()
		lpdb.set_next(VectorEntry())
		lpdb.set_next(SolutionEntry(x=VectorEntry()))

		lpdb2 = LocalPythonDatabase()
		lpdb2.ingest_dictionary(lpdb.dump_to_dictionary())
		self.assertIsInstance( lpdb2.get('data_fragment.0'), VectorEntry )
		self.assertIsInstance( lpdb2.get('data_fragment.1'), VectorEntry )
		self.assertIsInstance( lpdb2.get('solution.0'), SolutionEntry )

	def test_lpdb_yaml(self):
		lpdb = LocalPythonDatabase()
		lpdb.set_next(VectorEntry())
		lpdb.set_next(SolutionEntry(x=VectorEntry()))

		lpdb2 = LocalPythonDatabase()
		filename = os.path.join(os.getcwd(), 'test_db.yaml')
		self.__test_files.append(filename)
		lpdb2.ingest_yaml(lpdb.dump_to_yaml(filename))
		self.assertIsInstance( lpdb2.get('data_fragment.0'), VectorEntry )
		self.assertIsInstance( lpdb2.get('data_fragment.1'), VectorEntry )
		self.assertIsInstance( lpdb2.get('solution.0'), SolutionEntry )
