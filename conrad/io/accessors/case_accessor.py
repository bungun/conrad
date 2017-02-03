"""
Define :class:`CaseAccessor`
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
import yaml

from conrad.case import Case
from conrad.io.schema import CaseEntry, HistoryEntry, CONRAD_DB_ENTRY_PREFIXES
from conrad.io.accessors.base_accessor import ConradDBAccessor
from conrad.io.accessors.anatomy_accessor import AnatomyAccessor
from conrad.io.accessors.physics_accessor import PhysicsAccessor
from conrad.io.accessors.solver_accessor import SolverCacheAccessor
from conrad.io.accessors.history_accessor import HistoryAccessor

def validate_case_entry(entry):
	if not isinstance(entry, CaseEntry):
		raise ValueError(
				'argument `case_entry` must be of type {}'.format(CaseEntry))

class CaseAccessor(ConradDBAccessor):
	def __init__(self, database=None, filesystem=None):
		self.__anatomy_accessor = AnatomyAccessor(
				database=database, filesystem=filesystem)
		self.__physics_accessor = PhysicsAccessor(
				database=database, filesystem=filesystem)
		self.__solver_cache_accessor = SolverCacheAccessor()
		self.__history_accessor = HistoryAccessor()
		ConradDBAccessor.__init__(
				self, subaccessors=[
						self.__solver_cache_accessor, self.__physics_accessor,
						self.__history_accessor, self.__anatomy_accessor],
				database=database, filesystem=filesystem)

	@property
	def anatomy_accessor(self):
		return self.__anatomy_accessor

	@property
	def physics_accessor(self):
		return self.__physics_accessor

	@property
	def solver_cache_accessor(self):
		return self.__solver_cache_accessor

	@property
	def history_accessor(self):
		return self.__history_accessor

	def save_case(self, case, case_name, directory, overwrite=False,
				  case_ID=None):
		if not isinstance(case, Case):
			raise TypeError(
					'argument `case` must be of type {}'.format(Case))

		self.FS.check_dir(directory)
		subdir = self.FS.join_mkdir(directory, ['cases', case_name])

		rx = case.prescription.rx_list if case.prescription is None else None
		if case_ID is None or not self.DB.has_key(case_ID):
			case_ID = self.DB.next_available_key(CaseEntry)

		return self.DB.set(
				case_ID,
				CaseEntry(
						name=case_name, prescription=rx,
						physics=self.physics_accessor.save_physics(
								case.physics, subdir, overwrite),
						anatomy=self.anatomy_accessor.save_anatomy(
								case.anatomy)),
				overwrite=True)

	def update_case_entry(self, case_entry, case, directory, overwrite=False,
						  case_ID=None):
		if not isinstance(case, Case):
			raise TypeError(
					'argument `case` must be of type {}'.format(Case))
		if not isinstance(case_entry, CaseEntry):
			raise TypeError(
					'argument `case_entry` must be of type {}'
					''.format(CaseEntry))

		subdir = self.FS.join_mkdir(directory, ['cases', case_entry.name])
		case_entry.physics = self.physics_accessor.save_physics(
				case.physics, subdir, overwrite)
		case_entry.anatomy = self.anatomy_accessor.save_anatomy(case.anatomy)

		if case_ID is None or not self.DB.has_key(case_ID):
			case_ID = self.DB.next_available_key(CaseEntry)

		return self.DB.set(case_ID, case_entry, overwrite=True)

	def load_case(self, case_entry, frame='default'):
		case_entry = self.DB.get(case_entry)
		validate_case_entry(case_entry)
		if not case_entry.complete:
			raise ValueError('case incomplete')

		return Case(
			anatomy=self.anatomy_accessor.load_anatomy(case_entry.anatomy),
			physics=self.physics_accessor.load_physics(
					case_entry.physics, frame_name=frame),
			prescription=case_entry.prescription,
		)

	def load_frame(self, case_entry, frame_name):
		case_entry = self.DB.get(case_entry)
		validate_case_entry(case_entry)
		physics_entry = self.DB.get(case_entry.physics)
		frame = self.physics_accessor.frame_accessor.select_frame_entry(
				physics_entry.frames, frame_name)
		return self.physics_accessor.frame_accessor.load_frame(frame)

	def load_frame_mapping(self, case_entry, source_frame, target_frame):
		case_entry = self.DB.get(case_entry)
		validate_case_entry(case_entry)
		physics_entry = self.DB.get(case_entry.physics)

		fma = self.physics_accessor.frame_mapping_accessor
		return fma.load_frame_mapping(fma.select_frame_mapping_entry(
				physics_entry.frame_mappings, source_frame, target_frame))

	def save_solver_cache(self, case_entry, solver, cache_name, frame_name,
						  directory):
		case_entry = self.DB.get(case_entry)
		validate_case_entry(case_entry)
		cache_ID = self.solver_cache_accessor.save_solver_cache(
				solver, cache_name, frame_name, directory)
		case_entry.add_solver_caches(cache_ID)
		return cache_ID

	def load_solver_cache(self, case_entry, cache_name, frame_name):
		case_entry = self.DB.get(case_entry)
		validate_case_entry(case_entry)
		cache = self.solver_cache_accessor.select_solver_cache_entry(
				case_entry.solver_caches, cache_name, frame_name)
		return self.solver_cache_accessor.load_solver_cache(cache)

	# def save_solver_state(self, solver, frame_name, snapshot_name):
	#	pass

	# def load_solver_state(self, solver_state_list, frame_name, snapshot_name):
	# 	pass

	def __load_history(self, case_entry):
		if case_entry.history is not None:
			case_entry.history = self.history_accessor.load_history(
					case_entry.history)
		else:
			case_entry.history = HistoryEntry()

	def save_solution(self, case_entry, frame_name, solution_name, directory,
					  **solution_data):
		case_entry = self.DB.get(case_entry)
		validate_case_entry(case_entry)
		self.__load_history(case_entry)

		solution_ID = self.history_accessor.solution_accessor.save_solution(
				directory, solution_name, frame_name, **solution_data)
		case_entry.history.add_solutions(solution_ID)
		return solution_ID

	def load_solution(self, history_entry, frame_name, solution_name):
		history_entry = self.DB.get(history_entry)
		self.history_accessor.load_history(history_entry)
		sol = self.history_accessor.solution_accessor.select_solution_entry(
				history_entry.solutions, frame_name, solution_name)
		return self.history_accessor.solution_accessor.load_solution(sol)

	def load_case_yaml(self, yaml_file):
		# try multi-document specification
		self.DB.clear_log()
		self.DB.ingest_yaml(yaml_file)
		for e in self.DB.logged_entries:
			if CONRAD_DB_ENTRY_PREFIXES[CaseEntry] in e:
				return self.load_case(e)

		# try single-document specification:
		if os.path.exists(yaml_file):
			f = open(yaml_file, 'r')
			case_dictionary = yaml.safe_load(f)
			ce = CaseEntry(**case_dictionary)
			f.close()
			return self.load_case(ce)

		# otherwise, return None
		return None

	def write_case_yaml(self, case, case_name, directory, single_document=False):
		self.FS.check_dir(directory)
		self.DB.clear_log()
		ptr = self.save_case(case, case_name, directory)
		if single_document:
			if os.path.exists(directory):
				filename = os.path.join(directory, case_name + '.yaml')
				f = open(filename, 'w')
				entry = self.DB.get(ptr)
				f.write(yaml.safe_dump(
						entry.arborize(self.DB).nested_dictionary,
						default_flow_style=False))
				f.close()
				return filename
			else:
				raise ValueError('directory `{}` not found'.format(directory))
		else:
			return self.DB.dump_to_yaml(
					os.path.join(directory, case_name),
					logged_entries_only=True)
