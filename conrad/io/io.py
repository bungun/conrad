"""
Define :class:`ConradIO` for managing saving and loading of
:class:`Case` objects.
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

import yaml

from conrad.io.schema import cdb_util, CaseEntry
from conrad.io.filesystem import ConradFilesystemBase, LocalFilesystem
from conrad.io.database import ConradDatabaseBase, LocalPythonDatabase
from conrad.io.accessors import CaseAccessor

class CaseIO(object):
	def __init__(self, **options):
		self.__working_directory = None

		self.__DB = None
		self.__FS = None

		self.__case_accessor = CaseAccessor()
		self.__active_case_ID = None
		self.__active_case_entry = None
		self.__active_case_object = None
		self.__active_case_directory = None

		# map case names to case IDs
		self.__cases = {}

		DB_yaml = options.pop('DB_yaml', None)
		DB_dict = options.pop('DB_dict', None)
		DB_constructor = options.pop('DB_constructor', LocalPythonDatabase)

		if not issubclass(DB_constructor, ConradDatabaseBase):
			raise TypeError(
					'if keyword argument `DB_constructor` provided, it '
					'must be a type that inherits from {}'
					''.format(ConradDatabaseBase))
		self.DB = DB_constructor(dictionary=DB_dict, yaml_file=DB_yaml)

		FS = options.pop('filesystem', options.pop('FS', None))
		if isinstance(FS, ConradFilesystemBase):
			self.FS = FS
		else:
			FS_constructor = options.pop('FS_constructor', LocalFilesystem)
			if not issubclass(FS_constructor, ConradFilesystemBase):
				raise TypeError(
						'if keyword argument `FS_constructor` provided, '
						'it must be a type that inherits from {}'
						''.format(ConradFilesystemBase))
			self.FS = FS_constructor()

		if options.pop('load_default', False):
			if len(self.available_cases) > 0:
				self.load_case(self.available_cases[0])

	@property
	def DB(self):
		return self.__DB

	@DB.setter
	def DB(self, database):
		if isinstance(database, ConradDatabaseBase):
			self.__DB = database
			self.accessor.set_database(self.DB)

	@property
	def FS(self):
		return self.__FS

	@FS.setter
	def FS(self, filesystem):
		if isinstance(filesystem, ConradFilesystemBase):
			self.__FS = filesystem
			self.accessor.set_filesystem(self.FS)

	@property
	def working_directory(self):
		if self.active_case is not None:
			return self.__active_case_directory
		return self.__working_directory

	@working_directory.setter
	def working_directory(self, directory):
		self.FS.check_dir(directory)
		self.__working_directory = directory

	@property
	def accessor(self):
		return self.__case_accessor

	@property
	def active_case(self):
		return self.__active_case_object

	@property
	def active_meta(self):
		return self.__active_case_entry

	@property
	def active_frame_name(self):
		if self.active_case is not None:
			return self.active_case.physics.frame.name
		else:
			raise ValueError('no active acse')

	@property
	def available_cases(self):
		case_keys = self.DB.get_keys(entry_type=CaseEntry)
		self.__cases = {self.DB.get(k).name: k for k in case_keys}
		return self.__cases.keys()

	def select_case_entry(self, case_name, case_ID=None):
		if len(self.__cases) == 0:
			# load case names from database if not cached
			self.available_cases

		if case_name in self.__cases:
			case_ID = self.__cases[case_name]

		if cdb_util.is_database_pointer(case_ID, CaseEntry):
			return self.DB.get(case_ID)

		raise ValueError(
				'no case found for case name=`{}`, case_ID=`{}`'
				''.format(case_name, case_ID))

	def load_case(self, case_name, case_ID=None, case_entry=None):
		self.close_active_case()

		if isinstance(case_entry, CaseEntry):
			ce = case_entry
		else:
			ce = self.select_case_entry(case_name, case_ID)

		if ce.name in self.__cases:
			self.__active_case_ID = self.__cases[ce.name]
		else:
			self.__active_case_ID = None
		self.__active_case_entry = ce
		self.__active_case_object = self.accessor.load_case(ce)
		return self.active_case

	def save_new_case(self, case, case_name, directory=None, overwrite=False):
		self.close_active_case()
		if case_name in self.__cases:
			raise ValueError('case name `%s` already used' %case_name)

		if self.working_directory is None and directory is None:
			raise ValueError(
					'no directory specified. please call with keyword '
					'`directory`, or specify a default directory by '
					'setting attribute `CaseIO.working_directory`')
		elif directory is None:
			directory = self.working_directory

		self.__active_case_ID = self.accessor.save_case(
				case, case_name, directory, overwrite=overwrite)
		self.__active_case_entry = self.DB.get(self.__active_case_ID)
		self.__active_case_object = case
		self.__active_case_directory = self.FS.join_mkdir(directory, case_name)
		self.__cases[case_name] = self.__active_case_ID

	def save_active_case(self, directory=None, overwrite=False):
		if self.working_directory is None and directory is None:
			raise ValueError(
					'no directory specified. please call with keyword '
					'`directory`, or specify a default directory by '
					'setting attribute `CaseIO.working_directory`')
		elif directory is None:
			directory = self.working_directory

		if self.active_meta is None or self.active_case is None:
			raise ValueError('no active case')
		else:
			cid = self.accessor.update_case_entry(
					self.active_meta, self.active_case, directory,
					overwrite=overwrite, case_ID=self.__active_case_ID)
			self.__cases[self.active_meta.name] = self.__active_case_ID = cid
			self.__active_case_directory = self.FS.join_mkdir(
					directory, self.active_meta.name)

	def rename_active_case(self, name):
		if self.active_case is not None:
			old_name = self.active_meta.name
			self.active_meta.name = name
			self.__cases[name] = self.__cases.pop(old_name)

	def close_active_case(self):
		if self.active_case is not None:
			self.save_active_case(self.__active_case_directory)
		self.__active_case_ID = None
		self.__active_case_object = None
		self.__active_case_entry = None
		self.__active_case_directory = None

	def load_frame(self, frame_name):
		if self.active_meta is None or self.active_case is None:
			raise ValueError('no active case')

		entry = self.active_meta
		case = self.active_case

		if frame_name not in self.active_case.physics.available_frames:
			case.physics.add_dose_frame(
					frame_name,
					dose_frame=self.accessor.load_frame(entry, frame_name))
		case.physics.change_dose_frame(frame_name)

	def load_frame_mapping(self, source_frame, target_frame):
		if self.active_meta is None or self.active_case is None:
			raise ValueError('no active case')

		entry = self.active_meta
		case = self.active_case
		available_mappings = case.physics.available_frame_mappings
		if (source_frame, target_frame) not in available_mappings:
			case.physics.add_frame_mapping(self.accessor.load_frame_mapping(
					entry, source_frame, target_frame))

	def save_solver_cache(self, cache_name, directory=None, overwrite=False):
		if self.working_directory is None and directory is None:
			raise ValueError(
					'no directory specified. please call with keyword '
					'`directory`, or specify a default directory by '
					'setting attribute `CaseIO.working_directory`')
		elif directory is None:
			directory = self.working_directory

		if self.active_meta is None or self.active_case is None:
			raise ValueError('no active case')

		if self.active_case.problem.solver is None:
			raise ValueError('active case has no built solver')

		self.accessor.save_solver_cache(
				self.active_meta, self.active_case.problem.solver,
				cache_name, self.active_frame_name, directory,
				overwrite=overwrite)

	def load_solver_cache(self, cache_name):
		if self.active_meta is None or self.active_case is None:
			raise ValueError('no active case')

		return self.accessor.load_solver_cache(
				self.active_meta, cache_name, self.active_frame_name)

	def save_solution(self, solution_name, directory=None, frame_name=None,
					  **solution_data):
		if self.working_directory is None and directory is None:
			raise ValueError(
					'no directory specified. please call with keyword '
					'`directory`, or specify a default directory by '
					'setting attribute `CaseIO.working_directory`')
		elif directory is None:
			directory = self.working_directory

		if self.active_meta is None or self.active_case is None:
			raise ValueError('no active case')

		if frame_name is None:
			frame_name = self.active_frame_name

		self.accessor.save_solution(
				self.active_meta, frame_name, solution_name, directory,
				**solution_data)

	def load_solution(self, solution_name, frame_name=None):
		if self.active_meta is None or self.active_case is None:
			raise ValueError('no active case')

		if frame_name is None:
			frame_name = self.active_frame_name

		return self.accessor.load_solution(
				self.active_meta.history, frame_name, solution_name)

	def dump_active_to_YAML(self, directory=None, yaml_directory=None,
							overwrite=False):
		self.save_active_case(directory=directory, overwrite=overwrite)
		return self.accessor.write_case_yaml_from_entry(
				self.active_meta, self.__active_case_directory,
				single_document=True, yaml_directory=yaml_directory)

	def case_to_YAML(self, case, case_name, directory=None,
					 yaml_directory=None):
		if self.working_directory is None and directory is None:
			raise ValueError(
					'no directory specified. please call with keyword '
					'`directory`, or specify a default directory by '
					'setting attribute `CaseIO.working_directory`')
		elif directory is None:
			directory = self.working_directory

		return self.accessor.write_case_yaml(
				case, case_name, directory, single_document=True,
				yaml_directory=yaml_directory)

	def YAML_to_case(self, yaml_file):
		self.close_active_case()
		return self.accessor.load_case_yaml(yaml_file)
