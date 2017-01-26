"""
TOOO: DOCSTRING
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

import json, yaml
import abc
import os
import operator as op
from conrad.defs import is_vector, sparse_or_dense
from conrad.io.schema import *

class ConradDatabaseBase(ConradDatabaseSuper):
	__metaclass__ = abc.ABCMeta

	def __init__(self, dictionary=None, yaml_file=None, **args):
		ConradDatabaseSuper.__init__(self)
		if dictionary is not None:
			self.ingest_dictionary(dictionary)

		if yaml_file is not None:
			self.ingest_yaml(yaml_file)

	@staticmethod
	def raw_data_to_entry(data, allow_unsafe=False):
		if isinstance(data, ConradDatabaseEntry):
			return data
		if not isinstance(data, dict):
			raise TypeError(
					'raw data must be supplied as {}'.format(dict))

		if allow_unsafe:
			data = cdb_util.route_data_fragment(data)
			return ConradDatabaseBase.raw_data_to_entry(data)

		if not CONRAD_DB_TYPETAG in data:
			raise ValueError(
					'raw data must contain key-value pair `{}: '
					'<valid conrad database typestring>`'
					''.format(CONRAD_DB_TYPETAG))
		data_type = data[CONRAD_DB_TYPETAG]
		if data_type not in CONRAD_DB_TYPESTRING_TO_CONSTRUCTOR:
			raise ValueError('unknown data type {}'.format(data_type))

		return CONRAD_DB_TYPESTRING_TO_CONSTRUCTOR[data_type](**data)

	@staticmethod
	def join_key(data_type, key):
		if data_type in CONRAD_DB_ENTRY_PREFIXES:
			prefix = CONRAD_DB_ENTRY_PREFIXES[data_type]
		elif data_type in CONRAD_DB_ENTRY_TYPES.values():
			for k in CONRAD_DB_ENTRY_TYPES:
				if data_type == CONRAD_DB_ENTRY_TYPES[k]:
					prefix = CONRAD_DB_ENTRY_PREFIXES[k]
					break
		else:
			raise TypeError(
					'type `{}` does not correspond to a known ConRad '
					'database entry type')

		key = str(key)
		if key.startswith(prefix):
			return key
		else:
			return prefix + key

	@staticmethod
	def type_from_key(key):
		key = str(key)
		for k, prefix in CONRAD_DB_ENTRY_PREFIXES.items():
			if key.startswith(prefix):
				return CONRAD_DB_ENTRY_TYPES[k]
		return None

	@abc.abstractproperty
	def logged_entries(self):
		return NotImplemented

	@abc.abstractmethod
	def next_available_key(self, data_type):
		return NotImplemented

	@abc.abstractmethod
	def get(self, key):
		return NotImplemented

	@abc.abstractmethod
	def get_keys(self, entry_type=None):
		return NotImplemented

	@abc.abstractmethod
	def has_key(self, key):
		return NotImplemented

	@abc.abstractmethod
	def set(self, key, value, overwrite=False):
		return NotImplemented

	@abc.abstractmethod
	def set_next(self, value):
		return NotImplemented

	@abc.abstractmethod
	def ingest_dictionary(self, dictionary):
		return NotImplemented

	@abc.abstractmethod
	def dump_to_dictionary(self, logged_entries_only=False):
		return NotImplemented

	def ingest_yaml(self, yaml_file):
		if not os.path.exists(yaml_file):
			raise ValueError('file `{}` not located'.format(yaml_file))

		if yaml_file.endswith(('.yml', '.YML', '.yaml', '.YAML')):
			f = open(yaml_file)

			# safe_load_all in case yaml file contains multiple documents
			contents = yaml.safe_load_all(f)

			# process generator returned by safe_load_all
			dictionary = {}
			for c in contents:
				if isinstance(c, dict):
					dictionary.update(c)
			f.close()

			return self.ingest_dictionary(dictionary)
		else:
			raise ValueError(
					'input file must be YAML-formatted')

	def dump_to_yaml(self, yaml_file, logged_entries_only=False,
					 overwrite_file=False):
		dictionary = self.dump_to_dictionary(logged_entries_only)
		if not yaml_file.endswith(('.yml', '.YML', '.yaml', '.YAML')):
				yaml_file += '.yaml'
		if not os.path.exists(os.path.dirname(yaml_file)):
			raise ValueError(
					'cannot save file in directory {}'
					''.format(os.path.dirname(yaml_file)))

		yaml_docs = [{k: dictionary[k]} for k in dictionary]

		arg = 'w' if overwrite_file else 'a'
		f = open(yaml_file, arg)
		f.write(yaml.safe_dump_all(yaml_docs, default_flow_style=False))
		f.close()
		return yaml_file

	def clear_log(self):
		return NotImplemented

class LocalPythonDatabase(ConradDatabaseBase):
	def __init__(self, dictionary=None, yaml_file=None):
		# TRANSACTION LOG
		self.__log = []

		# TABLES
		self.__data_fragments = {}
		self.__frames = {}
		self.__frame_mappings = {}
		self.__solutions = {}
		self.__histories = {}
		self.__solver_caches = {}
		self.__physics_instances = {}
		self.__structures = {}
		self.__anatomies = {}
		self.__cases = {}

		self.__tables = {
				CONRAD_DB_ENTRY_TYPES[VectorEntry]: self.__data_fragments,
				CONRAD_DB_ENTRY_TYPES[
						DenseMatrixEntry]: self.__data_fragments,
				CONRAD_DB_ENTRY_TYPES[
						SparseMatrixEntry]: self.__data_fragments,
				CONRAD_DB_ENTRY_TYPES[
						DataDictionaryEntry]: self.__data_fragments,
				CONRAD_DB_ENTRY_TYPES[DoseFrameEntry]: self.__frames,
				CONRAD_DB_ENTRY_TYPES[
						DoseFrameMappingEntry]: self.__frame_mappings,
				CONRAD_DB_ENTRY_TYPES[
						PhysicsEntry]: self.__physics_instances,
				CONRAD_DB_ENTRY_TYPES[SolutionEntry]: self.__solutions,
				CONRAD_DB_ENTRY_TYPES[HistoryEntry]: self.__histories,
				CONRAD_DB_ENTRY_TYPES[
						SolverCacheEntry]: self.__solver_caches,
				CONRAD_DB_ENTRY_TYPES[StructureEntry]: self.__structures,
				CONRAD_DB_ENTRY_TYPES[AnatomyEntry]: self.__anatomies,
				CONRAD_DB_ENTRY_TYPES[CaseEntry]: self.__cases,
		}


		ConradDatabaseBase.__init__(
				self, dictionary=dictionary, yaml_file=yaml_file)

	def next_available_key(self, data_type):
		if isinstance(data_type, type) and issubclass(
				data_type, ConradDatabaseEntry):
			data_type = CONRAD_DB_ENTRY_TYPES[data_type]
		if data_type not in self.__tables:
			raise ValueError(
					'data type {} unrecognized, no key generated'
					''.format(data_type))

		table = self.__tables[data_type]
		key = len(table)
		while self.join_key(data_type, key) in table:
			key += 1
		return self.join_key(data_type, key)

	def set(self, key, value, overwrite=False):
		if value is None:
			return None
		elif not isinstance(value, ConradDatabaseEntry):
			pass
			raise TypeError(
					'value `{}` does not correspond to a known '
					'ConRad database entry type, could not store'
					''.format(value))
		data_type = CONRAD_DB_ENTRY_TYPES[type(value)]
		key = self.join_key(data_type, key)

		if key in self.__tables[data_type] and not overwrite:
			raise KeyError(
					'key `{}` already used for database type {}. '
					'use flag `overwrite=True` to overwrite'
					''.format(key, data_type))
		self.__tables[data_type][key] = value
		self.__log.append(key)
		return key

	def set_next(self, value):
		if value is None:
			return None
		elif not isinstance(value, ConradDatabaseEntry):
			raise TypeError(
					'value `{}` does not correspond to a known '
					'ConRad database entry type, could not store'
					''.format(value))
		key = self.next_available_key(type(value))
		return self.set(key, value)

	def has_key(self, key):
		return any(map(lambda t: key in t, self.__tables.values()))

	def get(self, key):
		data_type = self.type_from_key(key)

		if data_type in self.__tables:
			table = self.__tables[data_type]
			if key not in table:
				raise KeyError(
						'key `{}` does not correspond to a value in '
						'the table for {} entries in the ConRad '
						'database'.format(key, data_type))
			raw_data = table[key]
		else:
			raw_data = key

		if raw_data is None:
			return None
		elif isinstance(raw_data, (dict, ConradDatabaseEntry)):
				return self.raw_data_to_entry(raw_data, allow_unsafe=True)
		else:
			raise ValueError(
					'no corresponding database table/entry found for '
					'key `{}`'.format(key))

	def get_keys(self, entry_type=None):
		if entry_type is None:
			return reduce(
					operator.add, [t.keys() for t in self.__tables.values()])
		if issubclass(entry_type, ConradDatabaseEntry):
			return self.__tables[CONRAD_DB_ENTRY_TYPES[entry_type]].keys()
		if entry_type in self.__tables:
			return self.__tables[entry_type].keys()
		else:
			raise ValueError(
					'entry type `{}` does not correspond to a ConRad '
					'database entry type'.format(entry_type))

	@property
	def logged_entries(self):
		return self.__log

	def clear_log(self):
		self.__log = []

	def ingest_dictionary(self, dictionary):
		if not isinstance(dictionary, dict):
			raise TypeError(
					'argument `dictionary` must be of type {}'
					''.format(dict))

		table_names = [
				'data_fragments', 'frames', 'frame_mappings', 'solutions',
				'histories', 'solver_caches', 'physics_instances',
				'structures', 'anatomies', 'cases']
		for table_name in table_names:
			table = dictionary.pop(table_name, {})
			if isinstance(table, dict):
				for key in table:
					self.set(
							key, self.raw_data_to_entry(table[key]),
							overwrite=True)

	def __dump_table(self, dictionary, logged_entries_only=False, flatten=True):
		table = {}
		final = False

		while not final:
			keys = list(dictionary.keys())
			for k in keys:
				if k in table:
					continue
				elif bool(k in self.__log) >= logged_entries_only:
					entry = dictionary[k]
					if flatten:
						entry.flatten(self)
					table[k] = entry.nested_dictionary
			final = len(keys) == len(dictionary.keys())

		return table

	def dump_to_dictionary(self, logged_entries_only=False, flatten=True):
		return {
				# this order matters due to recursive nature of .flatten() call
				'cases': self.__dump_table(self.__cases),
				'anatomies': self.__dump_table(self.__anatomies),
				'structures': self.__dump_table(self.__structures),
				'physics_instances': self.__dump_table(
						self.__physics_instances),
				'frames': self.__dump_table(self.__frames),
				'frame_mappings': self.__dump_table(self.__frame_mappings),
				'histories': self.__dump_table(self.__histories),
				'solutions': self.__dump_table(self.__solutions),
				'solver_caches': self.__dump_table(self.__solver_caches),
				'data_fragments': self.__dump_table(self.__data_fragments),
		}
