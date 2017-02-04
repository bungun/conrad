"""
Define accessor base class :class:`ConradDBAccessor`
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

from conrad.io.schema import cdb_util, DataFragmentEntry
from conrad.io.filesystem import ConradFilesystemBase, LocalFilesystem
from conrad.io.database import ConradDatabaseBase, LocalPythonDatabase

class ConradDBAccessor(object):
	def __init__(self, subaccessors=None, database=None, filesystem=None):
		self.__DB = None
		self.__FS = None
		self.__subaccessors = []
		filter_ = lambda x: isinstance(x, ConradDBAccessor)
		if subaccessors is not None:
			self.__subaccessors += list(filter(filter_, subaccessors))
		self.set_filesystem(filesystem)
		self.set_database(database)
	@property
	def DB(self):
		return self.__DB

	@property
	def FS(self):
		return self.__FS

	def set_filesystem(self, filesystem_interface=None):
		if filesystem_interface is None:
			self.__FS = LocalFilesystem()
		elif isinstance(filesystem_interface, ConradFilesystemBase):
			self.__FS = filesystem_interface
		else:
			raise TypeError(
					'argument `filesystem` must be an object derived '
					'from {}'.format(ConradFilesystemBase))
		for s in self.__subaccessors:
			s.set_filesystem(self.FS)

	def set_database(self, database_interface=None):
		if database_interface is None:
			self.__DB = LocalPythonDatabase()
		elif isinstance(database_interface, ConradDatabaseBase):
			self.__DB = database_interface
		else:
			raise TypeError(
					'argument `database_interface` must be an object derived '
					'from {}'.format(ConradDatabaseBase))
		for s in self.__subaccessors:
			s.set_database(self.DB)

	def record_entry(self, directory, name, data, overwrite=False):
		written = self.FS.write_data(directory, name, data, overwrite)
		if isinstance(written, DataFragmentEntry):
			return self.DB.set_next(written)
		else:
			return written

	def pop_and_record(self, dictionary, key, directory, name_base='',
					  alternate_keys=None, overwrite=False):
		alternate_keys = [] if alternate_keys is None else list(alternate_keys)
		key_options = [key] + alternate_keys

		if isinstance(dictionary, dict):
			unwritten_val = cdb_util.try_keys(dictionary, *key_options)
		else:
			unwritten_val = None

		if name_base in ('', None):
			name = str(key)
		else:
			name = str(name_base) + '_' + str(key)

		return self.record_entry(directory, name, unwritten_val, overwrite)

	def load_entry(self, entry):
		if entry is None:
			return None
		entry = self.DB.get(entry)
		if isinstance(entry, DataFragmentEntry):
			entry = self.FS.read_data(entry)
		if isinstance(entry, dict):
			for k in entry:
				if isinstance(entry[k], str) and self.DB.has_key(entry[k]):
					entry[k] = self.load_entry(entry[k])
		return entry