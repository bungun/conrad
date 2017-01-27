"""
Define :class:`ConradFilesystemBase` and :class:`LocalFilesystem` for
loading and saving treatment planning cases.
"""
"""
# Copyright 2016 Baris Ungun, Anqi Fu

# This file is part of CONRAD.

# CONRAD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# CONRAD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CONRAD.  If not, see <http://www.gnu.org/licenses/>.
"""
from conrad.compat import *

import os
import abc
import numpy as np
import scipy.sparse as sp

from conrad.defs import sparse_or_dense, CONRAD_MATRIX_TYPES
from conrad.case import Case
from conrad.medicine import Anatomy, Structure, Prescription
from conrad.io.schema import *

class ConradFilesystemBase(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self):
		self.__DIGEST = {
				int : lambda number: number,
				float : lambda number: number,
				str : lambda string: string,
				dict: lambda dictionary: dictionary,
				type(None) : lambda none_var: None,
				DataDictionaryEntry : self.to_data_dictionary,
				VectorEntry : self.to_vector,
				DenseMatrixEntry : self.to_dense_matrix,
				SparseMatrixEntry : self.to_sparse_matrix,
				UnsafeFileEntry : self.to_unsafe_data,
		}

		self.__DUMP = {
				int : lambda directory, name, value, overwrite: value,
				float : lambda directory, name, value, overwrite: value,
				str : lambda directory, name, value, overwrite: value,
				type(None) : lambda directory, name, value, overwrite: value,
				dict : self.write_data_dictionary,
				np.ndarray : self.write_ndarray,
				sp.csr_matrix : self.write_sparse_matrix,
				sp.csc_matrix : self.write_sparse_matrix,
		}

	@abc.abstractmethod
	def check_dir(self, directory):
		raise NotImplementedError

	@abc.abstractmethod
	def join_mkdir(self, directory, *subdir):
		raise NotImplementedError

	@abc.abstractmethod
	def read(self, file, key):
		raise NotImplementedError

	@abc.abstractmethod
	def read_all(self, file):
		raise NotImplementedError

	@abc.abstractmethod
	def write(self, file, data, overwrite=False):
		raise NotImplementedError

	def read_data(self, data_fragment_entry):
		data_fragment_entry = cdb_util.route_data_fragment(data_fragment_entry)
		if type(data_fragment_entry) not in self.__DIGEST:
			raise TypeError(
					'no read method for data of type {}'
					''.format(type(data_fragment_entry)))
		return self.__DIGEST[type(data_fragment_entry)](data_fragment_entry)

	def write_data(self, directory, name, data, overwrite=False):
		if type(data) not in self.__DUMP:
			raise TypeError(
					'no write method for data of type {}'
					''.format(type(data)))
		return self.__DUMP[type(data)](directory, name, data, overwrite)

	def to_unsafe_data(self, unsafe_file_entry):
		if isinstance(unsafe_file_entry, dict):
			unsafe_file_entry = UnsafeFileEntry(**unsafe_file_entry)
		if not isinstance(unsafe_file_entry, UnsafeFileEntry):
			raise TypeError(
					'input should be of type (or parsable as) {}'
					''.format(UnsafeFileEntry))
		if not unsafe_file_entry.complete:
			raise ValueError(
					'no numpy file assigned to {}'.format(UnsafeFileEntry))
		typed_data = self.read_all(unsafe_file_entry.file)
		if isinstance(typed_data, np.ndarray):
			return typed_data
		if sparse_or_dense(typed_data):
			return typed_data
		if isinstance(typed_data, dict):
			sme = SparseMatrixEntry(**typed_data)
			if sme.complete:
				return self.to_sparse_matrix(sme)
			dme = DenseMatrixEntry(**typed_data)
			if dme.complete:
				return self.to_dense_matrix(dme)
			ve = VectorEntry(**typed_data)
			if ve.complete:
				return self.to_vector(ve)
		return typed_data

	def to_data_dictionary(self, data_dictionary_entry):
		if isinstance(data_dictionary_entry, dict):
			data_dictionary_entry = DataDictionaryEntry(**data_dictionary_entry)
		if not isinstance(data_dictionary_entry, DataDictionaryEntry):
			raise TypeError(
					'input should be of type (or parsable as) {}'
					''.format(DataDictionaryEntry))
		if not data_dictionary_entry.complete:
			raise ValueError(
					'data incomplete, could not form dictionary\n\n'
					'input:\n{}'
					''.format(data_dictionary_entry.nested_dictionary))
		return {
				k: self.read_data(data_dictionary_entry.entries[k]) for k in
				data_dictionary_entry.entries
		}

	def to_vector(self, vector_entry):
		if isinstance(vector_entry, dict):
			vector_entry = VectorEntry(**vector_entry)
		if not isinstance(vector_entry, VectorEntry):
			raise TypeError(
					'input should be of type (or parsable as) {}'
					''.format(VectorEntry))
		if not vector_entry.complete:
			raise ValueError(
					'data incomplete, could not form vector\n\ninput:\n'
					'{}'.format(vector_entry.nested_dictionary))
		return self.read(vector_entry.data_file, vector_entry.data_key)

	def to_dense_matrix(self, dense_matrix_entry):
		if isinstance(dense_matrix_entry, dict):
			dense_matrix_entry = DenseMatrixEntry(**dense_matrix_entry)

		if not isinstance(dense_matrix_entry, DenseMatrixEntry):
			raise TypeError(
					'input should be of type (or parsable as) {}'
					''.format(DenseMatrixEntry))

		if not dense_matrix_entry.complete:
			raise ValueError(
					'data incomplete, could not form dense matrix\n\n'
					'input:\n{}'
					''.format(dense_matrix_entry.nested_dictionary))
		data = self.read(
				dense_matrix_entry.data_file, dense_matrix_entry.data_key)
		order = 'C' if dense_matrix_entry.layout_rowmajor else 'F'
		return np.array(data, order=order)

	def to_sparse_matrix(self, sparse_matrix_entry):
		sm_entry = sparse_matrix_entry
		if isinstance(sm_entry, dict):
			sm_entry = SparseMatrixEntry(**sm_entry)
		if not isinstance(sm_entry, SparseMatrixEntry):
			raise TypeError(
					'input should be of type (or parsable as) {}'
					''.format(SparseMatrixEntry))
		if not sm_entry.complete:
			raise ValueError(
					'data incomplete, could not form sparse matrix\n\n'
					'input:\n{}'.format(sm_entry.nested_dictionary))
		constructor = sp.csr_matrix if sm_entry.layout_CSR else \
					  sp.csc_matrix
		values = self.read(sm_entry.data_values_file, sm_entry.data_values_key)
		indices = self.read(
				sm_entry.data_indices_file, sm_entry.data_indices_key)
		pointers = self.read(
				sm_entry.data_pointers_file, sm_entry.data_pointers_key)
		if sm_entry.layout_fortran_indexing:
			indices -=1
			pointers -=1

		return constructor((values, indices, pointers), shape=sm_entry.shape)

	def write_data_dictionary(self, directory, name, dictionary,
							  overwrite=False):
		saveable_type = lambda o: isinstance(o, CONRAD_MATRIX_TYPES)
		if any(map(saveable_type, dictionary.values())):
			dd = DataDictionaryEntry()
			dd.entries = {k: self.write_data(
					directory, name + '_' + str(k), v, overwrite)
					for k, v in dictionary.items()}
			return dd
		else:
			return dictionary

	def write_vector(self, directory, name, vector, overwrite=False):
		if not isinstance(vector, np.ndarray) or not len(vector.shape) == 1:
			raise TypeError(
					'vector to be written must be a 1-D {}'.format(np.ndarray))
		ddict = self.write(
						os.path.join(directory, name), vector, overwrite)
		return VectorEntry(**{
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[VectorEntry],
				'data': ddict,
		})

	def write_dense_matrix(self, directory, name, matrix, overwrite=False):
		if not isinstance(matrix, np.ndarray) or not len(matrix.shape) == 2:
			raise TypeError(
					'matrix to be written must be a 2-D {}'.format(np.ndarray))

		return DenseMatrixEntry(**{
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[DenseMatrixEntry],
				'layout_rowmajor': matrix.flags.c_contiguous,
				'data': self.write(
						os.path.join(directory, name), matrix, overwrite)
		})

	def write_ndarray(self, directory, name, array, overwrite=False):
		if not isinstance(array, np.ndarray):
			raise TypeError(
					'array to be written must be of type {}'
					''.format(np.ndarray))
		if len(array.shape) == 1:
			return self.write_vector(directory, name, array, overwrite)
		elif len(array.shape) == 2:
			return self.write_dense_matrix(directory, name, array, overwrite)
		else:
			raise ValueError(
					'array to be written must be 1-D or 2-D')

	def write_sparse_matrix(self, directory, name, matrix, overwrite=False):
		if not isinstance(matrix, (sp.csr_matrix, sp.csc_matrix)):
			raise TypeError(
					'sparse matrix to be written must be CSC/CSR '
					'formatted')
		return SparseMatrixEntry(**{
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[SparseMatrixEntry],
				'layout_CSR': isinstance(matrix, sp.csr_matrix),
				'layout_fortran_indexing': False,
				'shape': matrix.shape,
				'data': {
						'pointers': self.write(
								os.path.join(directory, name + '_pointers'),
								matrix.indptr, overwrite),
						'indices': self.write(
									os.path.join(directory, name + '_indices'),
									matrix.indices, overwrite),
						'values': self.write(
								os.path.join(directory, name + '_values'),
								matrix.data, overwrite)
				},
		})

	def write_matrix(self, directory, name, matrix, overwrite=False):
		if not sparse_or_dense(matrix):
			raise TypeError(
					'matrix to save must be one of {}'
					''.format(CONRAD_MATRIX_TYPES))
		if isinstance(matrix, np.ndarray):
			return self.write_dense_matrix(directory, name, matrix, overwrite)
		else:
			return self.write_sparse_matrix(directory, name, matrix, overwrite)

class LocalFilesystem(ConradFilesystemBase):
	def check_dir(self, directory):
		if not os.path.exists(directory):
			raise OSError('path {} does not exist'.format(directory))

	def join_mkdir(self, directory, *subdir):
		d = directory
		for s in subdir:
			if isinstance(s, str):
				if not s in d:
					d = os.path.join(d, s)
				if not os.path.exists(d):
					os.mkdir(d)
		return d

	def read(self, file, key=None):
		file = str(file)
		if not os.path.exists(file):
			raise OSError('file {} does not exist'.format(file))
		if file.endswith('.npy'):
			return np.load(file)
		elif file.endswith('.npz'):
			if key is None:
				raise ValueError('no key provided for `.npz` file')
			return np.load(filename)[key]
		elif file.endswith('.txt'):
			return np.loadtxt(filename)
		else:
			raise ValueError('file extension must be one of {}'.format(
							('.npz', '.npy', '.txt')))

	def read_all(self, file):
		file = str(file)
		if not os.path.exists(file):
			raise OSError('file {} does not exist'.format(file))
		if file.endswith(('.txt', '.npy')):
			return self.read(file)
		else:
			data = {}
			repository = np.load(filename)
			for k in repository.files:
				data[k] = self.read(str(k) + '.npy')
			return data

	def write(self, file, data, overwrite=False):
		file = str(file)
		extension = '.npz' if isinstance(data, dict) else '.npy'
		if not file.endswith(extension):
			file += extension

		if os.path.exists(file) and not overwrite:
			raise OSError('file `{}` exists; please specify keyword '
						  'argument `overwrite=True` to proceed with '
						  'save operation'.format(file))

		if isinstance(data, dict):
			np.savez(file, **data)
			return {key: {'file': file, 'key': key} for key in data}
		else:
			np.save(file, data)
			return {'file': file, 'key': None}