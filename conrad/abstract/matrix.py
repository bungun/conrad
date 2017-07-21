from conrad.compat import *

import numpy as np
import scipy.sparse as sp

from conrad.defs import vec, sparse_or_dense, CONRAD_MATRIX_TYPES

def csx_slice_compressed(matrix, indices):
	"""
	Slice the rows (CSR) or columns (CSC) of a CS[X] sparse matrix

	Arguments:
		matrix (:class:`scipy.sparse.csr_matrix` or
			:class:`scipy.sparse.csc_matrix`): Matrix to be sliced into
		indices (:obj:`list`): Indices of compressed dimension to
			include in submatrix

	Returns:
		(:class:`scipy.sparse.csr_matrix` or
		:class:`scipy.sparse.csc_matrix`): Submatrix
	"""
	indices = list(indices)

	if isinstance(matrix, sp.csr_matrix):
		m = len(indices)
		n = matrix.shape[1]
	else:
		m = matrix.shape[0]
		n = len(indices)

	val_full = matrix.data
	ind_full = matrix.indices
	ptr_full = matrix.indptr

	ptr_sub = np.hstack((0, np.diff(ptr_full)[indices].cumsum()))
	nnz_sub = ptr_sub[-1]

	val_sub = np.zeros(nnz_sub, dtype=val_full.dtype)
	ind_sub = np.zeros(nnz_sub, dtype=ind_full.dtype)

	for idx_sub, idx_full in enumerate(indices):
		ptr_0s, ptr_1s = ptr_sub[idx_sub], ptr_sub[idx_sub + 1]
		ptr_0f, ptr_1f = ptr_full[idx_full], ptr_full[idx_full + 1]
		val_sub[ptr_0s:ptr_1s] = val_full[ptr_0f:ptr_1f]
		ind_sub[ptr_0s:ptr_1s] = ind_full[ptr_0f:ptr_1f]

	return type(matrix)((val_sub, ind_sub, ptr_sub), shape=(m, n))

def csx_slice_uncompressed(matrix, indices):
	"""
	Slice the columns (CSR) or rows (CSC) of a CS[X] sparse matrix

	Arguments:
		matrix (:class:`scipy.sparse.csr_matrix` or
			:class:`scipy.sparse.csc_matrix`): Matrix to be sliced into
		indices (:obj:`list`): Indices of uncompressed dimension to
			include in submatrix

	Returns:
		(:class:`scipy.sparse.csr_matrix` or
		:class:`scipy.sparse.csc_matrix`): Submatrix

	"""
	indices = list(indices)

	if isinstance(matrix, sp.csr_matrix):
		m = matrix.shape[0]
		n = len(indices)
	else:
		m = len(indices)
		n = matrix.shape[1]

	val_full = matrix.data
	ind_full = matrix.indices
	ptr_full = matrix.indptr

	ptr_sub = np.zeros_like(ptr_full)
	included = np.zeros(matrix.nnz, dtype=bool)
	indices.sort()
	perm_inverse = {}
	for i0, i1 in enumerate(indices):
		perm_inverse[i1] = i0

	for k in xrange(len(ptr_full) - 1):
		ptr0, ptr1 = ptr_full[k], ptr_full[k + 1]
		ind_slice = ind_full[ptr0:ptr1]
		index_iter = iter(indices)
		i_target = next(index_iter)
		for j in np.argsort(ind_slice):
			i_sorted = ind_slice[j]
			while i_sorted > i_target and i_target < indices[-1]:
				i_target = next(index_iter)
			if i_sorted < i_target:
				continue
			elif i_sorted == i_target:
				ptr_sub[k + 1] += 1
				included[ptr0 + j] = True

	ptr_sub[1:] = ptr_sub[1:].cumsum()
	nnz_sub = ptr_sub[-1]
	val_sub = np.zeros(nnz_sub, dtype=val_full.dtype)
	ind_sub = np.zeros(nnz_sub, dtype=ind_full.dtype)

	head = 0
	for ptr, include in enumerate(included):
		if include:
			ind_sub[head] = perm_inverse[ind_full[ptr]]
			val_sub[head] = val_full[ptr]
			head += 1

	return type(matrix)((val_sub, ind_sub, ptr_sub), shape=(m, n))

class SliceCachingMatrix(object):
	def __init__(self, data):
		self.__dim1 = None
		self.__dim2 = None
		self.__data = None
		self.__row_slices = {}
		self.__column_slices = {}
		self.__double_slices = {}

		self.data = data

	def __contains__(self, comparator):
		if not isinstance(comparator, (int, tuple)):
			raise TypeError('comparator must be int or 2-tuple')
		if isinstance(comparator, tuple):
			if not len(comparator) == 2:
				raise TypeError('comparator must be int or 2-tuple')
			lookup = comparator[0]
			comparator = comparator[1]
		else:
			lookup = 'row'

		if lookup == 'both':
			return comparator in self.__double_slices
		elif lookup == 'column':
			return comparator in self.__column_slices
		else:
			return comparator in self.__row_slices

	@property
	def contiguous(self):
		return self.data is not None

	@property
	def row_dim(self):
		return self.__dim1

	@property
	def column_dim(self):
		return self.__dim2

	@property
	def shape(self):
		if self.__dim1 is None or self.__dim2 is None:
			return None
		else:
			return self.__dim1, self.__dim2

	@property
	def data(self):
		return self.__data

	def _preprocess_data(self, data):
		return data

	def __shape_check(self, shape):
		if self.shape is not None and self.shape != shape:
			raise ValueError(
					'input matrix shape does not match known shape '
					'of {}'.format(SliceCachingMatrix))

	@data.setter
	def data(self, data):
		data = self._preprocess_data(data)
		if isinstance(data, dict):
			labeled_by = data.pop('labeled_by', 'rows')
			data_contiguous = data.pop('contiguous', None)
			if data_contiguous is not None:
				self.data = data_contiguous
				if len(data) == 0:
					return

			if labeled_by not in ('rows', 'columns'):
				raise ValueError(
						'when data provided as a dictionary of '
						'matrices, the optional dictionary entry '
						'`labeled_by` must be one of `columns` or `rows` '
						'(default)')
			if not all(sparse_or_dense(m) for m in data.values()):
				raise TypeError(
						'when data provided as a dictionary of '
						'matrices, each value must be one of the '
						'following matrix types: {}'
						''.format(CONRAD_MATRIX_TYPES))
			if labeled_by == 'columns':
				for mat in data.values():
					rows = mat.shape[0]
					break
				if not all([m.shape[0] == rows for m in data.values()]):
					raise ValueError(
							'all submatrices must have consistent '
							'number of rows when a dictionary of '
							'horizontally concatenable matrices is '
							'provided')
				columns = sum([m.shape[1] for m in data.values()])
			else:
				rows = sum([m.shape[0] for m in data.values()])
				for mat in data.values():
					columns = mat.shape[1]
					break
				if not all([m.shape[1] == columns for m in data.values()]):
					raise ValueError(
							'all submatrices must have consistent '
							'number of columns when a dictionary of '
							'vertically concatenable matrices is '
							'provided')

			self.__shape_check((rows, columns))
			self.__dim1, self.__dim2 = rows, columns
			if labeled_by == 'columns':
				self.__column_slices.update(data)
			else:
				self.__row_slices.update(data)
		else:
			if not sparse_or_dense(data):
				raise TypeError(
						'when data provided as a singleton matrix, '
						'it must be formatted as one of the following '
						'matrix types {}'.format(CONRAD_MATRIX_TYPES))
			self.__shape_check(data.shape)
			self.__dim1, self.__dim2 = data.shape
			self.__data = data

	@staticmethod
	def __row_slice_generic(data, indices):
		if indices is None:
			raise ValueError(
					'argument `indices` cannot be `None` if requesting '
					'an uncached slice')

		if isinstance(data, np.ndarray):
			return data[indices, :]
		elif isinstance(data, sp.csr_matrix):
			return csx_slice_compressed(data, indices)
		else:
			return csx_slice_uncompressed(data, indices)

	def row_slice(self, label, indices):
		if label in self.__row_slices:
			return self.__row_slices[label]
		if self.data is None:
			raise AttributeError(
					'unified matrix for all rows not set/built, '
					'row slicing by label not possible. Either '
					'set `{}.data` or call `{}`.row_assemble if '
					'component matrices already provided'
					''.format(SliceCachingMatrix, SliceCachingMatrix))
		else:
			if callable(indices):
				indices = indices(label)
			self.__row_slices[label] = self.__row_slice_generic(
					self.data, indices)
			return self.__row_slices[label]

	def row_assemble(self, labels):
		raise NotImplementedError

	@staticmethod
	def __column_slice_generic(data, indices):
		if indices is None:
			raise ValueError(
					'argument `indices` cannot be `None` if requesting '
					'an uncached slice')
		if isinstance(data, np.ndarray):
			return data[:, indices]
		elif isinstance(data, sp.csr_matrix):
			return csx_slice_uncompressed(data, indices)
		else:
			return csx_slice_compressed(data, indices)

	def column_slice(self, label, indices):
		if label in self.__column_slices:
			return self.__column_slices[label]
		if self.data is None:
			raise AttributeError(
					'unified matrix for all columns not set/built, '
					'column slicing by label not possible. Either '
					'set `{}.data` or call `{}`.column_assemble if '
					'component matrices already provided'
					''.format(SliceCachingMatrix, SliceCachingMatrix))
		else:
			if callable(indices):
				indices = indices(label)
			self.__column_slices[label] = self.__column_slice_generic(
					self.data, indices)
			return self.__column_slices[label]

	def column_assemble(self, labels):
		raise NotImplementedError

	def slice(self, row_label=None, column_label=None,
			  row_indices=None, column_indices=None):
		if row_label is None and column_label is None:
			raise ValueError(
					'at least one of arguments `row_label` and '
					'`column_label` must not be `None`')
		elif row_label is not None and column_label is None:
			return self.row_slice(row_label, row_indices)
		elif row_label is None and column_label is not None:
			return self.column_slice(column_label, column_indices)
		else:
			key = (row_label, column_label)
			if key in self.__double_slices:
				# return precomputed (row, column)labeled submatrix, if cached
				return self.__double_slices[key]
			elif row_label in self.__row_slices:
				# use precomputed row-labeled submatrix, if cached
				if callable(column_indices):
					column_indices = column_indices(column_label)
				self.__double_slices[key] = self.__column_slice_generic(
						self.__row_slices[row_label], column_indices)
			elif column_label in self.__column_slices:
				# use precomputed column-labeled submatrix, if cached
				if callable(row_indices):
					row_indices = row_indices(row_label)
				self.__double_slices[key] = self.__row_slice_generic(
						self.__column_slices[column_label], row_indices)
			else:
				if isinstance(self.data, (np.ndarray, sp.csr_matrix)):
					slice1 = self.row_slice
					label = row_label
					indices1 = row_indices
					slice2 = self.__column_slice_generic
					indices2 = column_indices
				else:
					slice1 = self.column_slice
					label = column_label
					indices1 = column_indices
					slice2 = self.__row_slice_generic
					indices2 = row_indices

				self.__double_slices[key] = slice2(
						slice1(label, indices1), indices2)
			return self.__double_slices[key]

	@property
	def __cached_slices(self):
		return {
				'row': self.__row_slices.keys(),
				'column': self.__column_slices.keys(),
				'both': self.__double_slices.keys(),
		}

	@property
	def cached_slices(self):
		return self.__cached_slices

	@property
	def __manifest(self):
		manifest = {}
		if self.data is not None:
			manifest['contiguous'] = self.data
		if len(self.__row_slices) > 0:
			manifest['labeled_by'] = 'rows'
			manifest.update(self.__row_slices)
		elif len(self.__column_slices) > 0:
			manifest['labeled_by'] = 'columns'
			manifest.update(self.__column_slices)
		if len(manifest) == 0:
			raise ValueError(
					'{} not exportable as manifest: full matrix or '
					'major axis slices not set')
		return manifest

	@property
	def manifest(self):
		return self.__manifest
