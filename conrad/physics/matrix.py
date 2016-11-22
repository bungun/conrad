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

class WeightVector(object):
	def __init__(self, data):
		self.__size = None
		self.__data = None
		self.__slices = {}

		self.data = data

	@staticmethod
	def __validate_weights(weights):
		if sum(vec(weights) < 0) > 0:
			raise ValueError('contents of weight vectors must be nonnegative')

	def __contains__(self, comparator):
		return comparator in self.__slices

	@property
	def size(self):
		return self.__size

	@property
	def shape(self):
		return self.data.shape if self.data is not None else None

	@property
	def data(self):
		return self.__data

	@data.setter
	def data(self, data):
		if isinstance(data, dict):
			for w in data.values():
				self.__validate_weights(w)
			self.__slices = {l: vec(w).astype(float) for l, w in data.items()}
			self.__size = sum(w.size for w in self.__slices.values())
		else:
			self.__validate_weights(data)
			self.__data = vec(data).astype(float)
			self.__size = self.__data.size

	def slice(self, label, indices=None):
		# calculate and cache slice
		if not label in self:
			if self.data is None:
				raise AttributeError(
						'cannot build slice from weight vector if '
						'`WeightVector.data` is not set' )
			self.__slices[label] = self.data[indices]

		# return cached slice
		return self.__slices[label]

	def assemble(self):
		if len(self.__slices) == 0:
			raise AttributeError('no subvectors to assemble')
		self.__data = np.hstack(self.__slices.values())

class DoseMatrix(object):
	def __init__(self, data):
		self.__dim1 = None
		self.__dim2 = None
		self.__data = None
		self.__voxel_slices = {}
		self.__beam_slices = {}
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
			lookup = 'voxel'

		if lookup == 'both':
			return comparator in self.__double_slices
		elif lookup == 'beam':
			return comparator in self.__beam_slices
		else:
			return comparator in self.__voxel_slices

	@property
	def contiguous(self):
		return self.data is not None

	@property
	def voxel_dim(self):
		return self.__dim1

	@property
	def beam_dim(self):
		return self.__dim2

	@property
	def shape(self):
		return (self.__dim1, self.__dim2)

	@property
	def data(self):
		return self.__data

	@data.setter
	def data(self, data):
		if isinstance(data, dict):
			labeled_by = data.pop('labeled_by', 'voxels')
			if labeled_by not in ('voxels', 'beams'):
				raise ValueError(
						'when data provided as a dictionary of '
						'matrices, the optional dictionary entry '
						'`labeled_by` must be on of `beams` or `voxels` '
						'(default)')
			if not all(sparse_or_dense(m) for m in data.values()):
				raise TypeError(
						'when data provided as a dictionary of '
						'matrices, each value must be one of the '
						'following matrix types: {}'
						''.format(CONRAD_MATRIX_TYPES))
			if labeled_by == 'beams':
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
				self.__beam_slices.update(data)
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
				self.__voxel_slices.update(data)
		else:
			if not sparse_or_dense(data):
				raise TypeError(
						'when data provided as a singleton matrix, '
						'it must be formatted as one of the following '
						'matrix types {}'.format(CONRAD_MATRIX_TYPES))
			self.__data = data
			rows, columns = data.shape
		self.__dim1, self.__dim2 = rows, columns

	@staticmethod
	def __voxel_slice_generic(data, indices):
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

	def voxel_slice(self, label, indices):
		if label in self.__voxel_slices:
			return self.__voxel_slices[label]
		if self.data is None:
			raise AttributeError(
					'unified matrix for all voxels not set/built, '
					'voxel slicing by label not possible. Either '
					'set `{}.data` or call `{}`.voxel_assemble if '
					'component matrices already provided'
					''.format(DoseMatrix, DoseMatrix))
		else:
			self.__voxel_slices[label] = self.__voxel_slice_generic(
					self.data, indices)
			return self.__voxel_slices[label]

	def voxel_assemble(self, labels):
		return NotImplemented

	@staticmethod
	def __beam_slice_generic(data, indices):
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

	def beam_slice(self, label, indices):
		if label in self.__beam_slices:
			return self.__beam_slices[label]
		if self.data is None:
			raise AttributeError(
					'unified matrix for all beams not set/built, '
					'beam slicing by label not possible. Either '
					'set `{}.data` or call `{}`.beam_assemble if '
					'component matrices already provided'
					''.format(DoseMatrix, DoseMatrix))
		else:
			self.__beam_slices[label] = self.__beam_slice_generic(
					self.data, indices)
			return self.__beam_slices[label]

	def beam_assemble(self, labels):
		return NotImplemented

	@property
	def cached_slices(self):
		return {
				'voxel': self.__voxel_slices.keys(),
				'beam': self.__beam_slices.keys(),
				'both': self.__double_slices.keys(),
		}

	def slice(self, voxel_label=None, beam_label=None,
			  voxel_indices=None, beam_indices=None):
		if callable(voxel_indices) and voxel_label is not None:
			voxel_indices = voxel_indices(voxel_label)
		if callable(beam_indices) and beam_label is not None:
			beam_indices = beam_indices(beam_label)

		if voxel_label is None and beam_label is None:
			raise ValueError(
					'at least one of arguments `voxel_label` and '
					'`beam_label` must not be `None`')
		elif voxel_label is not None and beam_label is None:
			return self.voxel_slice(voxel_label, voxel_indices)
		elif voxel_label is None and beam_label is not None:
			return self.beam_slice(beam_label, beam_indices)
		else:
			key = (voxel_label, beam_label)
			if key in self.__double_slices:
				# return precomputed (voxel, beam)labeled submatrix, if cached
				return self.__double_slices[key]
			elif voxel_label in self.__voxel_slices:
				# use precomputed voxel-labeled submatrix, if cached
				self.__double_slices[key] = self.__beam_slice_generic(
						self.__voxel_slices[voxel_label], beam_indices)
			elif beam_label in self.__beam_slices:
				# use precomputed beam-labeled submatrix, if cached
				self.__double_slices[key] = self.__voxel_slice_generic(
						self.__beam_slices[beam_label], voxel_indices)
			else:
				if isinstance(self.data, (np.ndarray, sp.csr_matrix)):
					slice1 = self.voxel_slice
					label = voxel_label
					indices1 = voxel_indices
					slice2 = self.__beam_slice_generic
					indices2 = beam_indices
				else:
					slice1 = self.beam_slice
					label = beam_label
					indices1 = beam_indices
					slice2 = self.__voxel_slice_generic
					indices2 = voxel_indices

				self.__double_slices[key] = slice2(
						slice1(label, indices1), indices2)
			return self.__double_slices[key]