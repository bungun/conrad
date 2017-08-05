"""
Define classes describing permutation and clustering relations.
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

import numpy as np
import abc

from conrad.defs import vec, is_vector, sparse_or_dense
from conrad.abstract.vector import SliceCachingVector
from conrad.abstract.matrix import SliceCachingMatrix

# TODO: Change module to maps?
# TODO: Change this to DiscreteMap?
# TODO: left and right application to matrices
# TODO: HierarchicalMapping type or recursive capabilities for
#	DictionaryMapping
@add_metaclass(abc.ABCMeta)
class AbstractDiscreteMapping(object):
	@abc.abstractproperty
	def vec(self):
		raise NotImplementedError

	@abc.abstractproperty
	def manifest(self):
		raise NotImplementedError

	@abc.abstractproperty
	def n_frame0(self):
		raise NotImplementedError

	@abc.abstractproperty
	def n_frame1(self):
		raise NotImplementedError

	@property
	def shape(self):
		return (self.n_frame0, self.n_frame1)

	@abc.abstractmethod
	def frame0_to_1_inplace(self, in_, out_, clear_output=False, **options):
		raise NotImplementedError

	@abc.abstractmethod
	def frame0_to_1(self, in_, **options):
		raise NotImplementedError

	@abc.abstractmethod
	def frame1_to_0_inplace(self, in_, out_, clear_output=False, **options):
		raise NotImplementedError

	@abc.abstractmethod
	def frame1_to_0(self, in_, **options):
		raise NotImplementedError

class DiscreteMapping(AbstractDiscreteMapping):
	r"""
	Generic description of relations between two discrete, finite sets.

	In particular, the expected use is for mapping components from
	one vector space to the components of another, and vice-versa.

	Each :class:`DiscreteMapping` has a 'forward' sense, taken to be a
	one-to-one or many-to-one relation, while one-to-many relations are
	through the 'reverse' sense application of the mapping.

	The (forward) mapping can also be viewed as a linear
	transformation from

	:math: R^n --> R^m, \mbox{where},
	:math: n = |\mbox{set} 0| \mbox{and} m = |\mbox{set} 1|.

	The reverse mapping is the inverse of this transformation (or at
	least related to it by a second diagonal transformation).
	"""
	def __init__(self, map_vector, target_dimension=None):
		r"""
		Initialize a one-to-one or many-to-one discrete relation.

		The size of the first set/vector space is taken to be the
		length of ``map_vector``, and the size of the second set/vector
		space is taken to be implied by the (base-``0``) value of the
		largest entry in ``map_vector``.

		Arguments:
			map_vector: Vector-like array of :obj:`int`, representing
				mapping. Let v be the input vector. Then the relation
				maps the i'th entry of the first set to the v[i]'th
				entry of the second set.
		"""
		self.__forwardmap = vec(map_vector).astype(int)
		self.__n_frame0 = len(self.__forwardmap)
		self.__n_frame1 = self.__forwardmap.max() + 1

	@property
	def vec(self):
		""" Vector representation of forward mapping. """
		return self.__forwardmap

	@property
	def manifest(self):
		return {'data': self.vec, 'type': map_type_to_string(self)}

	def __getitem__(self, index):
		return self.vec[index]

	@property
	def n_frame0(self):
		""" Number of elements in first frame/discrete set. """
		return self.__n_frame0

	@property
	def n_frame1(self):
		""" Number of elements in second frame/discrete set. """
		return self.__n_frame1

	def frame0_to_1_inplace(self, in_, out_, clear_output=False, **options):
		"""
		Map elements of array ``in_`` to elements of array ``out_``.

		Procedure::
			# let forward_map be the map: SET_0 --> SET_1.
			# for each INDEX_0, INDEX_1 in forward_map, do
			#	out_[INDEX_1] += in_[INDEX_0]
			# end for

		If arrays are matrices, mapping operates on rows. If arrays are
		vectors, mapping operates on entries.

		Arguments:
			in_: Input array with :attr:`DiscreteMapping.n_frame0` rows
				and ``k`` >= ``1`` columns.
			out_: Output array with :attr:`DiscreteMapping.n_frame1`
				rows and ``k`` >= 1 columns. Modified in-place.
			clear_output (:obj:`bool`, optional): If ``True``, set
				output array to ``0`` before adding input values.

		Returns:
			Vector `out_`, after in-place modification.

		Raises:
			TypeError: If input and output arrays are not (jointly)
				vectors or matrices.
			ValueError: If input and output array dimensions are not
				compatible with each other or consistent with the
				dimensions of the mapping.
		"""
		vector_processing = is_vector(in_) and is_vector(out_)
		matrix_processing = sparse_or_dense(in_) and sparse_or_dense(out_)
		apply_from_left = apply_from_left=options.pop('apply_from_left', True)
		if not (vector_processing or matrix_processing):
			raise TypeError('arguments "in_" and "out_" be numpy or '
							'scipy vectors or matrices')

		if vector_processing:
			dim_in1, dim_in2 = in_.size, 1
			dim_out1, dim_out2 = out_.size, 1
		else:
			dim_in1, dim_in2 = in_.shape
			dim_out1, dim_out2 = out_.shape

		if apply_from_left:
			expectA = 'left'
			expectB = 'M x N, and K x N'
			expectC = '\nM = {}\nK = {}'.format(self.n_frame0, self.n_frame1)
			dimension_match = dim_in1 == self.n_frame0
			dimension_match &= dim_out1 == self.n_frame1
			dimension_match &= dim_in2 == dim_out2
		else:
			expectA = 'right'
			expectB = 'M x N, and M x K'
			expectC = '\nN = {}\nK = {}'.format(self.n_frame0, self.n_frame1)
			dimension_match = dim_in2 == self.n_frame0
			dimension_match &= dim_out2 == self.n_frame1
			dimension_match &= dim_in1 == dim_out1

		if not dimension_match:
			raise ValueError('to apply mapping transformation from {}, '
							 'arguments "in_" and "out_" be vectors or '
							 'matrices of dimensions {}, '
							 'respectively, with:\n'
							 'Provided:\n input: {}x{}\noutput: {}x{}'
							 ''.format(expectA, expectB, expectC,
							 dim_in1, dim_in2, dim_out1, dim_out2))
		if clear_output:
			out_ *= 0

		if apply_from_left:
			for idx_0, idx_1 in enumerate(self.vec):
				out_[idx_1, ...] += in_[idx_0, ...]
		else:
			for idx_0, idx_1 in enumerate(self.vec):
				out_[..., idx_1] += in_[..., idx_0]

		# TODO: use efficient slicing when input is sparse. warn when output sparse??
		return out_

	def frame0_to_1(self, in_, **options):
		"""
		Map elements of array ``in_`` to a new array.

		If input array is a matrix, mapping operates on rows. If array
		is a vector, mapping operates on entries.

		A new empty vector or matrix

		Arguments:
			in_: Input array with :attr:`DiscreteMapping.n_frame0` rows
				and ``k`` >= ``1`` columns.

		Returns:
			:class:`numpy.ndarray`: Array with
			:attr:`DiscreteMapping.n_frame1` rows and ``k`` columns.
			Input entries are mapped one-to-one or one-to-many *into*
			output.
		"""
		apply_from_left = options.pop('apply_from_left', True)
		if is_vector(in_):
			out_ = np.zeros(self.__n_frame1)
		elif sparse_or_dense(in_):
			dim1 = self.__n_frame1 if apply_from_left else in_.shape[0]
			dim2 = in_.shape[1] if apply_from_left else self.__n_frame1
			out_ = np.zeros((dim1, dim2))

		return self.frame0_to_1_inplace(
				in_, out_, apply_from_left=apply_from_left, **options)


	def frame1_to_0_inplace(self, in_, out_, clear_output=False, **options):
		"""
		Allocating version of :meth`DiscreteMapping.frame0_to_1`.

		Procedure::
			# let reverse_map be the map: SET_1 --> SET_0.
			# for each INDEX_1, INDEX_0 in forward_map, do
			#	out_[INDEX_0] += in_[INDEX_1]
			# end for

		If arrays are matrices, mapping operates on rows. If arrays are
		vectors, mapping operates on entries.

		Arguments:
			in_: Input array with :attr:`DiscreteMapping.n_frame1` rows
				and ``k`` >= ``1`` columns.
			out_: Output array with :attr:``DiscreteMapping.n_frame0``
				rows and ``k`` >= ``1`` columns. Modified in-place.
			clear_output (:obj:`bool`, optional): If ``True``, set
				output array to ``0`` before adding input values.

		Returns:
			Vector ``out_``, after in-place modification.

		Raises:
			TypeError: If input and output arrays are not (jointly)
				vectors or matrices.
			ValueError: If input and output array dimensions are not
				compatible with each other or consistent with the
				dimensions of the mapping.
		"""
		vector_processing = is_vector(in_) and is_vector(out_)
		matrix_processing = sparse_or_dense(in_) and sparse_or_dense(out_)
		apply_from_left = apply_from_left=options.pop('apply_from_left', True)
		if not (vector_processing or matrix_processing):
			raise TypeError('arguments "in_" and "out_" be numpy or '
							'scipy vectors or matrices')

		if vector_processing:
			dim_in1, dim_in2 = in_.size, 1
			dim_out1, dim_out2 = out_.size, 1
		else:
			dim_in1, dim_in2 = in_.shape
			dim_out1, dim_out2 = out_.shape

		if apply_from_left:
			expectA = 'left'
			expectB = 'K x N, and M x N'
			expectC = '\nK = {}\nM = {}'.format(self.n_frame1, self.n_frame0)
			dimension_match = dim_in1 == self.n_frame1
			dimension_match &= dim_out1 == self.n_frame0
			dimension_match &= dim_in2 == dim_out2
		else:
			expectA = 'right'
			expectB = 'M x K, and M x N'
			expectC = '\nK = {}\nN = {}'.format(self.n_frame1, self.n_frame0)
			dimension_match = dim_in2 == self.n_frame1
			dimension_match &= dim_out2 == self.n_frame0
			dimension_match &= dim_in1 == dim_out1

		if not dimension_match:
			raise ValueError('to apply mapping transformation from {}, '
							 'arguments "in_" and "out_" be vectors or '
							 'matrices of dimensions {}, '
							 'respectively, with:\n'
							 'Provided:\n input: {}x{}\noutput: {}x{}'
							 ''.format(expectA, expectB, expectC,
							 dim_in1, dim_in2, dim_out1, dim_out2))
		if clear_output:
			out_ *= 0

		if apply_from_left:
			for idx_0, idx_1 in enumerate(self.vec):
				out_[idx_0, ...] += in_[idx_1, ...]
		else:
			for idx_0, idx_1 in enumerate(self.vec):
				out_[..., idx_0] += in_[..., idx_1]

		# TODO: use efficient slicing when input is sparse. warn when output sparse??
		return out_

	def frame1_to_0(self, in_, **options):
		"""
		Allocating version of :meth`DiscreteMapping.frame1_to_0`.

		If input array is a matrix, mapping operates on rows. If array
		is a vector, mapping operates on entries.

		Arguments:
			in_: Input array with :attr:`DiscreteMapping.n_frame0` rows
				and ``k`` >= ``1`` columns.

		Returns:
			:class:`numpy.ndarray`: Array with
			:attr:`DiscreteMapping.n_frame0` rows and same number of
			columns as input. Input entries are mapped one-to-one or
			many-to-one *into* output.
		"""
		apply_from_left = options.pop('apply_from_left', True)
		if is_vector(in_):
			out_ = np.zeros(self.__n_frame0)
		elif sparse_or_dense(in_):
			dim1 = self.__n_frame0 if apply_from_left else in_.shape[0]
			dim2 = in_.shape[1] if apply_from_left else self.__n_frame0
			out_ = np.zeros((dim1, dim2))

		return self.frame1_to_0_inplace(
				in_, out_, apply_from_left=apply_from_left, **options)

class ClusterMapping(DiscreteMapping):
	""" Map ``M`` elements to ``K`` clusters, with ``K`` <= ``M``. """

	def __init__(self, clustering_vector):
		"""
		Initialize as :class`DiscreteMapping` instance.

		Determine cluster sizes by counting instances in which
		``cluster_vector`` names each cluster ``k`` as a target.

		Arguments:
			clustering_vector:  Vector-like array of :obj:`int` mapping
				entry indices to cluster indices. Let c be the input
				vector. Then the relation maps the i'th member of the
				first set to the v[i]'th cluster in the second set.
		"""
		DiscreteMapping.__init__(self, clustering_vector)
		self.__cluster_weights = np.zeros(self.n_clusters)
		for cluster_index in self.vec:
			self.__cluster_weights[cluster_index] += 1.
		self.__empty_clusters = np.sum(self.__cluster_weights == 0) > 0

	@property
	def n_clusters(self):
		""" Number of clusters in target set. """
		return self.n_frame1

	@property
	def n_points(self):
		""" Number of elements in source set. """
		return self.n_frame0

	@property
	def cluster_weights(self):
		""" Number of elements mapped to each cluster. """
		return self.__cluster_weights

	def __rescale_len_points(self, data, apply_from_left=True, **options):
		"""
		Scale input array's entries by corresponding cluster's weight.

		If input array is vector, method acts on entries. If input array
		is matrix, method acts on rows.

		Procedure::
			# Let the entries of vector c give the map:
			#	SET_FULL-->SET_CLUSTERED, and vector w give the cluster
			#	weights.
			#
			# for each i=INDEX_FULL, k=INDEX_CLUSTER in c, do
			#	data[i] *= 1 / w[k]
			# end for

		Arguments:
			data: Array to transform, with
				:attr:`ClusterMapping.n_points` rows. Modified in-place.

		Returns:
			None
		"""
		if apply_from_left:
			for idx_point, idx_cluster in enumerate(self.vec):
				w = self.cluster_weights[idx_cluster]
				if w > 0:
					data[idx_point, ...] *= 1. / w
		else:
			for idx_point, idx_cluster in enumerate(self.vec):
				w = self.cluster_weights[idx_cluster]
				if w > 0:
					data[..., idx_point] *= 1. / w

	def __rescale_len_clusters(self, data, apply_from_left=True, **options):
		"""
		Scale input array's entries by corresponding cluster's weight.

		If input array is vector, method acts on entries. If input array
		is matrix, method acts on rows.

		Procedure::
			# Let the entries of vector c give the map:
			#	SET_FULL-->SET_CLUSTERED, and vector w give the cluster
			#	weights.
			#
			# for each i=INDEX_FULL, k=INDEX_CLUSTER in c, do
			#	data[k] *= 1 / w[k]
			# end for

		Arguments:
			data: Array to transform, with
				:attr:`ClusterMapping.n_clusters` rows. Modified
				in-place.

		Returns:
			None
		"""
		if apply_from_left:
			for idx_cluster, w in enumerate(self.cluster_weights):
				if w > 0:
						data[idx_cluster, ...] *= 1. / w
		else:
			for idx_cluster, w in enumerate(self.cluster_weights):
				if w > 0:
						data[..., idx_cluster] *= 1. / w

	def downsample_inplace(self, in_, out_, rescale_output=True,
						   clear_output=False, **options):
		"""
		Downsample entries of ``in_``, add to entries of ``out_``.

		Let matrix ``C`` give the linear transformation described by
		this :class:`ClusterMapping`, mapping ``m`` points to ``k``
		clusters, with ``k`` <= ``m``. Then ``C`` is a matrix in
		R^{k \times m} with exactly one ``1`` per column.

		The downsampling operation is equivalent to::
			# without rescaling:
			# out_ += C * in_

			# with rescaling:
			# out_ += (C'C)^{-1} * C * in_

		Arguments:
			in_: Array with ``m`` rows and ``n`` >= ``1`` columns.
			out_: Array with ``k`` rows and ``n`` >= ``1`` columns.
			rescale_output (:obj:`bool`, optional): If ``True``,
				downsampled rows of ``in_`` are scaled by the cluster
				weights.
			clear_output (:obj:`bool`, optional): If ``True``, entries
				of ``out_`` are set to ``0`` before downsampled entries
				of ``in_`` added.

		Returns:
			Updated version of array ``out_``, modified in-place.
		"""
		out_ = self.frame0_to_1_inplace(
				in_, out_, clear_output=clear_output, **options)
		if rescale_output:
			self.__rescale_len_clusters(out_, **options)
		return out_

	def downsample(self, in_, rescale_output=True, **options):
		"""
		Allocating version of :meth:`ClusterMapping.downsample_inplace`.

		Downsamples an input array with ``m`` rows and ``n`` >= ``1``
		columns to an output array with ``k`` < ``m`` rows and ``n``
		columns.

		Arguments:
			in_: Array with ``m`` rows and ``n`` >= ``1`` columns.
			rescale_output (:obj:`bool`, optional): If ``True``,
				downsampled rows of ``in_`` are scaled by the cluster
				weights.

		Returns:
			A new array of size ``k`` x ``n`` containing downsampled
			data.
		"""
		out_ = self.frame0_to_1(in_, **options)
		if rescale_output:
			self.__rescale_len_clusters(out_, **options)
		return out_

	def upsample_inplace(self, in_, out_, rescale_output=False,
						 clear_output=False, **options):
		"""
		Upsample entries of ``in_``, add to entries of ``out_``.

		Let matrix ``C`` give the linear transformation described by this
		`ClusterMapping`, mapping ``m`` points to ``k`` clusters, with
		``k`` <= ``m``. Then ``C`` is a matrix in R^{k \times m} with
		exactly one ``1`` per column.

		The upsampling operation is equivalent to::
			# without rescaling:
			# out_ += C' * in_

			# with rescaling:
			# out_ += C' * (C'C)^{-1} * in_

		Arguments:
			in_: Array with ``k`` rows and ``n`` >= ``1`` columns.
			out_: Array with ``m`` rows and ``n`` >= ``1`` columns.
			rescale_output (:obj:`bool`, optional): If ``True``, rows of
				``in_`` are scaled by the cluster weights prior to
				upsampling.
			clear_output (:obj:`bool`, optional): If ``True``, entries
				of ``out_`` are set to 0 before upsampled entries of
				``in_`` added.

		Returns:
			Updated version of array ``out_``, modified in-place.
		"""
		out_ = self.frame1_to_0_inplace(
				in_, out_, clear_output=clear_output, **options)
		if rescale_output:
			self.__rescale_len_points(out_, **options)
		return out_

	def upsample(self, in_, rescale_output=False, **options):
		"""
		Allocating version of :meth:`ClusterMapping.upsample_inplace`.

		Upsamples an input array with ``k`` rows and ``n`` >= ``1``
		columns to an output array with ``m`` > ``k`` rows and ``n``
		columns.

		Arguments:
			in_: Array with ``m`` rows and ``n`` >= ``1`` columns.
			rescale_output (:obj:`bool`, optional): If ``True``, rows of
				``in_`` are scaled by the cluster weights prior to
				upsampling.

		Returns:
			A new array of size ``m`` x ``n`` containing upsampled data.
		"""
		out_ = self.frame1_to_0(in_, **options)
		if rescale_output:
			self.__rescale_len_points(out_, **options)
		return out_

	@property
	def contiguous(self):
		"""
		Rebase mapping to omit empty cluster indices.

		Arguments:
			None

		Returns:
			:class:`ClusterMapping`: Contiguous cluster mapping: ``N``
				points mapped to ``K`` clusters, with cluster indices
				ranging from ``k`` = ``0``, ..., ``K - 1``, and each
				cluster index corresponding to a non-empty cluster.
		"""
		if not self.__empty_clusters:
			return self

		vec = np.zeros(self.vec.size, dtype=int)
		vec += self.vec
		idx_order = vec.argsort()

		cluster_new = 0
		cluster_old = vec[idx_order[0]]

		for idx in idx_order:
			if vec[idx] != cluster_old:
				cluster_new += 1
				cluster_old = vec[idx]

			vec[idx] = cluster_new

		return ClusterMapping(vec)

class PermutationMapping(DiscreteMapping):
	""" Map ``N`` elements to each other, one-to-one. """

	def __init__(self, permutation_vector):
		"""
		Initialize as :class:`DiscreteMapping` instance.

		Arguments:
			permutation_vector: Vector of integer indices of length
				``N``. Each integer ``i`` = ``0``, ..., ``N - 1`` should
				appear exactly once.

		Raises:
			ValueError: If contents of ``permutation_vector`` imply
				input and output sets to have np.different sizes, or if
				each element in the input set is not represented exactly
				once in the output set.
		"""
		DiscreteMapping.__init__(self, permutation_vector)
		if self.n_frame0 != self.n_frame1:
			raise ValueError('{} requires input and output spaces to be '
							 'of same dimension'.format(PermutationMapping))
		if sum(np.diff(self.vec[self.vec.argsort()]) != 1) > 0:
			raise ValueError('{} requires 1-to-1 mapping between input '
							 'output spaces; some output indices were '
							 'skipped'.format(PermutationMapping))

class IdentityMapping(AbstractDiscreteMapping):
	def __init__(self, n):
		self.__n = max(1, int(n))

	@property
	def vec(self):
		return np.array(xrange(self.n_frame0))

	@property
	def manifest(self):
		return {'data': self.__n, 'type': map_type_to_string(self)}

	@property
	def n_frame0(self):
		return self.__n

	@property
	def n_frame1(self):
		return self.__n

	def frame0_to_1_inplace(self, in_, out_, clear_output=False, **options):
		if clear_output:
			out_ *= 0
		out_ += in_
		return out_

	def frame0_to_1(self, in_, **options):
		out_ = np.zeros_like(in_)
		return self.frame0_to_1_inplace(in_, out_, True, **options)

	def frame1_to_0_inplace(self, in_, out_, clear_output=False, **options):
		return self.frame0_to_1_inplace(in_, out_, clear_output **options)

	def frame1_to_0(self, in_, **options):
		return self.frame0_to_1(in_, **options)

class DictionaryMapping(AbstractDiscreteMapping):
	def __init__(self, mapping_dictionary, key_order=None):
		self.__maps = {}
		self.__key_order = None
		self.__vec = None
		self.__concatenated_map = None
		self.__n_frame0 = None
		self.__n_frame1 = None

		if isinstance(mapping_dictionary, DictionaryMapping):
			self.__maps.update(mapping_dictionary._DictionaryMapping__maps)
		else:
			for key in mapping_dictionary:
				mapping = mapping_dictionary[key]
				if isinstance(mapping, AbstractDiscreteMapping):
					self.__maps[key] = mapping
				elif isinstance(mapping, dict):
					self.__maps[key] = string_to_map_constructor(
							mapping['type'])(mapping['data'])


		self.__n_frame0 = sum([m.n_frame0 for m in self.mappings])
		self.__n_frame1 = sum([m.n_frame1 for m in self.mappings])
		self.__vec = np.zeros(self.__n_frame0, dtype=int)
		self.key_order = self.keys if key_order is None else key_order

	def __contains__(self, key):
		return key in self.__maps

	def __getitem__(self, key):
		return self.__maps[key]

	@property
	def key_order(self):
		if self.__key_order is None:
			return np.array(self.keys)
		else:
			return self.__key_order

	@key_order.setter
	def key_order(self, key_order):
		ko = [int(k) for k in key_order]
		if not (len(ko) == len(self.keys)) and all([k in self.keys for k in ko]):
			raise ValueError(
					'argument `key_order` must be a permutation of the '
					'keys in this `{}`'.format(DictionaryMapping))
		self.__key_order = ko
		self.__rebuild_vec()

	@property
	def keys(self):
		return self.__maps.keys()

	@property
	def mappings(self):
		return self.__maps.values()

	@property
	def vec(self):
		return self.__vec

	@property
	def manifest(self):
		return {
				'data': {l: m.manifest for l, m in self.__maps.items()},
				'type': map_type_to_string(self)
		}
		return self.__maps

	@property
	def n_frame0(self):
		return self.__n_frame0

	@property
	def n_frame1(self):
		return self.__n_frame1

	def __get_working_form(self, io_object, name, apply_from_left=True, **options):
		if isinstance(io_object, dict):
			return io_object
		if isinstance(io_object, SliceCachingVector):
			return io_object._SliceCachingVector__slices
		elif isinstance(io_object, SliceCachingVector):
			if apply_from_left:
				return io_object._SliceCachingMatrix__row_slices
			else:
				return io_object._SliceCachingMatrix__column_slices
		else:
			raise TypeError('argument `{}` must be of types {}'.format(
					name,
					(dict, SliceCachingVector, SliceCachingMatrix)))

	def frame0_to_1_inplace(self, in_, out_, clear_output=False, **options):
		rescale_output = options.pop('rescale_output', False)
		d_in = self.__get_working_form(in_, 'in_', **options)
		d_out = self.__get_working_form(out_, 'out_', **options)
		for key in d_in:
			m = self[key]
			if isinstance(m, ClusterMapping):
				m.downsample_inplace(
						d_in[key], d_out[key], clear_output=clear_output,
						rescale_output=rescale_output, **options)
			else:
				m.frame0_to_1_inplace(
						d_in[key], d_out[key], clear_output=clear_output,
						**options)

	def frame0_to_1(self, in_, **options):
		rescale_output = options.pop('rescale_output', False)
		d_in = self.__get_working_form(in_, 'in_', **options)
		out_ = {}
		for key in d_in:
			m = self[key]
			if isinstance(m, ClusterMapping):
				out_[key] = m.downsample(
						d_in[key], rescale_output=rescale_output, **options)
			else:
				out_[key] = m.frame0_to_1(
						d_in[key], **options)
		return type(in_)(out_)

	def frame1_to_0_inplace(self, in_, out_, clear_output=False, **options):
		rescale_output = options.pop('rescale_output', False)
		d_in = self.__get_working_form(in_, 'in_', **options)
		d_out = self.__get_working_form(out_, 'out_', **options)
		for key in d_in:
			m = self[key]
			if isinstance(m, ClusterMapping):
				m.upsample_inplace(
						d_in[key], d_out[key], clear_output=clear_output,
						rescale_output=rescale_output, **options)
			else:
				m.frame1_to_0_inplace(
						d_in[key], d_out[key], clear_output=clear_output,
						**options)


	def frame1_to_0(self, in_, **options):
		rescale_output = options.pop('rescale_output', False)
		d_in = self.__get_working_form(in_, 'in_', **options)
		out_ = {}
		for key in d_in:
			m = self[key]
			if isinstance(m, ClusterMapping):
				out_[key] = m.upsample(
						d_in[key], rescale_output=rescale_output, **options)
			else:
				out_[key] = m.frame1_to_0(
						d_in[key], **options)
		return type(in_)(out_)

	def __rebuild_vec(self):
		ko = self.key_order
		self.__vec *= 0
		ptr0 = 0
		ptr1 = 0
		for k in ko:
			self.vec[ptr0 : ptr0 + self[k].n_frame0] = self[k].vec + ptr1
			ptr0 += self[k].n_frame0
			ptr1 += self[k].n_frame1
		map_types = listmap(type, self.mappings)
		#
		if map_types.count(map_types[0]) == len(map_types):
			self.__concatenated_map = map_types[0](self.vec)
		else:
			self.__concatenated_map = DiscreteMapping(self.vec)

	@property
	def concatenated_map(self):
		return self.__concatenated_map

class DictionaryClusterMapping(DictionaryMapping):
	def __init__(self, mapping_dictionary):
		input_ = {}
		valid = True
		valid_types = (
				IdentityMapping, PermutationMapping, ClusterMapping)

		if isinstance(mapping_dictionary, DictionaryClusterMapping):
			input_.update(mapping_dictionary._DictionaryMapping__maps)
		else:
			# validate inputs as identity, permutation, or cluster mappings
			for key in mapping_dictionary:
				mapping = mapping_dictionary[key]
				if isinstance(mapping, valid_types):
					input_[key] = mapping
				elif isinstance(mapping, AbstractDiscreteMapping):
					valid = False
					break
				elif isinstance(mapping, dict):
					constructor = string_to_map_constructor(mapping['type'])
					if not constructor in valid_types:
						valid = False
						break
					input_[key] = constructor(mapping['data'])
			if not valid:
				raise TypeError(
						'input dictionary can only contain objects of '
						'types {} or dictionary manifests of those '
						'same object types'.format(valid_types))
		DictionaryMapping.__init__(self, input_)


	@property
	def n_clusters(self):
		""" Number of clusters in target set. """
		return self.n_frame1

	@property
	def n_points(self):
		""" Number of elements in source set. """
		return self.n_frame0

	@property
	def cluster_weights(self):
		""" Number of elements mapped to each cluster. """
		wts = {}
		for key in self._DictionaryMapping__maps:
			m = self[key]
			if isinstance(m, ClusterMapping):
				wts[key] = m.cluster_weights
			else:
				wts[key] = np.ones(m.n_frame1)

		return wts

	def downsample_inplace(self, in_, out_, rescale_output=False,
						   clear_output=False, **options):
		return self.frame0_to_1_inplace(
				in_, out_, rescale_output=rescale_output,
				clear_output=clear_output, **options)

	def downsample(self, in_, rescale_output=False, **options):
		return self.frame0_to_1(
				in_, rescale_output=rescale_output, **options)

	def upsample_inplace(self, in_, out_, rescale_output=False,
						 clear_output=False, **options):
		return self.frame1_to_0_inplace(
				in_, out_, rescale_output=rescale_output,
				clear_output=clear_output, **options)

	def upsample(self, in_, rescale_output=False):
		return self.frame1_to_0(
				in_, rescale_output=rescale_output, **options)

	@property
	def contiguous(self):
		contiguous = {}
		for k in self:
			m = self[k]
			if isinstance(m, ClusterMapping):
				contiguous[k] = m.contiguous
			else:
				contiguous[k] = m
		return DictionaryClusterMapping(contiguous)

	@property
	def concatenated_cluster_map(self):
		return ClusterMapping(self.vec)

def map_type_to_string(mapping):
	if isinstance(mapping, PermutationMapping):
		return 'permutation'
	elif isinstance(mapping, ClusterMapping):
		return 'cluster'
	elif isinstance(mapping, DiscreteMapping):
		return 'discrete'
	elif isinstance(mapping, IdentityMapping):
		return 'identity'
	elif isinstance(mapping, DictionaryMapping):
		return 'dictionary'
	elif isinstance(mapping, DictionaryClusterMapping):
		return 'cluster dictionary'
	else:
		raise TypeError('not a valid mapping')

def string_to_map_constructor(string):
	string = str(string)
	if len(string) > 0:
		if string in ('permutation, Permutation'):
			return PermutationMapping
		elif string in ('cluster', 'clustering', 'Cluster', 'Clustering'):
			return ClusterMapping
		elif string in ('identity', 'Identity'):
			return IdentityMapping
		elif string in ('dictionary', 'Dictionary'):
			return DictionaryMapping
		elif string in (
					'dictionary cluster', 'Dictionary Cluster',
					'dictionary clustering', 'Dictionary Clustering',
					'cluster dictionary', 'Cluster Dictionary',
					'clustering dictionary', 'Clustering Dictionary'):
			return DictionaryClusterMapping
	return DiscreteMapping
