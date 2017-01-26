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

from conrad.defs import vec, is_vector, sparse_or_dense

# TODO: Change module to maps?
# TODO: Change this to DiscreteMap?

class DiscreteMapping(object):
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

	def frame0_to_1_inplace(self, in_, out_, clear_output=False):
		"""
		Map elements of array ``in_`` to elements of array ``out_``.

		Procedure::
			# let forward_map be the map: SET_0 --> SET_1.
			# for each INDEX_0, INDEX_1 in forward_map, do
			# 	out_[INDEX_1] += in_[INDEX_0]
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
		if not (vector_processing or matrix_processing):
			raise TypeError('arguments "in_" and "out_" be numpy or '
							'scipy vectors or matrices')

		dim_in1 = in_.shape[0]
		dim_out1 = out_.shape[0]
		if vector_processing:
			dim_in2 = 1
			dim_out2 = 1
		else:
			dim_in2 = in_.shape[1]
			dim_out2 = out_.shape[1]

		if bool(
				dim_in1 != self.n_frame0 or
				dim_out1 != self.n_frame1 or
				dim_in2 != dim_out2):
			raise ValueError('arguments "in_" and "out_" be vectors or '
							 'matrices of dimensions M x N, and K x N, '
							 'respectively, with:\nM = {}\nK={}\n'
							 'Provided:\n input: {}x{}\noutput: {}x{}'
							 ''.format(self.n_frame0, self.n_frame1,
							 dim_in1, dim_in2, dim_out1, dim_out2))
		if clear_output:
			out_ *= 0

		if vector_processing:
			for idx_0, idx_1 in enumerate(self.vec):
				out_[idx_1] += in_[idx_0]
		else:
			for idx_0, idx_1 in enumerate(self.vec):
				out_[idx_1, :] += in_[idx_0, :]
		# TODO: use efficient slicing when input is sparse. warn when output sparse??

		return out_

	def frame0_to_1(self, in_):
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
		if is_vector(in_):
			out_ = np.zeros(self.__n_frame1)
		elif sparse_or_dense(in_):
			dim1 = self.__n_frame1
			dim2 = in_.shape[1]
			out_ = np.zeros((dim1, dim2))

		return self.frame0_to_1_inplace(in_, out_)


	def frame1_to_0_inplace(self, in_, out_, clear_output=False):
		"""
		Allocating version of :meth`DiscreteMapping.frame0_to_1`.

		Procedure::
			# let reverse_map be the map: SET_1 --> SET_0.
			# for each INDEX_1, INDEX_0 in forward_map, do
			# 	out_[INDEX_0] += in_[INDEX_1]
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

		if not (vector_processing or matrix_processing):
			raise TypeError('arguments "in_" and "out_" be numpy or '
							'scipy vectors or matrices')

		dim_in1 = in_.shape[0]
		dim_out1 = out_.shape[0]
		if vector_processing:
			dim_in2 = 1
			dim_out2 = 1
		else:
			dim_in2 = in_.shape[1]
			dim_out2 = out_.shape[1]

		if bool(
				dim_in1 != self.n_frame1 or
				dim_out1 != self.n_frame0 or
				dim_in2 != dim_out2):
			raise ValueError('arguments "in_" and "out_" be vectors or '
							 'matrices of dimensions K x N, and M x N, '
							 'respectively, with:\nK = {}\nM={}\n'
							 'Provided:\n input: {}x{}\noutput: {}x{}'
							 ''.format(self.n_frame1, self.n_frame0,
							 dim_in1, dim_in2, dim_out1, dim_out2))
		if clear_output:
			out_ *= 0

		if vector_processing:
			for idx_0, idx_1 in enumerate(self.vec):
				out_[idx_0] += in_[idx_1]

		if matrix_processing:
			for idx_0, idx_1 in enumerate(self.vec):
				out_[idx_0, :] += in_[idx_1, :]

		return out_

	def frame1_to_0(self, in_):
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
		if is_vector(in_):
			out_ = np.zeros(self.__n_frame0)
		elif sparse_or_dense(in_):
			dim1 = self.__n_frame0
			dim2 = in_.shape[1]
			out_ = np.zeros((dim1, dim2))

		return self.frame1_to_0_inplace(in_, out_)

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

	def __rescale_len_points(self, data):
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
		vector = data.shape[0] == data.size
		for idx_point, idx_cluster in enumerate(self.vec):
			w = self.cluster_weights[idx_cluster]
			if w > 0:
				if vector:
					data[idx_point] *= 1. / w
				else:
					data[idx_point, :] *= 1. / w

	def __rescale_len_clusters(self, data):
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
		vector = data.shape[0] == data.size
		for idx_cluster, w in enumerate(self.cluster_weights):
			if w > 0:
				if vector:
					data[idx_cluster] *= 1. / w
				else:
					data[idx_cluster, :] *= 1. / w

	def downsample_inplace(self, in_, out_, rescale_output=True,
						 clear_output=False):
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
		out_ = self.frame0_to_1_inplace(in_, out_, clear_output=clear_output)
		if rescale_output:
			self.__rescale_len_clusters(out_)
		return out_

	def downsample(self, in_, rescale_output=True):
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
		out_ = self.frame0_to_1(in_)
		if rescale_output:
			self.__rescale_len_clusters(out_)
		return out_

	def upsample_inplace(self, in_, out_, rescale_output=False,
						 clear_output=False):
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
		out_ = self.frame1_to_0_inplace(in_, out_, clear_output=clear_output)
		if rescale_output:
			self.__rescale_len_points(out_)
		return out_

	def upsample(self, in_, rescale_output=False):
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
		out_ = self.frame1_to_0(in_)
		if rescale_output:
			self.__rescale_len_points(out_)
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

def map_type_to_string(mapping):
	if isinstance(mapping, PermutationMapping):
		return 'permutation'
	elif isinstance(mapping, ClusterMapping):
		return 'cluster'
	elif isinstance(mapping, DiscreteMapping):
		return 'discrete'
	else:
		raise TypeError('not a valid mapping')

def string_to_map_constructor(string):
	string = str(string)
	if len(string) > 0:
		if string in ('permutation, Permutation'):
			return PermutationMapping
		elif string in ('cluster', 'clustering', 'Cluster', 'Clustering'):
			return ClusterMapping
	return DiscreteMapping

# class HierarchicalMapping
# class DictionaryMapping
#	- dictionary of labels->discrete mappings
#	- should probably also have a dictionary of labels->indices in source space
# 	- i guess something similar could be achieved by chaining selection mapping
# 	and another arbitrary mapping
# class SelectionMapping





