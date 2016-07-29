from numpy import zeros, ndarray, diff

from conrad.compat import *
from conrad.defs import vec, is_vector, sparse_or_dense

class AbstractMapping(object):
	def __init__(self, map_vector):
		self.__forwardmap = vec(map_vector).astype(int)
		self.__n_frame0 = len(self.__forwardmap)
		self.__n_frame1 = self.__forwardmap.max() + 1

	@property
	def vec(self):
		return self.__forwardmap

	@property
	def n_frame0(self):
		return self.__n_frame0

	@property
	def n_frame1(self):
		return self.__n_frame1

	def frame0_to_1_inplace(self, in_, out_, clear_output=False):
		vector_processing = is_vector(in_) and is_vector(out_)
		matrix_processing = sparse_or_dense(in_) and sparse_or_dense(out_)

		if not vector_processing or matrix_processing:
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
				dim_in_1 != self.n_frame0 or
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

		if matrix_processing:
			for idx_0, idx_1 in enumerate(self.vec):
				out_[idx_1, :] += in_[idx_0, :]

		return out_

	def frame0_to_1(self, in_):
		if is_vector(in_):
			out_ = zeros(self.__n_frame1)
		elif sparse_or_dense(in_):
			dim1 = self.__n_frame1
			dim2 = in_.shape[1]
			out_ = zeros((dim1, dim2))

		return self.frame0_to_1_inplace(in_, out_)


	def frame1_to_0_inplace(self, in_, out_, clear_output=False):
		vector_processing = is_vector(in_) and is_vector(out_)
		matrix_processing = sparse_or_dense(in_) and sparse_or_dense(out_)

		if not vector_processing or matrix_processing:
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
				dim_in_1 != self.n_frame1 or
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
			for idx_in, idx_out in enumerate(self.vec):
				out_[idx_0, :] += in_[idx_1, :]

		return out_

	def frame0_to_1(self, in_):
		if is_vector(in_):
			out_ = zeros(self.__n_frame0)
		elif sparse_or_dense(in_):
			dim1 = self.__n_frame0
			dim2 = in_.shape[1]
			out_ = zeros((dim1, dim2))

		return self.frame1_to_0_inplace(in_, out_)


class ClusterMapping(AbstractMapping):
	def __init__(self, clustering_vector):
		AbstractMapping.__init__(self, clustering_vector)
		self.__cluster_weights = zeros(self.n_clusters)
		for cluster_index in self.vec:
			self.__cluster_weights[cluster_index] += 1.

	@property
	def n_clusters(self):
		return self.n_frame1
	@property
	def n_points(self):
		return self.n_frame0

	@property
	def cluster_weights(self):
		return self.__cluster_weights

	def __rescale_len_points(self, data):
		vector = data.shape[0] == data.size
		for idx_point, idx_cluster in enumerate(self.vec):
			w = self.cluster_weights[idx_cluster]
			if w > 0:
				if vector:
					data[idx_point] *= 1. / w
				else:
					data[idx_point, :] *= 1. / w

	def __rescale_len_cluster(self, data):
		vector = data.shape[0] == data.size
		for idx_cluster, w in enumerate(self.cluster_weights):
			if w > 0:
				if vector:
					data[idx_cluster] *= 1. / w
				else:
					data[idx_cluster, :] *= 1. / w

	def downsample_inplace(self, in_, out_, rescale_output=True,
						 clear_output=False):
		out_ = self.frame0_to_1_inplace(in_, out_, clear_output=clear_output)
		if rescale_output:
			self.__rescale_len_cluster(out_)

	def downsample(self, in_, rescale_output=True):
		out_ = self.frame0_to_1(in_, out_, clear_output=clear_output)
		if rescale_output:
			self.__rescale_len_cluster(out_)

	def upsample_inplace(self, in_, out_, rescale_output=True,
						 clear_output=False):
		out_ = self.frame1_to_0_inplace(in_, out_, clear_output=clear_output)
		if rescale_output:
			self.__rescale_len_points(out_)

	def upsample(self, in_, rescale_output=True):
		out_ = self.frame1_to_0(in_, out_, clear_output=clear_output)
		if rescale_output:
			self.__rescale_len_points(out_)

	def to_contiguous(self):
		""" return contiguous cluster mapping, i.e., rebase to omit
			cluster indices with no mapped points

			return new ClusterMapping object
		"""
		vec = zeros(self.vec.size, dtype=int)
		vec += self.vec
		idx_order = vec.argsort()

		unit_incr = 0
		curr = 0

		for idx in idx_order:
			if vec[idx] != curr:
				unit_incr += 1
				curr = vec[idx]

			vec[idx] = unit_incr

		return ClusterMapping(vec)

class PermutationMapping(AbstractMapping):
	def __init__(self, permutation_vector):
		AbstractMapping.__init__(self, permutation_vector)
		if self.n_frame0 != self.n_frame1:
			raise ValueError('{} requires input and output spaces to be '
							 'of same dimension'.format(PermutationMapping))
		if sum(diff(self.vec.argsort()) != 1) > 0:
			raise ValueError('{} requires 1-to-1 mapping between input '
							 'output spaces; some output indices were '
							 'skipped'.format(PermutationMapping))

