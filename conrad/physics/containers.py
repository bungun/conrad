from conrad.compat import *

import numpy as np

from conrad.defs import vec
from conrad.abstract.vector import SliceCachingVector
from conrad.abstract.matrix import SliceCachingMatrix

class WeightVector(SliceCachingVector):
	def __init__(self, data):
		SliceCachingVector.__init__(self, data)

	def _validate(self, data):
		nonneg = lambda v: np.sum(v < 0) == 0
		if isinstance(data, dict):
			valid = all(map(nonneg, data.values()))
		else:
			valid = nonneg(data)
		if not valid:
			raise ValueError(
					'entries of weight vector must be nonnnegative')

class DoseMatrix(SliceCachingMatrix):
	def __init__(self, data):
		SliceCachingMatrix.__init__(self, data)

	def __contains__(self, comparator):
		if isinstance(comparator, tuple):
			comparator = tuple(
					item.replace('voxel', 'row').replace('beam', 'column') if
					isinstance(item, str) else item for item in comparator)
		return SliceCachingMatrix.__contains__(self, comparator)

	@property
	def voxel_dim(self):
		return self.row_dim

	@property
	def beam_dim(self):
		return self.column_dim

	def _preprocess_data(self, data):
		if isinstance(data, dict) and 'labeled_by' in data:
			data['labeled_by'] = data['labeled_by'].replace(
					'voxels', 'rows').replace('beams', 'columns')
		return data

	def voxel_slice(self, label, indices):
		return self.row_slice(label, indices)

	def beam_slice(self, label, indices):
		return self.column_slice(label, indices)

	def slice(self, voxel_label=None, beam_label=None, voxel_indices=None,
			  beam_indices=None):
		return SliceCachingMatrix.slice(
				self, voxel_label, beam_label, voxel_indices, beam_indices)

	@property
	def cached_slices(self):
		d = self._SliceCachingMatrix__cached_slices
		d['voxel'] = d.pop('row')
		d['beam'] = d.pop('column')
		return d

	@property
	def manifest(self):
		m = self._SliceCachingMatrix__manifest
		if 'labeled_by' in m:
			m['labeled_by'] = m['labeled_by'].replace(
					'rows', 'voxels').replace('columns', 'beams')
		return m
