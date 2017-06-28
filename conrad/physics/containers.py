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

	@property
	def unweighted(self):
		return self.data is not None and np.sum(self.data == 1) == self.size

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
