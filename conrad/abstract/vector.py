from conrad.compat import *

import numpy as np

from conrad.defs import vec

class SliceCachingVector(object):
	def __init__(self, data):
		self.__size = None
		self.__data = None
		self.__slices = {}
		self.data = data

	def __contains__(self, comparator):
		return comparator in self.__slices

	@property
	def keys(self):
		return self.__slices.keys()

	@property
	def size(self):
		return self.__size

	@property
	def shape(self):
		return self.data.shape if self.data is not None else None

	def _validate(self, data):
		return True

	@property
	def data(self):
		return self.__data

	@data.setter
	def data(self, data):
		self._validate(data)
		if isinstance(data, dict):
			data_contiguous = data.pop('contiguous', None)
			if data_contiguous is not None:
				self.data = data_contiguous
				if len(data) == 0:
					return

			for k in data:
				data[k] = vec(data[k]).astype(float)
			size = sum(w.size for w in data.values())
		else:
			data = vec(data).astype(float)
			size = data.size

		if self.size is not None and self.size != size:
			raise ValueError(
					'length of input data does not match known length '
					'of {}'.format(SliceCachingVector))
		else:
			self.__size = size

		if isinstance(data, dict):
			self.__slices.update(data)
			if data_contiguous is not None:
				self.data = data_contiguous
		else:
			self.__data = data

	def slice(self, label, indices=None):
		# calculate and cache slice
		if not label in self:
			if self.data is None:
				raise AttributeError(
						'cannot build slice from vector if '
						'`SliceCachingVector.data` is not set' )
			self.__slices[label] = self.data[indices]

		# return cached slice
		return self.__slices[label]

	def assemble(self):
		if len(self.__slices) == 0:
			raise AttributeError('no subvectors to assemble')
		self.__data = np.hstack(self.__slices.values())

	@property
	def manifest(self):
		manifest = {}
		if self.data is not None:
			manifest['contiguous'] = self.data
		if len(self.__slices) > 0:
			manifest.update(self.__slices)
		if len(manifest) == 0:
			raise ValueError(
					'{} not exportable as manifest: full vector or '
					'slices not set')
		return manifest