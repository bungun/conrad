from conrad.compat import *

import numpy as np
from six.moves import queue

from conrad.defs import vec

class VectorConstraintQueue(object):
	def __init__(self):
		self.constraint_queue = queue.Queue()
		self.relop_to_method = {}
		for key in [
				'<', '<=', 'lt', 'LT', 'leq', 'LEQ', 'less than',
				'less_than', 'less equals', 'less_equals']:
			self.relop_to_method[key] = self.satisfies_less_than
		for key in [
				'>', '>=', 'gt', 'GT', 'geq', 'GEQ', 'greater than',
				'greater_than', 'greater equals', 'greater_equals']:
			self.relop_to_method[key] = self.satisfies_greater_than
		for key in ['=', '==', 'eq', 'EQ', 'equals']:
			self.relop_to_method[key] = self.satisfies_equals

	def __iter__(self):
		return self.constraint_queue.queue.__iter__()

	def enqueue(self, relop, bound):
		self.constraint_queue.put((bound, self.relop_to_method[relop]))

	def enqueue_multi(self, *relop_bound_pairs):
		for relop, bound in bound_relop_pairs:
			self.enqueue(bound_relop_pairs)

	def dequeue(self):
		return self.constraint_queue.get()

	def clear(self):
		self.constraint_queue = queue.Queue()

	def dequeue_and_test(self, variable, reltol=1e-3, abstol=1e-4):
		bound, test = self.constraint_queue.get()
		return test(variable, reltol, abstol)

	def satisfies_all(self, variable, reltol=1e-3, abstol=1e-4):
		for bound, test in self.constraint_queue.queue:
			if not test(variable, bound, reltol, abstol):
				return False
		return True

	def tol(self, bound, reltol, abstol):
		return np.abs(reltol) * np.abs(bound) + np.abs(abstol)

	def satisfies_greater_than(self, variable, bound, reltol, abstol):
		return np.all(variable >= bound - self.tol(bound, reltol, abstol))

	def satisfies_less_than(self, variable, bound, reltol, abstol):
		return np.all(variable <= bound + self.tol(bound, reltol, abstol))

	def satisfies_equals(self, variable, bound, reltol, abstol):
		return np.all(np.abs(variable - bound) <= self.tol(
				bound, reltol, abstol))

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