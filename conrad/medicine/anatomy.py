from conrad.compat import *
from conrad.medicine.structure import Structure

class Anatomy(object):
	def __init__(self, structures=None, **options):
		self.__structures = {}
		self.__labels = None
		if structures:
			self.structures = structures

		# intialize from prescription object
		rx = options.pop('prescription', None)
		if rx:
			self.structures = rx.structure_dict

		# use labels if provided
		labels = options.pop('label_vector', None)
		if labels:
			self.labels = labels

	@property
	def structures(self):
		return self.__structures

	@structures.setter
	def structures(self):
		if not isinstance(structures, (dict)):
			raise TypeError('argument "structures" must be of type '
							'{}'.format(dict))

		if not all(listmap(
				lambda s: isinstance(s, Structure), structures.values())):
			raise TypeError('argument "structures" must be a dict of {}'
							'objects'.format(Structure))

		self.__structures = structures

	@property
	def n_structures(self):
		return len(self.structures.keys())

	@property
	def size(self):
		return sum(listmap(lambda s: s.size, self.structures.values()))

	@property
	def label_set(self):
	    return self.structures.keys()

	@property
	def labels(self):
	    return self.__labels

	@labels.setter
	def labels(self, label_vector):
		for label in self.label_set:
			size = sum(listmap(lambda l: int(l == label), label_vector))
			self.structures[label].size = size

		if self.size != len(label_vector):
			for label in self.label_set:
				self.structures[label].size = nan

			diff = len(label_vector) - self.size
			err = 'unused' if diff > 0 else 'repeated'
			raise ValueError('error digesting label vector: {} '
							 '{} entries'.format(err, diff))

		self.__labels = label_vector

	def import_dose_matrix(self, physics):
		if self.size is None:
			raise RuntimeError('cannot import dose matrix to {} object '
							   'before setting structure labels'.format(
							   type(self)))

		if not isinstance(physics, Physics):
			raise TypeError('argument "Physics" must be of type '
							'{}'.format(Physics))

		if physics.voxels != self.size:
			raise ValueError('number of voxels in argument "physics" '
							 '({}) must match size of patient geometry '
							 '({})'.format(physics.voxels, self.size))

		if physics.dose_matrix is None:
			raise ValueError('field "dose_matrix" of argument "physics"'
							 ' must be initialized before calling {}'
							 '.import_dose_matrix()'.format(type(self)))

		for label in self.label_set:
			indices = listmap(lambda x: x[0], listfilter(lambda x: x[1]==label,
												 enumerate(a)))
			self.structures[label].A_full = self.A[indices, :]
			self.structures[label].A_mean = None

	def dose_summary_data(self, percentiles = [2, 98]):
		d = {}
		for label, s in self.structures.items():
			d[label] = s.summary(percentiles=percentiles)
		return d

	@property
	def dose_summary_string(self):
		out = ''
		for s in self.structures.values():
			out += s.summary_string
		return out

	def __iadd__(self, other):
		if isinstance(other, Structure):
			self.__structures[other.label] = other
		elif isinstance(other, dict):
			for key, item in other.items:
				self.__structures[key] += item

		return self