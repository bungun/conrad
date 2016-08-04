from numpy import nan

from conrad.compat import *
from conrad.medicine.structure import Structure

class Anatomy(object):
	def __init__(self, structures=None, **options):
		self.__structures = {}

		if isinstance(structures, Anatomy):
			self.__structures = structures._Anatomy__structures
		elif structures:
			self.structures = structures

	def __getitem__(self, key):
		if key in self.__structures:
			return self.__structures[key]
		else:
			for s in self:
				if s.name == key:
					return s
		raise KeyError('key {} does not correspond to a structure label'
					   ' or name in this {} object'.format(key, Anatomy))


		return self.__structures[key]

	def __iter__(self):
		return self.__structures.values().__iter__()

	@property
	def structures(self):
		return self.__structures

	@property
	def list(self):
		return self.structures.values()

	@structures.setter
	def structures(self, structures):
		# check iterability
		try:
			_  = (s for s in structures)
		except TypeError:
			raise TypeError('argument "structures" must be iterable')

		if isinstance(structures, dict):
			structures = structures.values()

		for s in structures:
			if not isinstance(s, Structure):
				raise ValueError('each element of argument "structures"'
								 'must be of type {}'.format(Structure))
			self += s

	@property
	def is_empty(self):
		return self.n_structures == 0

	@property
	def n_structures(self):
		return len(self.structures)

	@property
	def size(self):
		if self.is_empty:
			return 0
		elif any([s.size is None for s in self]):
			return nan
		else:
			return sum([s.size for s in self])

	@property
	def labels(self):
	    return self.structures.keys()

	@property
	def plannable(self):
		if self.is_empty:
			return False

		# at least one target
		status = any([structure.is_target for structure in self])
		# at least one target
		status &= all([structure.plannable for structure in self])
		return status

	def clear_constraints():
		for s in self:
			s.constraints.clear()

	def calculate_doses(self, beam_intensities):
		for s in self:
			s.calculate_dose(beam_intensities)

	def dose_summary_data(self, percentiles = [2, 98]):
		d = {}
		for s in self:
			d[s.label] = s.summary(percentiles=percentiles)
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

	def __isub__(self, other):
		key = other
		for s in self:
			if s.name == other:
				key = s.label
				break

		s = self.__structures.pop(key, None)
		if s is None:
			print('argument "other"={} does not correspond to the label '
				  'or name of a {} in this {} object. no operation '
				  'performed'.format(other, Structure, Anatomy))

		return self