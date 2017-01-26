"""
Define :class:`Anatomy`, container for treatment planning structures.
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

from numpy import nan

from conrad.medicine.structure import Structure

class Anatomy(object):
	"""
	Iterable container class for treatment planning structures.

	Provides simple syntax via overloaded operators, including addition,
	retrieval, and removal of structures from anatomy::

		anatomy = Anatomy()

		# target structure with label = 4
		s1 = Structure(4, 'target', True)

		# non-target structure with label = 12
		s2 = Structure(12, 'non-target', False)

		# non-target structure with label = 7
		s3 = Structure(7, 'non-target 2', False)

		anatomy += s1
		anatomy += s2
		anatomy += s3

		# remove structure s3 by name
		anatomy -= 'non-target 2'

		# remove structure s2 by label
		anatomy -= 12

		# retrieve structure s1 by name
		anatomy[4]
		anatomy['target']
	"""
	def __init__(self, structures=None):
		"""
		Initialize :class:`Anatomy` object, empty by default.

		Arguments:
			structures (optional): Iterable collection of
				:class:`Structure` objects to append to
				:class:`Anatomy`. If ``structures`` is of type
				:class:`Anatomy`, initializer acts as a copy
				constructor.
		"""
		self.__structures = {}
		self.__label_order = None

		if isinstance(structures, Anatomy):
			self.__structures = structures._Anatomy__structures
		elif structures:
			self.structures = structures

	def __contains__(self, comparator):
		for s in self:
			if comparator in (s.name, s.label):
				return True
		return False

	def __getitem__(self, key):
		for s in self:
			if key in (s.name, s.label):
				return s
		raise KeyError('key {} does not correspond to a structure label'
					   ' or name in this {} object'.format(key, Anatomy))

	def __iter__(self):
		return self.__structures.values().__iter__()

	@property
	def structures(self):
		"""
		Dictionary of structures in anatomy, keyed by label.

		Setter method accepts any iterable collection of
		:class:`Structure` objects.

		Raises:
			TypeError: If input to setter is not iterable.
			ValueError: If input to setter contains elements of a type
				other than :class:`Structure`.
		"""
		return self.__structures

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
	def list(self):
		""" List of structures in :class:`Anatomy`. """
		return [self[label] for label in self.label_order]

	@property
	def label_order(self):
		"""
		Ranked list of labels of structures in :class:`Anatomy`.

		Raises:
			ValueError: If input to setter contains labels for
				structures not contained in anatomy, or if the length
				of the input list does not match `Anatomy.n_structures`.
		"""
		if self.__label_order is None:
			return [s.label for s in self]
		elif len(self.__label_order) != self.n_structures:
			return [s.label for s in self]
		else:
			return self.__label_order

	@label_order.setter
	def label_order(self, ordered_labels):
		if len(ordered_labels) != self.n_structures:
			raise ValueError('provided label ordering has length {}.\n'
							 'Anatomy contains {} structures'
							 ''.format(len(ordered_labels), self.n_structures))
		for label in ordered_labels:
			if label not in self.labels:
				raise ValueError('label {} does not exist in this {}'
								 ''.format(label, Anatomy))
		self.__label_order = list(ordered_labels)

	@property
	def is_empty(self):
		""" ``True`` if :class:`Anatomy` contains no structures. """
		return self.n_structures == 0

	@property
	def n_structures(self):
		""" Number of structures in :class:`Anatomy`. """
		return len(self.structures)

	@property
	def size(self):
		""" Total number of voxels in all structures in :class:`Anatomy`. """
		if self.is_empty:
			return 0
		elif any([s.size is None for s in self]):
			return nan
		else:
			return sum([s.size for s in self])

	@property
	def labels(self):
		""" List of labels of structures in :class:`Anatomy`. """
		return self.structures.keys()

	@property
	def plannable(self):
		"""
		``True`` if all structures plannable and at least one is a target.
		"""
		if self.is_empty:
			return False

		# at least one target
		status = any([structure.is_target for structure in self])

		# every structure plannable, i.e. has dose matrix and other
		# required data
		status &= all([structure.plannable for structure in self])
		return status

	def clear_constraints(self):
		"""
		Clear all constraints from all structures in :class:`Anatomy`.

		Arguments:
			None

		Returns:
			None
		"""
		for s in self:
			s.constraints.clear()

	def calculate_doses(self, beam_intensities):
		"""
		Calculate voxel doses to each structure in :class:`Anatomy`.

		Arguments:
			beam_intensities: Beam intensities to provide to each
				structure's `Structure.calculate_dose` method.

		Returns:
			None
		"""
		for s in self:
			s.calculate_dose(beam_intensities)

	def propagate_doses(self, voxel_doses):
		"""
		Assign pre-calculated voxel doses to each structure in
		:class:`Anatomy`

		Arguments:
			voxel_doses (:obj:`dict`): Dictionary mapping structure
				labels to voxel dose subvectors.

		Returns:
			None
		"""
		for s in self:
			s.assign_dose(voxel_doses[s.label])

	def dose_summary_data(self, percentiles=[2, 98]):
		"""
		Collimate dose summaries from each structure in :class:`Anatomy`.

		Arguments:
			percentiles (:obj:`list`): List of percentiles to include
				in dose summary queries.

		Returns:
			:obj:`dict`: Dictionary of dose summaries obtained by
			calling `Structure.summary` for each structure.
		"""
		d = {}
		for s in self:
			d[s.label] = s.summary(percentiles=percentiles)
		return d

	@property
	def dose_summary_string(self):
		"""
		Collimate dose summary strings from each structure in :class:`Anatomy`.

		Arguments:
			None

		Returns:
			:obj:`dict`: Dictionary of dose summaries obtained by
			calling `Structure.summary_string` for each structure.
		"""
		out = ''
		for s in self.structures.values():
			out += s.summary_string
		return out

	def __iadd__(self, other):
		"""
		Overload operator +=.

		Append structure(s) in argument to :class:`Anatomy`.

		Arguments:
			other: Singleton or iterable collection of
				:class:`Structure` objects.

		Returns:
		 	:class:`Anatomy`: Updated anatomy.
		"""
		if isinstance(other, Structure):
			self.__structures[other.label] = other
		else:
			for key, item in enumerate(other):
				self += item

		return self

	def __isub__(self, other):
		"""
		Overload operator -=.

		Arguments:
			other: Name or label of structure to remove from :class:`Anatomy`.

		Returns:
			:class:`Anatomy`: Downdated anatomy.
		"""
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

	def __str__(self):
		"""
		Collimate strings for each :class:`Structure` in :class:`Anatomy`.
		"""
		ret_string = str(
				'\n{} with {} structures:\n'.format(
						Anatomy, self.n_structures))
		for s in self:
			ret_string += str(s)
		return ret_string

	def plotting_data(self, constraints_only=False, maxlength=None):
		"""
		Dictionary of :mod:`matplotlib`-compatible plotting data for all
		structures.

		Args:
			constraints_only (:obj:`bool`, optional): If ``True``,
				return only the constraints associated with each
				structure.
			maxlength (:obj:`int`, optional): If specified, re-sample
				each structure's DVH plotting data to have a maximum
				series length of ``maxlength``.
		"""
		return {s.label: s.plotting_data(constraints_only=constraints_only,
										 maxlength=maxlength) for s in self}