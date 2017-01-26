"""
TOOO: DOCSTRING
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

from conrad.physics.string import dose_from_string
from conrad.medicine import Anatomy, Structure
from conrad.medicine.prescription import eval_constraint
from conrad.case import Case
from conrad.io.schema import AnatomyEntry, StructureEntry
from conrad.io.accessors.base_accessor import ConradDBAccessor

class AnatomyAccessor(ConradDBAccessor):
	def __init__(self, database=None, filesystem=None):
		self.__structure_accessor = StructureAccessor()
		ConradDBAccessor.__init__(
				self, subaccessors=[self.__structure_accessor],
				database=database, filesystem=filesystem)

	@property
	def structure_accessor(self):
		return self.__structure_accessor

	def save_anatomy(self, anatomy):
		if not isinstance(anatomy, Anatomy):
			raise TypeError(
					'argument `anatomy` must be of type {}'
					''.format(Anatomy))

		a = AnatomyEntry()
		a.add_structures(*map(
				self.structure_accessor.save_structure, anatomy))

		return self.DB.set_next(a)

	def load_anatomy(self, anatomy_entry):
		anatomy_entry = self.DB.get(anatomy_entry)
		if not isinstance(anatomy_entry, AnatomyEntry):
			raise ValueError(
					'argument `anatomy_entry` must be of type {}, '
					'or a dictionary representation of/ConRad database '
					'pointer to that type'.format(AnatomyEntry))
		if not anatomy_entry.complete:
			raise ValueError('anatomy incomplete')

		anatomy = Anatomy()
		for s in anatomy_entry.structures:
			s = self.DB.get(s)
			anatomy += self.structure_accessor.load_structure(s)
		return anatomy

class StructureAccessor(ConradDBAccessor):
	def __init__(self, database=None, filesystem=None):
		ConradDBAccessor.__init__(
				self, database=database, filesystem=filesystem)

	def save_structure(self, structure):
		if not isinstance(structure, Structure):
			raise TypeError(
					'argument `structure` must be of type {}'
					''.format(Structure))

		return self.DB.set_next(StructureEntry(
				label=structure.label, name=structure.name,
				target=structure.is_target, size=structure.size,
				rx=structure.dose_rx,
				constraints=list(map(str, structure.constraints.list))))

	def load_structure(self, structure_entry):
		structure_entry = self.DB.get(structure_entry)
		if not isinstance(structure_entry, StructureEntry):
			raise ValueError(
					'argument `structure_entry` must be of type {}, '
					'or a dictionary representation of/ConRad database '
					'pointer to that type'.format(StructureEntry))
		if not structure_entry.complete:
			raise ValueError('structure incomplete')

		s = Structure(
				structure_entry.label, structure_entry.name,
				structure_entry.target, size=structure_entry.size)

		if structure_entry.rx is not None:
			s.dose_rx = dose_from_string(structure_entry.rx)

		if structure_entry.constraints is not None:
			for c in structure_entry.constraints:
				s.constraints += eval_constraint(c)

		if structure_entry.objective is not None:
			pass
			# TODO: implement this

		return s