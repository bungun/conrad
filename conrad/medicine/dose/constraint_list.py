"""
Defines the :class:`ConstraintList` container object for convenient
manipulation of dose constraints.
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
from conrad.medicine.dose.constraints import *
from conrad.medicine.dose.parsing import eval_constraint

class ConstraintList(object):
	"""
	Container for :class:`Constraint` objects.

	Attributes:
		items (:obj:`dict`): Dictionary of constraints in container,
			keyed by hashed values generated upon addition of constraint
			to container.
		last_key: Key generated upon most recent addition of a
			constraint to the container.
	"""
	def __init__(self, constraints=None):
		"""
		Initialize bare :class:`ConstraintList` container.

		Arguments:
			None
		"""
		self.items = {}
		self.last_key = None
		if constraints is not None:
			self += constraints

	@staticmethod
	def __keygen(constraint):
		"""
		Build unique identifier for ``constraint``.

		Hash current time and constraint properties (dose, relop,
		threshold) to generate unique identifier, take first seven
		characters as a key with low probability of collision.

		Arguments:
			constraint (:class:`Constraint`): Dose constraint to be
				assigned a key.

		Returns:
			:obj:`str`: Seven character key.
		"""
		return hashlib.sha1(str(
				str(time.clock()) +
				str(constraint.dose) +
				str(constraint.threshold) +
				str(constraint.relop)
			).encode('utf-8')).hexdigest()[:7]

	def __contains__(self, comparator):
		if isinstance(comparator, Constraint):
			return self.contains(comparator)
		else:
			return comparator in self.items

	def __getitem__(self, key):
		""" Overload operator []. """
		return self.items[key]

	def __iter__(self):
		""" Python3-compatible iterator implementation. """
		return self.items.__iter__()

	def __iadd__(self, other):
		"""
		Overload operator +=.

		Enable syntax :class:`ConstraintList` += :class:`Constraint`.

		Arguments:
			other: Singleton, or iterable collection of
				:class:`Constraint` objects to append to this
				:class:`ConstraintList`.

		Returns:
			:class:`ConstraintList`: Updated version of this object.

		Raises:
			TypeError: If ``other`` is not a :class:`Constraint` or
				iterable collection of constraints.
		"""
		if isinstance(other, Constraint):
			key = self.__keygen(other)
			self.items[key] =  other
			self.last_key = key
			return self
		elif isinstance(other, str):
			try:
				# str of list of constraints
				self += eval(other)
			except SyntaxError:
				# str of constraint
				self += eval_constraint(other)
		elif isinstance(other, (list, tuple)):
			for constr in other:
				self += constr
			return self
		elif isinstance(other, ConstraintList):
			self += other.items
			return self
		elif isinstance(other, dict):
			for constr in other.values():
				self += constr
			return self
		else:
			raise TypeError('argument must be of type {} or {}'.format(
							Constraint, ConstraintList))

	def __isub__(self, other):
		"""
		Overload operator -=.

		Enables syntaxes
			:class:`ConstraintList` -= :class:`Constraint`, and
			:class:`ConstraintList` -= ``key``.

		Remove ``other`` from this :class:`ConstraintList` if it is a
		key with a corresponding :class:`Constraint`, *or* if it is a
		:class:`Constraint` for which an exactly equivalent
		:class:`Constraint` is found in the list.

		Arguments:
			other: :class:`Constraint` or key to a :class:`Constraint`
				to be removed from this :class:`ConstraintList`.

		Returns:
			:class:`ConstraintList`: Updated version of this object.
		"""
		if isinstance(other, Constraint):
			for key, constr in self.items.items():
				if other == constr:
					del self.items[key]
					return self
		else:
			if other in self.items:
				del self.items[other]
				return self

	@property
	def size(self):
		""" Number of constraints in list. """
		return len(self.items)

	@property
	def keys(self):
		""" Keys of constraints in list. """
		return self.items.keys()

	@property
	def list(self):
		"""
		:obj:`list` of :class:`Constraint` objects in :class:`ConstraintList`.
		"""
		return list(self.items.values())

	@property
	def mean_only(self):
		""" ``True`` if list exclusively contains mean constraints. """
		meantest = lambda c : isinstance(c, MeanConstraint)
		if self.size == 0:
			return True
		else:
			return all(listmap(meantest, self.items.values()))

	def contains(self, constr):
		"""
		Test whether search :class:`Constraint` exists in this :class:`ConstraintList`.

		Arguments:
			constr (:class:`Constraint`): Search term.

		Returns:
			:obj:`bool`: ``True`` if a :class:`Constraint` equivalent to
			``constr`` found in this :class:`ConstraintList`.
		"""
		return constr in self.items.values()

	def clear(self):
		"""
		Clear constraints from :class:`ConstraintList`.

		Arguments:
			None

		Returns:
			None
		"""
		self.items = {}

	@property
	def plotting_data(self):
		"""
		List of :mod:`matplotlib`-compatible data for all constraints.
		"""
		return [(key, dc.plotting_data) for key, dc in self.items.items()]

	def __str__(self):
		""" Stringify list by concatenating strings of each constraint. """
		out = '(keys):\t (constraints)\n'
		for key, constr in self.items.items():
			out += key + ':\t' + str(constr) + '\n'
		return out