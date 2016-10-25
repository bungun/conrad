"""
Define physical units used throughout :mod:`conrad`.

Each unit derives from the :class:`AbstractNonnegativeUnit` base class,
and is represented as an object with overloaded operators to support
syntax allowing intuitive unit manipulation, i.e.,

	$ (2 * MM()) * (2 * MM())

yields a :class:`MM2` object with :attr:`MM2.value`=``4``.

Attributes:
	mm: Exported instance of :class:`MM`.
	mm2: Exported instance of :class:`MM2`.
	mm3: Exported instance of :class:`MM3`.
	cm: Exported instance of :class:`CM`.
	cm2: Exported instance of :class:`CM2`.
	cm3: Exported instance of :class:`CM3`.
	percent: Exported instance of :class:`Percent`.
	Gy: Exported instance of :class:`Gray`.
	cGy: Exported instance of :class:`centiGray`.
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

class AbstractNonnegativeUnit(object):
	""" Base class for physical units """

	def __init__(self, value=1.0):
		"""
		Initializer for unit object.

		Arguments:
			value (:obj:`float`): Initial value, as multiple of the
				object's unit type.
		"""
		self.__value = nan
		self.value = value

	@property
	def value(self):
		"""
	 	Multiple of object's physical units.

		Raises:
			TypeError: If argument to setter is not :obj:`int` or
				:obj:`float`.
			ValueError: If argument to setter is negative.
		"""
		return self.__value

	@value.setter
	def value(self, value):
		if not isinstance(value, (int, float)):
			raise TypeError('argument "value" must be of type {} or '
							'{}'.format(int, float))
		elif value < 0:
			raise ValueError('argument "value" must be nonnegative')
		else:
			self.__value = float(value)

	def __float__(self):
		return self.value

class Percent(AbstractNonnegativeUnit):
	""" Specialize :obj:`AbstractNonnegativeUnit` to percent units. """

	def __init__(self, value=100):
		""" Initialize :obj:`Percent` as :obj:`AbstractNonnegativeUnit`. """
		AbstractNonnegativeUnit.__init__(self, value=value)

	@property
	def fraction(self):
		""" 1/100th of :attr:`Percent.value`. """
		return self.value / 100.

	@fraction.setter
	def fraction(self, fraction):
		self.value = 100. * fraction

	def __rmul__(self, other):
		"""
		Overload right multiplication.

		Multiplies :attr:`Percent.value` by ``other``.

		Arguments:
			other: Nonnegative scalar to right multiply by
				:class:`Percent.`

		Returns:
			:class:`Percent`: Product of multiplication.
		"""
		ret = Percent(self.value)
		if isinstance(other, Percent):
			if ret.value is nan:
				other = other.value
			else:
				other = other.fraction

		ret.value = other if ret.value is nan else ret.fraction * other
		return ret

	def __add__(self, other):
		"""
		Overload operator +.

		Arguments:
			other: Scalar or :class:`Percent` to add to this object's
				value.

		Returns:
			:class:`Percent`: Updated version of this :class:`Percent`.

		Raises:
			TypeError: If ``other`` is not of type :obj:`int`,
				:obj:`float` or :class:`Percent`.
		"""
		if not isinstance(other, (int, float, Percent)):
			raise TypeError('addition with {} object only defined when right '
							'operand is of type {}, {}, or {}'.format(Percent,
							int, float, Percent))
		if isinstance(other, (int, float)):
			self.value += other
		else:
			self.value += other.value

		return self

	def __iadd__(self, other):
		""" Overload operator +=, alias to operator +. """
		return self.__add__(other)

	def __eq__(self, other):
		"""
		Overload operator ==.

		Arguments:
			other (:class:`Percent`): Compared percentage.

		Returns:
			:obj:`bool`: ``True`` if ``other`` and ``self`` contain
			equal values.

		Raises:
			TypeError: If compared value not of type :class:`Percent`.
		"""
		if not isinstance(other, Percent):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of same type'.format(Percent))
		else:
			return self.value == other.value

	def __str__(self):
		""" String of contained value P as 'P%'. """
		return '{}%'.format(self.value)

class Length(AbstractNonnegativeUnit):
	""" Specialize :class:`AbstractNonnegativeUnit` to lengths. """
	def __init__(self, value=1.):
		"""
		Initialize :class:`Length` as :class:`AbstractNonnegativeUnit`.
		"""
		AbstractNonnegativeUnit.__init__(self, value=value)

	def to_mm(self):
		""" Unit conversion method, to be overridden when specialized. """
		raise ValueError('method "to_mm" not implemented.')

	def to_cm(self):
		""" Unit conversion method, to be overridden when specialized. """
		raise ValueError('method "to_cm" not implemented.')

class MM(Length):
	""" Specialize :class:`Length` to millimeters. """

	def __init__(self, value=1.):
		""" Initialize :class:`MM` as :class:`Length`. """
		Length.__init__(self, value)

	def __mul__(self, other):
		"""
		Overload operator * (left multiplication).

		:class:`MM` * :class:`Length` yields :class:`MM2`.
		:class:`MM` * :class:`Area` yields :class:`MM3`.

		Arguments:
			other: :class:`Length` or :class:`Area` to multiply. Assumed
				to have ``to_mm`` or ``to_mm2`` properties,
				respectively.

		Returns:
			Product in :class:`MM2` or :class:`MM3` units.

		Raises:
			TypeError: if ``other`` is not of type :class:`Length` or
				:class:`Area`.
		"""
		if isinstance(other, Length):
			return MM2(self.value * other.to_mm.value)
		elif isinstance(other, Area):
			return MM3(self.value * other.to_mm2.value)
		else:
			raise TypeError('Type {} only supports left multiplication '
							'of types {} or {}'.format(Length, Length, Area))

	def __rmul__(self, other):
		"""
		Overload right multiplication.

		Arguments:
			other: Nonnegative scalar to right multiply by :class:`MM.`

		Returns:
			:class:`MM`: Millimeter object with value set to original
			value times ``other``.
		"""
		if isinstance(other, Percent):
			other = other.fraction

		ret = MM(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		"""
		Overload operator ==.

		Compare values contained in ``self``, ``other`` by first
		converting both to millimeter units to establish a consistent
		basis of comparison.

		Arguments:
			other (:class:`Length`): Compared volume.

		Returns:
			:obj:`bool`: ``True`` if ``other`` and ``self`` contain
			equivalent values.

		Raises:
			TypeError: If ``other`` not of type :class:`Length`.
		"""
		if not isinstance(other, Length):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(MM, Length))
		else:
			return self.value == other.to_mm.value

	def __str__(self):
		""" String of length value L as 'L mm'. """
		return str('{} mm'.format(self.value))

	@property
	def to_mm(self):
		""" Convert to length units of mm (no-op). """
		return self

	@property
	def to_cm(self):
		""" Convert to length units of cm. """
		return CM(self.value * 1e-1)

class CM(Length):
	""" Specialize :class:`Length` to centimeters. """

	def __init__(self, value=1.):
		""" Initialize :class:`CM` object as :class:`Length`. """
		Length.__init__(self, value)

	def __mul__(self, other):
		"""
		Overload operator * (left multiplication).

		:class:`CM` * :class:`Length` yields :class:`CM2`.
		:class:`CM` * `Area` yields :class:`CM3`.

		Arguments:
			other: :class:`Length` or :class:`Area` to multiply. Assumed
				to have ``to_cm`` or ``to_cm2`` properties, respectively.

		Returns:
			Product in :class:`CM2` or :class:`CM3` units.

		Raises:
			TypeError: if ``other`` is not of type :class:`Length` or
				:class:`Area`.
		"""
		if isinstance(other, Length):
			return CM2(self.value * other.to_cm.value)
		elif isinstance(other, Area):
			return CM3(self.value * other.to_cm2.value)
		else:
			raise TypeError('Type {} only supports left multiplication '
							'of types {} or {}'.format(Length, Length, Area))

	def __rmul__(self, other):
		"""
		Overload right multiplication.

		Arguments:
			other: Nonnegative scalar to right multiply by :class:`CM.`

		Returns:
			:class:`CM`: Centimeter object with value set to original
			value times ``other``.
		"""
		if isinstance(other, Percent):
			other = other.fraction

		ret = CM(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		"""
		Overload operator ==.

		Compare values contained in ``self``, ``other`` by first
		converting both to centimeter units to establish a consistent
		basis of comparison.

		Arguments:
			other (:class:`Length`): Compared volume.

		Returns:
			:obj:`bool`: ``True`` if ``other`` and ``self`` contain
			equivalent values.

		Raises:
			TypeError: If ``other`` not of type :class:`Length`.
		"""
		if not isinstance(other, Length):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(CM, Length))
		else:
			return self.value == other.to_cm.value


	def __str__(self):
		""" String of length value L as 'L cm'. """
		return str('{} cm'.format(self.value))

	@property
	def to_mm(self):
		""" Convert to length units of mm. """
		return MM(self.value * 10)

	@property
	def to_cm(self):
		""" Convert to length units of cm (no-op). """
		return self

class Area(AbstractNonnegativeUnit):
	""" Extend abstract unit to form base class of :class:`Area` units. """

	def __init__(self, value=1.):
		""" Initialize :class:`Area` object. """
		AbstractNonnegativeUnit.__init__(self, value=value)

	def to_mm2(self):
		""" Unit conversion method, to be overridden when specialized. """
		raise ValueError('method "to_mm2" not implemented.')

	def to_cm2(self):
		""" Unit conversion method, to be overridden when specialized. """
		raise ValueError('method "to_cm2" not implemented.')

class MM2(Area):
	""" Specialize :class:`Area` to millimeters squared. """

	def __init__(self, value=1.):
		""" Initialize :class:`MM2` object as :class:`Area`. """
		Area.__init__(self, value)

	def __mul__(self, other):
		"""
		Overload operator * (left multiplication).

		:class:`MM2` * :class:`Area` yields :class:`MM3`.

		Arguments:
			Other: :class:`Length` to multiply.

		Returns:
			Product in :class:`MM3` units.

		Raises:
			TypeError: if ``other`` is not of type :class:`Length`.
		"""
		if isinstance(other, Length):
			return MM3(self.value * other.to_mm.value)
		else:
			raise TypeError('Type {} only supports left multiplication '
							'of type {}'.format(Area, Length))

	def __rmul__(self, other):
		"""
		Overload right multiplication.

		Arguments:
			other: Nonnegative scalar to right multiply by :class:`MM2.`

		Returns:
			:class:`MM2`: Millimeter^2 object with value set to original
			value times ``other``.
		"""
		if isinstance(other, Percent):
			other = other.fraction

		ret = MM2(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		"""
		Overload operator ==.

		Compare values contained in ``self``, ``other`` by first
		converting both to mm^2 units to establish a consistent basis of
		comparison.

		Arguments:
			other (:class:`Area`): Compared volume.

		Returns:
			:obj:`bool`:``True`` if ``other`` and ``self`` contain
			equivalent values.

		Raises:
			TypeError: If ``other`` not of type :class:`Area`.
		"""
		if not isinstance(other, Area):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(MM2, Area))
		else:
			return self.value == other.to_mm2.value

	def __str__(self):
		""" String of area value A as 'A mm^2'. """
		return str('{} mm^2'.format(self.value))

	@property
	def to_mm2(self):
		""" Convert to area units of mm^2 (no-op). """
		return self

	@property
	def to_cm2(self):
		""" Convert to area units of cm^2. """
		return CM2(self.value * 1e-2)

class CM2(Area):
	""" Specialize :class:`Area` to centimeters squared. """
	def __init__(self, value=1.):
		""" Initialize :class:`CM2` object as :class:`Area`. """
		Area.__init__(self, value)

	def __mul__(self, other):
		"""
		Overload operator * (left multiplication).

		:class:`CM2` * :class:`Length` yields :class:`CM3`.

		Arguments:
			Other: :class:`Length` to multiply.

		Returns:
			Product in :class:`CM3` units.

		Raises:
			TypeError: if ``other`` is not of type :class:`Length`.
		"""
		if isinstance(other, Length):
			return CM3(self.value * other.to_cm.value)
		else:
			raise TypeError('Type {} only supports left multiplication '
							'of type {}'.format(Area, Length))

	def __rmul__(self, other):
		"""
		Overload right multiplication.

		Arguments:
			other: Nonnegative scalar to right multiply by `CM2.`

		Returns:
			`CM2`: Centimeter^2 object with value set to original value
			times `other`.
		"""
		if isinstance(other, Percent):
			other = other.fraction

		ret = CM2(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		"""
		Overload operator ==.

		Compare values contained in ``self``, ``other`` by first
		converting both to cm^2 units to establish a consistent basis of
		comparison.

		Arguments:
			other (:class:`Area`): Compared volume.

		Returns:
			:obj:`bool`: ``True`` if ``other`` and ``self`` contain
			equivalent values.

		Raises:
			TypeError: If ``other`` not of type :class:`Area`.
		"""
		if not isinstance(other, Area):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(CM2, Area))
		else:
			return self.value == other.to_cm2.value

	def __str__(self):
		""" String of area value A as 'A cm^2'. """
		return str('{} cm^2'.format(self.value))

	@property
	def to_cm2(self):
		""" Convert to area units of cm^2 (no-op). """
		return self

	@property
	def to_mm2(self):
		""" Convert to area units of mm^2. """
		return MM2(self.value * 1e2)


class Volume(AbstractNonnegativeUnit):
	""" Extend abstract unit to form base class of :class:`Volume` units. """

	def __init__(self, value=1.):
		""" Initialize :class:`Volume` object. """
		AbstractNonnegativeUnit.__init__(self, value=value)

	def to_mm3(self):
		""" Unit conversion method, to be overridden when specialized. """
		raise ValueError('method "to_mm3" not implemented.')

	def to_cm3(self):
		""" Unit conversion method, to be overridden when specialized. """
		raise ValueError('method "to_cm3" not implemented.')


class MM3(Volume):
	""" Specialize :class:`Volume` to millimeters cubed. """
	def __init__(self, value=1.):
		""" Initialize :class:`MM3` as :class:`Volume`. """
		Volume.__init__(self, value)

	def __rmul__(self, other):
		"""
		Overload right multiplication.

		Arguments:
			other: Nonnegative scalar to right multiply by :class:`MM3.`

		Returns:
			:class:`MM3`: Millimeter^3 object with value set to original
			value times ``other``.
		"""
		if isinstance(other, Percent):
			other = other.fraction

		ret = MM3(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		"""
		Overload operator ==.

		Compare values contained in `self`, `other` by first converting
		both to mm^3 units to establish a consistent basis of
		comparison.

		Arguments:
			other (:class:`Volume`): Compared volume.

		Returns:
			:obj:`bool`: ``True`` if ``other`` and ``self`` contain
			equivalent values.

		Raises:
			TypeError: If ``other`` not of type :class:`Volume`.
		"""
		if not isinstance(other, Volume):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(MM3, Volume))
		else:
			return self.value == other.to_mm3.value

	def __str__(self):
		""" String of volume value V as 'V mm^3'. """
		return str('{} mm^3'.format(self.value))

	@property
	def to_mm3(self):
		""" Convert to volume units of mm^3 (no-op). """
		return self

	@property
	def to_cm3(self):
		""" Convert to volume units of cm^3. """
		return CM3(self.value * 1e-3)

class CM3(Volume):
	""" Specialize :class:`Area` to centimeters cubed. """

	def __init__(self, value=1.):
		""" Initialize :class:`CM3` as :class:`Volume`. """
		Volume.__init__(self, value)

	def __rmul__(self, other):
		"""
		Overload right multiplication.

		Arguments:
			other: Nonnegative scalar to right multiply by :class:`CM3.`

		Returns:
			:class:`CM3`: Centimeter^3 object with value set to original
			value times ``other``.
		"""
		if isinstance(other, Percent):
			other = other.fraction

		ret = CM3(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		"""
		Overload operator ==.

		Compare values contained in ``self``, ``other`` by first
		converting both to cm^3 units to establish a consistent basis of
		comparison.

		Arguments:
			other (:class:`Volume`): Compared volume.

		Returns:
			:obj:`bool`: ``True`` if ``other`` and ``self`` contain
			equivalent values.

		Raises:
			TypeError: If ``other`` not of type :class:`Volume`.
		"""
		if not isinstance(other, Volume):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(CM3, Volume))
		else:
			return self.value == other.to_cm3.value

	def __str__(self):
		""" String of volume value V as 'V cm^3'. """
		return str('{} cm^3'.format(self.value))

	@property
	def to_mm3(self):
		""" Convert to volume units of mm^3. """
		return MM3(self.value * 1e3)

	@property
	def to_cm3(self):
		""" Convert to volume units of cm^3 (no-op). """
		return self

class DeliveredDose(AbstractNonnegativeUnit):
	"""
	Extend :obj:`AbstractNonnegativeUnit` to form base class of dose units.
	"""

	def __init__(self, value=1.):
		"""
		Initialize :class:`DeliveredDose` as :class:`AbstractNonnegativeUnit`
		"""
		AbstractNonnegativeUnit.__init__(self, value=value)

	def to_cGy(self):
		""" Unit conversion method, to be overridden when specialized. """
		raise ValueError('method "to_cGy" not implemented.')

	def to_Gy(self):
		""" Unit conversion method, to be overridden when specialized. """
		raise ValueError('method "to_Gy" not implemented.')

class Gray(DeliveredDose):
	""" Specialize :class:`DeliveredDose` to radiological units of Gray. """

	def __init__(self, value=1.):
		""" Initialize `Gray` object as type `DeliveredDose`. """
		DeliveredDose.__init__(self, value=value)

	def __rmul__(self, other):
		"""
		Overload right multiplication.

		Arguments:
			other: Nonnegative scalar to right multiply by :class:`Gray.`

		Returns:
			:class:`Gray`: Gray object with value set to original value
			times ``other``.

		Raises:
			TypeError: If ``other`` is not of type :obj:`int`,
				:obj:`float` or :class:`Percent`.
		"""
		if not isinstance(other, (int, float, Percent)):
			raise TypeError('right multiplication by {} object only defined '
							'when left operand is of type {}, {} or {}'.format(
							int, float, Percent))

		if isinstance(other, Percent):
			other = other.fraction

		ret = Gray(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __add__(self, other):
		"""
		Overload operator +.

		Arguments:
			other: Scalar or :class:`DeliveredDose` to add to this
				object's value.

		Returns:
			:class:`Gray`: Updated version of this dose object.

		Raises:
			TypeError: If ``other`` is not of type :obj:`int`,
				:obj:`float` or :class:`DeliveredDose`.
		"""
		if not isinstance(other, (int, float, DeliveredDose)):
			raise TypeError('addition with {} object only defined when right '
							'operand is of type {}, {}, {}, or {}'.format(
							Gray, int, float, Gray, centiGray))
		if isinstance(other, (int, float)):
			self.value += other
		else:
			self.value += other.to_Gy.value
		return self

	def __iadd__(self, other):
		""" Overload operator +=, alias to operator +. """
		return self.__add__(other)

	def __eq__(self, other):
		"""
		Overload operator ==.

		Compare values contained in ``self``, ``other`` by first
		converting both to Gray units to establish a consistent basis of
		comparison.

		Arguments:
			other (:class:`DeliveredDose`): Compared dose.

		Returns:
			:obj:`bool`: ``True`` if ``other`` and ``self`` contain
			equivalent values.

		Raises:
			TypeError: If ``other`` not of type :class:`Delivered Dose`.
		"""
		if not isinstance(other, DeliveredDose):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(Gray,
							DeliveredDose))
		else:
			return self.value == other.to_Gy.value

	def __str__(self):
		""" String of dose value D as 'D Gy'. """
		return '{:.1f} Gy'.format(self.value)

	@property
	def to_Gy(self):
		""" Convert to dose units of Gray (no-op). """
		return self

	@property
	def to_cGy(self):
		""" Convert to dose units of centiGray. """
		return centiGray(100 * self.value)

class centiGray(DeliveredDose):
	""" Specialize :class:`DeliveredDose` class to centiGrays. """

	def __init__(self, value=1.):
		"""
		Initialize :class:`centiGray` object as type :class:`DeliveredDose`.
		"""
		DeliveredDose.__init__(self, value=value)

	def __rmul__(self, other):
		"""
		Overload right multiplication.

		Arguments:
			other: Nonnegative scalar to right multiply by
				:class:`centiGray.`

		Returns:
			:class:`centiGray`: centiGray object with value set to
			original value times ``other``.

		Raises:
			TypeError: If ``other`` is not of type :obj:`int`,
				:obj:`float` or :class:`Percent`.
		"""
		if not isinstance(other, (int, float, Percent)):
			raise TypeError('right multiplication by {} object only defined '
							'when left operand is of type {}, {} or {}'.format(
							centiGray, int, float, Percent))

		if isinstance(other, Percent):
			other = other.fraction

		ret = centiGray(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __add__(self, other):
		"""
		Overload operator +.

		Arguments:
			other: Scalar or :class:`DeliveredDose` to add to this
				object's value.

		Returns:
			:class:`centiGray`: Updated version of this dose object.

		Raises:
			TypeError: If ``other`` is not of type :obj:`int`,
				:obj:`float` or :class:`DeliveredDose`.
		"""
		if not isinstance(other, (int, float, DeliveredDose)):
			raise TypeError('addition with {} object only defined when right '
							'operand is of type {}, {}, {}, or {}'.format(
							centiGray, int, float, Gray, centiGray))
		if isinstance(other, (int, float)):
			self.value += other
		else:
			self.value += other.to_cGy.value

		return self

	def __iadd__(self, other):
		""" Overload operator +=, alias to operator +. """
		return self.__add__(self)

	def __eq__(self, other):
		"""
		Overload operator ==.

		Compare values contained in ``self``, ``other`` by first
		converting both to centiGray units to establish a consistent
		basis of comparison.

		Arguments:
			other (:class:`DeliveredDose`): Compared dose.

		Returns:
			:obj:`bool`: ``True`` if ``other`` and ``self`` contain
			equivalent values.

		Raises:
			TypeError: If ``other`` not of type :class:`Delivered Dose`.
		"""
		if not isinstance(other, DeliveredDose):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(centiGray,
							DeliveredDose))
		else:
			return self.value == other.to_cGy.value

	def __str__(self):
		""" String of dose value D as 'D cGy'. """
		return '{} cGy'.format(self.value)

	@property
	def to_Gy(self):
		""" Convert to dose units of Gray. """
		return Gray(0.01 * self.value)

	@property
	def to_cGy(self):
		""" Convert to dose units of centriGray (no-op). """
		return self

mm = MM()
mm2 = MM2()
mm3 = MM3()
cm = CM()
cm2 = CM2()
cm3 = CM3()
percent = Percent()
Gy = Gray()
cGy = centiGray()