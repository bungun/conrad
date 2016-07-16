from numpy import nan
from conrad.compat import *

class AbstractNonnegativeUnit(object):
	def __init__(self, value=nan):
		self.__value = nan
		self.value = value

	@property
	def value(self):
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

class Percent(AbstractNonnegativeUnit):
	def __init__(self, value=nan):
		AbstractNonnegativeUnit.__init__(self, value=value)

	@property
	def fraction(self):
		return self.value / 100.

	@fraction.setter
	def fraction(self, fraction):
		self.value = 100. * fraction

	def __rmul__(self, other):
		ret = Percent(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __add__(self, other):
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
		return self.__add__(other)

	def __eq__(self, other):
		if not isinstance(other, Percent):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of same type'.format(Percent))
		else:
			return self.value == other.value

	def __str__(self):
		return '{}%'.format(self.value)

class Length(AbstractNonnegativeUnit):
	def __init__(self, value=nan):
		AbstractNonnegativeUnit.__init__(self, value=value)

class MM(Length):
	def __init__(self, value=nan):
		Length.__init__(self, value)

	def __mul__(self, other):
		if isinstance(other, Length):
			if isinstance(other, CM):
				return MM2(self.value * 10. * other.value)
			else:
				return MM2(self.value * other.value)
		elif isinstance(other, Area):
			if isinstance(other, CM2):
				return MM3(self.value * 100. * other.value)
			else:
				return MM3(self.value * other.value)
		else:
			raise TypeError('Type {} only supports left multiplication '
							'of types {} or {}'.format(Length, Length, Area))

	def __rmul__(self, other):
		ret = MM(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		if not isinstance(other, Length):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(MM, Length))
		else:
			return self.value == other.to_mm.value

	def __str__(self):
		return str('{} mm'.format(self.value))

	@property
	def to_mm(self):
		return self

	@property
	def to_cm(self):
		return CM(self.value * 1e-1)

class CM(Length):
	def __init__(self, value=nan):
		Length.__init__(self, value)

	def __mul__(self, other):
		if isinstance(other, Length):
			if isinstance(other, CM):
				return CM2(self.value * other.value)
			else:
				return CM2(self.value * 0.1 * other.value)
		elif isinstance(other, Area):
			if isinstance(other, CM2):
				return CM3(self.value * other.value)
			else:
				return CM3(self.value * 0.01 * other.value)
		else:
			raise TypeError('Type {} only supports left multiplication '
							'of types {} or {}'.format(Length, Length, Area))

	def __rmul__(self, other):
		ret = CM(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		if not isinstance(other, Length):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(CM, Length))
		else:
			return self.value == other.to_cm.value


	def __str__(self):
		return str('{} cm'.format(self.value))

	@property
	def to_mm(self):
		return MM(self.value * 10)

	@property
	def to_cm(self):
		return self

class Area(AbstractNonnegativeUnit):
	def __init__(self, value=nan):
		AbstractNonnegativeUnit.__init__(self, value=value)

class MM2(Area):
	def __init__(self, value=nan):
		Area.__init__(self, value)

	def __mul__(self, other):
		if isinstance(other, Length):
			if isinstance(other, CM):
				return MM3(self.value * 10. * other.value)
			else:
				return MM3(self.value * other.value)
		else:
			raise TypeError('Type {} only supports left multiplication '
							'of type {}'.format(Area, Length))

	def __rmul__(self, other):
		ret = MM2(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		if not isinstance(other, Area):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(MM2, Area))
		else:
			return self.value == other.to_mm2.value

	def __str__(self):
		return str('{} mm^2'.format(self.value))

	@property
	def to_mm2(self):
		return self

	@property
	def to_cm2(self):
		return CM2(self.value * 1e-2)

class CM2(Area):
	def __init__(self, value=nan):
		Area.__init__(self, value)

	def __mul__(self, other):
		if isinstance(other, Length):
			if isinstance(other, CM):
				return CM3(self.value * other.value)
			else:
				return CM3(self.value * 0.1 * other.value)
		else:
			raise TypeError('Type {} only supports left multiplication '
							'of type {}'.format(Area, Length))

	def __rmul__(self, other):
		ret = CM2(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		if not isinstance(other, Area):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(CM2, Area))
		else:
			return self.value == other.to_cm2.value

	def __str__(self):
		return str('{} cm^2'.format(self.value))

	@property
	def to_cm2(self):
		return self

	@property
	def to_mm2(self):
		return MM2(self.value * 1e2)


class Volume(object):
	def __init__(self, value=nan):
		self.__value = nan
		self.value = value

	@property
	def value(self):
		return self.__value

	@value.setter
	def value(self, value):
		if not isinstance(value, (int, float)):
			raise TypeError('argument "value" must be of type {} or '
							'{}'.format(int, float))
		elif value < 0:
			raise ValueError('argument "value" must be nonnegative')
		else:
			self.__value = value

class MM3(Volume):
	def __init__(self, value=nan):
		Volume.__init__(self, value)

	def __rmul__(self, other):
		ret = MM3(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		if not isinstance(other, Volume):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(MM3, Volume))
		else:
			return self.value == other.to_mm3.value

	def __str__(self):
		return str('{} mm^3'.format(self.value))

	@property
	def to_mm3(self):
		return self

	@property
	def to_cm3(self):
		return CM3(self.value * 1e-3)

class CM3(Volume):
	def __init__(self, value=nan):
		Volume.__init__(self, value)

	def __rmul__(self, other):
		ret = CM3(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __eq__(self, other):
		if not isinstance(other, Volume):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(CM3, Volume))
		else:
			return self.value == other.to_cm3.value

	def __str__(self):
		return str('{} cm^3'.format(self.value))

	@property
	def to_mm3(self):
		return MM3(self.value * 1e3)

	@property
	def to_cm3(self):
		return self

class DeliveredDose(AbstractNonnegativeUnit):
	def __init__(self, value=nan):
		AbstractNonnegativeUnit.__init__(self, value=value)

class Gray(DeliveredDose):
	def __init__(self, value=nan):
		DeliveredDose.__init__(self, value=value)

	def __rmul__(self, other):
		if not isinstance(other, (int, float, Percent)):
			raise TypeError('right multiplication by {} object only defined '
							'when left operand is of type {}, {} or {}'.format(
							int, float, Percent))

		other = other.fraction if isinstance(other, Percent) else other

		ret = Gray(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __add__(self, other):
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
		return self.__add__(other)

	def __eq__(self, other):
		if not isinstance(other, DeliveredDose):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(Gray,
							DeliveredDose))
		else:
			return self.value == other.to_Gy.value

	def __str__(self):
		return '{} Gy'.format(self.value)

	@property
	def to_Gy(self):
	    return self

	@property
	def to_cGy(self):
		return centiGray(100 * self.value)

class centiGray(DeliveredDose):
	def __init__(self, value=nan):
		DeliveredDose.__init__(self, value=value)

	def __rmul__(self, other):
		if not isinstance(other, (int, float, Percent)):
			raise TypeError('right multiplication by {} object only defined '
							'when left operand is of type {}, {} or {}'.format(
							centiGray, int, float, Percent))

		other = other.fraction if isinstance(other, Percent) else other

		ret = centiGray(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __add__(self, other):
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
		return self.__add__(self)

	def __eq__(self, other):
		if not isinstance(other, DeliveredDose):
			raise TypeError('equality comparison with {} object only defined '
							'for objects of type {}'.format(centiGray,
							DeliveredDose))
		else:
			return self.value == other.to_cGy.value

	def __str__(self):
		return '{} cGy'.format(self.value)

	@property
	def to_Gy(self):
	    return Gray(0.01 * self.value)

	@property
	def to_cGy(self):
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