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
			self.__value = value

		self.__value = float(value)

	def __mul__(self, other):
		pass

	def __rmul__(self, other):
		pass

	def __str__(self):
		pass

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

	def __str__(self):
		return '{} \%'.format(self.value)

class Length(AbstractNonnegativeUnit):
	def __init__(self, value=nan):
		AbstractNonnegativeUnit.__init__(self, value=value)

	def __mul__(self, other):
		pass

	def __rmul__(self, other):
		pass

	def __str__(self):
		pass

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

	def __str__(self):
		return str('{} mm'.format(self.value))

	def to_mm(self):
		return self

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

	def __str__(self):
		return str('{} cm'.format(self.value))

	def to_mm(self):
		return MM(self.value * 10)

	def to_cm(self):
		return self

class Area(AbstractNonnegativeUnit):
	def __init__(self, value=nan):
		AbstractNonnegativeUnit.__init__(self, value=value)

	def __mul__(self, other):
		pass

	def __rmul__(self, other):
		pass

	def __str__(self):
		pass

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

	def __str__(self):
		return str('{} mm^2'.format(self.value))

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

	def __str__(self):
		return str('{} cm^2'.format(self.value))

	def to_cm3(self):
		return self

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

	def __rmul__(self, other):
		pass

class MM3(Volume):
	def __init__(self, value=nan):
		Volume.__init__(self, value)

	def __rmul__(self, other):
		ret = MM3(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __str__(self):
		return str('{} mm^3'.format(self.value))

	def to_cm3(self):
		return CM3(self.value * 1e-3)

class CM3(Volume):
	def __init__(self, value=nan):
		Volume.__init__(self, value)

	def __rmul__(self, other):
		ret = CM3(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __str__(self):
		return str('{} cm^3'.format(self.value))

	def to_cm3(self):
		return self

class DeliveredDose(AbstractNonnegativeUnit):
		def __init__(self, value=nan):
		AbstractNonnegativeUnit.__init__(self, value=value)

	def __mul__(self, other):
		pass

	def __rmul__(self, other):
		pass

	def __str__(self):
		pass

class Gray(DeliveredDose):
	def __init__(self, value=nan):
		DeliveredDose.__init__(self, value=value)

	def __rmul__(self, other):
		if not isinstance(other, (int, float, Percent)):
			raise TypeError('right multiplication by {} object only defined '
							'when left operand is of type {}, {} or {}'.format(
							int, float, Percent))

		other = other.value if isinstance(other, Percent) else other

		ret = Gray(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret


	def __str__(self):
		return '{} Gy'.format(self.value)

	@property
	def to_Gy(self):
	    return self

	@property
	def to_cGy(self):
		return self(0.01 * self.value)

class centiGray(DeliveredDose):
	def __init__(self, value=nan):
		DeliveredDose.__init__(self, value=value)

	def __rmul__(self, other):
		if not isinstance(other, (int, float, Percent)):
			raise TypeError('right multiplication by {} object only defined '
							'when left operand is of type {}, {} or {}'.format(
							int, float, Percent))

		other = other.value if isinstance(other, Percent) else other

		ret = centiGray(self.value)
		ret.value = other if ret.value is nan else ret.value * other
		return ret

	def __str__(self):
		return '{} cGy'.format(self.value)

	@property
	def to_Gy(self):
	    return Gray(100. * self.value)

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