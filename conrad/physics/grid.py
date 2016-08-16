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
from numpy import nan
from operator import mul

from conrad.compat import *
from conrad.physics.units import mm, mm2, cm3, Length

class AbstractGrid(object):
	def __init__(self):
		self.__x = 0
		self.__y = 0
		self.__z = 0
		self.__x_unit_length = nan * mm
		self.__y_unit_length = nan * mm
		self.__z_unit_length = nan * mm
		self.__order = ''
		self.__dims = []

		# dictionary for index->position calculations
		self.__pos = {}

	@staticmethod
	def validate_nonnegative_int(var, name):
		if not isinstance(var, int):
			raise TypeError(
					'argument "{}" must be of type {}'.format(name, int))
		elif var < 0:
			raise ValueError('argument "{}" must be >= 0'.format(name))

	@staticmethod
	def validate_positive_int(var, name):
		if not isinstance(var, int):
			raise TypeError(
					'argument "{}" must be of type {}'.format(name, int))
		elif var <= 0:
			raise ValueError('argument "{}" must be >= 1'.format(name))

	@staticmethod
	def validate_length(var, name):
		if not isinstance(var, Length):
			raise TypeError(
					'argument "{}" must be of type {}'.format(name, Length))

	def calculate_strides(self):
		span = 1
		lengths = {}
		strides = {}
		for i, d in enumerate(self.dims):
			lengths[d] = self.shape[i]

		for dim in self.order:
			strides[dim] = span
			span *= lengths[dim]

		self.strides = strides

	@property
	def order(self):
		return self.__order

	@property
	def dims(self):
		return self.__dims

	@property
	def shape(self):
		pass

	@property
	def x_unit_length(self):
		return self.__x_unit_length

	@property
	def y_unit_length(self):
		return self.__y_unit_length

	@property
	def z_unit_length(self):
		pass

	@property
	def x_length(self):
		return self.__z * self.__x_unit_length

	@property
	def y_length(self):
		return self.__y * self.__y_unit_length

	@property
	def z_length(self):
		pass

class Grid2D(AbstractGrid):
	def __init__(self, x=None, y=None):
		AbstractGrid.__init__(self)
		self._AbstractGrid__order = 'xy'
		self._AbstractGrid__dims = ['x', 'y']
		self.__unit_area = nan * mm2
		self.__pos = self._AbstractGrid__pos
		self.set_shape(x, y)

	def set_order(self, order='xy'):
		if not isinstance(order, str):
			raise TypeError('argument "order" must be of type {}'.format(str))
		elif len(order) != 2 or not order in ('xy', 'yx'):
			raise ValueError('argument "order" must be "xy" or "yx"')
		else:
			self._AbstractGrid__order = order
			self.calculate_strides()

	def set_shape(self, x=None, y=None):
		self.validate_positive_int(x, 'x')
		self.validate_positive_int(y, 'y')
		self._AbstractGrid__x = x
		self._AbstractGrid__y = y
		self.calculate_strides()

	def set_scale(self, x_length=None, y_length=None):
		self.validate_length(x_length, 'x_length')
		self.validate_length(y_length, 'y_length')
		self._AbstractGrid__x_unit_length = x_length
		self._AbstractGrid__y_unit_length = y_length
		self.__unit_area = (x_length * y_length)

	@property
	def dims(self):
		return ('x', 'y')

	@property
	def shape(self):
		return self._AbstractGrid__x, self._AbstractGrid__y

	@property
	def z_unit_length(self):
		raise AttributeError('2D grids have no "z" dimension')

	@property
	def z_length(self):
		raise AttributeError('2D grids have no "z" dimension')

	def index2position(self, index):
		index = int(index)
		self.validate_nonnegative_int(index, 'index')

		gridsize = reduce(mul, self.shape)
		if index >= gridsize:
			raise ValueError('index {} outside of geometry with {} '
							 'elements'.format(index, gridsize))

		for i in xrange(2):
			self.__pos[self.order[1 - i]] = int(
					index / self.strides[self.order[1 - i]])
			index = index % self.strides[self.order[1 - i]]

		return self.__pos['x'], self.__pos['y']

	def position2index(self, x, y):
		for arg in [(x, 'x'), (y, 'y')]:
			self.validate_nonnegative_int(arg[0], arg[1])

		if x >= self._AbstractGrid__x or y >= self._AbstractGrid__y:
			raise ValueError('position ({}, {}) outside of geometry with '
							 'dimesions ({}, {})'.format(x, y, *self.shape))

		return x * self.strides['x'] + y * self.strides['y']

	def __str__(self):
		args = [self.voxels]
		args += list(self.shape)
		args += list(self.unit_dimensions)
		args += [self.unit_volume, self.order]
		return str('Geometry: \n\tvoxels: {}\n\tshape: {} x {}'
				   '\n\tvoxel dimensions: {} x {}\n\tvoxel volume: {}'
				   '\n\ttraversal order: {}'.format(*args))

class Grid3D(AbstractGrid):
	def __init__(self, x=None, y=None, z=None):
		AbstractGrid.__init__(self)
		self.__unit_volume = nan * cm3
		self._AbstractGrid__order = 'xyz'
		self._AbstractGrid__dims = ['x', 'y', 'z']
		self.__pos = self._AbstractGrid__pos
		self.set_shape(x, y, z)

	def set_shape(self, x=None, y=None, z=None):
		self.validate_positive_int(x, 'x')
		self.validate_positive_int(y, 'y')
		self.validate_positive_int(z, 'z')
		self._AbstractGrid__x = x
		self._AbstractGrid__y = y
		self._AbstractGrid__z = z
		self.calculate_strides()

	def set_scale(self, x_length=None, y_length=None, z_length=None):
		self.validate_length(x_length, 'x_length')
		self.validate_length(y_length, 'y_length')
		self.validate_length(z_length, 'z_length')
		self._AbstractGrid__x_unit_length = x_length
		self._AbstractGrid__y_unit_length = y_length
		self._AbstractGrid__z_unit_length = z_length
		self.__unit_volume = (x_length * y_length * z_length)

	def set_order(self, order='xyz'):
		contains_xyz = all(listmap(lambda v: v in order, ('x', 'y', 'z')))

		if not isinstance(order, str):
			raise TypeError('argument "order" must be of type {}'.format(str))
		elif len(order) != 3 or not contains_xyz:
			raise ValueError('argument "order" must be a permutation of "xyz"')
		else:
			self._AbstractGrid__order = order
			self.calculate_strides()

	@property
	def shape(self):
		return (self._AbstractGrid__x, self._AbstractGrid__y,
				self._AbstractGrid__z)

	@property
	def unit_dimensions(self):
		return self.x_unit_length, self.y_unit_length, self.z_unit_length

	@property
	def unit_volume(self):
		return self.__unit_volume.to_cm3

	@property
	def total_dimensions(self):
		return self.x_length, self.y_length, self.z_length

	@property
	def z_unit_length(self):
		return self._AbstractGrid__z_unit_length

	@property
	def z_length(self):
		return self._AbstractGrid__z * self._AbstractGrid__z_unit_length

	def index2position(self, index):
		index = int(index)
		self.validate_nonnegative_int(index, 'index')

		gridsize = reduce(mul, self.shape)
		if index >= gridsize:
			raise ValueError('index {} outside of geometry with {} '
							 'elements'.format(index, gridsize))

		for i in xrange(3):
			self.__pos[self.order[2 - i]] = int(
					index / self.strides[self.order[2 - i]])
			index = index % self.strides[self.order[2 - i]]

		return self.__pos['x'], self.__pos['y'], self.__pos['z']

	def position2index(self, x, y, z):
		for arg in [(x, 'x'), (y, 'y'), (z, 'z')]:
			self.validate_nonnegative_int(arg[0], arg[1])

		if bool(
				x >= self._AbstractGrid__x or
				y >= self._AbstractGrid__y or
				z >= self._AbstractGrid__z):
			raise ValueError('position ({}, {}, {}) outside of geometry with '
							 'dimesions ({}, {}, {})'.format(x, y, z,
							 *self.shape))

		return int(x * self.strides['x'] + y * self.strides['y'] +
				   z * self.strides['z'])

	def __str__(self):
		args = [self.voxels]
		args += list(self.shape)
		args += list(self.unit_dimensions)
		args += [self.unit_volume, self.order]
		return str('Geometry: \n\tvoxels: {}\n\tshape: {} x {} x {}'
				   '\n\tvoxel dimensions: {} x {} x {}\n\tvoxel volume: {}'
				   '\n\ttraversal order: {}'.format(*args))