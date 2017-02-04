"""
Define base classes for abstract, 2-D, and 3-D regular grids.
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

import abc
import numpy as np
import operator as op

from conrad.physics.units import mm, mm2, cm3, Length

@add_metaclass(abc.ABCMeta)
class AbstractGrid(object):
	""" Base class for regular grids. """

	def __init__(self):
		"""
		Initialize :class:`AbstractGrid`.

		By default, up to 3 dimensions allowed, all dimensions assigned
		size zero, all unit cell lengths set to ``nan`` (with length
		units of millimeters).

		The grid traversal order is indefinite upon initialization.
		"""
		self.__x = 0
		self.__y = 0
		self.__z = 0
		self.__x_unit_length = np.nan * mm
		self.__y_unit_length = np.nan * mm
		self.__z_unit_length = np.nan * mm
		self.__order = ''
		self.__dims = []
		self.__strides = {}

		# dictionary for index->position calculations
		self.__pos = {}

	@staticmethod
	def validate_nonnegative_int(var, name):
		"""
		Check whether ``var`` is :obj:`int` >= 0.

		Arguments:
			var: Variable to validate.
			name: Name of variable to use in exception, if triggered.

		Returns:
			None

		Raises:
			TypeError: If ``var`` is not an integer.
			ValueError: If ``var`` is negative.
		"""
		if not isinstance(var, int):
			raise TypeError(
					'argument `{}` must be of type {}'.format(name, int))
		elif var < 0:
			raise ValueError('argument `{}` must be >= 0'.format(name))

	@staticmethod
	def validate_positive_int(var, name):
		"""
		Check whether ``var`` is :obj:`int` > 0.

		Arguments:
			var: Variable to validate.
			name: Name of variable to use in exception, if triggered.

		Returns:
			None

		Raises:
			TypeError: If ``var`` is not an integer.
			ValueError: If ``var`` is not positive.
		"""
		if not isinstance(var, int):
			raise TypeError(
					'argument `{}` must be of type {}'.format(name, int))
		elif var <= 0:
			raise ValueError('argument `{}` must be >= 1'.format(name))

	@staticmethod
	def validate_length(var, name):
		"""
		Check whether ``var`` is of type :class:`Length`.

		Arguments:
			var: Variable to validate.
			name: Name of variable to use in exception, if triggered.

		Returns:
			None

		Raises:
			TypeError: If ``var`` is not of type :class:`Length`.
		"""
		if not isinstance(var, Length):
			raise TypeError(
					'argument `{}` must be of type {}'.format(name, Length))

	def calculate_strides(self):
		"""
		Trigger calculation of grid strides.

		Grid strides calculated based on sizes and order of grid
		dimensions.

		Arguments:
			None

		Returns:
			None
		"""
		if self.order == '':
			raise ValueError('dimension order not set')

		span = 1
		lengths = {}
		for i, d in enumerate(self.dims):
			lengths[d] = self.shape[i]

		for dim in self.order:
			self.__strides[dim] = span
			span *= lengths[dim]


	@property
	def order(self):
		""" String listing order of dimensions. """
		return self.__order

	@property
	def dims(self):
		""" List of names/labels of dimensions, e.g., ['x', 'y']. """
		return self.__dims

	@property
	def strides(self):
		""" Dictionary of strides by dimension label. """
		return self.__strides

	@abc.abstractproperty
	def shape(self):
		""" Grid shape, or tuple of dimension lengths. """
		raise NotImplementedError

	@property
	def x_unit_length(self):
		""" Length of x-dimension of grid's unit cell. """
		return self.__x_unit_length

	@property
	def y_unit_length(self):
		""" Length of y-dimension of grid's unit cell. """
		return self.__y_unit_length

	@property
	def z_unit_length(self):
		""" Length of z-dimension of grid's unit cell. """
		pass

	@property
	def x_length(self):
		""" Length of x-dimension of entire grid. """
		return self.__z * self.__x_unit_length

	@property
	def y_length(self):
		""" Length of y-dimension of entire grid. """
		return self.__y * self.__y_unit_length

	@property
	def z_length(self):
		""" Length of z-dimension of entire grid. """
		pass

class Grid2D(AbstractGrid):
	""" Specialize :class:`AbstractGrid` to two-dimensional regular grids. """

	def __init__(self, x=None, y=None):
		"""
		Initialize :class:`Grid2D`.

		Arguments:
			x (:obj:`int`, optional): Size of grid's x-dimension.
			y (:obj:`int`, optional): Size of grid's y-dimension.
		"""
		AbstractGrid.__init__(self)
		self._AbstractGrid__order = 'xy'
		self._AbstractGrid__dims = ('x', 'y')
		self.__unit_area = np.nan * mm2
		self.__pos = self._AbstractGrid__pos
		self.set_shape(x, y)

	def set_order(self, order='xy'):
		"""
		Grid traversal order.

		Traversal order determines grid strides. For instance, a 5 x 3
		grid in order `xy` has unit stride in the x-dimension and stride
		5 in the y dimension.

		Arguments:
			order (:obj:`str`): Traversal order of the grid.

		Returns:
			None

		Raises:
			TypeError: If ``order`` not of type :obj:`str`.
			ValueError: If ``order`` not one of {'xy', 'yx'}.
		"""
		if not isinstance(order, str):
			raise TypeError('argument `order` must be of type {}'.format(str))
		elif len(order) != 2 or not order in ('xy', 'yx'):
			raise ValueError('argument `order` must be `xy` or `yx`')
		else:
			self._AbstractGrid__order = order
			self.calculate_strides()

	def set_shape(self, x=None, y=None):
		"""
		Set grid dimensions.

		Trigger recalculation of the grid traversal strides, given the
		:class:`Grid2D` object's current traversal order.

		Arguments:
			x (:obj:`int`): Number of grid units in x dimension.
			y (:obj:`int`): Number of grid units in y dimension.

		Returns:
			None
		"""
		self.validate_positive_int(x, 'x')
		self.validate_positive_int(y, 'y')
		self._AbstractGrid__x = x
		self._AbstractGrid__y = y
		self.calculate_strides()

	def set_scale(self, x_length=None, y_length=None):
		"""
		Set size of grid unit cell.

		Trigger calculation of unit cell area.

		Arguments:
			x_length (:class:`Length`): Length of grid unit cell in x
				dimension.
			y_length (:class:`Length`): Length of grid unit cell in y
				dimension.

		Returns:
			None
		"""
		self.validate_length(x_length, 'x_length')
		self.validate_length(y_length, 'y_length')
		self._AbstractGrid__x_unit_length = x_length
		self._AbstractGrid__y_unit_length = y_length
		self.__unit_area = (x_length * y_length)


	@property
	def shape(self):
		""" Tuple of grid dimensions. """
		return self._AbstractGrid__x, self._AbstractGrid__y

	@property
	def z_unit_length(self):
		""" Length of grid unit cell in z dimension: error. """
		raise AttributeError('2D grids have no "z" dimension')

	@property
	def z_length(self):
		""" Total length of grid in z dimension: error. """
		raise AttributeError('2D grids have no "z" dimension')

	def index2position(self, index):
		"""
		Convert ``index`` to (x, y) position in grid.

		Arguments:
			index (:obj:`int`): Index of grid element.

		Returns:
			:obj:`tuple` of :obj:`int`: Tuple representing (x, y) grid
			position of element at requested index.

		Raises:
			ValueError: If ``index`` exceeds grid bounds.
		"""
		index = int(index)
		self.validate_nonnegative_int(index, 'index')

		gridsize = reduce(op.mul, self.shape)
		if index >= gridsize:
			raise ValueError('index {} outside of geometry with {} '
							 'elements'.format(index, gridsize))

		for i in xrange(2):
			self.__pos[self.order[1 - i]] = int(
					index / self.strides[self.order[1 - i]])
			index = index % self.strides[self.order[1 - i]]

		return self.__pos['x'], self.__pos['y']

	def position2index(self, x, y):
		"""
		Convert (``x``, ``y``) position in grid to grid element index.

		Arguments:
			x (:obj:`int`): x position of grid element.
			y (:obj:`int`): y position of grid element.

		Returns:
			:obj:`int`: Index of grid element at requested position.

		Raises:
			ValueError: If (``x``, ``y``) exceeds grid bounds.
		"""
		for arg in [(x, 'x'), (y, 'y')]:
			self.validate_nonnegative_int(arg[0], arg[1])

		if x >= self._AbstractGrid__x or y >= self._AbstractGrid__y:
			raise ValueError('position ({}, {}) outside of geometry with '
							 'dimesions ({}, {})'.format(x, y, *self.shape))

		return x * self.strides['x'] + y * self.strides['y']

	def __str__(self):
		""" String description of grid dimensions. """
		args = [self.voxels]
		args += list(self.shape)
		args += list(self.unit_dimensions)
		args += [self.unit_volume, self.order]
		return str('Geometry: \n\tvoxels: {}\n\tshape: {} x {}'
				   '\n\tvoxel dimensions: {} x {}\n\tvoxel volume: {}'
				   '\n\ttraversal order: {}'.format(*args))

class Grid3D(AbstractGrid):
	"""
	Specialize :class:`AbstractGrid` to three-dimensional regular grids.
	"""

	def __init__(self, x=None, y=None, z=None):
		"""
		Initialize :class:`Grid3D`.

		Arguments:
			x (:obj:`int`, optional): Size of grid's x-dimension.
			y (:obj:`int`, optional): Size of grid's y-dimension.
			z (:obj:`int`, optional): Size of grid's z-dimension.
		"""
		AbstractGrid.__init__(self)
		self.__unit_volume = np.nan * cm3
		self._AbstractGrid__order = 'xyz'
		self._AbstractGrid__dims = ('x', 'y', 'z')
		self.__pos = self._AbstractGrid__pos
		self.set_shape(x, y, z)

	def set_shape(self, x=None, y=None, z=None):
		"""
		Set grid dimensions.

		Trigger recalculation of the grid traversal strides, given the
		:class:`Grid3D` object's current traversal order.

		Arguments:
			x (:obj:`int`): Number of grid units in x dimension.
			y (:obj:`int`): Number of grid units in y dimension.
			z (:obj:`int`): Number of grid units in z dimension.

		Returns:
			None
		"""
		self.validate_positive_int(x, 'x')
		self.validate_positive_int(y, 'y')
		self.validate_positive_int(z, 'z')
		self._AbstractGrid__x = x
		self._AbstractGrid__y = y
		self._AbstractGrid__z = z
		self.calculate_strides()

	def set_scale(self, x_length=None, y_length=None, z_length=None):
		"""
		Set size of grid unit cell.

		Trigger calculation of unit cell volume.

		Arguments:
			x_length (:class:`Length`): Length of grid unit cell in x
				dimension.
			y_length (:class:`Length`): Length of grid unit cell in y
				dimension.
			z_length (:class:`Length`): Length of grid unit cell in z
				dimension.

		Returns:
			None
		"""
		self.validate_length(x_length, 'x_length')
		self.validate_length(y_length, 'y_length')
		self.validate_length(z_length, 'z_length')
		self._AbstractGrid__x_unit_length = x_length
		self._AbstractGrid__y_unit_length = y_length
		self._AbstractGrid__z_unit_length = z_length
		self.__unit_volume = (x_length * y_length * z_length)

	def set_order(self, order='xyz'):
		"""
		Grid traversal order.

		Traversal order determines grid strides. For instance, a
		5 x 3 x 4 grid in order 'xyz' has unit stride in the
		x-dimension, stride 5 in the y dimension, and stride 15 in the
		z-dimension.

		Arguments:
			order (:obj:`str`): Traversal order of the grid.

		Returns:
			None

		Raises:
			TypeError: If ``order`` not of type :obj:`str`.
			ValueError: If ``order`` not a permutation of 'xyz'.
		"""
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
		""" Tuple of grid dimensions. """
		return (self._AbstractGrid__x, self._AbstractGrid__y,
				self._AbstractGrid__z)

	@property
	def unit_dimensions(self):
		""" Tuple of lengths of grid's unit cell. """
		return self.x_unit_length, self.y_unit_length, self.z_unit_length

	@property
	def unit_volume(self):
		""" Volume of grid's unit cell. """
		return self.__unit_volume.to_cm3

	@property
	def total_dimensions(self):
		""" Tuple of total lengths along each grid dimension. """
		return self.x_length, self.y_length, self.z_length

	@property
	def z_unit_length(self):
		""" Length of z-dimension of grid's unit cell. """
		return self._AbstractGrid__z_unit_length

	@property
	def z_length(self):
		""" Length of z-dimension of entire grid. """
		return self._AbstractGrid__z * self._AbstractGrid__z_unit_length

	def index2position(self, index):
		"""
		Convert ``index`` to (x, y, z) position in grid.

		Arguments:
			index (:obj:`int`): Index of grid element.

		Returns:
			:obj:`tuple` of :obj:`int`: Tuple representing (x, y, z)
				grid position of element at requested index.

		Raises:
			ValueError: If ``index`` exceeds grid bounds.
		"""
		index = int(index)
		self.validate_nonnegative_int(index, 'index')

		gridsize = reduce(op.mul, self.shape)
		if index >= gridsize:
			raise ValueError('index {} outside of geometry with {} '
							 'elements'.format(index, gridsize))

		for i in xrange(3):
			self.__pos[self.order[2 - i]] = int(
					index / self.strides[self.order[2 - i]])
			index = index % self.strides[self.order[2 - i]]

		return self.__pos['x'], self.__pos['y'], self.__pos['z']

	def position2index(self, x, y, z):
		"""
		Convert (``x``,``y``,``z``) position in grid to grid element index.

		Arguments:
			x (:obj:`int`): x position of grid element.
			y (:obj:`int`): y position of grid element.
			z (:obj:`int`): z position of grid element.

		Returns:
			int: Index of grid element at requested position.

		Raises:
			ValueError: If (``x``, ``y``, ``z``) exceeds grid bounds.
		"""
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
		""" String description of grid dimensions. """
		args = [self.voxels]
		args += list(self.shape)
		args += list(self.unit_dimensions)
		args += [self.unit_volume, self.order]
		return str('Geometry: \n\tvoxels: {}\n\tshape: {} x {} x {}'
				   '\n\tvoxel dimensions: {} x {} x {}\n\tvoxel volume: {}'
				   '\n\ttraversal order: {}'.format(*args))