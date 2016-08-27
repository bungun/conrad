"""
Define :class:`VoxelGrid` to describe dose grids used in dose
calculations, or other regular voxel grids used in treatment planning,
such as CT/MRI/PET scan data sets.
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

from conrad.physics.grid import Grid3D

class VoxelGrid(Grid3D):
	""" Specialize :class:`Grid3D` to (regular) voxel grids. """

	def __init__(self, x_voxels=None, y_voxels=None, z_voxels=None, grid=None):
		"""
		Initialize :class:`VoxelGrid` as :class:`Grid3D` instance.

		Arguments:
			x_voxels (int, optional): Number of voxels in grid's
				x-dimension.
			y_voxels (int, optional): Number of voxels in grid's
				y-dimension.
			z_voxels (int, optional): Number of voxels in grid's
				z-dimension.
			grid (:class:`Grid3D`, optional): Pre-existing
				three-dimensional grid from which to initialize grid
				shape.
		"""
		if isinstance(grid, Grid3D):
			Grid3D.__init__(self, *grid.shape)
		else:
			Grid3D.__init__(self, x=x_voxels, y=y_voxels, z=z_voxels)

	@property
	def voxels(self):
		""" Number of voxels in grid. """
		if self.x_voxels and self.y_voxels and self.z_voxels:
			return self.x_voxels * self.y_voxels * self.z_voxels
		else:
			return 0

	@property
	def total_volume(self):
		""" Total volume of grid; undefine if unit volume unknown. """
		if self.unit_volume.value is nan:
			return nan * self.unit_volume
		else:
			return self.voxels * self.unit_volume

	@property
	def x_voxels(self):
		""" Width of grid's x-dimension, in voxels. """
		return self._AbstractGrid__x

	@property
	def y_voxels(self):
		""" Width of grid's y-dimension, in voxels. """
		return self._AbstractGrid__y

	@property
	def z_voxels(self):
		""" Width of grid's z-dimension, in voxels. """
		return self._AbstractGrid__z