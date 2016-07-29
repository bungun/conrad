from numpy import nan

from conrad.compat import *
from conrad.physics.grid import Grid3D

class VoxelGrid(Grid3D):
	def __init__(self, x_voxels=None, y_voxels=None, z_voxels=None, grid=None):
		if isinstance(grid, Grid3D):
			Grid3D.__init__(self, *grid.shape)
		else:
			Grid3D.__init__(self, x=x_voxels, y=y_voxels, z=z_voxels)

	@property
	def voxels(self):
		if self.x_voxels and self.y_voxels and self.z_voxels:
			return self.x_voxels * self.y_voxels * self.z_voxels
		else:
			return 0

	@property
	def total_volume(self):
		if self.unit_volume.value is nan:
			return nan * self.unit_volume
		else:
			return self.voxels * self.unit_volume

	@property
	def x_voxels(self):
		return self._AbstractGrid__x

	@property
	def y_voxels(self):
		return self._AbstractGrid__y

	@property
	def z_voxels(self):
		return self._AbstractGrid__z