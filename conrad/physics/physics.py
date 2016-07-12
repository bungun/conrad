from operator import add
from numpy import nan, ndarray
from scipy.sparse import isspmatrix

from conrad.compat import *
from conrad.physics.grid import Grid2D, Grid3D

class BeamTypes(object):
	ELECTRON = 'electron'
	PARTICLE = 'particle'
	PHOTON = 'photon'
	PROTON = 'proton'
	__types = (ELECTRON, PARTICLE, PHOTON, PROTON)

	def validate(self, beamtype):
		return beamtype in self.__types

BEAM_TYPES = BeamTypes()

class Physics(object):
	def __init__(self, voxels, beams, dose_matrix=None):
		self.__dose_grid = None
		self.__beam_set = None
		self.__dose_matrix = None

		if not isinstance(voxels, VoxelGrid):
			raise TypeError('explain')

		if not isinstance(beams, BeamSet):
			raise TypeError('explain')

		self.__dose_grid = voxels
		self.__beam_set = beams

	@property
	def beams(self):
		return self.__beam_set.beam_count

	@property
	def voxels(self):
	    return self.__dose_grid.voxels

	@property
	def dose_matrix(self):
		return self.__dose_matrix

	@dose_matrix.setter
	def dose_matrix(self, dose_matrix):
		if not (isinstance(dose_matrix, ndarray) or isspmatrix(dose_matrix)):
			raise TypeError('explain')
		elif len(dose_matrix.shape) != 2:
			raise ValueError('explain')
		elif dose_matrix.shape != (self.voxels, self.beams)
			raise ValueError('explain')

		self.__dose_matrix = dose_matrix

class BeamSet(object):
	def __init__(self, beams=None, n_beams=None):
		self.__beams = []
		if beams:
			self.beams = beams

		if isinstance(n_beams, int):
			self.beams = max(0, n_beams) * [Beam()]

	@property
	def beam_count(self):
		return 0 + reduce(add, listmap(lambda b : b.count, self.__beams))

	@beams.setter
	def beams(self, beams):
		self.beams = beams

	# def __iadd__(self, other):
		# return self

class AbstractBeam(object):
	def __init__(self):
		self.__type = '<unknown beam type>'
		self.__energy = nan
		self.__limit = nan

	@property
	def count(self):
		return 1

	@property
	def energy(self):
		return self.__energy

	@energy.setter(self, energy):
		self.__energy = float(energy)

		# position (x, y, z, or r, phi, theta)
		# unit normal (orientation, cartesian?)

	@property
	def type(self):
		return self.__type

	@type.setter
	def type(self, beam_type):
		if  BEAM_TYPES.validate(beam_type):
			self.__type = beam_type

	@property
	def limit(self):
	    return self.__limit

	@limit.setter
	def limit(self, limit):
		if not isinstance(limit, (int, float)):
			raise TypeError('beam limit must be an {} or {}'.format(
							int, float))
		elif limit <= 0:
			raise ValueError('beam limit must be positive')

	   self.__limit = float(limit)



class Path(AbstractBeam):
	def __init__(self, stations=None):
		AbstractBeam.__init__(self)
		self.__stations = []

		stations if stations else []

	@staticmethod
	def valid_beam(beam):
		return isinstance(b, (Aperture, FluenceMap))

	@property
	def count(self):
		return len(self.__stations)

	@stations.setter
	def stations(self, stations):
		if not isinstance(stations, (list, tuple)):
			raise TypeError('explain')
		elif not all(listmap(self.valid_beam, stations)):
			raise TypeError('explain')
		else:
			self.__stations = list(stations)

	def __iadd__(self, other):
		if not isinstance(other, (list, tuple, Aperture, FluenceMap)):
			raise TypeError('explain')

		if bool(isinstance(self, (list, tuple)):
			if not all(listmap(self.valid_beam, other))):
				raise TypeError('explain')
			else:
				self.stations += list(other)
		else:
			self.stations.append(other)

		return self

class Arc(Path):
	def __init__(self, stations=None):
		Path.__init__(self, stations)


class Beam(AbstractBeam):
	def __init__(self):
		AbstractBeam.__init__(self)

	@property
	def count(self):
		return 1

	# convert between this and fluence map

class Beamlet(AbstractBeam):
	def __init__(self):
		AbstractBeam.__init__(self)

	@property
	def count(self):
		return 1

class Aperture(AbstractBeam):
	def __init__(self):
		AbstractBeam.__init__(self)

	# convert between this and fluence map

class FluenceMap(AbstractBeam):
	def __init__(self, size1, size2):
		AbstractBeam.__init__(self)
		self.__bixel_grid = BixelGrid(size1, size2)

	# active
	# banned
	# methods for converting to aperture

class VoxelGrid(Grid3D):
	def __init__(self, x_voxels=None, y_voxels=None, z_voxels=None):
		Grid3D.__init__(self, x=x_voxels, y=y_voxels, z=z_voxels)

	@property
	def voxels(self):
		return self.x_voxels * self.y_voxels * self.z_voxels

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

class BixelGrid(Grid2D):
	def __init__(self, x_bixels=None, y_bixels=None):
		Grid2D.__init__(self, x=x_bixels, y=y_bixels)

	@property
	def bixels(self):
		return self.x_bixels * self.y_bixels

	@property
	def x_bixels(self):
		return self._AbstractGrid__x

	@property
	def y_bixels(self):
		return self._AbstractGrid__y