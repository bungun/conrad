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

	@energy.setter
	def energy(self, energy):
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

# class Path(BeamSet):
# 	def __init__(self, stations=None):
# 		BeamSet.__init__(self)
# 		self.__stations = []

# 		stations if stations else []

# 	@staticmethod
# 	def valid_beam(beam):
# 		return isinstance(b, (Aperture, FluenceMap))

# 	@property
# 	def count(self):
# 		return len(self.__stations)

# 	@property
# 	def stations(self):
# 		return self.__stations

# 	@stations.setter
# 	def stations(self, stations):
# 		if not isinstance(stations, (list, tuple)):
# 			raise TypeError('explain')
# 		elif not all(listmap(self.valid_beam, stations)):
# 			raise TypeError('explain')
# 		else:
# 			self.__stations = list(stations)

# 	def __iadd__(self, other):
# 		if not isinstance(other, (list, tuple, Aperture, FluenceMap)):
# 			raise TypeError('explain')

# 		if bool(isinstance(self, (list, tuple))):
# 			if not all(listmap(self.valid_beam, other)):
# 				raise TypeError('explain')
# 			else:
# 				self.stations += list(other)
# 		else:
# 			self.stations.append(other)

# 		return self

# class Arc(Path):
# 	def __init__(self, stations=None):
# 		Path.__init__(self, stations)


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

class FluenceMap(AbstractBeam):
	def __init__(self, size1, size2):
		AbstractBeam.__init__(self)
		self.__bixel_grid = BixelGrid(size1, size2)

	@property
	def count(self):
		return self.__bixel_grid.bixels

	# methods for converting to aperture

class BeamSet(AbstractBeam):
	def __init__(self, beams=None):
		AbstractBeam.__init__(self)
		self.__beams = []
		if beams:
			if isinstance(beams, int):
				self.beams = [Beam() for b in xrange(max(0, beams))]
			else:
				self.beams = beams

	@property
	def count(self):
		if len(self.beams) == 0:
			return 0
		else:
			return sum([b.count for b in self.beams])

	@property
	def beams(self):
		return self.__beams

	@beams.setter
	def beams(self, beams):
		if isinstance(beams, BeamSet):
			self.__beams = beams.beams
		else:
			try:
				beam_orig = self.beams
				self.__beams = []
				for b in beams:
					self += b
			except:
				self.__beams = beam_orig
				raise TypeError('argument "beams" must be an iterable '
								'collection of {} objects, or a {}'
								''.format(AbstractBeam, BeamSet))

	def __iadd__(self, other):
		if isinstance(other, AbstractBeam):
			self.__beams.append(other)
		else:
			raise TypeError('addition for {} only defined for objects '
							'inheriting from {}'.format(BeamSet, AbstractBeam))
		return self