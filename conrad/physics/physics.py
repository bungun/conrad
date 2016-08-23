"""
Define DoseFrame and Physics classes for treatment planning.

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
from numpy import ndarray, ones, nan

from conrad.compat import *
from conrad.defs import vec, sparse_or_dense, CONRAD_MATRIX_TYPES
from conrad.physics.beams import BeamSet
from conrad.physics.voxels import VoxelGrid

class DoseFrame(object):
	"""
	Describe a reference frame (voxels x beams) for dosing physics.

	A `DoseFrame` object provides a description of the mathematical
	basis of the dosing physics, which usually consists of a matrix in
	R^{voxels x beams}, mapping the space of beam intensities, R^{beams}
	to the space of doses delivered to each voxel, R^{voxels}.

	For a given plan, we may require conversions between several related
	representations of the dose matrix. For instance, the beams may in
	fact be beamlets that can be coalesced into apertures, or---in order
	to accelerate the treatment plan optimization---may be clustered or
	sampled. Similarly, voxels may be clustered or sampled. For voxels,
	there is also a geometric frame, with X * Y * Z voxels, where the
	tuple (X, Y, Z) gives the dimensions of a regularly discretized
	grid, the so-called dose grid used in Monte Carlo simulations or ray
	tracing calculations. Since many of the voxels in this rectangular
	volume necessarily lie outside of the patient volume, there is only
	some number of voxels m < X * Y * Z that are actually relevant to
	treatment planning.

	Accordingly, each `DoseFrame` is intended to capture one such
	configuration of beams and voxels, with corresponding data on labels
	and/or weights attached to the configuration. Voxel labels allow
	each voxel to be mapped to an anatomical or clinical structure used
	in planning. The concept of beam labels is defined to allow beams to
	be gathered in logical groups (e.g. beamlets in fluence maps, or
	apertures in arcs) that may be constrained jointly or treated as a
	unit in some other way in an optimization context. Voxel and beam
	weights are defined for accounting purposes: if a `DoseFrame`
	represents a set of clustered voxels or beams, the associated
	weights give the number of unitary voxels or beams in each cluster,
	so that optimization objective terms can be weighted appropriately.
	"""

	def __init__(self, voxels=None, beams=None, data=None, voxel_labels=None,
				 beam_labels=None, voxel_weights=None, beam_weights=None):
		"""
		Initialize `DoseFrame`.

		Arguments:
			voxels (int, optional): Number of voxels in frame.
			beams (int, optional): Number of beams in frame.
			data (optional): Dose matrix.
			voxel_labels (optional): Vector of labels mapping each voxel
				to a structure.
			beam_labels (optional): Vector of labels mapping each beam
				to a group.
			voxel_weights (optional): Vector of weights, e.g., number of
				voxels in each cluster if working in a voxel-clustered
				frame.
			beam_weights (optional): Vector of weights, e.g., number of
				beams in each cluster if working in a beam-clustered
				frame.

		Raises:
			ValueError: If dimensions implied by arguments are
				inconsistent.
		"""
		self.__voxels = nan
		self.__beams = nan
		self.__data = None
		self.__voxel_labels = None
		self.__beam_labels = None
		self.__voxel_weights = None
		self.__beam_weights = None

		if isinstance(beams, BeamSet):
			beams = beams.count

		if data is not None:
			self.data = data
			if voxels is not None:
				if self.voxels != voxels:
					raise ValueError('when arguments "voxels" and "data"'
									 ' are both provided, the size'
									 ' specified by "voxels" must match '
									 ' first dimension of "data"\n'
									 ' {} != {}'.format(voxels, self.voxels))
			if beams is not None:
				if self.beams != beams:
					raise ValueError('when arguments "beams" and "data"'
									 ' are both provided, the size'
									 ' specified by "beams" must match '
									 ' second dimension of "data"\n'
									 ' {} != {}'.format(beams, self.beams))
		else:
			self.voxels = voxels
			self.beams = beams

		if voxel_labels is not None:
			self.voxel_labels = voxel_labels
		if beam_labels is not None:
			self.beam_labels = beam_labels

		if voxel_weights is not None:
			self.voxel_weights = voxel_weights
		if beam_weights is not None:
			self.beam_weights = beam_weights

	@property
	def plannable(self):
		""" True if dose matrix and voxel labels assigned to frame. """
		return self.data is not None and self.voxel_labels is not None

	@property
	def shape(self):
		""" Frame dimensions, {R^voxels x R^beams}. """
		if self.voxels is None or self.beams is None:
			return None
		else:
			return self.voxels, self.beams

	@property
	def data(self):
		"""
		Dose matrix.

		Setter will also use dimensions of input matrix to set any
		dimensions (`DoseFrame.voxels` or `DoseFrame.beams`) that are
		not already assigned at call time.

		Raises:
			TypeError: If input to setter is not a sparse or dense
				matrix type recognized by CONRAD.
			ValueError: If provided matrix dimensions inconsistent with
				known frame dimensions.
		"""
		return self.__data

	@data.setter
	def data(self, data):
		if not sparse_or_dense(data):
			raise TypeError('argument "data" must be a dense or sparse '
							'matrix, in the form of a {}, {}, or {}'
							''.format(*CONRAD_MATRIX_TYPES))

		if self.voxels not in (None, nan):
			if data.shape[0] != self.voxels:
				raise ValueError('argument "data" must be a matrix '
								 'with {} rows'.format(self.voxels))
		else:
			self.voxels = data.shape[0]

		if self.beams not in (None, nan):
			if data.shape[1] != self.beams:
				raise ValueError('argument "data" must be a matrix '
								 'with {} columns'.format(self.beams))
		else:
			self.beams = data.shape[1]

		self.__data = data

	@property
	def voxels(self):
		"""
		Number of voxels in dose frame.

		If `DoseFrame.voxel_weights` has not been assigned at call time,
		the setter will initialize it to the 1 vector.

		Raises:
			ValueError: If `DoseFrame.voxels` already determined. Voxel
				dimension is a write-once property.
		"""
		return self.__voxels

	@voxels.setter
	def voxels(self, voxels):
		if self.voxels not in (None, nan):
			raise ValueError('{} property "voxels" cannot be changed '
							 'once set'.format(DoseFrame))
		if voxels is not None:
			self.__voxels = int(voxels)
			if self.voxel_weights is None:
				self.voxel_weights = ones(self.voxels, dtype=int)

	@property
	def beams(self):
		"""
		Number of beams in dose frame.

		If `DoseFrame.beam_weights` has not been assigned at call time,
		the setter will initialize it to the 1 vector.

		Raises:
			ValueError: If `DoseFrame.beams` already determined. Beam
				dimension is a write-once property.
		"""
		return self.__beams

	@beams.setter
	def beams(self, beams):
		if self.beams not in (None, nan):
			raise ValueError('{} property "beams" cannot be changed '
							 'once set'.format(DoseFrame))
		if beams is not None:
			beams = beams.count if isinstance(beams, BeamSet) else beams
			self.__beams = int(beams)
			if self.beam_weights is None:
				self.beam_weights = ones(self.beams, dtype=int)

	@property
	def voxel_labels(self):
		"""
		Vector of labels mapping voxels to structures.

		Setter will also use dimension of input vector to set voxel
		dimensions (`DoseFrame.voxels`) if not already assigned at call
		time.

		Raises:
			ValueError: If provided vector dimensions inconsistent with
				known frame dimensions.
		"""
		return self.__voxel_labels

	@voxel_labels.setter
	def voxel_labels(self, voxel_labels):
		if self.voxels in (None, nan):
			self.voxels = len(voxel_labels)
		if len(voxel_labels) != self.voxels:
			raise ValueError('length of "voxel labels" ({}) must match '
							 'number of voxels in frame ({})'
							 ''.format(len(voxel_labels), self.voxels))
		self.__voxel_labels = vec(voxel_labels).astype(int)

	@property
	def beam_labels(self):
		"""
		Vector of labels mapping beams to beam groups.

		Setter will also use dimension of input vector to set beam
		dimensions (`DoseFrame.beams`) if not already assigned at call
		time.

		Raises:
			ValueError: If provided vector dimensions inconsistent with
				known frame dimensions.
		"""
		return self.__beam_labels

	@beam_labels.setter
	def beam_labels(self, beam_labels):
		if self.beams in (None, nan):
			self.beams = len(beam_labels)
		if len(beam_labels) != self.beams:
			raise ValueError('length of "beam labels" ({}) must match '
							 'number of beams in frame ({})'
							 ''.format(len(beam_labels), self.beams))
		self.__beam_labels = vec(beam_labels).astype(int)

	@property
	def voxel_weights(self):
		"""
		Vector of weights assigned to each (meta-)voxel.

		Setter will also use dimension of input vector to set voxel
		dimensions (`DoseFrame.voxels`) if not already assigned at call
		time.

		Raises:
			ValueError: If provided vector dimensions inconsistent with
				known frame dimensions.
		"""
		return self.__voxel_weights

	@voxel_weights.setter
	def voxel_weights(self, voxel_weights):
		if self.voxels in (None, nan):
			self.voxels = len(voxel_weights)
		if len(voxel_weights) != self.voxels:
			raise ValueError('length of "voxel_weights" ({}) must match '
							 'number of voxels in frame ({})'
							 ''.format(len(voxel_weights), self.voxels))
		self.__voxel_weights = vec(voxel_weights).astype(float)

	@property
	def beam_weights(self):
		"""
		Vector of weights assigned to each (meta-)beam.

		Setter will also use dimension of input vector to set voxel
		dimensions (`DoseFrame.beams`) if not already assigned at call
		time.

		Raises:
			ValueError: If provided vector dimensions inconsistent with
				known frame dimensions.
		"""
		return self.__beam_weights

	@beam_weights.setter
	def beam_weights(self, beam_weights):
		if self.beams in (None, nan):
			self.beams = len(beam_weights)
		if len(beam_weights) != self.beams:
			raise ValueError('length of "beam_weights" ({}) must match '
							 'number of beams in frame ({})'
							 ''.format(len(beam_weights), self.beams))
		self.__beam_weights = vec(beam_weights).astype(float)

	@staticmethod
	def indices_by_label(label_vector, label, vector_name):
		"""
		Retrieve indices of vector entries corresponding to a given value.

		Arguments:
			label_vector: Vector of values to search for entries
				corresponding
			label: Value to find.
			vector_name (:obj:`str`): Name of vector, for use in
				exception messages.

		Returns:
			`numpy.ndarray`: Vector of indices at which the entries of
				`label_vector` are equal to `label`.

		Raises:
			ValueError: If `label_vector` is `None`.
			KeyError: If `label` not found in `label_vector`.

		"""
		if label_vector is None:
			raise ValueError('`{}.{}` not set, retrieval by label '
							 'impossible'.format(DoseFrame, vector_name))

		indices = listmap(lambda x: x[0], listfilter(
				lambda x: x[1] == label, enumerate(label_vector)))
		if len(indices) == 0:
			raise KeyError('label {} not found in entries of field '
						   '"{}"'.format(label, vector_name))

		return vec(indices)

	def voxel_lookup_by_label(self, label):
		""" Get indices of voxels labeled `label` in this `DoseFrame`. """
		indices = self.indices_by_label(
				self.voxel_labels, label, 'voxel_labels')
		return indices

	def beam_lookup_by_label(self, label):
		""" Get indices of beam labeled `label` in this `DoseFrame`. """
		indices = self.indices_by_label(self.beam_labels, label, 'beam_labels')
		return indices


	def __str__(self):
		""" String of `DoseFrame` dimensions. """
		return str('Dose Frame: {} VOXELS by {} BEAMS'.format(
				self.voxels, self.beams))

class Physics(object):
	"""
	Class managing all dose-related information for treatment planning.

	A `Physics` instance includes one or more `DoseFrames`, each with
	attached data including voxel dimensions, beam dimensions, a
	voxel-to-structure mapping, and a dose influence matrix. The class
	also provides an interface for adding and switching between frames,
	and extracting data from the active frame.

	A `Physics` instance optionally has an associated `VoxelGrid` that
	represents the dose grid used for dose matrix calculation, and that
	provides the necessary geometric information for reconstructing and
	rendering the 3-D dose distribution (or 2-D slices thereof).
	"""

	def __init__(self, voxels=None, beams=None, dose_matrix=None,
				 dose_grid=None, voxel_labels=None, **options):
		"""
		Initialize `Physics`.

		Arguments:
			voxels (int or :class:`Physics`, optional): Number of voxels
				in initial `Physics.frame`. If argument is of type
				`Physics`, initializer acts as a copy constructor.
			beams (int or :class:`BeamSet`, optional): Number of beams
				or `BeamSet` object describing beam configuration in
				initial `Physics.frame`.
			dose_matrix (optional): Dose matrix assigned to initial
				`Physics.frame`.
			dose_grid (`VoxelGrid`, optional): Three dimensional grid,
				defines number and layout of voxels in geometric dose
				frame. Used for, e.g., visualizing 2-D slices of the
				dose distribution.
			**options: Arbitrary keyword arguments, passed to
				`DoseFrame` initializer to determine properties of
				initial `Physics.frame`.
		"""
		self.__frames = {}
		self.__dose_grid = None
		self.__dose_frame = None
		self.__beams = None
		self.__FRAME_LOAD_FLAG = False

		# copy constructor
		if isinstance(voxels, Physics):
			physics_in = voxels
			self.__frames = physics_in._Physics__frames
			self.__dose_grid = physics_in._Physics__dose_grid
			self.__dose_frame = physics_in._Physics__dose_frame
			self.__beams = physics_in._Physics__beams
			self.__FRAME_LOAD_FLAG = physics_in._Physics__FRAME_LOAD_FLAG
			return

		# normal initialization
		if dose_grid is not None:
			self.dose_grid = dose_grid

		if voxels is None and self.dose_grid is not None:
			voxels = self.dose_grid.voxels

		f = DoseFrame(voxels, beams, dose_matrix, voxel_labels=voxel_labels,
					  **options)

		if self.beams is None and isinstance(f.beams, int):
			self.__beams = BeamSet(f.beams)
		elif isinstance(beams, BeamSet):
			self.__beams = beams

		self.__dose_frame = f
		self.__frames[0] = f
		self.__frames['full'] = f

		if self.dose_grid is not None:
			if self.dose_grid.voxels == self.voxels:
				self.__frames['geometric'] = f

	@property
	def frame(self):
		"""
		Handle to `DoseFrame` representing current dosing configuration.
		"""
		return self.__dose_frame

	@property
	def data_loaded(self):
		""" True if a client has seen data from the current dose frame. """
		return self.__FRAME_LOAD_FLAG

	def mark_data_as_loaded(self):
		""" Allow clients to mark dose frame data as seen. """
		self.__FRAME_LOAD_FLAG = True

	def add_dose_frame(self, key, **frame_args):
		"""
		Add new `DoseFrame` representation of a dosing configuration.

		Arguments:
			key: A new `DoseFrame` will be added to the `Physics`
				object's dictionary with the key `key`.
			**frame_args: Keyword arguments passed to `DoseFrame`
				initializer.

		Returns:
			None

		Raises:
			ValueError: If `key` corresponds to an existing key in the
				`Physics` object's dictionary of dose frames.
		"""
		if key in self.__frames:
			raise ValueError('key "{}" already exists in {} frame '
							 'dictionary'.format(key, Physics))
		self.__frames[key] = DoseFrame()

	def change_dose_frame(self, key):
		"""
		Switch between dose frames already attached to `Physics`.
		"""
		if not key in self.__frames:
			raise KeyError('no dose data frame found for key {}')
		self.__dose_frame = self.__frames[key]
		self.__FRAME_LOAD_FLAG = False

	@property
	def available_frames(self):
		""" List of keys to dose frames already attached to `Physics`. """
		return self.__frames.keys()

	@property
	def plannable(self):
		""" True if current `Physics.frame.plannable`. """
		return self.frame.plannable

	@property
	def beams(self):
		""" Number of beams in current `Physics.frame`. """
		return self.frame.beams

	@property
	def voxels(self):
		""" Number of voxels in current `Physics.frame`. """
		return self.frame.voxels

	@property
	def dose_grid(self):
		""" Three-dimensional grid . """
		return self.__dose_grid

	@dose_grid.setter
	def dose_grid(self, grid):
		self.__dose_grid = VoxelGrid(grid=grid)

	@property
	def dose_matrix(self):
		""" Dose influence matrix for current `Physics.frame`. """
		return self.frame.data

	@dose_matrix.setter
	def dose_matrix(self, dose_matrix):
		self.frame.data = dose_matrix
		if self.__beams is None and isinstance(self.frame.beams, int):
			self.__beams = BeamSet(self.frame.beams)

	@property
	def voxel_labels(self):
		"""
		Vector mapping voxels to structures in current `Physics.frame`.
		"""
		return self.frame.voxel_labels

	@voxel_labels.setter
	def voxel_labels(self, voxel_labels):
		self.frame.voxel_labels = voxel_labels

	def dose_matrix_by_label(self, voxel_label=None, beam_label=None):
		"""
		Submatrix of dose matrix, filtered by voxel and beam labels.

		Arguments:
			voxel_label (optional): Label for which to build/retrieve
				submatrix of current `Physics.dose_matrix` based on
				row indices for which `voxel_label` matches the entries
				of `Physics.voxel_labels`. All rows returned if no label
				provided.
			beam_label (optional): Label for which to build/retrieve
				submatrix of current `Physics.dose_matrix` based on
				column indices for which `beam_label` matches the
				entries of `Physics.frame.beam_labels`. All columns
				returned if no label provided.

		Returns:
			Submatrix of dose matrix attached to current `Physics.frame`,
			based on row indices for which `Physics.frame.voxel_labels`
			matches the queried `voxel_label`, and column indices for
			which `Physics.frame.beam_labels` matches the queried
			`beam_label`.
		"""
		if voxel_label is not None:
			v_indices = self.frame.voxel_lookup_by_label(voxel_label)
		else:
			v_indices = xrange(self.frame.voxels)

		if beam_label is not None:
			b_indices = self.frame.beam_lookup_by_label(beam_label)
		else:
			b_indices = xrange(self.frame.beams)

		return self.dose_matrix[v_indices, :][:, b_indices]

	def voxel_weights_by_label(self, label):
		""" Subvector of voxel weights, filtered by `label`. """
		indices = self.frame.voxel_lookup_by_label(label)
		return self.frame.voxel_weights[indices]

	def beam_weights_by_label(self, label):
		""" Subvector of beam weights, filtered by `label`. """
		indices = self.frame.beam_lookup_by_label(label)
		return self.frame.beam_weights[indices]