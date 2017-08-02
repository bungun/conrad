"""
Define :class:`DoseFrame` and :class:`Physics` classes for treatment
planning.
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

import numpy as np
import scipy.sparse as sp

from conrad.defs import vec
from conrad.abstract.mapping import AbstractDiscreteMapping, DiscreteMapping, \
									map_type_to_string
from conrad.physics.beams import BeamSet
from conrad.physics.voxels import VoxelGrid
from conrad.physics.containers import WeightVector, DoseMatrix

class DoseFrame(object):
	r"""
	Describe a reference frame (voxels x beams) for dosing physics.

	A :class:`DoseFrame` provides a description of the mathematical
	basis of the dosing physics, which usually consists of a matrix in
	:math:`\mathbf{R}^{\mbox{voxels} \times \mbox{beams}}`, mapping the
	space of beam intensities, :math:`\mathbf{R}^\mbox{beams}` to the
	space of doses delivered to each voxel,
	:math:`\mathbf{R}^\mbox{voxels}`.

	For a given plan, we may require conversions between several related
	representations of the dose matrix. For instance, the beams may in
	fact be beamlets that can be coalesced into apertures, or---in order
	to accelerate the treatment plan optimization---may be clustered or
	sampled. Similarly, voxels may be clustered or sampled. For voxels,
	there is also a geometric frame, with ``X`` * ``Y`` * ``Z`` voxels,
	where the tuple (``X``, ``Y``, ``Z``) gives the dimensions of a
	regularly discretized grid, the so-called dose grid used in Monte
	Carlo simulations or ray tracing calculations. Since many of the
	voxels in this rectangular volume necessarily lie outside of the
	patient volume, there is only some number of voxels ``m`` < ``X`` *
	``Y`` * ``Z`` that are actually relevant to treatment planning.

	Accordingly, each :class:`DoseFrame` is intended to capture one such
	configuration of beams and voxels, with corresponding data on labels
	and/or weights attached to the configuration. Voxel labels allow
	each voxel to be mapped to an anatomical or clinical structure used
	in planning. The concept of beam labels is defined to allow beams to
	be gathered in logical groups (e.g. beamlets in fluence maps, or
	apertures in arcs) that may be constrained jointly or treated as a
	unit in some other way in an optimization context. Voxel and beam
	weights are defined for accounting purposes: if a :class:`DoseFrame`
	represents a set of clustered voxels or beams, the associated
	weights give the number of unitary voxels or beams in each cluster,
	so that optimization objective terms can be weighted appropriately.
	"""

	def __init__(self, voxels=None, beams=None, data=None, voxel_labels=None,
				 beam_labels=None, voxel_weights=None, beam_weights=None,
				 frame_name=None, **options):
		"""
		Initialize :class:`DoseFrame`.

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
		self.__voxels = np.nan
		self.__beams = np.nan
		self.__dose_matrix = None
		self.__voxel_labels = None
		self.__beam_labels = None
		self.__voxel_weights = None
		self.__beam_weights = None
		self.__name = 'unnamed_frame'

		if isinstance(beams, BeamSet):
			beams = beams.count

		if data is not None:
			self.dose_matrix = data
			if voxels is not None:
				if self.voxels != voxels:
					raise ValueError('when arguments `voxels` and `data`'
									 ' are both provided, the size'
									 ' specified by `voxels` must match '
									 ' first dimension of `data`\n'
									 ' {} != {}'.format(voxels, self.voxels))
			if beams is not None:
				if self.beams != beams:
					raise ValueError('when arguments `beams` and `data`'
									 ' are both provided, the size'
									 ' specified by `beams` must match '
									 ' second dimension of `data`\n'
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

		if frame_name is not None:
			self.name = frame_name

	@property
	def plannable(self):
		"""
		True if both dose matrix and voxel label data loaded.

		This can be achieved by having a contiguous matrix and a vector
		of voxel labels indicating the identity of each row of the
		matrix, or a dictionary of submatrices that map label keys to
		submatrix values.
		"""
		return bool(
				self.dose_matrix is not None and
				self.dose_matrix.contiguous <= (self.voxel_labels is not None))

	@property
	def shape(self):
		r"""
		Frame dimensions, :math:`\{\mathbf{R}^\mbox{voxels} \times \mathbf{R}^\mbox{beams}\}`.
		"""
		if self.voxels is None or self.beams is None:
			return None
		else:
			return self.voxels, self.beams

	@property
	def dose_matrix(self):
		"""
		Dose matrix.

		Setter will also use dimensions of input matrix to set any
		dimensions (:attr:`DoseFrame.voxels` or :attr:`DoseFrame.beams`)
		that are not already assigned at call time.

		Raises:
			TypeError: If input to setter is not a sparse or dense
				matrix type recognized by :mod:`conrad`.
			ValueError: If provided matrix dimensions inconsistent with
				known frame dimensions.
		"""
		return self.__dose_matrix

	@dose_matrix.setter
	def dose_matrix(self, data):
		mat = DoseMatrix(data)

		if self.voxels not in (None, np.nan):
			if mat.voxel_dim != self.voxels:
				raise ValueError('argument `data` must be a matrix '
								 'with {} rows'.format(self.voxels))
		else:
			self.voxels = mat.voxel_dim

		if self.beams not in (None, np.nan):
			if mat.beam_dim != self.beams:
				raise ValueError('argument `data` must be a matrix '
								 'with {} columns'.format(self.beams))
		else:
			self.beams = mat.beam_dim

		self.__dose_matrix = mat

	@property
	def voxels(self):
		"""
		Number of voxels in dose frame.

		If :attr:`DoseFrame.voxel_weights` has not been assigned at call
		time, the setter will initialize it to the 1 vector.

		Raises:
			ValueError: If :attr:`DoseFrame.voxels` already determined.
				Voxel dimension is a write-once property.
		"""
		return self.__voxels

	@voxels.setter
	def voxels(self, voxels):
		if self.voxels not in (None, np.nan):
			raise ValueError('{} property `voxels` cannot be changed '
							 'once set'.format(DoseFrame))
		if voxels is not None:
			self.__voxels = int(voxels)
			if self.voxel_weights is None:
				self.voxel_weights = np.ones(self.voxels, dtype=int)

	@property
	def beams(self):
		"""
		Number of beams in dose frame.

		If :attr:`DoseFrame.beam_weights` has not been assigned at call
		time, the setter will initialize it to the 1 vector.

		Raises:
			ValueError: If :attr:`DoseFrame.beams` already determined.
				Beam dimension is a write-once property.
		"""
		return self.__beams

	@beams.setter
	def beams(self, beams):
		if self.beams not in (None, np.nan):
			raise ValueError('{} property `beams` cannot be changed '
							 'once set'.format(DoseFrame))
		if beams is not None:
			beams = beams.count if isinstance(beams, BeamSet) else beams
			self.__beams = int(beams)
			if self.beam_weights is None:
				self.beam_weights = np.ones(self.beams, dtype=int)

	@property
	def voxel_labels(self):
		"""
		Vector of labels mapping voxels to structures.

		Setter will also use dimension of input vector to set voxel
		dimensions (:attr:`DoseFrame.voxels`) if not already assigned at
		call time.

		Raises:
			ValueError: If provided vector dimensions inconsistent with
				known frame dimensions.
		"""
		return self.__voxel_labels

	@voxel_labels.setter
	def voxel_labels(self, voxel_labels):
		if self.voxels in (None, np.nan):
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
		dimensions (:attr:`DoseFrame.beams`) if not already assigned at
		call time.

		Raises:
			ValueError: If provided vector dimensions inconsistent with
				known frame dimensions.
		"""
		return self.__beam_labels

	@beam_labels.setter
	def beam_labels(self, beam_labels):
		if self.beams in (None, np.nan):
			self.beams = len(beam_labels)
		if len(beam_labels) != self.beams:
			raise ValueError('length of `beam labels` ({}) must match '
							 'number of beams in frame ({})'
							 ''.format(len(beam_labels), self.beams))
		self.__beam_labels = vec(beam_labels).astype(int)

	@property
	def voxel_weights(self):
		"""
		Vector of weights assigned to each (meta-)voxel.

		Setter will also use dimension of input vector to set voxel
		dimensions (:attr:`DoseFrame.voxels`) if not already assigned at
		call time.

		Raises:
			ValueError: If provided vector dimensions inconsistent with
				known frame dimensions.
		"""
		if not isinstance(self.__voxel_weights, WeightVector):
			return None
		else:
			return self.__voxel_weights

	@voxel_weights.setter
	def voxel_weights(self, voxel_weights):
		voxel_weights = WeightVector(voxel_weights)
		if self.voxels in (None, np.nan):
			self.voxels = voxel_weights.size
		if voxel_weights.size != self.voxels:
			raise ValueError('length of `voxel_weights` ({}) must match '
							 'number of voxels in frame ({})'
							 ''.format(voxel_weights.size, self.voxels))
		self.__voxel_weights = voxel_weights

	@property
	def beam_weights(self):
		"""
		Vector of weights assigned to each (meta-)beam.

		Setter will also use dimension of input vector to set voxel
		dimensions (:attr:`DoseFrame.beams`) if not already assigned at
		call time.

		Raises:
			ValueError: If provided vector dimensions inconsistent with
				known frame dimensions.
		"""
		if not isinstance(self.__beam_weights, WeightVector):
			return None
		else:
			return self.__beam_weights

	@beam_weights.setter
	def beam_weights(self, beam_weights):
		beam_weights = WeightVector(beam_weights)
		if self.beams in (None, np.nan):
			self.beams = beam_weights.size
		if beam_weights.size != self.beams:
			raise ValueError('length of `beam_weights` ({}) must match '
							 'number of beams in frame ({})'
							 ''.format(beam_weights.size, self.beams))
		self.__beam_weights = beam_weights

	@property
	def name(self):
		return self.__name

	@name.setter
	def name(self, name):
		if name is not None:
			self.__name = str(name)


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
			:class:`~numpy.ndarray`: Vector of indices at which the
			entries of ``label_vector`` are equal to ``label``.

		Raises:
			ValueError: If ``label_vector`` is ``None``.
			KeyError: If ``label`` not found in ``label_vector``.

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
		"""
		Get indices of voxels labeled ``label`` in this :class:`DoseFrame`.
		"""
		return self.indices_by_label(self.voxel_labels, label, 'voxel_labels')

	def beam_lookup_by_label(self, label):
		"""
		Get indices of beam labeled ``label`` in this :class:`DoseFrame`.
		"""
		return self.indices_by_label(self.beam_labels, label, 'beam_labels')

	def submatrix(self, voxel_label=None, beam_label=None):
		if self.dose_matrix is None:
			raise AttributeError(
					'`{}.dose_matrix` must be set tp slice into '
					'submatrices'.format(DoseFrame))
		return self.dose_matrix.slice(
				voxel_label, beam_label, self.voxel_lookup_by_label,
				self.beam_lookup_by_label)

	def __str__(self):
		""" String of :class:`DoseFrame` dimensions. """
		return str('Dose Frame: {} VOXELS by {} BEAMS'.format(
				self.voxels, self.beams))

class DoseFrameMapping(object):
	def __init__(self, source_name, target_name, voxel_map=None, beam_map=None):
		self.__source = str(source_name)
		self.__target = str(target_name)
		self.__voxel_map = None
		self.__beam_map = None

		if voxel_map is not None:
			self.voxel_map = voxel_map
		if beam_map is not None:
			self.beam_map = beam_map

	@property
	def source(self):
		return self.__source

	@property
	def target(self):
		return self.__target

	@property
	def voxel_map(self):
		return self.__voxel_map

	@voxel_map.setter
	def voxel_map(self, voxel_map):
		if not isinstance(voxel_map, AbstractDiscreteMapping):
			voxel_map = DiscreteMapping(voxel_map)

		if not isinstance(voxel_map, AbstractDiscreteMapping):
			raise TypeError(
					'`voxel_map` must be derived from {} (or castable '
					'as {})'
					''.format(AbstractDiscreteMapping, DiscreteMapping))
		else:
			self.__voxel_map = voxel_map

	@property
	def voxel_map_type(self):
		if self.voxel_map is None:
			return None
		return map_type_to_string(self.voxel_map)

	@property
	def beam_map(self):
		return self.__beam_map

	@beam_map.setter
	def beam_map(self, beam_map):
		if not isinstance(beam_map, AbstractDiscreteMapping):
			beam_map = AbstractDiscreteMapping(beam_map)

		if not isinstance(beam_map, AbstractDiscreteMapping):
			raise TypeError(
					'`voxel_map` must be derived from {} (or castable '
					'as {})'
					''.format(AbstractDiscreteMapping, DiscreteMapping))
		else:
			self.__beam_map = beam_map

	@property
	def beam_map_type(self):
		if self.beam_map is None:
			return None
		return map_type_to_string(self.beam_map)

class Physics(object):
	"""
	Class managing all dose-related information for treatment planning.

	A :class:`Physics` instance includes one or more
	:class:`DoseFrames`, each with attached data including voxel
	dimensions, beam dimensions, a voxel-to-structure mapping, and a
	dose influence matrix. The class also provides an interface for
	adding and switching between frames, and extracting data from the
	active frame.

	A :class:`Physics` instance optionally has an associated
	:class:`VoxelGrid` that represents the dose grid used for dose
	matrix calculation, and that provides the necessary geometric
	information for reconstructing and rendering the 3-D dose
	distribution (or 2-D slices thereof).
	"""

	def __init__(self, voxels=None, beams=None, dose_matrix=None,
				 dose_grid=None, voxel_labels=None, **options):
		"""
		Initialize :class:`Physics`.

		Arguments:
			voxels (int or :class:`Physics`, optional): Number of voxels
				in initial :attr:`Physics.frame`. If argument is of type
				:class:`Physics`, initializer acts as a copy
				constructor.
			beams (:obj:`int` or :class:`BeamSet`, optional): Number of
				beams or :class:`BeamSet` object describing beam
				configuration in initial :attr:`Physics.frame`.
			dose_matrix (optional): Dose matrix assigned to initial
				:attr:`Physics.frame`.
			dose_grid (:class:`VoxelGrid`, optional): Three dimensional
				grid, defines number and layout of voxels in geometric
				dose frame. Used for, e.g., visualizing 2-D slices of
				the dose distribution.
			**options: Arbitrary keyword arguments, passed to
				:class:`DoseFrame` initializer to determine properties
				of initial :attr:`Physics.frame`.
		"""
		self.__frames = {}
		self.__frame_mappings = []
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

		f = options.pop('dose_frame', None)
		if not isinstance(f, DoseFrame):
			f = DoseFrame(
					voxels, beams, dose_matrix, voxel_labels=voxel_labels,
					**options)

		if self.__beams is None and isinstance(f.beams, int):
			self.__beams = BeamSet(f.beams)
		elif isinstance(beams, BeamSet):
			self.__beams = beams

		if 'unnamed' in f.name:
			f.name = options.pop('frame0_name', DEFAULT_FRAME0_NAME)
		self.__dose_frame = f
		self.__frames[f.name] = f

		if self.dose_grid is not None:
			if self.dose_grid.voxels == f.voxels:
				self.__frames['geometric'] = f

	@property
	def frame(self):
		"""
		Handle to :class:`DoseFrame` representing current dosing configuration.
		"""
		return self.__dose_frame

	@property
	def plannable(self):
		"""
		True if current frame has both dose matrix and voxel label data
		"""
		return self.frame.plannable

	@property
	def data_loaded(self):
		""" ``True`` if a client has seen data from the current dose frame. """
		return self.__FRAME_LOAD_FLAG

	def mark_data_as_loaded(self):
		""" Allow clients to mark dose frame data as seen. """
		self.__FRAME_LOAD_FLAG = True

	def add_dose_frame(self, key, **frame_args):
		"""
		Add new :class:`DoseFrame` representation of a dosing configuration.

		Arguments:
			key: A new :class:`DoseFrame` will be added to the
				:class:`Physics` object's dictionary with the key
				``key``.
			**frame_args: Keyword arguments passed to :class:`DoseFrame`
				initializer.

		Returns:
			None

		Raises:
			ValueError: If ``key`` corresponds to an existing key in the
				:class:`Physics` object's dictionary of dose frames.
		"""
		if key in self.__frames:
			raise ValueError('key `{}` already exists in {} frame '
							 'dictionary'.format(key, Physics))

		f = frame_args.pop('dose_frame', None)
		if not isinstance(f, DoseFrame):
			f = DoseFrame(**frame_args)

		self.__frames[key] = f
		self.__frames[key].name = key

	def change_dose_frame(self, key):
		"""
		Switch between dose frames already attached to :class:`Physics`.
		"""
		if not key in self.__frames:
			raise KeyError('no dose data frame found for key "{}"'.format(key))
		self.__dose_frame = self.__frames[key]
		self.__FRAME_LOAD_FLAG = False

	@property
	def available_frames(self):
		"""
		List of keys to dose frames already attached to :class:`Physics`.
		"""
		return self.__frames.keys()

	@property
	def unique_frames(self):
		"""
		List of unique dose frames attached to :class:`Physics`.
		"""
		return list(set(self.__frames.values()))

	@property
	def beams(self):
		""" Number of beams in current :attr:`Physics.frame`. """
		return self.frame.beams

	@property
	def voxels(self):
		""" Number of voxels in current :attr:`Physics.frame`. """
		return self.frame.voxels

	@property
	def dose_grid(self):
		""" Three-dimensional grid. """
		return self.__dose_grid

	@dose_grid.setter
	def dose_grid(self, grid):
		self.__dose_grid = VoxelGrid(grid=grid)

	@property
	def dose_matrix(self):
		""" Dose influence matrix for current :attr:`Physics.frame`. """
		return self.frame.dose_matrix

	@dose_matrix.setter
	def dose_matrix(self, dose_matrix):
		self.frame.dose_matrix = dose_matrix
		if self.__beams is None and isinstance(self.frame.beams, int):
			self.__beams = BeamSet(self.frame.beams)

	@property
	def voxel_labels(self):
		"""
		Vector mapping voxels to structures in current :attr:`Physics.frame`.
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
				submatrix of current :attr:`Physics.dose_matrix` based
				on row indices for which ``voxel_label`` matches the
				entries of :attr:`Physics.voxel_labels`. All rows
				returned if no label provided.
			beam_label (optional): Label for which to build/retrieve
				submatrix of current :attr:`Physics.dose_matrix` based
				on column indices for which ``beam_label`` matches the
				entries of :attr:`Physics.frame.beam_labels`. All
				columns returned if no label provided.

		Returns:
			Submatrix of dose matrix attached to current
			:attr:`Physics.frame`, based on row indices for which
			:attr:`Physics.frame.voxel_labels` matches the queried
			``voxel_label``, and column indices for which
			:attr:`Physics.frame.beam_labels` matches the queried
			``beam_label``.
		"""
		return self.frame.submatrix(voxel_label, beam_label)

	def voxel_weights_by_label(self, label):
		""" Subvector of voxel weights, filtered by ``label``. """
		if label not in self.frame.voxel_weights:
			indices = self.frame.voxel_lookup_by_label(label)
		else:
			indices = None
		return self.frame.voxel_weights.slice(label, indices)

	def beam_weights_by_label(self, label):
		""" Subvector of beam weights, filtered by ``label``. """
		if label not in self.frame.beam_weights:
			indices = self.frame.beam_lookup_by_label(label)
		else:
			indices = None
		return self.frame.beam_weights.slice(label, indices)

	def split_dose_by_label(self, dose_vector, labels):
		if isinstance(dose_vector, dict):
			doses = dose_vector
		else:
			y = vec(dose_vector)
			if y.size != self.voxels:
				raise ValueError(
						'input vector must match voxel dimension of '
						'current dose frame')
			doses = {label: y[self.frame.voxel_lookup_by_label(label)]}
		return doses

	@property
	def available_frame_mappings(self):
		return [(fm.source, fm.target) for fm in self.__frame_mappings]

	def add_frame_mapping(self, mapping):
		if not isinstance(mapping, DoseFrameMapping):
			raise TypeError(
					'argument `mapping` must be of type {}'
					''.format(DoseFrameMapping))
		for fm in self.__frame_mappings:
			if fm.source == mapping.source and fm.target == mapping.target:
				raise ValueError(
						'frame mapping for source=`{}`, target=`{}` '
						'already attached to {}'
						''.format(fm.source, fm.target, Physics))
		self.__frame_mappings.append(mapping)

	def retrieve_frame_mapping(self, source_frame, target_frame):
		if source_frame == target_frame:
			raise ValueError(
					'arguments `source_frame` and `target_frame` are '
					'identical, no frame mapping retrieved')
		for fm in self.__frame_mappings:
			if fm.source == source_frame and fm.target == target_frame:
				return fm
		raise ValueError(
				'no frame mapping found for source=`{}`, target=`{}`'
				''.format(source_frame, target_frame))

DEFAULT_FRAME0_NAME = 'frame0'
