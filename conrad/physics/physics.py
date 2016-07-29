from numpy import ndarray, ones

from conrad.compat import *
from conrad.defs import vec, sparse_or_dense, CONRAD_MATRIX_TYPES
from conrad.physics.beams import BeamSet
from conrad.physics.voxels import VoxelGrid

class DoseFrame(object):
	def __init__(self, voxels, beams, data, voxel_labels=None,
				 beam_labels=None, voxel_weights=None,
				 beam_weights=None):
		self.__voxels = None
		self.__beams = None
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
		elif self.voxels is not None:
			self.voxel_weights = ones(self.voxels, dtype=int)

		if beam_weights is not None:
			self.beam_weights = beam_weights
		elif self.beams is not None:
			self.beam_weights = ones(self.beams, dtype=int)

		if self.voxels is None or self.beams is None:
			raise ValueError('minimal requirements for initialization:'
							 '\n-argument "data" is a matrix OR '
							 '\n-other arguments combine to '
							 'specify/imply the VOXELS x BEAMS '
							 'dimensions of the {}'
							 ''.format(DoseFrame))

	@property
	def plannable(self):
		return self.data is not None and self.voxel_labels is not None

	@property
	def shape(self):
		if self.voxels is None or self.beams is None:
			return None
		else:
			return self.voxels, self.beams

	@property
	def data(self):
		return self.__data

	@data.setter
	def data(self, data):
		if not sparse_or_dense(data):
			raise TypeError('argument "data" must be a dense or sparse '
							'matrix, in the form of a {}, {}, or {}'
							''.format(*CONRAD_MATRIX_TYPES))

		if self.voxels:
			if data.shape[0] != self.voxels:
				raise ValueError('argument "data" must be a matrix '
								 'with {} rows'.format(self.voxels))
		else:
			self.voxels = data.shape[0]

		if self.beams:
			if data.shape[1] != self.beams:
				raise ValueError('argument "data" must be a matrix '
								 'with {} columns'.format(self.beams))
		else:
			self.beams = data.shape[1]

		self.__data = data

	@property
	def voxels(self):
		return self.__voxels

	@voxels.setter
	def voxels(self, voxels):
		if self.voxels:
			raise ValueError('{} property "voxels" cannot be changed '
							 'once set'.format(DoseFrame))
		if voxels:
			self.__voxels = int(voxels)

	@property
	def beams(self):
		return self.__beams

	@beams.setter
	def beams(self, beams):
		if self.beams:
			raise ValueError('{} property "beams" cannot be changed '
							 'once set'.format(DoseFrame))
		if beams:
			self.__beams = int(beams)

	@property
	def voxel_labels(self):
		return self.__voxel_labels

	@voxel_labels.setter
	def voxel_labels(self, voxel_labels):
		if self.voxels is None:
			self.voxels = len(voxel_labels)
		if len(voxel_labels) != self.voxels:
			raise ValueError('length of "voxel labels" ({}) must match '
							 'number of voxels in frame ({})'
							 ''.format(len(voxel_labels), self.voxels))
		self.__voxel_labels = vec(voxel_labels).astype(int)

	@property
	def beam_labels(self):
		return self.__beam_labels

	@beam_labels.setter
	def beam_labels(self, beam_labels):
		if self.beams is None:
			self.beams = len(beam_labels)
		if len(beam_labels) != self.beams:
			raise ValueError('length of "beam labels" ({}) must match '
							 'number of beams in frame ({})'
							 ''.format(len(beam_labels), self.beams))
		self.__beam_labels = vec(beam_labels).astype(int)

	@property
	def voxel_weights(self):
		return self.__voxel_weights

	@voxel_weights.setter
	def voxel_weights(self, voxel_weights):
		if self.voxels is None:
			self.voxels = len(voxel_weights)
		if len(voxel_weights) != self.voxels:
			raise ValueError('length of "voxel_weights" ({}) must match '
							 'number of voxels in frame ({})'
							 ''.format(len(voxel_weights), self.voxels))
		self.__voxel_weights = vec(voxel_weights).astype(float)

	@property
	def beam_weights(self):
		return self.__beam_weights

	@beam_weights.setter
	def beam_weights(self, beam_weights):
		if self.beams is None:
			self.beams = len(beam_weights)
		if len(beam_weights) != self.beams:
			raise ValueError('length of "beam_weights" ({}) must match '
							 'number of beams in frame ({})'
							 ''.format(len(beam_weights), self.beams))
		self.__beam_weights = vec(beam_weights).astype(float)

	@staticmethod
	def indices_by_label(label_vector, label, vector_name):
		if label_vector is None:
			raise ValueError('{} object field "{}" must be '
							 'set to perform retrieval by label'
							 ''.format(DoseFrame, name))

		indices = listmap(lambda x: x[0], listfilter(
				lambda x: x[1] == label, enumerate(label_vector)))
		if len(indices) == 0:
			raise KeyError('label {} not found in entries of field '
						   '"{}"'.format(label, name))

		return vec(indices)

	def voxel_lookup_by_label(self, label):
		indices = self.indices_by_label(
				self.voxel_labels, label, 'voxel_labels')
		return indices

	def beam_lookup_by_label(self, label):
		indices = self.indices_by_label(self.beam_labels, label, 'beam_labels')
		return indices


	def __str__(self):
		return str('Dose Frame: {} VOXELS by {} BEAMS'.format(
				self.voxels, self.beams))

class Physics(object):
	def __init__(self, voxels=None, beams=None, dose_matrix=None,
				 dose_grid=None, voxel_labels=None, **options):
		self.__frames = {}
		self.__dose_grid = None
		self.__dose_frame = None
		self.__beams = None
		self.__FRAME_LOAD_FLAG = False

		# copy constructor
		if isinstance(voxels, Physics):
			self.__frames = voxels._Physics__frames
			self.__dose_grid = voxels._Physics__dose_grid
			self.__dose_frame = voxels._Physics__dose_frame
			self.__beams = voxels._Physics__beams
			self.__FRAME_LOAD_FLAG = voxels._Physics__FRAME_LOAD_FLAG
			return

		# normal initialization
		if dose_grid is not None:
			self.dose_grid = dose_grid

		if voxels is None and self.dose_grid is not None:
			voxels = self.dose_grid.voxels

		f = DoseFrame(voxels, beams, dose_matrix, voxel_labels=voxel_labels,
					  **options)

		if self.__beams is None:
			self.__beams = BeamSet(f.beams)

		self.__dose_frame = f
		self.__frames[0] = f
		self.__frames['full'] = f

		if self.dose_grid is not None:
			if self.dose_grid.voxels == self.voxels:
				self.__frames['geometric'] = f

	@property
	def frame(self):
		return self.__dose_frame

	@property
	def data_loaded(self):
		return self.__FRAME_LOAD_FLAG

	def mark_data_as_loaded(self):
		self.__FRAME_LOAD_FLAG = True

	def add_dose_frame(self, key):
		if key in self.frames:
			raise ValueError('key "{}" already exists in {} frame '
							 'dictionary'.format(key, Physics))

	def change_dose_frame(self, key):
		if not key in self.frames:
			raise KeyError('no dose data frame found for key {}')
		self.__dose_frame = self.__frames[key]
		self.__FRAME_LOAD_FLAG = False

	@property
	def available_frames(self):
		return self.__frames.keys()

	@property
	def plannable(self):
		return self.frame.plannable

	@property
	def beams(self):
		return self.frame.beams

	@property
	def voxels(self):
		return self.frame.voxels

	@property
	def dose_grid(self):
		return self.__dose_grid

	@dose_grid.setter
	def dose_grid(self, grid):
		self.__dose_grid = VoxelGrid(grid=grid)

	@property
	def dose_matrix(self):
		return self.frame.data

	@dose_matrix.setter
	def dose_matrix(self, dose_matrix):
		self.frame.data = dose_matrix

	@property
	def voxel_labels(self):
		return self.frame.voxel_labels

	@voxel_labels.setter
	def voxel_labels(self, voxel_labels):
		self.frame.voxel_labels = voxel_labels

	def dose_matrix_by_label(self, voxel_label=None, beam_label=None):
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
		indices = self.frame.voxel_lookup_by_label(label)
		return self.frame.voxel_weights[indices]

	def beam_weights_by_label(self, label):
		indices = self.frame.beam_lookup_by_label(label)
		return self.frame.beam_weights[indices]