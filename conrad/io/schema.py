"""
TOOO: DOCSTRING
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
import ast
import numpy as np
import scipy.sparse as sp

class ConradDatabaseUtilties(object):
	@staticmethod
	def is_database_pointer(string, entry_type):
		if entry_type in CONRAD_DB_ENTRY_PREFIXES and isinstance(string, str):
			return string.startswith(CONRAD_DB_ENTRY_PREFIXES[entry_type])
		return False

	def isinstance_or_db_pointer(self, value, entry_type):
		if entry_type in CONRAD_DB_ENTRY_PREFIXES:
			if isinstance(value, str):
				return self.is_database_pointer(value, entry_type)
			else:
				return isinstance(value, entry_type)
		return False

	@staticmethod
	def expand_if_db_entry(value, field=None):
		if isinstance(value, ConradDatabaseEntry):
			retval = value.nested_dictionary
			if field is not None and isinstance(retval, dict):
				if field in retval:
					retval = retval[field]
			return retval
		else:
			return value

	def expand_list_if_db_entries(self, value_list, field=None):
		return [self.expand_if_db_entry(v, field) for v in value_list]

	def check_flat(self, values, db_type=None):
		flat = True
		for v in values:
			if isinstance(v, (type(None), int, float)):
				continue
			if isinstance(v, tuple):
				v_, db_type_ = v
				flat &= self.check_flat([v_], db_type_)
			elif db_type in CONRAD_DB_ENTRY_PREFIXES:
				flat &= self.is_database_pointer(v, db_type)
			else:
				flat &= isinstance(v, str)
		return flat

	def route_data_fragment(self, value):
		if isinstance(value, dict):
			if CONRAD_DB_TYPETAG in value:
				db_type = value[CONRAD_DB_TYPETAG]
				if db_type in CONRAD_DB_TYPESTRING_TO_CONSTRUCTOR:
					return CONRAD_DB_TYPESTRING_TO_CONSTRUCTOR[db_type](
							**value)
			else:
				for k in value:
					value[k] = self.route_data_fragment(value[k])
		elif isinstance(value, str):
			if value.endswith(('.npz', '.npy')):
				return UnsafeFileEntry(filename=value)
		return value

	@staticmethod
	def validate_db(conrad_db):
		if not isinstance(conrad_db, ConradDatabaseSuper):
			raise TypeError(
					'argument `conrad_db` must inherit from abstract '
					'base class `ConradDBBase`')

	def try_keys(self, dictionary, *keys):
		if not isinstance(dictionary, dict):
			raise TypeError(
					'argument `dictionary` must be of type {}'.format(dict))
		for k in keys:
			if isinstance(k, list):
				if len(k) == 1:
					return cdb_util.try_keys(k[0])
				elif len(k) == 2:
					subkey = k[1]
				else:
					subkey = k[1:]
				if k[0] in dictionary:
					return cdb_util.try_keys(dictionary[k[0]], subkey)
			else:
				if k in dictionary:
					return dictionary[k]
		return None

cdb_util = ConradDatabaseUtilties()

class ConradDatabaseSuper(object):
	pass

class ConradDatabaseEntry(object):
	@property
	def complete(self):
		return False

	@abc.abstractmethod
	def flatten(self, conrad_db):
		return NotImplemented

	@abc.abstractmethod
	def arborize(self, conrad_db):
		return NotImplemented

	@abc.abstractproperty
	def nested_dictionary(self):
		return NotImplemented

	@abc.abstractproperty
	def flat_dictionary(self):
		return NotImplemented

class CaseEntry(ConradDatabaseEntry):
	__metaclass__ = abc.ABCMeta

	def __init__(self, **entry_dictionary):
		ConradDatabaseEntry.__init__(self)
		self.__name = None
		self.__frame = None
		self.__prescription = None
		self.__anatomy = None
		self.__physics = None
		self.__history = None
		self.__solver_caches = []
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		complete = isinstance(self.name, str)
		complete &= cdb_util.isinstance_or_db_pointer(
				self.physics, PhysicsEntry)
		complete &= cdb_util.isinstance_or_db_pointer(
				self.anatomy, AnatomyEntry)
		return complete

	@property
	def name(self):
		return self.__name

	@name.setter
	def name(self, name):
		if isinstance(name, str):
			self.__name = name

	@property
	def prescription(self):
		return self.__prescription

	@prescription.setter
	def prescription(self, prescription_list_or_string):
		if isinstance(prescription_list_or_string, str):
			prescription = ast.literal_eval(prescription_list_or_string)
		elif isinstance(prescription_list_or_string, list):
			prescription = prescription_list_or_string
		else:
			prescription = None

		if isinstance(prescription, list):
			self.__prescription = prescription

	@property
	def anatomy(self):
		return self.__anatomy

	@anatomy.setter
	def anatomy(self, anatomy_entry):
		if isinstance(anatomy_entry, list):
			anatomy_entry = AnatomyEntry(structures=anatomy_entry)
		if isinstance(anatomy_entry, dict):
			anatomy_entry = AnatomyEntry(**anatomy_entry)
		if cdb_util.isinstance_or_db_pointer(anatomy_entry, AnatomyEntry):
			self.__anatomy = anatomy_entry

	@property
	def physics(self):
		return self.__physics

	@physics.setter
	def physics(self, physics_entry):
		if isinstance(physics_entry, dict):
			physics_entry = PhysicsEntry(**physics_entry)
		if cdb_util.isinstance_or_db_pointer(physics_entry, PhysicsEntry):
			self.__physics = physics_entry

	@property
	def history(self):
		return self.__history

	@history.setter
	def history(self, history_entry):
		if isinstance(history_entry, dict):
			history_entry = HistoryEntry(**history_entry)
		if cdb_util.isinstance_or_db_pointer(history_entry, HistoryEntry):
			self.__history = history_entry

	@property
	def solver_caches(self):
		return self.__solver_caches

	@solver_caches.setter
	def solver_caches(self, solver_cache_list):
		if solver_cache_list is None:
			return
		self.__solver_caches = []
		if isinstance(solver_cache_list, str):
			solver_cache_list = ast.literal_eval(solver_cache_list)
		self.add_solver_caches(*solver_cache_list)

	def add_solver_caches(self, *solver_caches):
		safe_list = []
		for s in solver_caches:
			if isinstance(s, dict):
				s = SolverCacheEntry(**s)
			if cdb_util.isinstance_or_db_pointer(s, SolverCacheEntry):
				safe_list.append(s)
			else:
				raise ValueError(
						'argument `solver_caches` must be a list of {} '
						'objects or corresponding database pointer '
						'strings'.format(SolverCacheEntry))
		self.__solver_caches += safe_list

	def ingest_dictionary(self, **case_dictionary):
		self.name = case_dictionary.pop('name', None)
		self.prescription = cdb_util.try_keys(
				case_dictionary, 'prescription', 'rx')
		self.anatomy = cdb_util.try_keys(
				case_dictionary, 'anatomy', 'structures')
		self.physics = cdb_util.try_keys(case_dictionary, 'physics')
		self.history = cdb_util.try_keys(
				case_dictionary, 'history', 'solutions')
		self.solver_caches = cdb_util.try_keys(
				case_dictionary, 'solver_caches')

	def flatten(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if isinstance(self.anatomy, AnatomyEntry):
			self.anatomy = conrad_db.set_next(self.anatomy.flatten(conrad_db))
		if isinstance(self.physics, PhysicsEntry):
			self.physics = conrad_db.set_next(self.physics.flatten(conrad_db))
		if isinstance(self.history, HistoryEntry):
			self.history = conrad_db.set_next(self.history.flatten(conrad_db))
		for i, sc in enumerate(self.__solver_caches):
			if isinstance(sc, SolverCacheEntry):
				self.__solver_caches[i] = conrad_db.set_next(
						sc.flatten(conrad_db))
		return self

	def arborize(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if self.anatomy is not None:
			self.anatomy = conrad_db.get(self.anatomy).arborize(conrad_db)
		if self.physics is not None:
			self.physics = conrad_db.get(self.physics).arborize(conrad_db)
		if self.history is not None:
			self.history = conrad_db.get(self.history).arborize(conrad_db)
		for i, sc in enumerate(self.__solver_caches):
			self.__solver_caches[i] = conrad_db.get(sc).arborize(conrad_db)
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'name': self.name,
				'prescription': cdb_util.expand_if_db_entry(self.prescription),
				'anatomy': cdb_util.expand_if_db_entry(
						self.anatomy, field='anatomy'),
				'physics': cdb_util.expand_if_db_entry(self.physics),
				'history': cdb_util.expand_if_db_entry(
						self.history, field='history'),
				'solver_caches': cdb_util.expand_list_if_db_entries(
						self.solver_caches),
		}

	@property
	def flat_dictionary(self):
		checklist = [
				(self.anatomy, AnatomyEntry),
				(self.physics, PhysicsEntry),
				(self.history, HistoryEntry),]
		checklist += [(s, SolverCacheEntry) for s in self.solver_caches]
		if not cdb_util.check_flat(checklist):
			raise ValueError(
					'cannot emit flat dictionary from {}: history, '
					'physics, anatomy, and solver caches must all be '
					'[lists of] database pointer strings, not database '
					'entry objects'.format(CaseEntry))
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'name': self.name,
				'prescription': str(self.prescription),
				'anatomy': self.anatomy,
				'physics': self.physics,
				'history': self.history,
				'solver_caches': str(self.solver_caches),
		}

class PhysicsEntry(ConradDatabaseEntry):
	def __init__(self, **entry_dictionary):
		ConradDatabaseEntry.__init__(self)
		self.__voxel_grid = {'x': None, 'y': None, 'z': None}
		self.__voxel_bitmask = None
		self.__beam_set = {
				'type': None,
				'control_points': {
						'number': None,
						'groups': None,
						'group_sizes': None,
				},
				'max': None,
				'max_active': None,
				'adjacency_matrix': None,
		}
		self.__frames = []
		self.__frame_mappings = []
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		return len(self.frames) > 0

	@property
	def voxel_grid(self):
		return self.__voxel_grid

	@voxel_grid.setter
	def voxel_grid(self, voxel_grid_dictionary):
		if isinstance(voxel_grid_dictionary, str):
			voxel_grid_dictionary = ast.literal_eval(voxel_grid_dictionary)
		if isinstance(voxel_grid_dictionary, dict):
			self.__voxel_grid['x'] = cdb_util.try_keys(
					voxel_grid_dictionary, 'x', 'x_voxels')
			self.__voxel_grid['y'] = cdb_util.try_keys(
					voxel_grid_dictionary, 'y', 'y_voxels')
			self.__voxel_grid['z'] = cdb_util.try_keys(
					voxel_grid_dictionary, 'z', 'z_voxels')

	@property
	def voxel_bitmask(self):
		return self.__voxel_bitmask

	@voxel_bitmask.setter
	def voxel_bitmask(self, voxel_bitmask):
		voxel_bitmask = cdb_util.route_data_fragment(voxel_bitmask)
		if cdb_util.isinstance_or_db_pointer(voxel_bitmask, DataFragmentEntry):
			self.__voxel_bitmask = voxel_bitmask


	@property
	def beam_set(self):
		return self.__beam_set

	@beam_set.setter
	def beam_set(self, beam_set_dictionary):
		if isinstance(beam_set_dictionary, str):
			beam_set_dictionary = ast.literal_eval(beam_set_dictionary)
		if isinstance(beam_set_dictionary, dict):
			beam_type = beam_set_dictionary.pop('type', None)
			if beam_type is not None:
				self.beam_set['type'] = str(beam_type)

			# control points: number
			n_control_points = cdb_util.try_keys(
					beam_set_dictionary, 'control_points_number',
					['control_points', 'number'])
			if n_control_points is not None:
				self.beam_set['control_points']['number'] = int(
						n_control_points)

			# control points: groups
			n_control_groups = cdb_util.try_keys(
					beam_set_dictionary, 'control_points_groups',
					['control_points', 'groups'])
			if n_control_groups is not None:
				self.beam_set['control_points']['groups'] = int(
						n_control_groups)

			# control points: group sizes
			control_group_sizes = cdb_util.try_keys(
					beam_set_dictionary, 'control_points_group_sizes',
					['control_points', 'group_sizes'])
			if control_group_sizes is not None:
				if isinstance(control_group_sizes, str):
					control_group_sizes = ast.literal_eval(control_group_sizes)
				self.beam_set['control_points']['group_sizes'] = [
						int(cgs) for cgs in control_group_sizes]

			beam_max = beam_set_dictionary.pop('max', None)
			if beam_max is not None:
				self.beam_set['max'] = float(beam_max)

			beam_max_active = beam_set_dictionary.pop('max_active', None)
			if beam_max_active is not None:
				self.beam_set['max_active'] = bool(beam_max_active)

			adjacency_matrix = cdb_util.route_data_fragment(
					beam_set_dictionary.pop('adjacency_matrix', None))
			if cdb_util.isinstance_or_db_pointer(
					adjacency_matrix, DataFragmentEntry):
				self.beam_set['adjacency_matrix'] = adjacency_matrix

	@property
	def frames(self):
		return self.__frames

	@frames.setter
	def frames(self, frame_list):
		if frame_list is None:
			return
		self.__frames = []
		if isinstance(frame_list, str):
			frame_list = ast.literal_eval(frame_list)
		self.add_frames(*frame_list)

	def add_frames(self, *frames):
		safe_list = []
		for f in frames:
			if isinstance(f, dict):
				f = DoseFrameEntry(**f)
			if cdb_util.isinstance_or_db_pointer(f, DoseFrameEntry):
				safe_list.append(f)
			else:
				raise ValueError(
						'argument `frames` must be a list of {} '
						'objects or corresponding database pointer '
						'strings'.format(DoseFrameEntry))
		self.__frames += safe_list

	@property
	def frame_mappings(self):
		return self.__frame_mappings

	@frame_mappings.setter
	def frame_mappings(self, frame_mapping_list):
		if frame_mapping_list is None:
			return
		self.__frame_mappings = []
		if isinstance(frame_mapping_list, str):
			frame_mapping_list = ast.literal_eval(frame_mapping_list)
		self.add_frame_mappings(*frame_mapping_list)

	def add_frame_mappings(self, *frame_mappings):
		safe_list = []
		for fm in frame_mappings:
			if isinstance(fm, dict):
				fm = DoseFrameMappingEntry(**fm)
			if cdb_util.isinstance_or_db_pointer(fm, DoseFrameMappingEntry):
				safe_list.append(fm)
			else:
				raise ValueError(
						'argument `frame_mappings` must be a list of {} '
						'objects or corresponding database pointer '
						'strings'.format(DoseFrameMappingEntry))
		self.__frame_mappings += safe_list

	def ingest_dictionary(self, **physics_dictionary):
		self.voxel_grid = physics_dictionary.pop('voxel_grid', None)
		self.voxel_bitmask = physics_dictionary.pop('voxel_bitmask', None)
		self.beam_set = physics_dictionary.pop('beam_set', None)
		self.frames = physics_dictionary.pop('frames', None)
		self.frame_mappings = physics_dictionary.pop('frame_mappings', None)

	def flatten(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if isinstance(self.voxel_bitmask, ConradDatabaseEntry):
			self.voxel_bitmask = conrad_db.set_next(
					self.voxel_bitmask.flatten(conrad_db))
		for i, f in enumerate(self.__frames):
			if isinstance(f, DoseFrameEntry):
				self.__frames[i] = conrad_db.set_next(f.flatten(conrad_db))
		for i, fm in enumerate(self.__frame_mappings):
			if isinstance(fm, DoseFrameMappingEntry):
				self.__frame_mappings[i] = conrad_db.set_next(
						fm.flatten(conrad_db))
		return self

	def arborize(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if self.voxel_bitmask is not None:
			self.voxel_bitmask = conrad_db.get(
					self.voxel_bitmask).arborize(conrad_db)
		for i, f in enumerate(self.__frames):
			self.__frames[i] = conrad_db.get(f).arborize(conrad_db)
		for i, fm in enumerate(self.__frame_mappings):
			self.__frame_mappings[i] = conrad_db.get(fm).arborize(conrad_db)
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'voxel_grid': self.voxel_grid,
				'voxel_bitmask': cdb_util.expand_if_db_entry(
						self.voxel_bitmask),
				'beam_set': self.beam_set,
				'frames': cdb_util.expand_list_if_db_entries(self.frames),
				'frame_mappings': cdb_util.expand_list_if_db_entries(
						self.frame_mappings),
		}

	@property
	def flat_dictionary(self):
		checklist = [(self.voxel_bitmask, DataFragmentEntry)]
		checklist += [(f, DoseFrameEntry) for f in self.frames]
		checklist += [
			(fm, DoseFrameMappingEntry) for fm in self.frame_mappings]
		if not cdb_util.check_flat(checklist):
			raise ValueError(
					'cannot emit flat dictionary from {}: '
					'voxel_bitmask, dose frames and dose frame '
					'mappings must all be [lists of] database pointer '
					'strings, not database entry objects'
					''.format(PhysicsEntry))
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'voxel_grid': self.voxel_grid,
				'voxel_bitmask': self.voxel_bitmask,
				'beam_set': self.beam_set,
				'frames': str(self.frames),
				'frame_mappings': str(self.frame_mappings),
		}

class DoseFrameEntry(ConradDatabaseEntry):
	def __init__(self, **entry_dictionary):
		ConradDatabaseEntry.__init__(self)
		self.__name = None
		self.__n_voxels = None
		self.__n_beams = None
		self.__dose_matrix = None
		self.__voxel_labels = None
		self.__voxel_weights = None
		self.__beam_labels = None
		self.__beam_weights = None
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		complete = isinstance(self.name, str)
		complete &= isinstance(self.n_voxels, int)
		complete &= isinstance(self.n_beams, int)
		complete &= cdb_util.isinstance_or_db_pointer(
				self.dose_matrix, DataFragmentEntry)
		return complete

	@property
	def name(self):
		return self.__name

	@name.setter
	def name(self, name):
		if isinstance(name, str):
			self.__name = name

	@property
	def n_voxels(self):
		return self.__n_voxels

	@n_voxels.setter
	def n_voxels(self, n_voxels):
		if n_voxels is not None:
			self.__n_voxels = int(n_voxels)

	@property
	def n_beams(self):
		return self.__n_beams

	@n_beams.setter
	def n_beams(self, n_beams):
		if n_beams is not None:
			self.__n_beams = int(n_beams)

	@property
	def dose_matrix(self):
		return self.__dose_matrix

	@dose_matrix.setter
	def dose_matrix(self, dose_matrix):
		dose_matrix = cdb_util.route_data_fragment(dose_matrix)
		if cdb_util.isinstance_or_db_pointer(dose_matrix, DataFragmentEntry):
			self.__dose_matrix = dose_matrix

	@property
	def voxel_labels(self):
		return self.__voxel_labels

	@voxel_labels.setter
	def voxel_labels(self, voxel_labels):
		voxel_labels = cdb_util.route_data_fragment(voxel_labels)
		if cdb_util.isinstance_or_db_pointer(voxel_labels, DataFragmentEntry):
			self.__voxel_labels = voxel_labels

	@property
	def voxel_weights(self):
		return self.__voxel_weights

	@voxel_weights.setter
	def voxel_weights(self, voxel_weights):
		voxel_weights = cdb_util.route_data_fragment(voxel_weights)
		if cdb_util.isinstance_or_db_pointer(voxel_weights, DataFragmentEntry):
			self.__voxel_weights = voxel_weights

	@property
	def beam_labels(self):
		return self.__beam_labels

	@beam_labels.setter
	def beam_labels(self, beam_labels):
		beam_labels = cdb_util.route_data_fragment(beam_labels)
		if cdb_util.isinstance_or_db_pointer(beam_labels, DataFragmentEntry):
			self.__beam_labels = beam_labels

	@property
	def beam_weights(self):
		return self.__beam_weights

	@beam_weights.setter
	def beam_weights(self, beam_weights):
		beam_weights = cdb_util.route_data_fragment(beam_weights)
		if cdb_util.isinstance_or_db_pointer(beam_weights, DataFragmentEntry):
			self.__beam_weights = beam_weights

	def ingest_dictionary(self, **dose_frame_dictionary):
		self.name = dose_frame_dictionary.pop('name', None)
		self.n_voxels = cdb_util.try_keys(
				dose_frame_dictionary, 'voxels', 'n_voxels')
		self.n_beams = cdb_util.try_keys(
				dose_frame_dictionary, 'beams', 'n_beams')
		self.dose_matrix = dose_frame_dictionary.pop('dose_matrix', None)
		self.voxel_labels = dose_frame_dictionary.pop('voxel_labels', None)
		self.voxel_weights = dose_frame_dictionary.pop('voxel_weights', None)
		self.beam_labels = dose_frame_dictionary.pop('beam_labels', None)
		self.beam_weights = dose_frame_dictionary.pop('beam_weights', None)

	def flatten(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if isinstance(self.dose_matrix, ConradDatabaseEntry):
			self.dose_matrix = conrad_db.set_next(
					self.dose_matrix.flatten(conrad_db))
		if isinstance(self.voxel_labels, ConradDatabaseEntry):
			self.voxel_labels = conrad_db.set_next(
					self.voxel_labels.flatten(conrad_db))
		if isinstance(self.voxel_weights, ConradDatabaseEntry):
			self.voxel_weights = conrad_db.set_next(
					self.voxel_weights.flatten(conrad_db))
		if isinstance(self.beam_labels, ConradDatabaseEntry):
			self.beam_labels = conrad_db.set_next(
					self.beam_labels.flatten(conrad_db))
		if isinstance(self.beam_weights, ConradDatabaseEntry):
			self.beam_weights = conrad_db.set_next(
					self.beam_weights.flatten(conrad_db))
		return self

	def arborize(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if self.dose_matrix is not None:
			self.dose_matrix = conrad_db.get(self.dose_matrix).arborize(
					conrad_db)
		if self.voxel_labels is not None:
			self.voxel_labels = conrad_db.get(self.voxel_labels).arborize(
					conrad_db)
		if self.voxel_weights is not None:
			self.voxel_weights = conrad_db.get(self.voxel_weights).arborize(
					conrad_db)
		if self.beam_labels is not None:
			self.beam_labels = conrad_db.get(self.beam_labels).arborize(
					conrad_db)
		if self.beam_weights is not None:
			self.beam_weights = conrad_db.get(self.beam_weights).arborize(
					conrad_db)
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'name': self.name,
				'n_voxels': self.n_voxels,
				'n_beams': self.n_beams,
				'dose_matrix': cdb_util.expand_if_db_entry(self.dose_matrix),
				'voxel_labels': cdb_util.expand_if_db_entry(self.voxel_labels),
				'voxel_weights': cdb_util.expand_if_db_entry(
						self.voxel_weights),
				'beam_labels': cdb_util.expand_if_db_entry(self.beam_labels),
				'beam_weights': cdb_util.expand_if_db_entry(self.beam_weights),
		}

	@property
	def flat_dictionary(self):
		checklist = [self.dose_matrix, self.voxel_labels, self.voxel_weights,
					 self.beam_labels, self.beam_weights]
		if not cdb_util.check_flat(checklist, DataFragmentEntry):
			raise ValueError(
					'cannot emit flat dictionary from {}: '
					'`dose_matrix`, `voxel_labels`, `voxel_weights`, '
					'`beam_labels`, and `beam_weights` must all be '
					'[lists of] database pointer strings, not database '
					'entry objects'.format(DoseFrameEntry))
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'name': self.name,
				'n_voxels': self.n_voxels,
				'n_beams': self.n_beams,
				'dose_matrix': self.dose_matrix,
				'voxel_labels': self.voxel_labels,
				'voxel_weights': self.voxel_weights,
				'beam_labels': self.beam_labels,
				'beam_weights': self.beam_weights,
		}

class DoseFrameMappingEntry(ConradDatabaseEntry):
	def __init__(self, **entry_dictionary):
		ConradDatabaseEntry.__init__(self)
		self.__source_frame = None
		self.__target_frame = None
		self.__voxel_map = None
		self.__voxel_map_type = None
		self.__beam_map = None
		self.__beam_map_type = None
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		complete = isinstance(self.source_frame, str)
		complete &= isinstance(self.target_frame, str)

		# require map types provided if maps provided
		complete &= bool(
				cdb_util.isinstance_or_db_pointer(
						self.voxel_map, DataFragmentEntry) <=
				isinstance(self.voxel_map_type, str))
		complete &= bool(
				cdb_util.isinstance_or_db_pointer(
						self.beam_map, DataFragmentEntry) <=
				isinstance(self.beam_map_type, str))
		return complete

	@property
	def source_frame(self):
		return self.__source_frame

	@source_frame.setter
	def source_frame(self, source_frame_name):
		if isinstance(source_frame_name, str):
			self.__source_frame = source_frame_name

	@property
	def target_frame(self):
		return self.__target_frame

	@target_frame.setter
	def target_frame(self, target_frame_name):
		if isinstance(target_frame_name, str):
			self.__target_frame = target_frame_name

	@property
	def voxel_map(self):
		return self.__voxel_map

	@voxel_map.setter
	def voxel_map(self, voxel_map):
		voxel_map = cdb_util.route_data_fragment(voxel_map)
		if cdb_util.isinstance_or_db_pointer(voxel_map, DataFragmentEntry):
			self.__voxel_map = voxel_map

	@property
	def voxel_map_type(self):
		return self.__voxel_map_type

	@voxel_map_type.setter
	def voxel_map_type(self, voxel_map_type_string):
		if isinstance(voxel_map_type_string, str):
			self.__voxel_map_type = voxel_map_type_string

	@property
	def beam_map(self):
		return self.__beam_map

	@beam_map.setter
	def beam_map(self, beam_map):
		beam_map = cdb_util.route_data_fragment(beam_map)
		if cdb_util.isinstance_or_db_pointer(beam_map, DataFragmentEntry):
			self.__beam_map = beam_map

	@property
	def beam_map_type(self):
		return self.__beam_map_type

	@beam_map_type.setter
	def beam_map_type(self, beam_map_type_string):
		if isinstance(beam_map_type_string, str):
			self.__beam_map_type = beam_map_type_string

	def ingest_dictionary(self, **frame_mapping_dictionary):
		self.source_frame = frame_mapping_dictionary.pop('source_frame', None)
		self.target_frame = frame_mapping_dictionary.pop('target_frame', None)
		self.voxel_map = frame_mapping_dictionary.pop('voxel_map', None)
		self.voxel_map_type = frame_mapping_dictionary.pop(
				'voxel_map_type', None)
		self.beam_map = frame_mapping_dictionary.pop('beam_map', None)
		self.beam_map_type = frame_mapping_dictionary.pop(
				'beam_map_type', None)

	def flatten(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if isinstance(self.voxel_map, ConradDatabaseEntry):
			self.voxel_map = conrad_db.set_next(self.voxel_map.flatten(conrad_db))
		if isinstance(self.beam_map, ConradDatabaseEntry):
			self.beam_map = conrad_db.set_next(self.beam_map.flatten(conrad_db))
		return self

	def arborize(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if self.voxel_map is not None:
			self.voxel_map = conrad_db.get(self.voxel_map).arborize(conrad_db)
		if self.beam_map is not None:
			self.beam_map = conrad_db.get(self.beam_map).arborize(conrad_db)
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'source_frame': self.source_frame,
				'target_frame': self.target_frame,
				'voxel_map': cdb_util.expand_if_db_entry(self.voxel_map),
				'voxel_map_type': self.voxel_map_type,
				'beam_map': cdb_util.expand_if_db_entry(self.beam_map),
				'beam_map_type': self.beam_map_type,
		}

	@property
	def flat_dictionary(self):
		if not cdb_util.check_flat(
				[self.voxel_map, self.beam_map], DataFragmentEntry):
			raise ValueError(
					'cannot emit flat dictionary from {}: entries '
					'`voxel_map` and `beam_map` must be database '
					'pointer strings, not database entry objects'
					''.format(DoseFrameMappingEntry))
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'source_frame': self.source_frame,
				'target_frame': self.target_frame,
				'voxel_map': self.voxel_map,
				'voxel_map_type': self.voxel_map_type,
				'beam_map': self.beam_map,
				'beam_map_type': self.beam_map_type,
		}

class DataFragmentEntry(ConradDatabaseEntry):
	__metaclass__ = abc.ABCMeta

	def __init__(self):
		ConradDatabaseEntry.__init__(self)
		self.__type = None
		self.__entries = None
		self.__layout_rowmajor = None
		self.__layout_CSR = None
		self.__layout_fortran_indexing = None
		self.__shape = None
		self.__data_file = None
		self.__data_key = None
		self.__data_pointers_file = None
		self.__data_pointers_key = None
		self.__data_indices_file = None
		self.__data_indices_key = None
		self.__data_values_file = None
		self.__data_values_key = None

	@staticmethod
	def __check_npyz_file_key(file, key):
		valid = isinstance(file, str)
		if valid and '.npz' in file:
			valid &= isinstance(key, str)
		elif valid:
			valid &= '.npy' in file
		return valid

	@property
	def type(self):
		return self.__type

	@property
	def fragment_flat_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: 'data_dictionary: '+ str(self.__type),
				'entries': self.__entries,
				'layout_rowmajor': self.__layout_rowmajor,
				'layout_CSR': self.__layout_CSR,
				'layout_fortran_indexing': self.__layout_fortran_indexing,
				'shape': self.__shape,
				'data_file': self.__data_file,
				'data_key': self.__data_key,
				'data_pointers_file': self.__data_pointers_file,
				'data_pointers_key': self.__data_pointers_key,
				'data_indices_file': self.__data_indices_file,
				'data_indices_key': self.__data_indices_key,
				'data_values_file': self.__data_values_file,
				'data_values_key': self.__data_values_key,
		}

	def ingest_dictionary(self, **data_fragment_dictionary):
		return NotImplemented

class UnsafeFileEntry(DataFragmentEntry):
	def __init__(self, **unsafe_dictionary):
		DataFragmentEntry.__init__(self)
		self.__unsafe_file = None


	@property
	def complete(self):
		return self.file is not None

	@property
	def file(self):
		return self.__unsafe_file

	@file.setter
	def file(self, filename):
		if isinstance(filename, str):
			if filename.endswith(('.npy', '.npz')):
				self.__unsafe_file = filename

	def ingest_dictionary(self, **unsafe_dictionary):
		self.file = cdb_util.try_keys(unsafe_dictionary, 'file', 'filename')

class DataDictionaryEntry(DataFragmentEntry):
	def __init__(self, **entry_dictionary):
		DataFragmentEntry.__init__(self)
		self._DataFragmentEntry__type = 'dictionary'
		self._DataFragmentEntry__entries = {}
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		return isinstance(self.entries, dict) and len(self.entries) > 0

	@property
	def entries(self):
		return self._DataFragmentEntry__entries

	@entries.setter
	def entries(self, entries):
		if isinstance(entries, str):
			entries = ast.literal_eval(entries)
		if isinstance(entries, dict):
			entries = {
					k: cdb_util.route_data_fragment(entries[k])
					for k in entries}
			self._DataFragmentEntry__entries = entries

	def ingest_dictionary(self, **data_dictionary):
		self.entries = data_dictionary.pop('entries', {})

	def flatten(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		for k, v in self.entries.items():
			if isinstance(v, ConradDatabaseEntry):
				self.entries[k] = conrad_db.set_next(v.flatten(conrad_db))
		return self

	def arborize(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		for k, v in self.entries.items():
			if cdb_util.isinstance_or_db_pointer(v, DataFragmentEntry):
				self.entries[k] = conrad_db.get(v).arborize(conrad_db)
		return self

	@property
	def nested_dictionary(self):
		dictionary = {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'entries': {
						k: cdb_util.expand_if_db_entry(self.entries[k])
						for k in self.entries},
		}
		return dictionary

	@property
	def flat_dictionary(self):
		if not cdb_util.check_flat(self.entries.values(), DataFragmentEntry):
			raise ValueError(
					'cannot emit flat dictionary from {}: all entries '
					'in data dictionary must be database pointer '
					'strings, not database entry objects'
					''.format(DataDictionaryEntry))
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'entries': str(self.entries),
		}

class DenseArrayEntry(DataFragmentEntry):
	__metaclass__ = abc.ABCMeta

	def __init__(self):
		DataFragmentEntry.__init__(self)

	@property
	def __complete(self):
		return self._DataFragmentEntry__check_npyz_file_key(
				self.data_file, self.data_key)

	@property
	def data_file(self):
		return self._DataFragmentEntry__data_file

	@data_file.setter
	def data_file(self, data_file):
		if data_file is not None:
			self._DataFragmentEntry__data_file = str(data_file)

	@property
	def data_key(self):
		return self._DataFragmentEntry__data_key

	@data_key.setter
	def data_key(self, data_key):
		if data_key is not None:
			self._DataFragmentEntry__data_key = str(data_key)

	def __ingest_dictionary(self, **dense_array_dictionary):
		self.data_file = cdb_util.try_keys(
				dense_array_dictionary, 'data_file', ['data', 'file'])
		self.data_key = cdb_util.try_keys(
				dense_array_dictionary, 'data_key', ['data', 'key'])

class VectorEntry(DenseArrayEntry):
	def __init__(self, **entry_dictionary):
		DenseArrayEntry.__init__(self)
		self._DataFragmentEntry__type = 'vector'
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		return self._DenseArrayEntry__complete

	def ingest_dictionary(self, **vector_dictionary):
		self._DenseArrayEntry__ingest_dictionary(**vector_dictionary)

	def flatten(self, conrad_db):
		return self

	def arborize(self, conrad_db):
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'data': {
						'file': self.data_file,
						'key': self.data_key,
				},
		}

	@property
	def flat_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'data_file': self.data_file,
				'data_key': self.data_key,
		}

class DenseMatrixEntry(DenseArrayEntry):
	def __init__(self, **entry_dictionary):
		DenseArrayEntry.__init__(self)
		self._DataFragmentEntry__type = 'dense matrix'
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		return bool(
				isinstance(self.layout_rowmajor, bool) and
				self._DenseArrayEntry__complete)

	@property
	def layout_rowmajor(self):
		return self._DataFragmentEntry__layout_rowmajor

	@layout_rowmajor.setter
	def layout_rowmajor(self, row_major):
		if row_major is not None:
			self._DataFragmentEntry__layout_rowmajor = bool(row_major)

	def ingest_dictionary(self, **densemat_dictionary):
		self.layout_rowmajor = cdb_util.try_keys(
				densemat_dictionary, 'layout_rowmajor',
				['layout', 'rowmajor'])
		self._DenseArrayEntry__ingest_dictionary(**densemat_dictionary)

	def flatten(self, conrad_db):
		return self

	def arborize(self, conrad_db):
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'layout': {
						'rowmajor': self.layout_rowmajor,
				},
				'data': {
						'file': self.data_file,
						'key': self.data_key,
				},
		}

	@property
	def flat_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'layout_rowmajor': self.layout_rowmajor,
				'data_file': self.data_file,
				'data_key': self.data_key,
		}

class SparseMatrixEntry(DataFragmentEntry):
	def __init__(self, **entry_dictionary):
		DataFragmentEntry.__init__(self)
		self._DataFragmentEntry__type = 'sparse_matrix'
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		complete = isinstance(self.layout_CSR, bool)
		complete &= isinstance(self.layout_fortran_indexing, bool)
		complete &= isinstance(self.shape, (list, tuple))
		complete &= self._DataFragmentEntry__check_npyz_file_key(
				self.data_pointers_file, self.data_pointers_key)
		complete &= self._DataFragmentEntry__check_npyz_file_key(
				self.data_indices_file, self.data_indices_key)
		complete &= self._DataFragmentEntry__check_npyz_file_key(
				self.data_values_file, self.data_values_key)
		return complete

	@property
	def layout_CSR(self):
		return self._DataFragmentEntry__layout_CSR

	@layout_CSR.setter
	def layout_CSR(self, layout_is_csr):
		if layout_is_csr is not None:
			self._DataFragmentEntry__layout_CSR = bool(layout_is_csr)

	@property
	def layout_fortran_indexing(self):
		return self._DataFragmentEntry__layout_fortran_indexing

	@layout_fortran_indexing.setter
	def layout_fortran_indexing(self, indices_are_base_1):
		if indices_are_base_1 is not None:
			self._DataFragmentEntry__layout_fortran_indexing = bool(
					indices_are_base_1)

	@property
	def shape(self):
		return self._DataFragmentEntry__shape

	@shape.setter
	def shape(self, shape):
		if isinstance(shape, (list, tuple)):
			self._DataFragmentEntry__shape = tuple(
					map(lambda x: int(x), shape))

	@property
	def data_pointers_file(self):
		return self._DataFragmentEntry__data_pointers_file

	@data_pointers_file.setter
	def data_pointers_file(self, data_pointers_file):
		if data_pointers_file is not None:
			self._DataFragmentEntry__data_pointers_file = str(
					data_pointers_file)

	@property
	def data_pointers_key(self):
		return self._DataFragmentEntry__data_pointers_key

	@data_pointers_key.setter
	def data_pointers_key(self, data_pointers_key):
		if data_pointers_key is not None:
			self._DataFragmentEntry__data_pointers_key = str(data_pointers_key)

	@property
	def data_indices_file(self):
		return self._DataFragmentEntry__data_indices_file

	@data_indices_file.setter
	def data_indices_file(self, data_indices_file):
		if data_indices_file is not None:
			self._DataFragmentEntry__data_indices_file = str(data_indices_file)

	@property
	def data_indices_key(self):
		return self._DataFragmentEntry__data_indices_key

	@data_indices_key.setter
	def data_indices_key(self, data_indices_key):
		if data_indices_key is not None:
			self._DataFragmentEntry__data_indices_key = str(data_indices_key)

	@property
	def data_values_file(self):
		return self._DataFragmentEntry__data_values_file

	@data_values_file.setter
	def data_values_file(self, data_values_file):
		if data_values_file is not None:
			self._DataFragmentEntry__data_values_file = str(data_values_file)

	@property
	def data_values_key(self):
		return self._DataFragmentEntry__data_values_key

	@data_values_key.setter
	def data_values_key(self, data_values_key):
		if data_values_key is not None:
			self._DataFragmentEntry__data_values_key = str(data_values_key)

	def ingest_dictionary(self, **sparsemat_dictionary):
		self.layout_CSR = cdb_util.try_keys(
				sparsemat_dictionary, 'layout_CSR', ['layout', 'CSR'])
		self.layout_fortran_indexing = cdb_util.try_keys(
				sparsemat_dictionary, 'layout_fortran_indexing',
				['layout', 'fortran_indexing'])
		self.shape = sparsemat_dictionary.pop('shape', None)
		self.data_pointers_file = cdb_util.try_keys(
				sparsemat_dictionary, 'data_pointers_file',
				['data', 'pointers', 'file'])
		self.data_pointers_key = cdb_util.try_keys(
				sparsemat_dictionary, 'data_pointers_key',
				['data', 'pointers', 'key'])
		self.data_indices_file = cdb_util.try_keys(
				sparsemat_dictionary, 'data_indices_file',
				['data', 'indices', 'file'])
		self.data_indices_key = cdb_util.try_keys(
				sparsemat_dictionary, 'data_indices_key',
				['data', 'indices', 'key'])
		self.data_values_file = cdb_util.try_keys(
				sparsemat_dictionary, 'data_values_file',
				['data', 'values', 'file'])
		self.data_values_key = cdb_util.try_keys(
				sparsemat_dictionary, 'data_values_key',
				['data', 'values', 'key'])

	def flatten(self, conrad_db):
		return self

	def arborize(self, conrad_db):
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'layout': {
						'CSR': self.layout_CSR,
						'fortran_indexing': self.layout_fortran_indexing,
				},
				'data': {
						'pointers': {
								'file': self.data_pointers_file,
								'key': self.data_pointers_key,
						},
						'indices': {
								'file': self.data_indices_file,
								'key': self.data_indices_key,
						},
						'values': {
								'file': self.data_values_file,
								'key': self.data_values_key,
						},
				},
		}

	@property
	def flat_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'layout_CSR': self.layout_CSR,
				'layout_fortran_indexing': self.layout_fortran_indexing,
				'shape': self.shape,
				'data_pointers_file': self.data_pointers_file,
				'data_pointers_key': self.data_pointers_key,
				'data_indices_file': self.data_indices_file,
				'data_indices_key': self.data_indices_key,
				'data_values_file': self.data_values_file,
				'data_values_key': self.data_values_key
		}

class HistoryEntry(ConradDatabaseEntry):
	def __init__(self, **entry_dictionary):
		ConradDatabaseEntry.__init__(self)
		self.__solutions = []
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		return len(self.solutions) > 0

	@property
	def solutions(self):
		return self.__solutions

	@solutions.setter
	def solutions(self, solution_list):
		if solution_list is None:
			return
		self.__solutions = []
		if isinstance(solution_list, str):
			solution_list = ast.literal_eval(solution_list)
		self.add_solutions(*solution_list)

	def add_solutions(self, *solutions):
		safe_list = []
		for s in solutions:
			if isinstance(s, dict):
				s = SolutionEntry(**s)
			if cdb_util.isinstance_or_db_pointer(s, SolutionEntry):
				safe_list.append(s)
			else:
				raise ValueError(
						'argument `solutions` must be a list of {} '
						'objects or corresponding database pointer '
						'strings'.format(SolutionEntry))
		self.__solutions += safe_list

	def ingest_dictionary(self, **history_dictionary):
		self.solutions = cdb_util.try_keys(history_dictionary,
				'history', 'solutions')
		# elif isinstance(history_dictionary, list):
			# self.solutions = history_dictionary

	def flatten(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		for i, s in enumerate(self.__solutions):
			if isinstance(s, SolutionEntry):
				self.__solutions[i] = conrad_db.set_next(s.flatten(conrad_db))
		return self

	def arborize(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		for i, s in enumerate(self.__solutions):
			self.__solutions[i] = conrad_db.get(s).arborize(conrad_db)
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'history': cdb_util.expand_list_if_db_entries(self.solutions)
		}

	@property
	def flat_dictionary(self):
		if not cdb_util.check_flat(self.solutions, SolutionEntry):
			raise ValueError(
					'cannot emit flat dictionary from {}: all listed '
					'solutions must be database pointer strings, not '
					'database entry objects of type {}'
					''.format(HistoryEntry, SolutionEntry))
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'history': str(self.solutions),
		}

class SolutionEntry(ConradDatabaseEntry):
	def __init__(self, **entry_dictionary):
		ConradDatabaseEntry.__init__(self)
		self.__name = None
		self.__frame = None
		self.__x = None
		self.__y = None
		self.__x_dual = None
		self.__y_dual = None
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		complete = isinstance(self.name, str)
		complete &= isinstance(self.frame, str)
		complete &= bool(
				cdb_util.isinstance_or_db_pointer(self.x, DataFragmentEntry) or
				cdb_util.isinstance_or_db_pointer(self.y, DataFragmentEntry))
		return complete

	@property
	def name(self):
		return self.__name

	@name.setter
	def name(self, name):
		if name is not None:
			name = str(name)
			if len(name) == 0:
				raise ValueError('solution name cannot be empty string')
			self.__name = name

	@property
	def frame(self):
		return self.__frame

	@frame.setter
	def frame(self, frame_name):
		if frame_name is not None:
			frame = str(frame_name)
			if len(frame) == 0:
				raise ValueError(
						'name of frame associated with solution cannot '
						'be empty string')
			self.__frame = frame

	@property
	def x(self):
		return self.__x

	@x.setter
	def x(self, x_entry):
		x_entry = cdb_util.route_data_fragment(x_entry)
		if cdb_util.isinstance_or_db_pointer(x_entry, DataFragmentEntry):
			self.__x = x_entry

	@property
	def y(self):
		return self.__y

	@y.setter
	def y(self, y_entry):
		y_entry = cdb_util.route_data_fragment(y_entry)
		if cdb_util.isinstance_or_db_pointer(y_entry, DataFragmentEntry):
			self.__y = y_entry

	@property
	def x_dual(self):
		return self.__x_dual

	@x_dual.setter
	def x_dual(self, x_dual_entry):
		x_dual_entry = cdb_util.route_data_fragment(x_dual_entry)
		if cdb_util.isinstance_or_db_pointer(x_dual_entry, DataFragmentEntry):
			self.__x_dual = x_dual_entry

	@property
	def y_dual(self):
		return self.__y_dual

	@y_dual.setter
	def y_dual(self, y_dual_entry):
		y_dual_entry = cdb_util.route_data_fragment(y_dual_entry)
		if cdb_util.isinstance_or_db_pointer(y_dual_entry, DataFragmentEntry):
			self.__y_dual = y_dual_entry

	def ingest_dictionary(self, **solution_dictionary):
		self.name = solution_dictionary.pop('name', None)
		self.frame = solution_dictionary.pop('frame', None)
		self.x = cdb_util.try_keys(
				solution_dictionary, 'x', 'beam_intensities', 'beam_weights')
		self.y = cdb_util.try_keys(
				solution_dictionary, 'y', 'voxel_doses')
		self.x_dual = cdb_util.try_keys(
				solution_dictionary, 'x_dual', 'mu', 'beam_prices')
		self.y_dual = cdb_util.try_keys(
				solution_dictionary, 'y_dual', 'nu', 'voxel_prices')

	def flatten(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if isinstance(self.x, ConradDatabaseEntry):
			self.x = conrad_db.set_next(self.x.flatten(conrad_db))
		if isinstance(self.y, ConradDatabaseEntry):
			self.y = conrad_db.set_next(self.y.flatten(conrad_db))
		if isinstance(self.x_dual, ConradDatabaseEntry):
			self.x_dual = conrad_db.set_next(self.x_dual.flatten(conrad_db))
		if isinstance(self.y_dual, ConradDatabaseEntry):
			self.y_dual = conrad_db.set_next(self.y_dual.flatten(conrad_db))
		return self

	def arborize(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if self.x is not None:
			self.x = conrad_db.get(self.x).arborize(conrad_db)
		if self.y is not None:
			self.y = conrad_db.get(self.y).arborize(conrad_db)
		if self.x_dual is not None:
			self.x_dual = conrad_db.get(self.x_dual).arborize(conrad_db)
		if self.y_dual is not None:
			self.y_dual = conrad_db.get(self.y_dual).arborize(conrad_db)
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'name': self.name,
				'frame': self.frame,
				'x': cdb_util.expand_if_db_entry(self.x),
				'y': cdb_util.expand_if_db_entry(self.y),
				'x_dual': cdb_util.expand_if_db_entry(self.x_dual),
				'y_dual': cdb_util.expand_if_db_entry(self.y_dual),
		}

	@property
	def flat_dictionary(self):
		checklist = [self.x, self.y, self.x_dual, self.y_dual]
		if not cdb_util.check_flat(checklist, DataFragmentEntry):
			raise ValueError(
				'cannot emit flat dictionary from {}: solution fields '
				'`x`, `y`, `x_dual` and `y_dual`, if provided, must be '
				'database pointer strings, not database entry objects'
				''.format(SolutionEntry))
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'name': self.name,
				'frame': self.frame,
				'x': self.x,
				'y': self.y,
				'x_dual': self.x_dual,
				'y_dual': self.y_dual,
		}

class SolverCacheEntry(ConradDatabaseEntry):
	def __init__(self, **entry_dictionary):
		ConradDatabaseEntry.__init__(self)
		self.__name = None
		self.__frame = None
		self.__solver = None
		self.__left_preconditioner = None
		self.__matrix = None
		self.__right_preconditioner = None
		self.__projector_type = None
		self.__projector_matrix = None
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		complete = isinstance(self.name, str)
		complete &= isinstance(self.frame, str)
		complete &= isinstance(self.solver, str)
		complete &= cdb_util.isinstance_or_db_pointer(
				self.left_preconditioner, DataFragmentEntry)
		complete &= cdb_util.isinstance_or_db_pointer(
				self.matrix, DataFragmentEntry)
		complete &= cdb_util.isinstance_or_db_pointer(
				self.right_preconditioner, DataFragmentEntry)
		complete &= isinstance(self.projector_type, str)
		if self.projector_matrix is not None:
			complete &= cdb_util.isinstance_or_db_pointer(
					self.projector_matrix, DataFragmentEntry)
		return complete

	@property
	def name(self):
		return self.__name

	@name.setter
	def name(self, name):
		if isinstance(name, str):
			self.__name = name

	@property
	def frame(self):
		return self.__frame

	@frame.setter
	def frame(self, frame_name):
		if isinstance(frame_name, str):
			self.__frame = frame_name

	@property
	def solver(self):
		return self.__solver

	@solver.setter
	def solver(self, solver_name):
		if isinstance(solver_name, str):
			self.__solver = solver_name

	@property
	def left_preconditioner(self):
		return self.__left_preconditioner

	@left_preconditioner.setter
	def left_preconditioner(self, left_preconditioner_entry):
		left_preconditioner_entry = cdb_util.route_data_fragment(
				left_preconditioner_entry)
		if cdb_util.isinstance_or_db_pointer(
				left_preconditioner_entry, DataFragmentEntry):
			self.__left_preconditioner = left_preconditioner_entry

	@property
	def matrix(self):
		return self.__matrix

	@matrix.setter
	def matrix(self, matrix_entry):
		matrix_entry = cdb_util.route_data_fragment(matrix_entry)
		if cdb_util.isinstance_or_db_pointer(matrix_entry, DataFragmentEntry):
			self.__matrix = matrix_entry

	@property
	def right_preconditioner(self):
		return self.__right_preconditioner

	@right_preconditioner.setter
	def right_preconditioner(self, right_preconditioner_entry):
		right_preconditioner_entry = cdb_util.route_data_fragment(
				right_preconditioner_entry)
		if cdb_util.isinstance_or_db_pointer(
				right_preconditioner_entry, DataFragmentEntry):
			self.__right_preconditioner = right_preconditioner_entry

	@property
	def projector_type(self):
		return self.__projector_type

	@projector_type.setter
	def projector_type(self, projector_type_string):
		if isinstance(projector_type_string, str):
			self.__projector_type = projector_type_string

	@property
	def projector_matrix(self):
		return self.__projector_matrix

	@projector_matrix.setter
	def projector_matrix(self, projector_matrix_entry):
		projector_matrix_entry = cdb_util.route_data_fragment(
				projector_matrix_entry)
		if cdb_util.isinstance_or_db_pointer(
				projector_matrix_entry, DataFragmentEntry):
			self.__projector_matrix = projector_matrix_entry

	def ingest_dictionary(self, **solver_cache_dictionary):
		self.name = cdb_util.try_keys(
				solver_cache_dictionary, 'name', 'cache_name')
		self.frame = solver_cache_dictionary.pop('frame', None)
		self.solver = solver_cache_dictionary.pop('solver', None)
		self.left_preconditioner = solver_cache_dictionary.pop(
				'left_preconditioner', None)
		self.matrix = solver_cache_dictionary.pop('matrix', None)
		self.right_preconditioner = solver_cache_dictionary.pop(
				'right_preconditioner', None)
		self.projector_type = cdb_util.try_keys(
				solver_cache_dictionary, 'projector_type',
				['projector', 'type'])
		self.projector_matrix = cdb_util.try_keys(
				solver_cache_dictionary, 'projector_matrix',
				['projector', 'matrix'])

	def flatten(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if isinstance(self.left_preconditioner, ConradDatabaseEntry):
			self.left_preconditioner = conrad_db.set_next(
					self.left_preconditioner.flatten(conrad_db))
		if isinstance(self.matrix, ConradDatabaseEntry):
			self.matrix = conrad_db.set_next(self.matrix.flatten(conrad_db))
		if isinstance(self.right_preconditioner, ConradDatabaseEntry):
			self.right_preconditioner = conrad_db.set_next(
					self.right_preconditioner.flatten(conrad_db))
		if isinstance(self.projector_matrix, ConradDatabaseEntry):
			self.projector_matrix = conrad_db.set_next(
					self.projector_matrix.flatten(conrad_db))
		return self

	def arborize(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		if self.left_preconditioner is not None:
			self.left_preconditioner = conrad_db.get(
					self.left_preconditioner).arborize(conrad_db)
		if self.matrix is not None:
			self.matrix = conrad_db.get(self.matrix).arborize(conrad_db)
		if self.right_preconditioner is not None:
			self.right_preconditioner = conrad_db.get(
					self.right_preconditioner).arborize(conrad_db)
		if self.projector_matrix is not None:
			self.projector_matrix = conrad_db.get(
					self.projector_matrix).arborize(conrad_db)
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'name': self.name,
				'frame': self.frame,
				'solver': self.solver,
				'left_preconditioner': cdb_util.expand_if_db_entry(
						self.left_preconditioner),
				'matrix': cdb_util.expand_if_db_entry(self.matrix),
				'right_preconditioner': self.right_preconditioner,
				'projector': {
						'type': self.projector_type,
						'matrix': cdb_util.expand_if_db_entry(
								self.projector_matrix),
				}
		}

	@property
	def flat_dictionary(self):
		checklist = [self.left_preconditioner, self.matrix,
					 self.right_preconditioner, self.projector_matrix]
		if not cdb_util.check_flat(checklist, DataFragmentEntry):
			raise ValueError(
					'cannot emit flat dictionary from {}: entries for '
					'preconditioners and matrices must be database '
					'pointer strings, not database entry objects'
					''.format(SolverCacheEntry))
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'name': self.name,
				'frame': self.frame,
				'solver': self.solver,
				'left_preconditioner': self.left_preconditioner,
				'matrix': self.matrix,
				'right_preconditioner': self.right_preconditioner,
				'projector_type': self.projector_type,
				'projector_matrix': self.projector_matrix,
		}

class AnatomyEntry(ConradDatabaseEntry):
	def __init__(self, **entry_dictionary):
		ConradDatabaseEntry.__init__(self)
		self.__structures = []
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		return len(self.structures) > 0

	@property
	def structures(self):
		return self.__structures

	@structures.setter
	def structures(self, structure_list):
		if structure_list is None:
			return
		self.__structures = []
		if isinstance(structure_list, str):
			structure_list = ast.literal_eval(structure_list)
		self.add_structures(*structure_list)

	def add_structures(self, *structures):
		safe_list = []
		for s in structures:
			if isinstance(s, dict):
				s = StructureEntry(**s)
			if cdb_util.isinstance_or_db_pointer(s, StructureEntry):
				safe_list.append(s)
			else:
				raise ValueError(
						'argument `structures` must be a list of {} '
						'objects or corresponding database pointer '
						'strings'.format(StructureEntry))
		self.__structures += safe_list

	def ingest_dictionary(self, **anatomy_dictionary):
		self.structures = cdb_util.try_keys(anatomy_dictionary,
				'anatomy', 'structures')
		# elif isinstance(anatomy_dictionary, list):
			# self.structures = anatomy_dictionary

	def flatten(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		for i, s in enumerate(self.__structures):
			if isinstance(s, ConradDatabaseEntry):
				self.__structures[i] = conrad_db.set_next(s.flatten(conrad_db))
		return self

	def arborize(self, conrad_db):
		cdb_util.validate_db(conrad_db)
		for i, s in enumerate(self.__structures):
			self.__structures[i] = conrad_db.get(s).arborize(conrad_db)
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'anatomy': cdb_util.expand_list_if_db_entries(self.structures)
		}

	@property
	def flat_dictionary(self):
		if not cdb_util.check_flat(self.structures, StructureEntry):
			raise ValueError(
					'cannot emit flat dictionary from {}: all listed '
					'structures must be database pointer strings, not '
					'database entry objects of type {}'
					''.format(AnatomyEntry, StructureEntry))
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'anatomy': str(self.structures)
		}

class StructureEntry(ConradDatabaseEntry):
	def __init__(self, **entry_dictionary):
		self.__name = None
		self.__label = None
		self.__target = None
		self.__rx = None
		self.__size = None
		self.__constraints = []
		self.__objective = {'type': None, 'weight': None, 'parameters': None}
		self.ingest_dictionary(**entry_dictionary)

	@property
	def complete(self):
		complete = isinstance(self.name, str)
		complete &= isinstance(self.label, int)
		complete &= isinstance(self.target, bool)
		complete &= isinstance(self.rx, str) if self.target else True
		return complete

	@property
	def name(self):
		return self.__name

	@name.setter
	def name(self, name):
		if name is not None:
			name = str(name)
			if len(name) == 0:
				raise ValueError('structure name cannot be empty string')
			self.__name = name

	@property
	def label(self):
		return self.__label

	@label.setter
	def label(self, label):
		if label is not None:
			self.__label = int(label)

	@property
	def target(self):
		return self.__target

	@target.setter
	def target(self, is_target):
		if is_target is not None:
			self.__target = bool(is_target)

	@property
	def rx(self):
		return self.__rx

	@rx.setter
	def rx(self, rx_dose):
		if rx_dose is not None:
			self.__rx = str(rx_dose)

	@property
	def size(self):
		return self.__size

	@size.setter
	def size(self, size):
		if size is not None:
			self.__size = int(size)

	@property
	def constraints(self):
		return self.__constraints

	@constraints.setter
	def constraints(self, constraint_list):
		if constraint_list is None:
			return
		elif isinstance(constraint_list, string):
			constraint_list = ast.literal_eval(constraint_list)

		if not all(map(lambda s: isinstance(s, string), constraint_list)):
			raise ValueError(
					'`constraint_list` must be a list of ConRad '
					'dose constraint strings')
		self.__constraints = [c for c in constraint_list]

	@property
	def objective(self):
		return self.__objective

	@objective.setter
	def objective(self, objective_dict):
		type_ = objective_dict.pop('type', None)
		if type_ is not None:
			self.__objectives['type'] = str(type_)
		weight = objective_dict.pop('weight', None)
		if weight is not None:
			self.__objectives['weight'] = float(weight)
		parameters = objective_dict.pop('parameters', None)
		if parameters not in (None, 'None'):
			if isinstance(parameters, str):
				parameters = ast.literal_eval(parameters)
			self.__objectives['parameters'] = [float(p) for p in parameters]

	def ingest_dictionary(self, **structure_dictionary):
		self.name = structure_dictionary.pop('name', None)
		self.label = structure_dictionary.pop('label', None)
		self.target = structure_dictionary.pop('target', None)
		self.rx = cdb_util.try_keys(
				structure_dictionary, 'rx', 'dose', 'rx_dose')
		self.constraint_list = structure_dictionary.pop('constraints', None)
		self.objective = {
				'type': cdb_util.try_keys(
						structure_dictionary, 'objective_type',
						['objective', 'type']),
				'weight': cdb_util.try_keys(
						structure_dictionary, 'objective_weight',
						['objective', 'weight']),
				'parameters': cdb_util.try_keys(
						structure_dictionary, 'objective_parameters',
						['objective', 'parameters']),
		}

	def flatten(self, conrad_db):
		return self

	def arborize(self, conrad_db):
		return self

	@property
	def nested_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'name': self.name,
				'label': self.label,
				'target': self.target,
				'rx': self.rx,
				'size': self.size,
				'constraints': self.constraints,
				'objective': self.objective,
		}

	@property
	def flat_dictionary(self):
		return {
				CONRAD_DB_TYPETAG: CONRAD_DB_TYPESTRING[type(self)],
				'name': self.name,
				'label': self.label,
				'target': self.target,
				'rx': self.rx,
				'size': self.size,
				'constraints': str(self.constraints),
				'objective_type': self.objective['type'],
				'objective_weight': self.objective['weight'],
				'objective_parameters': str(self.objective['parameters']),
		}

CONRAD_DB_ENTRY_PREFIXES = {
		DataFragmentEntry: 'data_fragment.',
		DataDictionaryEntry: 'data_fragment.',
		VectorEntry: 'data_fragment.',
		DenseMatrixEntry: 'data_fragment.',
		SparseMatrixEntry: 'data_fragment.',
		PhysicsEntry: 'physics.',
		AnatomyEntry: 'anatomy.',
		StructureEntry: 'structure.',
		HistoryEntry: 'history.',
		SolutionEntry: 'solution.',
		SolverCacheEntry: 'solver_cache.',
		DoseFrameEntry: 'frame.',
		DoseFrameMappingEntry: 'frame_mapping.',
		CaseEntry: 'case.',
}

CONRAD_DB_ENTRY_TYPES = {
	k: CONRAD_DB_ENTRY_PREFIXES[k].replace('.', '') for
	k in CONRAD_DB_ENTRY_PREFIXES
}

CONRAD_DB_TYPETAG = 'conrad_db_type'

CONRAD_DB_TYPESTRING = {
		DataDictionaryEntry: 'data_fragment: dictionary',
		VectorEntry: 'data_fragment: vector',
		DenseMatrixEntry: 'data_fragment: dense matrix',
		SparseMatrixEntry: 'data_fragment: sparse matrix',
		DoseFrameEntry: 'frame',
		DoseFrameMappingEntry: 'frame_mapping',
		PhysicsEntry: 'physics',
		StructureEntry: 'structure',
		AnatomyEntry: 'anatomy',
		SolutionEntry: 'solution',
		HistoryEntry: 'history',
		SolverCacheEntry: 'solver_cache',
		CaseEntry: 'case',
}

CONRAD_DB_TYPESTRING_TO_CONSTRUCTOR = {
		'data_fragment: dictionary': DataDictionaryEntry,
		'data_fragment: vector': VectorEntry,
		'data_fragment: dense matrix': DenseMatrixEntry,
		'data_fragment: sparse matrix': SparseMatrixEntry,
		'frame': DoseFrameEntry,
		'frame_mapping': DoseFrameMappingEntry,
		'physics': PhysicsEntry,
		'structure': StructureEntry,
		'anatomy': AnatomyEntry,
		'solution': SolutionEntry,
		'history': HistoryEntry,
		'solver_cache': SolverCacheEntry,
		'case': CaseEntry,
}