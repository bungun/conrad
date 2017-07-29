"""
Define :class:`DoseFrameAccessor`, :class:`FrameMappingAccessor` and
:class:`PhysicsAccessor`.
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

from conrad.abstract.mapping import string_to_map_constructor
from conrad.physics.physics import Physics, DoseFrame, DoseFrameMapping
from conrad.physics.physics import DEFAULT_FRAME0_NAME
from conrad.case import Case
from conrad.io.schema import DoseFrameEntry, DoseFrameMappingEntry
from conrad.io.schema import PhysicsEntry
from conrad.io.accessors.base_accessor import ConradDBAccessor

class DoseFrameAccessor(ConradDBAccessor):
	def __init__(self, database=None, filesystem=None):
		ConradDBAccessor.__init__(
				self, database=database, filesystem=filesystem)

	def save_frame(self, frame, directory, overwrite=False):
		if not isinstance(frame, DoseFrame):
			raise TypeError(
					'argument `frame` must be of type {}'
					''.format(DoseFrame))

		self.FS.check_dir(directory)
		subdir = self.FS.join_mkdir(directory, 'frames', frame.name)

		if frame.dose_matrix is not None:
			dm = self.record_entry(
					subdir, 'dose_matrix', frame.dose_matrix.manifest,
					overwrite=overwrite)
		else:
			dm = None

		if frame.voxel_weights is not None:
			vw = self.record_entry(
					subdir, 'voxel_weights', frame.voxel_weights.manifest,
					overwrite=overwrite)
		else:
			vw = None

		if frame.beam_weights is not None:
			bw = self.record_entry(
					subdir, 'beam_weights', frame.beam_weights.manifest,
					overwrite=overwrite)
		else:
			bw = None

		return self.DB.set_next(DoseFrameEntry(
				name=frame.name, n_voxels=frame.voxels, n_beams=frame.beams,
				dose_matrix=dm, voxel_weights=vw, beam_weights=bw,
				voxel_labels=self.record_entry(
						subdir, 'voxel_labels', frame.voxel_labels, overwrite),
				beam_labels=self.record_entry(
						subdir, 'beam_labels', frame.beam_labels, overwrite),

		))

	def load_frame(self, frame_entry):
		frame_entry = self.DB.get(frame_entry)
		if not isinstance(frame_entry, DoseFrameEntry):
			raise ValueError(
					'argument `frame_entry` must be of type {}, '
					'or a dictionary representation of/ConRad database '
					'pointer to that type'.format(DoseFrameEntry))
		if not frame_entry.complete:
			raise ValueError('dose frame incomplete')

		frame = DoseFrame(
				voxels=frame_entry.n_voxels, beams=frame_entry.n_beams,
				frame_name=frame_entry.name)

		if frame_entry.dose_matrix is not None:
			frame.dose_matrix = self.load_entry(frame_entry.dose_matrix)
		if frame_entry.voxel_labels is not None:
			frame.voxel_labels = self.load_entry(frame_entry.voxel_labels)
		if frame_entry.voxel_weights is not None:
			frame.voxel_weights = self.load_entry(frame_entry.voxel_weights)
		if frame_entry.beam_labels is not None:
			frame.beam_labels = self.load_entry(frame_entry.beam_labels)
		if frame_entry.beam_weights is not None:
			frame.beam_weights = self.load_entry(frame_entry.beam_weights)

		return frame

	def select_frame_entry(self, frame_list, frame_name='default'):
		if frame_name == 'default':
			frame_name = DEFAULT_FRAME0_NAME

		for frame_entry in map(self.DB.get, frame_list):
			if frame_entry.name == frame_name:
				return frame_entry

		raise ValueError(
				'not found: frame entry for frame `{}`'
				''.format(frame_name))

class FrameMappingAccessor(ConradDBAccessor):
	def __init__(self, database=None, filesystem=None):
		ConradDBAccessor.__init__(
				self, database=database, filesystem=filesystem)

	def save_frame_mapping(self, frame_mapping, directory, overwrite=False):
		if not isinstance(frame_mapping, DoseFrameMapping):
			raise TypeError(
					'argument `frame_mapping` must be of type {}'
					''.format(DoseFrameMapping))

		map_name = frame_mapping.source + '_to_' + frame_mapping.target

		self.FS.check_dir(directory)
		subdir = self.FS.join_mkdir(directory, 'frame_mappings', map_name)

		fm = frame_mapping
		vmap = fm.voxel_map.manifest if fm.voxel_map is not None else None
		bmap = fm.beam_map.manifest if fm.beam_map is not None else None

		return self.DB.set_next(DoseFrameMappingEntry(
				source_frame=frame_mapping.source,
				target_frame=frame_mapping.target,
				voxel_map=self.record_entry(
						subdir, 'voxel_map', vmap, overwrite=overwrite),
				voxel_map_type=frame_mapping.voxel_map_type,
				beam_map=self.record_entry(
						subdir, 'beam_map', bmap, overwrite=overwrite),
				beam_map_type=frame_mapping.beam_map_type
		))

	def load_frame_mapping(self, frame_mapping_entry):
		frame_mapping_entry = self.DB.get(frame_mapping_entry)
		if not isinstance(frame_mapping_entry, DoseFrameMappingEntry):
			raise ValueError(
					'argument `frame_mapping_entry` must be of type {}, '
					'or a dictionary representation of/ConRad database '
					'pointer to that type'.format(DoseFrameEntry))

		if not frame_mapping_entry.complete:
			raise ValueError('dose frame mapping incomplete')

		vmap = self.load_entry(frame_mapping_entry.voxel_map)
		if vmap is not None:
			vmap = string_to_map_constructor(
					frame_mapping_entry.voxel_map_type)(vmap['data'])

		bmap = self.load_entry(frame_mapping_entry.beam_map)
		if bmap is not None:
			bmap = string_to_map_constructor(
					frame_mapping_entry.beam_map_type)(bmap['data'])

		return DoseFrameMapping(
				frame_mapping_entry.source_frame,
				frame_mapping_entry.target_frame, vmap, bmap)

	def select_frame_mapping_entry(self, frame_mapping_list,
								   source_frame='default',
								   target_frame='default'):
		if source_frame == target_frame:
			raise ValueError(
					'source and target frame names must be different to '
					'retrieve a DoseFrame->DoseFrame mapping')

		for frame_mapping in map(self.DB.get, frame_mapping_list):
			if bool(frame_mapping.source_frame == source_frame and
					frame_mapping.target_frame == target_frame):
				return frame_mapping

		raise ValueError(
				'not found: frame mapping for source frame=`{}` and '
				'target frame=`{}`'
				''.format(source_frame, target_frame))

class PhysicsAccessor(ConradDBAccessor):
	def __init__(self, database=None, filesystem=None):
		self.__frame_accessor = DoseFrameAccessor(
				database=database, filesystem=filesystem)
		self.__frame_mapping_accessor = FrameMappingAccessor(
				database=database, filesystem=filesystem)
		self.__frame_cache = []
		self.__frame_mapping_cache = []
		ConradDBAccessor.__init__(self, subaccessors=[
				self.__frame_accessor, self.__frame_mapping_accessor],
				database=database, filesystem=filesystem)

	@property
	def frame_accessor(self):
		return self.__frame_accessor

	@property
	def frame_mapping_accessor(self):
		return self.__frame_mapping_accessor

	def save_physics(self, physics, directory, overwrite=False):
		if not isinstance(physics, Physics):
			raise TypeError(
					'argument `physics` must be of type {}'
					''.format(Physics))
		self.FS.check_dir(directory)

		if physics.dose_grid is not None:
			grid = {
					'x': physics.dose_grid.x_voxels,
					'y': physics.dose_grid.y_voxels,
					'z': physics.dose_grid.z_voxels,
			}
		else:
			grid = None

		frames = [
			self.frame_accessor.save_frame(f, directory, overwrite)
			for f in physics.unique_frames
		]

		mappings = [
				self.frame_mapping_accessor.save_frame_mapping(fm, directory, overwrite)
				for fm in [
						physics.retrieve_frame_mapping(s, t)
						for s, t in physics.available_frame_mappings
			]
		]

		return self.DB.set_next(PhysicsEntry(
				voxel_grid=grid, frames=frames, frame_mappings=mappings
		))

	def load_physics(self, physics_entry, frame_name='default'):
		physics_entry = self.DB.get(physics_entry)
		if not isinstance(physics_entry, PhysicsEntry):
			raise ValueError(
					'argument `physics_entry` must be of type {}, '
					'or a dictionary representation of/ConRad database '
					'pointer to that type'.format(PhysicsEntry))
		if not physics_entry.complete:
			raise ValueError('physics incomplete')

		if all([dim is not None for dim in physics_entry.voxel_grid.values()]):
			grid = physics_entry.voxel_grid
		else:
			grid = None

		self.__frame_cache = [self.DB.get(f) for f in physics_entry.frames]
		self.__frame_mapping_cache = [
			self.DB.get(fm) for fm in physics_entry.frame_mappings]

		frame_names = [f.name for f in self.__frame_cache]
		frame_names.sort()
		if frame_name == 'default':
			frame_name = frame_names[0]

		return Physics( dose_grid=grid, dose_frame=self.load_frame(frame_name))

	def load_frame(self, frame_name='default'):
		return self.frame_accessor.load_frame(
				self.frame_accessor.select_frame_entry(
						self.__frame_cache, frame_name))

	def load_frame_mapping(self, source_frame='default',
						   target_frame='default'):
		return self.frame_mapping_accessor.load_frame_mapping(
				self.frame_mapping_accessor.select_frame_mapping_entry(
						self.__frame_mapping_cache, source_frame,
						target_frame))

	@property
	def available_frames(self):
		return [f.name for f in self.__frame_cache]

	@property
	def available_frame_mappings(self):
		return [(fm.source, fm.target) for f in self.__frame_mapping_cache]

