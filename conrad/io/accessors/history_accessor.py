"""
Define :class:`SolutionAccessor` and :class:`HistoryAccessor`
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

from conrad.physics.physics import DEFAULT_FRAME0_NAME
from conrad.io.schema import HistoryEntry, SolutionEntry, cdb_util
from conrad.io.accessors.base_accessor import ConradDBAccessor

class SolutionAccessor(ConradDBAccessor):
	def __init__(self, database=None, filesystem=None):
		ConradDBAccessor.__init__(
				self, database=database, filesystem=filesystem)

	def save_solution(self, directory, solution_name, frame_name='default',
					  overwrite=False, **solution_components):
		if frame_name in 'default':
			frame_name = DEFAULT_FRAME0_NAME

		self.FS.check_dir(directory)
		subdir = self.FS.join_mkdir(
				directory, 'solutions', 'frame_{}'.format(frame_name),
				solution_name)

		s = SolutionEntry(frame=frame_name, name=solution_name)
		s.x = self.pop_and_record(
				solution_components, 'x', subdir,
				alternate_keys=['beam_intensities', 'beam_weights'],
				overwrite=overwrite)
		s.y = self.pop_and_record(
				solution_components, 'y', subdir,
				alternate_keys=['voxel_doses'],
				overwrite=overwrite)
		s.x_dual = self.pop_and_record(
				solution_components, 'x_dual', subdir,
				alternate_keys=['mu', 'beam_prices'],
				overwrite=overwrite)
		s.y_dual = self.pop_and_record(
				solution_components, 'y_dual', subdir,
				alternate_keys=['nu', 'voxel_prices'],
				overwrite=overwrite)
		return self.DB.set_next(s)

	def load_solution(self, solution_entry):
		solution_entry = self.DB.get(solution_entry)
		if not isinstance(solution_entry, SolutionEntry):
			raise ValueError(
					'argument `solution_entry` must be of type {}, '
					'or a dictionary representation of/ConRad database '
					'pointer to that type'.format(SolutionEntry))
		if not solution_entry.complete:
			raise ValueError('solution incomplete')

		x = self.load_entry(solution_entry.x)
		y = self.load_entry(solution_entry.y)
		x_dual = self.load_entry(solution_entry.x_dual)
		y_dual = self.load_entry(solution_entry.y_dual)
		return {
				'x': x,
				'y': y,
				'x_dual': x_dual,
				'y_dual': y_dual,
		}

	def select_solution_entry(self, solution_list, frame_name, solution_name):
		for sol in map(self.DB.get, solution_list):
			if sol.frame == frame_name and sol.name == solution_name:
				return sol

		raise ValueError(
				'solution with name `{}` associated with frame `{}` '
				'not found'.format(solution_name, frame_name))

class HistoryAccessor(ConradDBAccessor):
	def __init__(self, database=None, filesystem=None):
		self.solution_accessor = SolutionAccessor(
				database=database, filesystem=filesystem)
		self.__solution_cache = {}
		ConradDBAccessor.__init__(
				self, subaccessors=[self.solution_accessor], database=database,
				filesystem=filesystem)

	def save_history(self, history_dictionary, directory, overwrite=False):
		self.FS.check_dir(directory)

		h = HistoryEntry()
		for key in history_dictionary:
			subdir = self.FS.join_mkdir(directory, ['solutions', key])

			solution_dictionary = history_dictionary[key]
			h.add_solutions(self.solution_accessor.save_solution(
					subdir, key, solution_dictionary.pop('frame', 'default'),
					overwrite=overwrite, **solution_dictionary))
		return self.DB.set_next(h)

	def load_history(self, history_entry):
		history_entry = self.DB.get(history_entry)
		if not isinstance(history_entry, HistoryEntry):
			raise TypeError(
					'argument `history_entry` must be of type {}, '
					'or a dictionary representation of/ConRad database '
					'pointer to that type'.format(HistoryEntry))
		if not history_entry.complete:
			raise ValueError('history incomplete')

		for sol in history_entry.solutions:
			sol = self.DB.get(sol)
			self.__solution_cache[sol.name] = sol

		return history_entry

	def load_solution(self, frame_name, solution_name):
		if len(self.__solution_cache) == 0:
			raise ValueError('no history data loaded')

		return self.solution_accessor.load_solution(
				self.solution_accessor.select_solution_entry(
						self.__solution_cache.values(), frame_name,
						solution_name))
