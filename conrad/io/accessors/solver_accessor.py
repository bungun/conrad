"""
Define :class:`SolverCacheAccessor`
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

from conrad.optimization.solvers.solver_optkit import SolverOptkit, \
											  PROJECTOR_POGS_DENSE_DIRECT
from conrad.io.schema import SolverCacheEntry
from conrad.io.accessors.base_accessor import ConradDBAccessor

class SolverCacheAccessor(ConradDBAccessor):
	def __init__(self, database=None, filesystem=None):
		ConradDBAccessor.__init__(
				self, database=database, filesystem=filesystem)

	def __save_pogs_solver_cache(self, solver, cache_name, frame_name,
								 directory, overwrite=False):
		solver_cache_raw = solver.cache
		solver_cache_entry = SolverCacheEntry(
				name=cache_name, frame=frame_name, solver='POGS')

		self.FS.check_dir(directory)
		subdir = self.FS.join_mkdir(
				directory, 'solver_caches', 'pogs_solver_caches', frame_name)

		# save equilibration
		solver_cache_entry.matrix = self.pop_and_record(
				solver_cache_raw, 'matrix', subdir, overwrite=overwrite)
		solver_cache_entry.left_preconditioner = self.pop_and_record(
				solver_cache_raw, 'left_preconditioner', subdir,
				overwrite=overwrite)
		solver_cache_entry.right_preconditioner = self.pop_and_record(
				solver_cache_raw, 'right_preconditioner', subdir,
				overwrite=overwrite)

		# save projector
		solver_cache_entry.projector_matrix = self.pop_and_record(
				solver_cache_raw, 'projector_matrix', subdir,
				alternate_keys=[['projector', 'matrix']], overwrite=overwrite)
		solver_cache_entry.projector_type = solver_cache_raw.pop(
				'projector_type', solver_cache_raw.pop('projector').pop('type'))
		return self.DB.set_next(solver_cache_entry)

	def __load_pogs_solver_cache(self, solver_cache_entry):
		solver_cache_entry = self.DB.get(solver_cache_entry)
		if not solver_cache_entry.solver == 'POGS':
			raise ValueError(
					'method expects a POGS solver. solver = {}'
					''.format(solver_cache_entry.solver))

		d = self.load_entry(
				solver_cache_entry.left_preconditioner)
		A = self.load_entry(solver_cache_entry.matrix)
		e = self.load_entry(
				solver_cache_entry.right_preconditioner)
		if solver_cache_entry.projector_type == PROJECTOR_POGS_DENSE_DIRECT:
			L = self.load_entry(
					solver_cache_entry.projector_matrix)
		else:
			# e.g., projector_type == 'indirect'
			L = None
		return {
				'left_preconditioner': d,
				'matrix': A,
				'right_preconditioner': e,
				'projector_matrix': L,
		}

	# TODO: add solver name field?
	def save_solver_cache(self, solver, cache_name, frame_name, directory,
							   overwrite=False):
		self.FS.check_dir(directory)

		if isinstance(solver, SolverOptkit):
			return self.__save_pogs_solver_cache(
				solver, cache_name, frame_name, directory, overwrite)
		else:
			raise TypeError(
					'solver caching only implemented for solvers of '
					'type {}'.format(SolverOptkit))

	# TODO: add cache name
	def load_solver_cache(self, solver_cache_entry):
		solver_cache_entry = self.DB.get(solver_cache_entry)
		if not isinstance(solver_cache_entry, SolverCacheEntry):
			raise ValueError(
					'argument `solver_cache_entry` must be of type {}, '
					'or a dictionary representation of/ConRad database '
					'pointer to that type'.format(SolverCacheEntry))
		if not solver_cache_entry.complete:
			raise ValueError('solver_cache incomplete')

		if solver_cache_entry.solver == 'POGS':
			return self.__load_pogs_solver_cache(solver_cache_entry)
		else:
			raise ValueError(
					'solver caching only implemented for solvers of '
					'type {}'.format(SolverOptkit))

	# TODO: add cache name
	def select_solver_cache_entry(self, solver_cache_list, cache_name,
								  frame_name):
		for cache in map(self.DB.get, solver_cache_list):
			if cache.name == cache_name and cache.frame == frame_name:
				return cache

		raise ValueError(
				'solver cache for cache name `{}` and frame `{}` not '
				'found'.format(cache_name, frame_name))