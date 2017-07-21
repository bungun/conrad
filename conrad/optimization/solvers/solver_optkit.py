"""
Define POGS-based solver using :mod:`optkit`, if available.

For information on POGS, see:
https://foges.github.io/pogs/

For infromation on :mod:`optkit`, see:
https://github.com/bungun/optkit

If :func:`conrad.defs.module_installed` does not find the :mod:`optkit`,
the variable ``SolverOptkit`` is still defined in the module
namespace as a lambda returning ``None`` with the same method signature
as the initializer for :class:`SolverOptkit`. If :mod:`optkit` is found,
the class is defined normally.

# TODO: change backend switching syntax to check flag .precision_is_64bit
instead of current .precision_is_32bit when optkit api updated
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

from conrad.defs import module_installed, CONRAD_DEBUG_PRINT
from conrad.medicine.anatomy import Anatomy
from conrad.optimization.preprocessing import ObjectiveMethods
from conrad.optimization.solvers.solver_base import *

if module_installed('optkit'):
	import optkit as ok

	class SolverOptkit(Solver):
		r"""
		Interface between :mod:`conrad` and :mod:`optkit`'s POGS
		implementation.

		:class:`SolverOptkit` interprets :mod:`conrad`: treatment
		planning problems (based on structures with attached objectives,
		dose constraints, and dose matrices) to build equivalent convex
		optimization problems using POGS' syntax.

		The class provides an interface to modify, run, and retrieve
		solutions from optimization problems that can be executed on
		a CPU or GPU.

		:class:`SolverOptkit` does not (currently) support planning
		problems with dose constraints.

		Attributes:
			objective_voxels (:class:`optkit.PogsObjective`): POGS
				description of the fully-separable objective function
				:math:`f: \mathbf{R}^\mbox{voxels}\rightarrow\mathbf{R}`
				applied to the vector of voxel doses. Can be modified
				between solver runs at little cost.
			objective_beams (:class:`optkit.PogsObjective`): POGS
				description of the fully-separable objective function
				:math:`g: \mathbf{R}^\mbox{beams}\rightarrow\mathbf{R}`
				applied to the vector of beam intensities. Can be
				modified between solver runs at little cost.
			pogs_solver (:class:`optkit.PogsSolver`): POGS solver with
				fixed representation of the problem matrix. Must be
				rebuilt each time the dose matrix is changed.
		"""

		def __init__(self):
			"""
			Initialize empty :class:`SolverOpkit` as a :class:`Solver`.

			Arguments:
				None
			"""
			Solver.__init__(self)
			self.objective_voxels = None
			self.objective_beams = None
			self.pogs_solver = None
			self.__A_current = None
			self.__A_dict = {}
			self.__n_beams = None
			self.__curr_config = None
			self.__resume = False


		def init_problem(self, n_beams=None, **options):
			"""
			Initialize problem---no-op for :class:`SolverOptkit`.

			Method defined to match public methods of
			:class:`~optkit.optimization.solver_cvxpy.SolverCVXPY`.

			Arguments:
				n_beams (:obj:`int`, optional): Number of beams in plan.
				**options: Arbitrary keyword arguments.
			"""
			if n_beams is not None:
				self.__n_beams = int(n_beams)

		@property
		def n_beams(self):
			""" Number of candidate beams in solver's problem. """
			return self.__n_beams

		@staticmethod
		def can_solve(structures):
			"""
			Test if :class:`Structure` objects compatible with solver.

			Arguments:
				structures: An iterable collection of :class:`Structure`
					objects.

			Returns:
				:obj:`bool`: ``True`` if none of the structures have
				dose constraints.
			"""
			return all([s.constraints.size == 0 for s in structures])

		def clear(self):
			"""
			Destroy backend representation of solver.

			Sets :attr:`SolverOptkit.pogs_solver` to ``None``. Invoke
			:mod:`optkit` call to reset GPU (freeing memory allocated to
			solver), if applicable.

			Arguments:
				None

			Returns:
				None
			"""
			if self.pogs_solver:
				del self.pogs_solver
				ok.backend.reset_device()
				self.pogs_solver = None

		@staticmethod
		def __percentile_constraint_restricted(A, constr, slack=False):
			"""
			Form convex restriction to DVH constraint. Not implemented.

			Arguments:
				A: Structure-specific dose matrix to use in constraint.
				constr: Percentile-type dose constraint.
				slack (bool, optional): Flag to allow slack variable in
					 constraint formulation.

			Returns:
				None

			Raises:
				ValueError: Always.
			"""
			raise ValueError(
					'dose constraints not supported for SolverOptkit')

		@staticmethod
		def __percentile_constraint_exact(A, y, constr, had_slack=False):
			"""
			Form exact version of DVH constraint. Not implemented.

			Arguments:
				A: Structure-specific dose matrix to use in constraint.
				y: Vector of doses, feasible with respect to constraint
					`constr`.
				constr: Percentile-type dose constraint.
				slack (bool, optional): Flag to allow slack variable in
					 constraint formulation.

			Returns:
				None

			Raises:
				ValueError: Always.
			"""
			raise ValueError(
					'dose constraints not supported for SolverOptkit')


		def __add_constraints(self, structure, exact=False):
			"""
			Add constraints from ``structure`` to problem. Not implemented.

			Arguments:
				structure (:class:`~conrad.medicine.Structure`):
					Structure from which to read dose matrix and dose
					constraints.
				exact (:obj:`bool`, optional): If ``True``, treat
					percentile-type dose constraints as exact
					constraints. Otherwise, use convex restrictions.

			Returns:
				None

			Raises:
				ValueError: Always.
			"""
			raise ValueError(
					'dose constraints not supported for SolverOptkit')

		def get_slack_value(self, constr_id):
			"""
			Get slack variable for queried constraint. Not implemented.

			Arguments:
				constr_id (:obj:`str`): ID tag for queried constraint.

			Returns:
				float: NaN, as :attr:`numpy.np.nan`.
			"""
			return np.nan

		def get_dual_value(self, constr_id):
			"""
			Get dual variable for queried constraint. Not implemented.

			Arguments:
				constr_id (:obj:`str`): ID tag for queried constraint.

			Returns:
				float: NaN, as :attr:`numpy.np.nan`.
			"""
			return np.nan

		def get_dvh_slope(self, constr_id):
			"""
			Get slope for queried constraint. Not implemented.

			Arguments:
				constr_id (:obj:`str`): ID tag for queried constraint.

			Returns:
				float: NaN, as :attr:`numpy.np.nan`.
			"""
			return np.nan

		def __assert_solver_exists(self, property_name):
			"""
			Assert :attr:`SolverPogs.pogs_solver` is not ``None``.

			Helper method for getters of various properties that are
			retrievable only when a POGS solver is built.

			Arguments:
				property_name (:obj:`str`): Name of property to
					retrieve, display in exception message if raised.

			Returns:
				None

			Raises:
				ValueError: If :attr:`SolverOptkit.pogs_solver` is
					``None``.
			"""
			if self.pogs_solver is None:
				raise ValueError(
						'no POGS solver built; cannot retrieve '
						'property SolverOptkit.{}.\n Call '
						'SolverOptkit.build() at least once to build a '
						'solver in the backend'
						''.format(property_name))

		@property
		def x(self):
			r"""
			Vector variable of beam intensities, :math:`x`.
			"""
			self.__assert_solver_exists('x')
			return self.pogs_solver.output.x

		@property
		def x_dual(self):
			r"""
			Dual variable corresponding to constraint :math:`x \ge 0`.
			"""
			self.__assert_solver_exists('x_dual')
			return self.pogs_solver.output.mu

		@property
		def y_dual(self):
			r"""
			Dual variable corresponding to constraint :math:`Ax = y`.
			"""
			self.__assert_solver_exists('y_dual')
			return self.pogs_solver.output.nu

		@property
		def solvetime(self):
			""" Solver run time. """
			self.__assert_solver_exists('solvetime')
			return self.pogs_solver.info.c.solve_time

		@property
		def setuptime(self):
			""" Solver run time. """
			self.__assert_solver_exists('setuptime')
			return self.pogs_solver.info.c.setup_time

		@property
		def status(self):
			self.__assert_solver_exists('status')
			return self.pogs_solver.info.err

		@property
		def objective_value(self):
			""" Objective value at end of solve. """
			self.__assert_solver_exists('objective_value')
			return self.pogs_solver.info.objval

		@property
		def solveiters(self):
			""" Number of solver iterations performed. """
			self.__assert_solver_exists('solveiters')
			return int(self.pogs_solver.info.iters)

		def __preprocess_solver_cache(self, solver_cache):
			cache_options = {
					'bypass_initialization': False,
					'solver_cache': {
							'A_equil': None,
							'd': None,
							'e': None,
							'LLT': None,
					}
			}
			if isinstance(solver_cache, dict):
				cache_options['solver_cache']['A_equil'] = solver_cache.pop(
						'A_equil', solver_cache.pop('matrix', None))
				cache_options['solver_cache']['d'] = solver_cache.pop(
						'd', solver_cache.pop('left_preconditioner', None))
				cache_options['solver_cache']['e'] = solver_cache.pop(
						'e', solver_cache.pop('right_preconditioner', None))
				cache_options['solver_cache']['LLT'] = solver_cache.pop(
						'LLT', solver_cache.pop('projector_matrix', None))
			cache_options['bypass_initialization'] = bool(
					cache_options['solver_cache']['A_equil'] is not None and
					cache_options['solver_cache']['d'] is not None and
					cache_options['solver_cache']['e'] is not None)
			return cache_options

		def __check_for_updates(self, structures):
			A_dict_curr = {s.label: None for s in structures}
			for s in structures:
				if s.collapsable:
					A_dict_curr[s.label] = s.A_mean
				else:
					A_dict_curr[s.label] = s.A_full

			updated = True
			if len(self.__A_dict) > 0:
				updated = any([
						self.__A_dict[label] is not A_dict_curr[label]
						for label in A_dict_curr])
			self.__A_dict.update(A_dict_curr)
			return updated

		def __build_matrix(self, structures):
			r"""Gather dose matrix from ``structures``.

			Procedure ::
				# Set A = [] empty matrix with 0 rows and N columns.
				#
				# for each structure in structures do
				#	if structure is collapsable (mean/no dose constraints):
				#		append structure's 1 x N mean dose vector to A.
				#	else:
				#		append structure's M_structure x N dose matrix to A.
				# end for

			Arguments:
				structures: Iterable collection of
					:class:`~conrad.medicine.Structure` objects.

			Returns:
				:class:`np.ndarray`: Dose matrix
			"""
			cols = self._Solver__check_dimensions(structures)
			rows = sum([s.size if not s.collapsable else 1 for s in structures])
			A = np.zeros((rows, cols))
			CONRAD_DEBUG_PRINT('BUILT MATRIX SIZE: {}'.format(A.size))

			ptr = 0
			for s in structures:
				if s.collapsable:
					A[ptr, :] = s.A_mean[:]
					ptr += 1
				else:
					A[ptr : ptr + s.size, :] += s.A_full
					ptr += s.size

			return A

		def __build_voxel_objective(self, structures):
			rows = sum([s.size if not s.collapsable else 1 for s in structures])
			self.objective_voxels = ok.api.PogsObjective(rows)
			self.__update_voxel_objective(structures)

		def __update_voxel_objective(self, structures):
			self._Solver__set_scaling(structures)
			ptr = 0
			for s in structures:
				obj_sub = ObjectiveMethods.primal_expr_pogs(s)
				self.objective_voxels.copy_from(obj_sub, ptr)
				ptr += 1 if s.collapsable else s.size

		def __build_beam_objective(self, structures):
			cols = self._Solver__check_dimensions(structures)
			self.objective_beams = ok.api.PogsObjective(cols, h='IndGe0')
			self.__update_beam_objective(structures)

		def __update_beam_objective(self, structures):
			pass

		def __update_backend(self, gpu=False, double=False):
			changed = bool(gpu != ok.api.backend.device_is_gpu)
			changed |= bool(double == ok.api.backend.precision_is_32bit)
			if not changed:
				return
			else:
				self.clear()
				ok.set_backend(gpu=gpu, double=double)

		def build(self, structures, solver_cache=None, **options):
			"""
			Build POGS optimization problem from structure data.

			Extract dose matrix, target doses, and objective weights
			from structures.

			Use doses and weights to update POGS objectives
			(:attr:`SolverOptkit.objective_voxels` and
			:attr:`SolverOptkit.objective_beams`). POGS solver's dose
			matrix only updated if matrix gathered from structures has
			changed.

			Arguments:
				structures: Iterable collection of :class:`Structure`
					objects.
				solver_cache (:obj:`dict`, optional): If provided,
					solver will try to skip equilibration and
					factorization based on provided data.
				**options: Keyword arguments.

			Returns:
				:obj:`str`: String documenting how data in
				``structures`` were parsed to form an optimization
				problem.

			Raises:
				ValueError: If :meth:`SolverOptkit.can_solve` returns
					``False`` for ``structures``.
			"""
			if not self.can_solve(structures):
				raise ValueError(
						'SolverOptkit does not support dose constraints')

			if isinstance(structures, Anatomy):
				structures = structures.list

			self.__update_backend(
					options.pop(
							'gpu', ok.api.backend.device_is_gpu),
					options.pop(
							'double', not ok.api.backend.precision_is_32bit))

			matrix_updated = self.__check_for_updates(structures)
			if self.__A_current is None or matrix_updated:
				A = self.__A_current = self.__build_matrix(structures)
			else:
				A = self.__A_current

			n_voxels, n_beams = A.shape

			rebuild_f = bool(self.objective_voxels is None or
						   self.objective_voxels.size != n_voxels)
			rebuild_g = bool(self.objective_beams is None or
						   self.objective_beams.size != n_beams)

			if rebuild_f:
				self.__build_voxel_objective(structures)
			else:
				self.__update_voxel_objective(structures)

			if rebuild_g:
				self.__build_beam_objective(structures)
			else:
				self.__update_beam_objective(structures)

			if self.pogs_solver is None or matrix_updated:
				cache_options = self.__preprocess_solver_cache(solver_cache)
				self.pogs_solver = ok.api.PogsSolver(A, **cache_options)
				self.__resume = False
			else:
				self.__resume = True

			return self._Solver__construction_report(structures)

		def solve(self, **options):
			"""
			Execute optimization of a previously built planning problem.

			:meth:`SolverOptkit.build` must be called after
			initialization (and after any invocation of
			:meth:`SolverOptkit.clear`) for there to be a POGS solver
			instance to perform the requested optimization.

			Arguments:
				**options: Keyword arguments specifying solver options,
					passed to :meth:`optkit.PogsSolver.solve`.

			Returns:
				:obj:`bool`: ``True`` if POGS solver converged.

			Raises:
				ValueError: If :attr:`SolverOptkit.pogs_solver` has
					not been built.
			"""
			if self.pogs_solver is None:
				raise ValueError(
						'no POGS solver built; cannot perform '
						'treatment plan optimization.\n Call '
						'SolverOptkit.build() at least once to build a '
						'solver in the backend')

			options['abstol'] = options.pop('abstol', ABSTOL_DEFAULT)
			options['reltol'] = options.pop('reltol', RELTOL_DEFAULT)
			options['verbose'] = options.pop('verbose', VERBOSE_DEFAULT)
			options['suppress'] = options.pop('suppress', False)
			options['maxiters'] = options.pop(
					'maxiters', options.pop('maxiter', MAXITER_DEFAULT))
			options['resume'] = self.__resume


			scale_doses = options.pop('scale_doses', True)
			scale_doses &= self.global_dose_scaling != 1.
			if scale_doses:
				self.objective_voxels._Objective__b /= self.global_dose_scaling
				if 'x0' in options:
					options['x0'] /= self.global_dose_scaling

			self.pogs_solver.solve(
					self.objective_voxels, self.objective_beams, **options)

			if scale_doses:
				self.pogs_solver.output.x *= self.global_dose_scaling
				self.pogs_solver.output.y *= self.global_dose_scaling
			return self.pogs_solver.info.converged

		@property
		def cache(self):
			if self.pogs_solver is None:
				return None

			cache = self.pogs_solver.cache
			if cache.pop('direct'):
				projector_type = PROJECTOR_POGS_DENSE_DIRECT
			else:
				projector_type = 'indirect'

			return {
					'matrix': cache.pop('A_equil'),
					'left_preconditioner': cache.pop('d'),
					'right_preconditioner': cache.pop('e'),
					'projector': {
							'type': projector_type,
							'matrix': cache.pop('LLT'),
					},
					'state_variables': cache,
			}
else:
	SolverOptkit = lambda: None
