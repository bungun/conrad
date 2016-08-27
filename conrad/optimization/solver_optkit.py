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

from numpy import nan

from conrad.defs import module_installed
from conrad.optimization.solver_base import *

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
			self.__n_beams = None

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
			raise ValueError('dose constraints not supported for SolverOptkit')

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
			raise ValueError('dose constraints not supported for SolverOptkit')


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
			raise ValueError('dose constraints not supported for SolverOptkit')


		def get_slack_value(self, constr_id):
			"""
			Get slack variable for queried constraint. Not implemented.

			Arguments:
				constr_id (:obj:`str`): ID tag for queried constraint.

			Returns:
				float: NaN, as :attr:`numpy.nan`.
			"""
			return nan

		def get_dual_value(self, constr_id):
			"""
			Get dual variable for queried constraint. Not implemented.

			Arguments:
				constr_id (:obj:`str`): ID tag for queried constraint.

			Returns:
				float: NaN, as :attr:`numpy.nan`.
			"""
			return nan

		def get_dvh_slope(self, constr_id):
			"""
			Get slope for queried constraint. Not implemented.

			Arguments:
				constr_id (:obj:`str`): ID tag for queried constraint.

			Returns:
				float: NaN, as :attr:`numpy.nan`.
			"""
			return nan

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
				AttributeError: If :attr:`SolverOptkit.pogs_solver` is
					``None``.
			"""
			if self.pogs_solver is None:
				raise AttributeError('no POGS solver built; cannot '
									 'retrieve property SolverOptkit.{}'
									 '.\n Call SolverOptkit.build() at '
									 'least once to build a solver in '
									 'the backend'.format(property_name))

		@property
		def x(self):
			r"""
			Vector variable of beam intensities, :math:`x`.
			"""
			if self.pogs_solver is None:
				raise ValueError()
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

		def build(self, structures, **options):
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

			Returns:
				:obj:`str`: String documenting how data in
				``structures`` were parsed to form an optimization
				problem.

			Raises:
				ValueError: If :meth:`SolverOptkit.can_solve` returns
					``False`` for ``structures``.
			"""
			if not self.can_solve:
				raise ValueError(
						'SolverOptkit does not support dose constraints')

			A, dose, weight_abs, weight_lin = \
					self._Solver__gather_matrix_and_coefficients(structures)

			if self.__A_current is not None:
				matrix_updated = (self.__A_current != A).sum() > 0
			else:
				matrix_updated = True

			self.__A_current = A

			n_voxels, n_beams = A.shape

			rebuild_f = bool(self.objective_voxels is None or
						   self.objective_voxels.size != n_voxels)
			rebuild_g = bool(self.objective_beams is None or
						   self.objective_beams.size != n_beams)

			# f(y) = weight_abs|y - dose\ + weight_lin(y - dose)
			if rebuild_f:
				self.objective_voxels = ok.PogsObjective(
						n_voxels, h='Abs', b=dose, c=weight_abs, d=weight_lin)
			else:
				self.objective_voxels.set(b=dose, c=weight_abs, d=weight_lin)

			# g(x) = Ind { x >= 0 }
			if rebuild_g:
				self.objective_beams = ok.PogsObjective(n_beams, h='IndGe0')

			if self.pogs_solver is None or matrix_updated:
				self.clear()
				self.pogs_solver = ok.PogsSolver(A)

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
				AttributeError: If :attr:`SolverOptkit.pogs_solver` has
					not been built.
			"""
			if self.pogs_solver is None:
				raise AttributeError('no POGS solver built; cannot '
									 'perform treatment plan '
									 'optimization.\n Call '
									 'SolverOptkit.build() at least '
									 'once to build a solver in the '
									 'backend')

			self.pogs_solver.solve(self.objective_voxels, self.objective_beams,
								   **options)
			return self.pogs_solver.info.converged

else:
	SolverOptkit = lambda: None
