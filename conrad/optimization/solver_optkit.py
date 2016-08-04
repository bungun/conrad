from numpy import nan

from conrad.compat import *
from conrad.defs import module_installed
from conrad.optimization.solver_base import *

if module_installed('optkit'):
	import optkit as ok

	class SolverOptkit(Solver):
		""" TODO: docstring """

		def __init__(self):
			""" TODO: docstring """
			Solver.__init__(self)
			self.objective_voxels = None
			self.objective_beams = None
			self.pogs_solver = None
			self.__A_current = None
			self.__n_beams = None

		# methods:
		def init_problem(self, n_beams, **options):
			""" TODO: docstring """
			self.__n_beams = int(n_beams)

		@property
		def n_beams(self):
			return self.__n_beams

		@staticmethod
		def can_solve(structures):
			return all([s.constraints.size == 0 for s in structures])

		def clear(self):
			""" TODO: docstring """
			if self.pogs_solver:
				del self.pogs_solver
				ok.backend.reset_device()
				self.pogs_solver = None

		def build(self, structures, **options):
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
				self.pogs_solver = ok.PogsSolver(A)

			return self._Solver__construction_report(structures)

		@staticmethod
		def __percentile_constraint_restricted(A, constr, slack=False):
			""" Form the upper (or lower) DVH constraint:

				upper constraint:

					\sum (beta + (Ax - (b + slack)))_+ <= beta * vox_limit

				lower constraint:

					\sum (beta - (Ax - (b - slack)))_+ <= beta * vox_limit

			"""
			raise ValueError('dose constraints not supported for SolverOptkit')

		@staticmethod
		def __percentile_constraint_exact(A, y, constr, had_slack=False):
			# """ TODO: docstring """
			raise ValueError('dose constraints not supported for SolverOptkit')


		def __add_constraints(self, structure, exact=False):
			# """ TODO: docstring """
			raise ValueError('dose constraints not supported for SolverOptkit')


		def get_slack_value(self, constr_id):
			return nan

		def get_dual_value(self, constr_id):
			return nan

		def get_dvh_slope(self, constr_id):
			return nan

		def __assert_solver_exists(self, property_name):
			if self.pogs_solver is None:
				raise ValueError('no POGS solver built; cannot retrieve '
								 'property {}.\noCall {}.build() at least '
								 'once to build a solver in the backend'
								 ''.format(property_name, SolverOptkit))

		@property
		def x(self):
			if self.pogs_solver is None:
				raise ValueError()
			self.__assert_solver_exists('x')
			return self.pogs_solver.output.x

		@property
		def x_dual(self):
			self.__assert_solver_exists('x_dual')
			return self.pogs_solver.output.mu

		@property
		def y_dual(self):
			self.__assert_solver_exists('y_dual')
			return self.pogs_solver.output.nu

		@property
		def solvetime(self):
			self.__assert_solver_exists('solvetime')
			return self.pogs_solver.info.c.solve_time

		@property
		def status(self):
			self.__assert_solver_exists('status')
			return self.pogs_solver.info.err

		@property
		def objective_value(self):
			self.__assert_solver_exists('objective_value')
			return self.pogs_solver.info.objval

		@property
		def solveiters(self):
			self.__assert_solver_exists('solveiters')
			return int(self.pogs_solver.info.iters)

		def solve(self, **options):
			self.pogs_solver.solve(self.objective_voxels, self.objective_beams,
								   **options)
			return self.pogs_solver.info.converged

else:
	SolverOptkit = lambda: None
