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
import numpy as np

from conrad.defs import module_installed
from conrad.optimization.clustering_base import ClusteredProblem

OPTKIT_INSTALLED = module_installed('optkit')
if OPTKIT_INSTALLED:
	import optkit as ok
else:
	ok = NotImplemented

@add_metaclass(abc.ABCMeta)
class BeamClusteredProblem(ClusteredProblem):
	def __init__(self, case, reference_frame_name, clustered_frame_name):
		ClusteredProblem.__init__(
				self, case, reference_frame_name, clustered_frame_name)
		self.__cluster_mapping = case.retrieve_frame_mapping(
				self.reference_frame, self.clustered_frame)

	@property
	def cluster_mapping(self):
		return self.cluster_mapping

	def split_voxel_prices(self, voxel_prices, structure_order,
						   rows_per_structure):
		voxel_price_dict = {}
		offset = 0
		for label in structure_order:
			size = rows_per_structure[label]
			voxel_price_dict[label] = voxel_prices[offset : offset + size]
			offset += size

	def join_voxel_prices(self, voxel_price_dict, structure_order):
		return np.vstack([voxel_price_dict[label] for label in structure_order])

	def dose_objective_dual_eval(self, anatomy, voxel_price_dict):
		return sum(listmap(
				lambda s: s.dual_objective_eval(voxel_price_dict[s.label]),
				anatomy))

	def update_voxel_prices(self, voxel_price_dict, voxel_price_update):
		for label in voxel_price_dict:
			voxel_price_dict[label] += voxel_price_update[label]

class UnconstrainedBeamClusteredProblem(BeamClusteredProblem):
	def __init__(self, case, reference_frame_name, clustered_frame_name):
		BeamClusteredProblem.__init__(
				self, case, reference_frame_name, clustered_frame_name)

	def clear(self):
		self.__case.problem.solver.clear()

	def build_A_infeas(self, reference_anatomy, beam_prices, k, tol):
		mu = beam_prices

		n_infeas = min(sum(mu < -tol), k)
		rank = (mu.argsort())[:n_infeas]

		sizes = [1 if s.collapsable else s.size for s in reference_anatomy]
		offsets = np.roll(np.cumsum(sizes), 1)
		offsets[0] = 0
		A_infeas = np.zeros((sum(sizes), n_infeas))

		for idx, s in enumerate(reference_anatomy.list):
			offset = offsets[i]
			size = sizes[i]
			if s.collapsable:
				for j_new, j in enumerate(rank):
					A_infeas[offset, j_new] += s.A_mean[j]
			else:
				for j_new, j in enumerate(rank):
					A_infeas[offset : offset + size, j_new] += s.A[:, j]

		return A_infeas

	def build_A_infeas_augmented(self, A_infeas, voxel_prices):
		m, n = A_infeas.shape
		A_aug = np.zeros((n, m + 1))
		A_aug[:, :m] = A_infeas.T
		A_aug[:, -1] = A_infeas.T.dot(voxel_prices)
		return A_aug


	def solve_dual_infeas_pogs(self, A_infeas, voxel_prices, reference_anatomy,
				**solver_options):
		"""
		Problem:

		max f_conj(nu + delta)
		st  delta >= 0; nu + delta \in dom(f_conj);
			A^T(nu + delta) >= 0

		with:
			parameter nu:= current (infeasible) voxel prices,
			optimization variable delta:= nonnegative change to prices.

		Rephrase as POGS-compatible graph form problem:

			max f_conj(z') + g_conj(A^Tz'),

		where,
			z' = nu + delta

		and f_conj is the dual objective and also contains the constraints
			delta >= 0
			z' \in dom(f_conj)

		while g_conj enforces the constraints
			A^T(nu + delta) >= 0

		We have:

			A_infeas^T \in R^{n x m}
			A_infeas^T\nu \in R^{n x 1},

		and define

			A_aug = [A_infeas^T, A_infeas^T \nu];

		then,

			A_aug * [delta; 1] = A_infeas^T delta + A_infeas^T \nu

		so

			A_aug * [delta; 1] >= 0 <--> A_infeas^T delta + A_infeas^T \nu >= 0

		The problem is then

			max f_conj(z) + g_conj(A_aug * z)

		with

			z = [delta; 1]
			f_conj defined as above for the first m entries to express the
				dual objective and enforce dual domain constraints
			f_conj enforcing z_{m + 1} == 1 for the final entry
			g_conj enforcing nonnegativity as above

		Since this is an inner routine, the optkit/POGS backends are not
		changed.
		"""
		if not OPTKIT_INSTALLED:
			raise NotImplementedError(
					'module `optkit` not installed, cannot call '
					'`solve_dual_infeas_pogs()`')

		n_voxels, n_beams = A_infeas.shape


		# build f_conj
		objective_voxels_conjugate = ok.PogsObjective(
				n_voxels + 1, h='IndBox01')
		ptr = 0

		# structure dual objectives and constraints
		for s in reference_anatomy:
			obj_sub = self.methods.dual_fused_expr_constraints_pogs(
					s, nu_offset=voxel_prices, nonnegative=True)
			objective_voxels_conjugate.copy_from(obj_sub, ptr)
			ptr += 1 if s.collapsable else s.size

		# extra term to enforce final term z_{m+1} == 1
		objective_voxels_conjugate.set(start=-1, h='IndEq0', b=1, d=0)

		# build g_conj
		objective_beams_conjugate = ok.PogsObjective(n_beams, h='IndGe0')

		A_aug = self.build_A_infeas_augmented(A_infeas, voxel_prices)

		dual = ok.PogsSolver(A_aug)
		dual.solve(
				objective_beams_conjugate, objective_voxels_conjugate,
				**solver_options)

		solve_time = dual.info.c.solve_time + dual.info.c.setup_time
		delta_star = dual.output.x[:-1]

		del dual
		return delta_star, solve_time

	def dual_iterative(self, voxel_prices_dict, k, reference_anatomy, tol,
					   tol_infeas=1e-2, **solver_options):
		n_beams = np.size(next(iter(reference_anatomy)).A_mean)
		sizes = [np.size(nu) for nu in voxel_prices_dict.values()]

		TMAX = int(np.ceil(float(n)/float(k)))
		TOL_N = tol_infeas * n_beams

		nu_t = voxel_prices_dict
		mu_t = self.methods.beam_prices(reference_anatomy, nu_t)
		t = 0
		solve_time_total = 0.
		offset = 0.

		VERBOSE = True
		if 'verbose' in solver_options:
			VERBOSE = solver_options['verbose'] > 0

		while sum(mu_t < - tol) > TOL_N and t < TMAX:
			A_infeas = build_A_infeas(reference_anatomy, mu, k, tol)
			delta_star, solve_time = solve_dual_infeas_pogs(
					A_infeas, nu_t, reference_anatomy, **solver_options)
			delta_t = split_voxel_prices(
					delta_star, reference_anatomy.label_order, sizes)
			solve_time_total += solve_time

			if VERBOSE:
				print('# infeas before: {}'.format(sum(mu_t < - tol)))

			self.update_voxel_prices(nu_t, delta_t)
			mu_t = self.methods.beam_prices(reference_anatomy, nu_t)

			if VERBOSE:
				print('# infeas after: {}'.format(sum(mu_t < - tol)))

			offset += self.dose_objective_dual_eval(
					reference_anatomy, delta_star)

			if VERBOSE:
				print('CUMULATIVE OFFSET: {}'.format(offset))

			t += 1

		return nu_t, solve_time_total, t

	def solve_and_bound_clustered_problem(self, case, reference_anatomy,
										  cluster_mapping, **solver_options):
		_, run = case.plan(**solver_options)

		x_star_bclu = run.output.x
		x_star_bclu_upsampled = cluster_mapping.upsample(
				x_star_bclu, rescale_output=True)
		nu_star_bclu = run.output.nu
		nu_star_bclu_dict = self.split_voxel_prices(
				nu_star_bclu, run.profile.label_order,
				run.profile.representation_sizes)

		k = np.size(x_star_bclu)
		mu = self.methods.beam_prices(reference_anatomy, nu_star_bclu_dict)
		n = np.size(mu)
		tol = np.abs(mu.min()) * np.log10(n / k)

		reference_anatomy.calculate_doses(x_star_bclu_upsampled)
		obj_ub = self.dose_objective_primal_eval(reference_anatomy)


		nu_feas, solve_time_total, n_subproblems = self.dual_iterative(
				nu_star_bclu_dict, k, reference_anatomy, tol, **solver_options)
		obj_lb = self.dose_objective_dual_eval(reference_anatomy, nu_feas)

		mu = self.methods.beam_prices(reference_anatomy, nu_feas)
		print('TOL {}'.format(tol))
		print('MAX VIOLATION: {}'.format(np.abs(mu.min())))

		run.output.optimal_variables['x_full'] = x_star_bclu_upsampled
		run.output.optimal_variables['nu_feasible'] = self.join_voxel_prices(
				nu_feas)
		run.output.solver_info['primal_solve_time'] = float(
				case.problem.solver.solvetime + case.problem.solver.setuptime)
		run.output.solver_info['dual_time'] = solve_time_total

		return obj_ub, obj_lb, run

	def cluster_and_bound(self, **solver_options):
		self.reload_clustered_frame()
		return self.solve_and_bound_clustered_problem(
				self.case, self.reference_anatomy, self.cluster_mapping,
				**solver_options)

# class ConstrainedBeamClusteredProblem(VoxelClusteredProblem):
	# pass







