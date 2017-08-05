"""
TODO: DOCSTRING
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
import numpy as np

from conrad.optimization.solvers.environment import OPTKIT_INSTALLED
from conrad.optimization.clustering.clustering_base import *

if OPTKIT_INSTALLED:
	import optkit as ok
else:
	ok = NotImplemented

@add_metaclass(abc.ABCMeta)
class BeamClusteredProblem(ClusteredProblem):
	def __init__(self, case, reference_frame_name, clustered_frame_name):
		ClusteredProblem.__init__(
				self, case, reference_frame_name, clustered_frame_name)
		self.__cluster_mapping = case.physics.retrieve_frame_mapping(
				self.reference_frame, self.clustered_frame).beam_map
	@property
	def cluster_mapping(self):
		return self.__cluster_mapping

class UnconstrainedBeamClusteredProblem(BeamClusteredProblem):
	def __init__(self, case, reference_frame_name, clustered_frame_name):
		BeamClusteredProblem.__init__(
				self, case, reference_frame_name, clustered_frame_name)
		self.__infeasible_frame = None

	@property
	def infeasible_frame(self):
		return self.__infeasible_frame

	@property
	def infeasible_anatomy(self):
		if self.infeasible_frame is None:
			raise ValueError('no infeasible frame built')
		return self.case.anatomy_for_frame(self.infeasible_frame)

	def build_infeasible_frame(self, beam_prices, k=None):
		k_ = self.cluster_mapping.n_clusters if k is None else int(k)
		k = min(k_, self.cluster_mapping.n_clusters)

		# UPDATE: IGNORE TOLERANCE, JUST STRICT FEASIBILITY
		voxels = self.reference_anatomy.total_working_size
		infeas_beams = min(sum(beam_prices < 0), k)
		rank = (beam_prices.argsort())[:infeas_beams]

		A_dict_infeas = {
				s.label: s.A_mean[rank][:, None].T if s.collapsable
				else s.A[:, rank]
				for s in self.reference_anatomy}
		self.__infeasible_frame = 'infeas'
		self.case.physics.add_dose_frame(
				'infeas', data=A_dict_infeas,
				voxel_weights=self.case.physics.frame.voxel_weights.manifest)

	def teardown_infeasible_frame(self):
		self.case.change_dose_frame(self.clustered_frame)
		self.case.delete_dose_frame(self.infeasible_frame)
		self.__infeasible_frame = None

	def build_A_infeas(self, beam_prices, tol, k=None):
		self.build_infeasible_frame(beam_prices, k=k)
		self.case.physics.change_dose_frame(self.infeasible_frame)
		self.case.plannable
		voxels, infeas_beams = self.case.n_voxels, self.case.n_beams

		ptr = 0
		A_infeas = np.zeros((voxels, infeas_beams))
		for s in self.infeasible_anatomy.list:
			A_infeas[ptr:ptr + s.working_size, :] = s.A_working
			ptr += s.working_size
		return A_infeas

	def build_A_infeas_augmented(self, A_infeas, voxel_prices):
		m, n = A_infeas.shape
		A_aug = np.zeros((n, m + 1))
		A_aug[:, :m] = A_infeas.T
		A_aug[:, -1] = A_infeas.T.dot(voxel_prices)
		return A_aug

	def solve_dual_infeas_pogs(self, A_infeas, voxel_prices, **solver_options):
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
		objective_voxels_conjugate = ok.api.PogsObjective(
				n_voxels + 1, h='IndBox01')
		ptr = 0
		# structure dual objectives and constraints
		for s in self.infeasible_anatomy:
			obj_sub = self.methods.dual_fused_expr_constraints_pogs(
					s, nu_offset=voxel_prices[s.label], nonnegative=True)
			objective_voxels_conjugate.copy_from(obj_sub, ptr)
			ptr += s.working_size

		# extra term to enforce final term z_{m+1} == 1
		objective_voxels_conjugate.set(start=-1, h='IndEq0', b=1, d=0)

		# build g_conj
		objective_beams_conjugate = ok.api.PogsObjective(n_beams, h='IndGe0')


		A_aug = self.build_A_infeas_augmented(
				A_infeas, self.join_voxel_prices(voxel_prices))

		dual = ok.api.PogsSolver(A_aug)
		dual.solve(
				objective_beams_conjugate, objective_voxels_conjugate,
				**solver_options)

		solve_time = dual.info.c.solve_time + dual.info.c.setup_time
		delta_star = dual.output.x[:-1]

		del dual
		return delta_star, solve_time

	def in_dual_domain(self, voxel_prices_dict):
		for label in voxel_prices_dict:
			nu = voxel_prices_dict[label]
			s = self.reference_anatomy[label]
			if not self.methods.in_dual_domain(s, nu):
				return False
		return True

	def in_dual_cone(self, beam_prices, tol=1e-2):
		return np.min(beam_prices) >= -tol

	def dual_iterative(self, voxel_prices_dict, tol, tol_infeas=1e-2,
					   **solver_options):
		# TODO: ADD OPTION TO SWTICH BETWEEN CVXPY AND OPTKIT-BASED METHODS
		# FOR NOW, ONLY USES OPTKIT MODULE/POGS SOLVER
		reference_anatomy = self.reference_anatomy
		n_beams = self.cluster_mapping.n_points

		label_order = reference_anatomy.label_order
		sizes = {s.label: s.working_size for s in reference_anatomy}
		k = self.cluster_mapping.n_clusters

		TMAX = int(np.ceil(float(n_beams)/float(k)))
		TOL_N = tol_infeas * n_beams

		nu_t = voxel_prices_dict
		mu_t = self.methods.beam_prices(reference_anatomy, nu_t)
		t = 0
		solve_time_total = 0.
		offset = 0.
		VERBOSE = True
		if 'verbose' in solver_options:
			VERBOSE = solver_options['verbose'] > 0
		TEST_FEAS = solver_options.pop('test_feasibility', False)

		# while sum(mu_t < - tol) > TOL_N and t < TMAX:
		while not self.in_dual_cone(mu_t, tol) and t < TMAX:
			A_infeas = self.build_A_infeas(mu_t, tol)
			delta_star, solve_time = self.solve_dual_infeas_pogs(
					A_infeas, nu_t, **solver_options)
			delta_t = self.split_voxel_prices(delta_star, label_order, sizes)
			solve_time_total += solve_time

			if VERBOSE:
				print('# infeas before: {}'.format(sum(mu_t < - tol)))

			self.update_voxel_prices(nu_t, delta_t)
			if TEST_FEAS:
				if not self.in_dual_domain(nu_t):
					raise ValueError('dual variable out of domain')

			mu_t = self.methods.beam_prices(reference_anatomy, nu_t)

			if VERBOSE:
				print('# infeas after: {}'.format(sum(mu_t < - tol)))
				print('mu min: {}'.format(np.min(mu_t)))
			offset += self.dose_objective_dual_eval(reference_anatomy, delta_t)

			if VERBOSE:
				print('CUMULATIVE OFFSET: {}'.format(offset))
			self.teardown_infeasible_frame()
			t += 1

		return nu_t, solve_time_total, t

	def solve_clustered(self, **solver_options):
		self.methods.apply_joint_scaling(self.reference_anatomy)
		self.methods.apply_joint_scaling(self.clustered_anatomy)

		_, run = self.case.plan(frame=self.clustered_frame, **solver_options)

		x_star_bclu = run.output.x
		x_star_bclu_upsampled = self.cluster_mapping.upsample(
				x_star_bclu, rescale_output=True)
		nu_star_bclu = run.output.optimal_variables['nu']
		nu_star_bclu_dict = self.split_voxel_prices(
				nu_star_bclu, run.profile.label_order,
				run.profile.representation_sizes)
		return run, x_star_bclu, x_star_bclu_upsampled, nu_star_bclu_dict

	def solve_and_bound_clustered_problem(self, **solver_options):
		k = self.cluster_mapping.n_clusters
		n = self.cluster_mapping.n_points

		run, x_bclu, x_bclu_upsampled, nu_bclu_dict = self.solve_clustered(
				**solver_options)
		mu_clu = self.methods.beam_prices(self.clustered_anatomy, nu_bclu_dict)
		tol = np.abs(mu_clu.min()) * np.sqrt(float(n) / k)

		print('LB INFEAS {}'.format(self.dose_objective_dual_eval(
				self.reference_anatomy, nu_bclu_dict)))

		self.reference_anatomy.calculate_doses(x_bclu_upsampled)
		obj_ub = self.dose_objective_primal_eval(self.reference_anatomy)

		nu_feas, solve_time_total, n_subproblems = self.dual_iterative(
				nu_bclu_dict, tol, **solver_options)
		obj_lb = self.dose_objective_dual_eval(self.reference_anatomy, nu_feas)

		mu = self.methods.beam_prices(self.reference_anatomy, nu_feas)
		print('TOL {}'.format(tol))
		print('MU MIN: {}'.format(mu.min()))

		run.output.optimal_variables['x_full'] = x_bclu_upsampled
		run.output.optimal_variables['nu_feasible'] = self.join_voxel_prices(
				nu_feas)
		run.output.solver_info['primal_solve_time'] = float(
				self.clustered_problem.solver.solvetime +
				self.clustered_problem.solver.setuptime)
		run.output.solver_info['dual_time'] = solve_time_total

		return obj_ub, obj_lb, run

# TODO: objective methods - build primal objective; build dual objective;
# build dose matrix; build dual feasibility matrix, build dual feasibility objective;
# pogs versions thereof



# class ConstrainedBeamClusteredProblem(VoxelClusteredProblem):
	# pass


# class ColumnClusteringMethods(object):
# 	@staticmethod
# 	def cluster(beam_set, desired_compression):
# 		pass

# 	@staticmethod
# 	def compress_beams(case, desired_compression):
# 		pass
# 		# target compression in (int, float, dict)

# 	@staticmethod
# 	def generate_feasible_dual(case, full_frame, compressed_frame, nu_0):
# 		pass

# 	@staticmethod
# 	def solve_compressed(case, full_frame, compressed_frame):
# 		pass
# 		# return {x: --, y: --, upper_bound: --, lower_bound: --}

