"""
Unit tests for :mod:`conrad.optimization.problem`.
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

from conrad.physics.units import Gy
from conrad.medicine import Structure, Anatomy
from conrad.medicine.dose import D
from conrad.abstract.mapping import *
from conrad.physics.physics import DoseFrameMapping
from conrad.case import Case
from conrad.optimization.environment import OPTKIT_INSTALLED
from conrad.optimization.clustering.clustering_base import *
from conrad.optimization.clustering.voxel_clustering import *
from conrad.optimization.clustering.beam_clustering import *
from conrad.tests.base import *

class ClusteredProblemGenericTestCase(ConradTestCase):
	def build_simple_case(self):
		case = Case()
		case += Structure(0, 'Target', True)
		case += Structure(1, 'Avoid1', False)
		case += Structure(2, 'Avoid2', False)

		A_full = {i: np.eye(10) for i in xrange(3)}
		vmap = DictionaryClusterMapping({
				0: ClusterMapping([0, 0, 0, 1, 1, 1, 1, 2, 2, 2]),
				1: ClusterMapping([0, 0, 1, 1, 2, 2, 2, 2, 2, 3]),
				2: IdentityMapping(10),
		})
		dfmap_vclu = DoseFrameMapping('full', 'vclu', voxel_map=vmap)
		A_vclu = {
				0: np.array([3, 4, 3]) * np.eye(3),
				1: np.array([2, 2, 5, 1]) * np.eye(3),
				2: np.eye(10),
		}
		bmap = ClusterMapping([0, 0, 1, 1, 1, 1, 2, 3, 3, 3])
		dfmap_bclu =
		A_bclu = {
				0: np.array([2, 4, 1, 3]) * np.eye(4),
				1: np.array([2, 4, 1, 3]) * np.eye(4),
				2: np.array([2, 4, 1, 3]) * np.eye(4),
		}

		case.physics.frame.name = 'full'
		case.physics.frame.dose_matrix = A_full

		case.physics.add_dose_frame(
				'vclu', data=A_vclu, voxel_weights=vmap.cluster_weights)
		case.physics.add_dose_frame(
				'bclu', data=A_bclu, voxel_weights=bmap.cluster_weights)

		case.physics.add_frame_mapping(
				DoseFrameMapping('full', 'vclu', voxel_map=vmap))
		case.physics.add_frame_mapping(
				DoseFrameMapping('full', 'bclu', beam_map=bmap))
		return case

	def build_fancy_case(self):
		case = Case()
		case += Structure(0, 'Target', True)
		case += Structure(1, 'Avoid1', False)
		case += Structure(2, 'Avoid2', False)

		m_full = m_bclu = (400, 400, 50)
		n_full = n_vclu = 200
		m_vclu = (50, 50, 50)
		n_bclu = 50

		vec = n_bclu * np.random.rand(n_full)
		vec[-1] = n_bclu -1

		map_fb = ClusterMapping(np.random.rand())

		vec0 = m_bclu[0] * np.random.rand(m_full[0])
		vec0[-1] = m_bclu[0] - 1
		vec1 = m_bclu[1] * np.random.rand(m_full[1])
		vec1[-1] = m_bclu[1] - 1
		map_fv = DictionaryClusterMapping({
				0: ClusterMapping(vec0),
				1: ClusterMapping(vec1),
				2: IdentityMapping(m_vclu[2]),
			})

		scaling = (10., 1., 1.)
		A_bvclu = {
				[i]: scaling[i] * np.random.rand(m_vclu[i], n_bclu)
				for i in xrange(3)
		}

		A_vclu = {
			i: map_fb.upsample(A_bvclu[i].T).T
			for i in A_bvclu
		}
		A_vclu_noisy = {
			key: A_vclu[key] + (0.1 * np.random.rand(A_vclu[key].shape) - 0.05)
			for k in A_bclu
		}

		A_bclu = map_fv.upsample(A_bvclu)
		A_bclu_noisy = {
			key: A_bclu[key] + (0.1 * np.random.rand(A_bclu[key].shape) - 0.05)
			for k in A_bclu
		}

		A_full = map_fv.upsample(A_vclu)

		case.physics.frame.name = 'full'
		case.physics.frame.dose_matrix = A_full

		case.physics.add_dose_frame(
				'vclu', data=A_vclu, voxel_weights=map_fv.cluster_weights)
		case.physics.add_dose_frame(
				'bclu', data=A_bclu, voxel_weights=map_fb.cluster_weights)
		case.physics.add_dose_frame(
				'vclu_noisy', data=A_vclu_noisy, voxel_weights=map_fv.cluster_weights)
		case.physics.add_dose_frame(
				'bclu_noisy', data=A_bclu_noisy, voxel_weights=map_fb.cluster_weights)

		case.physics.add_frame_mapping(DoseFrameMapping(
				'full', 'vclu', voxel_map=map_fv))
		case.physics.add_frame_mapping(DoseFrameMapping(
				'full', 'bclu', beam_map=map_fb))
		case.physics.add_frame_mapping(DoseFrameMapping(
				'full', 'vclu_noisy', voxel_map=map_fv))
		case.physics.add_frame_mapping(DoseFrameMapping(
				'full', 'bclu_noisy', beam_map=map_fb))
		return case

class ClusteredProblemMock(ClusteredProblem):
	def clear(self):
		raise NotImplementedError
	def solve_and_bound_clustered_problem(self, **solver_options):
		raise NotImplementedError

class BeamClusteredProblemMock(BeamClusteredProblem):
	def clear(self):
		raise NotImplementedError
	def solve_and_bound_clustered_problem(self, **solver_options):
		raise NotImplementedError

class ClusteredProblemTestCase(ClusteredProblemGenericTestCase):
	def test_clustered_problem(self):
		case = self.build_simple_case()
		cp = ClusteredProblemMock(case, 'full', 'vclu')
		self.assertIs( cp.case, case )
		for s in (0, 1, 2, 'Target', 'Avoid1', 'Avoid2'):
			s in cp.reference_anatomy

		cp.reload_reference_frame()
		self.assertEqual( cp.case.frame.name, 'full' )
		cp.case.calculate_doses(np.random.rand(10))
		obj_reference = cp.dose_objective_primal_eval(cp.case.anatomy)

		cp.reload_clustered_frame()
		self.assertEqual( cp.case.frame.name, 'vclu' )
		cp.case.calculate_doses(np.random.rand(10))
		obj_clustered = cp.dose_objective_primal_eval(cp.case.anatomy)

		self.assert_scalar_equal( obj_reference, obj_clustered )

class VoxelClusteredProblemTestCase(ClusteredProblemGenericTestCase):
	pass

class UnconstrainedVoxClusProblemTestCase(ClusteredProblemGenericTestCase):
	@classmethod
	def setUpClass(self):
		self.case = self.build_fancy_case()
		self.vcp = UnconstrainedVoxelClusteredProblem(
				self.case, 'full', 'vclu')
		self.vcp_noisy = UnconstrainedVoxelClusteredProblem(
				self.case, 'full', 'vclu_noisy')

	def test_unconstrained_voxel_clustered_problem(self):
		self.vcp.reload_reference_frame()
		_, run = self.vcp.case.plan()
		obj = self.vcp.dose_objective_primal_eval(case.anatomy)

		ub_exact, lb_exact, run_vclu_exact = self.vcp.cluster_and_bound()
		ub_noisy, lb_noisy, run_vclu_noisy = self.vcp_noisy.cluster_and_bound()

		# make some problem
		# solve it
		# approximate, solve, get bounds, show valid
		self.assert_scalar_equal( lb_exact, obj )
		self.assert_scalar_equal( ub_exact, obj )
		self.assertTrue( ub_noisy >= obj )
		self.assertTrue( lb_noisy <= obj )

		run_vclu = self.vcp.plan()
		run_noisy = self.vcp_noisy.plan()
		self.assertTrue(
				run_vclu.output.optimal_variables['suboptimality'] >
				run_noisy.output.optimal_variables['suboptimality'] )

class BeamClusteredProblemTestCase(ClusteredProblemGenericTestCase):
	def test_beam_clustered_problem(self):
		case = self.build_simple_case()
		cp = BeamClusteredProblemMock(case, 'full', 'bclu')

		# init
		self.assertIsInstance( cp.cluster_mapping, ClusterMapping )

		# split voxel prices
		price_vec = np.random.rand(30)
		price_dict = cp.split_voxel_prices(
				price_vec, [0,1,2], {0:10, 1:10, 2:10})
		for i in xrange(3):
			self.assert_vector_equal(
					price_dict[i], price_vec[i * 10: (i+1) * 10] )

		# join voxel prices
		pv_recon = cp.join_voxel_prices(price_dict, [0, 1, 2])
		self.assert_vector_equal( pv_recon, price_vec )

		# dual eval
		obj_dual = cp.dose_objective_dual_eval(cp.reference_anatomy, price_dict)
		self.assertIsInstance(obj_dual, float)

		# update prices
		update_dict = {i: np.ones(10) for i in xrange(3)}
		cp.update_voxel_prices(price_dict, update_dict)
		for i in xrange(3):
			self.assert_vector_equal(
					price_dict[i], price_vec[i * 10: (i+1) * 10] + 1 )

class UnconstrainedBeamClusProblemTestCase(ClusteredProblemGenericTestCase):
	@classmethod
	def setUpClass(self):
		self.case = self.build_fancy_case()
		self.bcp = UnconstrainedBeamClusteredProblem(self.case, 'full', 'bclu')
		self.bcp_noisy = UnconstrainedBeamClusteredProblem(
				self.case, 'full', 'bclu_noisy')

	def test_build_A_infeas(self):
		prices = np.random.rand(200)
		prices[(3, 7, 9)] -= 10
		A_infeas = self.bcp.test_build_A_infeas(
				self.bcp.reference_anatomy, prices, 4, 1e-3)
		self.assertEqual( A_infeas.shape, (402, 3) )
		A_infeas = self.bcp.test_build_A_infeas(
				self.bcp.reference_anatomy, prices, 2, 1e-3)
		self.assertEqual( A_infeas.shape, (402, 2) )

	def test_build_A_infeas_augented(self):
		prices = np.random.rand(200)
		voxel_prices = np.random.rand(850)
		prices[(3, 7, 9)] -= 10
		A_infeas = self.bcp.test_build_A_infeas(
				self.bcp.reference_anatomy, prices, 4, 1e-3)
		A_aug = self.bcp.build_A_infeas_augmented(A_infeas, voxel_prices)
		m, n = A_infeas.shape
		A_aug_expect = np.zeros((n+1, m))
		A_aug_expect[:n, :] += A_infeas.T
		A_aug_expect[-1, :] += A_infeas.T.dot(voxel_prices)

		self.assertEqual( A_aug.shape, A_aug_expect.shape )
		x = np.random.rand(A_aug.shape[1])
		self.assert_vector_equal( A_aug.dot(x), A_aug_expect.dot(x) )

	def test_solve_dual_infeas_pogs(self):
		if not OPTKIT_INSTALLED:
			return

		k = self.bcp.cluster_mapping.n_clusters
		TOL = 1e-3

		voxel_prices = {
				s.label: -np.ones(s.size)
				for s in self.bcp.reference_anatomy}
		self.bcp.load_reference_frame()
		n_beams = self.bcp.case.n_beams

		beam_prices = self.bcp.methods.beam_prices(
				self.bcp.reference_anatomy, voxel_prices)
		n_infeas = np.sum(beam_prices <= 0)

		self.assertEqual( n_infeas, n_beams )

		A_infeas = self.bcp.build_A_infeas(beam_prices, 1e-3)
		voxel_price_update, _ = self.bcp.solve_dual_infeas_pogs(
				A_infeas, voxel_prices)

		self.bcp.update_voxel_prices(voxel_prices, voxel_price_update)
		beam_prices = self.bcp.methods.beam_prices(
				self.bcp.reference_anatomy, voxel_prices)
		n_infeas_updated = np.sum(beam_prices <= -TOL)

		# make sure # infeasible decreased by >= k each time
		self.assertTrue( n_infeas_updated <= n_infeas - k )

	def test_dual_iterative(self):
		if not OPTKIT_INSTALLED:
			return

		k = self.bcp.cluster_mapping.n_clusters
		self.bcp.load_reference_frame()
		n_beams = self.bcp.case.n_beams
		TOL = 1e-3

		voxel_prices = {
				s.label: -np.ones(s.size)
				for s in self.bcp.reference_anatomy}

		voxel_prices_feasible, _, epochs = self.bcp.dual_iterative(
				voxel_prices, 1e-3)

		beam_prices = self.bcp.methods.beam_prices(
				self.bcp.reference_anatomy, voxel_prices)
		n_infeas = np.sum(beam_prices <= -TOL)
		self.assertTrue( n_infeas == 0 )
		self.assertTrue( epochs * k <= n_beams / k)

	def test_unconstrained_beam_clustered_problem(self):
		if not OPTKIT_INSTALLED:
			return

		self.bcp.reload_reference_frame()

		# solve full
		_, run = self.bcp.case.plan()
		obj = self.bcp.dose_objective_primal_eval(case.anatomy)

		ub_exact, lb_exact, run_vclu_exact = self.bcp.cluster_and_bound()
		ub_noisy, lb_noisy, run_vclu_noisy = self.bcp_noisy.cluster_and_bound()

		# solve approximations, test validity of bounds
		self.assert_scalar_equal( lb_exact, obj )
		self.assert_scalar_equal( ub_exact, obj )
		self.assertTrue( ub_noisy >= obj )
		self.assertTrue( lb_noisy <= obj )

		# repeat test for .plan() method
		run_bclu = self.bcp.plan()
		run_noisy = self.bcp_noisy.plan()
		self.assertTrue(
				run_bclu.output.optimal_variables['suboptimality'] >
				run_noisy.output.optimal_variables['suboptimality'] )
