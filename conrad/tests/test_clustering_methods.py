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
from conrad.abstract.vector import VectorConstraintQueue
from conrad.medicine import Structure, Anatomy
from conrad.medicine.dose import D
from conrad.abstract.mapping import *
from conrad.physics.physics import DoseFrameMapping
from conrad.case import Case
from conrad.optimization.solvers.environment import OPTKIT_INSTALLED
from conrad.optimization.clustering.clustering_base import *
from conrad.optimization.clustering.voxel_clustering import *
from conrad.optimization.clustering.beam_clustering import *
from conrad.tests.base import *

PRINT_FRIENDLY_TESTING = False
# OPEN QUESTIONS:
# 	- why does manual evaluation give a different answer than the obj_primal


class ClusteredProblemGenericTestCase(ConradTestCase):
	@property
	def SOLVER_OPTIONS(self):
		return {'maxiters': 10000, 'verbose': 0}

	@staticmethod
	def build_simple_case():
		case = Case(physics={'frame_name':'full'})
		case.anatomy += Structure(0, 'Target', True)
		case.anatomy += Structure(1, 'Avoid1', False)
		case.anatomy += Structure(2, 'Avoid2', False)

		A_full = {i: np.eye(10) for i in xrange(3)}
		vmap = DictionaryClusterMapping({
				0: ClusterMapping([0, 0, 0, 1, 1, 1, 1, 2, 2, 2]),
				1: ClusterMapping([0, 0, 1, 1, 2, 2, 2, 2, 2, 3]),
				2: IdentityMapping(10),
		})
		A_vclu = {
				0: vmap[0].downsample(np.eye(10), rescale_output=True),
				1: vmap[1].downsample(np.eye(10), rescale_output=True),
				2: np.eye(10),
		}
		bmap = ClusterMapping([0, 0, 1, 1, 1, 1, 2, 3, 3, 3])
		A_bclu = {
				0: np.array([2, 4, 1, 3]) * np.eye(4),
				1: np.array([2, 4, 1, 3]) * np.eye(4),
				2: np.array([2, 4, 1, 3]) * np.eye(4),
		}

		case.physics.frame.dose_matrix = A_full

		case.physics.add_dose_frame(
				'vclu', data=A_vclu, voxel_weights=vmap.cluster_weights)
		case.physics.add_dose_frame('bclu', data=A_bclu)

		case.physics.add_frame_mapping(
				DoseFrameMapping('full', 'vclu', voxel_map=vmap))
		case.physics.add_frame_mapping(
				DoseFrameMapping('full', 'bclu', beam_map=bmap))
		return case

	@staticmethod
	def build_fancy_case():
		case = Case(physics={'frame_name':'full'})
		case.anatomy += Structure(0, 'Target', True)
		case.anatomy += Structure(1, 'Avoid1', False)
		case.anatomy += Structure(2, 'Avoid2', False)

		if PRINT_FRIENDLY_TESTING:
			m_full = m_bclu = (8, 8, 5)
			n_full = n_vclu = 5
			m_vclu = (4, 4, 5)
			n_bclu = 3
		else:
			m_full = m_bclu = (400, 400, 50)
			n_full = n_vclu = 2000
			m_vclu = (50, 50, 50)
			n_bclu = 500

		vec = n_bclu * np.random.rand(n_full)
		vec[:n_bclu] = xrange(n_bclu) # ensure each cluster represented once in full

		map_fb = ClusterMapping(vec)

		vec0 = m_vclu[0] * np.random.rand(m_full[0])
		vec0[:m_vclu[0]] = xrange(m_vclu[0])
		vec1 = m_vclu[1] * np.random.rand(m_full[1])
		vec1[:m_vclu[0]] = xrange(m_vclu[1])
		map_fv = DictionaryClusterMapping({
				0: ClusterMapping(vec0),
				1: ClusterMapping(vec1),
				2: IdentityMapping(m_vclu[2]),
			})

		scaling = (10., 1., 1.)
		A_full = {
				i: scaling[i] * np.random.rand(m_full[i], n_full)
				for i in xrange(3)
		}
		A_vclu = map_fv.downsample(A_full, rescale_output=True)
		A_bclu = {
			i: map_fb.downsample(A_full[i].T, rescale_output=True).T
			for i in A_full
		}
		A_vclu_full = map_fv.upsample(A_vclu)
		A_bclu_full = {
			i: map_fb.upsample(A_bclu[i].T).T
			for i in A_bclu
		}

		case.physics.frame.dose_matrix = A_full

		case.physics.add_dose_frame(
				'vclu', data=A_vclu, voxel_weights=map_fv.cluster_weights)
		case.physics.add_dose_frame('bclu', data=A_bclu)
		case.physics.add_dose_frame('vclu_recon', data=A_vclu_full)
		case.physics.add_dose_frame('bclu_recon', data=A_bclu_full)

		case.physics.add_frame_mapping(DoseFrameMapping(
				'full', 'vclu', voxel_map=map_fv))
		case.physics.add_frame_mapping(DoseFrameMapping(
				'full', 'bclu', beam_map=map_fb))
		case.physics.add_frame_mapping(DoseFrameMapping(
				'vclu_recon', 'vclu', voxel_map=map_fv))
		case.physics.add_frame_mapping(DoseFrameMapping(
				'bclu_recon', 'bclu', beam_map=map_fb))
		return case

	def default_primal_eval(self, w_abs, w_lin, dose, y):
		# Primal objective, assuming standard piecewise linear
		# objective for targets and linear objective for non-targets
		return np.dot(w_abs, np.abs(y - dose)) + np.dot(w_lin, y - dose)

	def exercise_problem_scaling_primal(self, exact_clustered_problem, beams=False):
		ecp = exact_clustered_problem
		ecp.methods.apply_joint_scaling(ecp.reference_anatomy)
		ecp.methods.apply_joint_scaling(ecp.clustered_anatomy)

		x = np.random.rand(ecp.case.n_beams)
		if beams:
			fm = ecp.case.physics.retrieve_frame_mapping(
					ecp.reference_frame, ecp.clustered_frame)
			map_ = fm.beam_map
			x_upscaled = map_.upsample(x, rescale_output=True)
			ecp.reference_anatomy.calculate_doses(x_upscaled)
		else:
			ecp.reference_anatomy.calculate_doses(x)
		ecp.clustered_anatomy.calculate_doses(x)
		y_clu_expect = ecp.methods.voxel_doses(ecp.clustered_anatomy)
		y_full_expect = ecp.methods.voxel_doses(ecp.reference_anatomy)
		obj_clu = ecp.dose_objective_primal_eval(ecp.clustered_anatomy)
		obj_full = ecp.dose_objective_primal_eval(ecp.reference_anatomy)

		print('OBJ CLUSTERED == OBJ EXACT ? (PRIMAL)')
		print('{} == {}'.format(obj_clu, obj_full))
		self.assert_scalar_equal( obj_clu, obj_full )

		if OPTKIT_INSTALLED:
			ecp.case.problem.solver_pogs.build(ecp.reference_anatomy.list)
			f_full = ecp.case.problem.solver_pogs.objective_voxels
			if beams:
				y = np.dot(ecp.case.problem.solver_pogs.A_current, x_upscaled)
			else:
				y = np.dot(ecp.case.problem.solver_pogs.A_current, x)

			self.assert_vector_equal( y, y_full_expect )
			obj_full_pogs = self.default_primal_eval(
					f_full.c, f_full.d, f_full.b, y)

			ecp.case.problem.solver_pogs.build(ecp.clustered_anatomy.list)
			f_clu = ecp.case.problem.solver_pogs.objective_voxels
			y = np.dot(ecp.case.problem.solver_pogs.A_current, x)
			self.assert_vector_equal( y, y_clu_expect )
			obj_clu_pogs = self.default_primal_eval(
					f_clu.c, f_clu.d, f_clu.b, y)

			print('POGS: OBJ CLUSTERED == OBJ EXACT ? (PRIMAL)')
			print('{} == {}'.format(obj_clu_pogs, obj_full_pogs))
			self.assert_scalar_equal( obj_clu_pogs, obj_full_pogs )

			print('POGS vs. EVAL AGREEMENT (PRIMAL)')
			print('{} == {}'.format(obj_full_pogs, obj_full ))
			self.assert_scalar_equal( obj_full_pogs, obj_full )

	def default_dual_dict_eval(self, clustered_problem, nu_dict, ref_frame=True):
		# Dual objective, assuming standard piecewise linear
		# objective for targets and linear objective for non-targets
		cp = clustered_problem
		anat = cp.reference_anatomy if ref_frame else cp.clustered_anatomy
		weights = [cp.methods.dual_expr_pogs(anat[label]).d for label in nu_dict]
		nus = [
				cp.methods.weighted_mean(anat[label], nu_dict[label])
				if anat[label].collapsable
				else nu_dict[label]
				for label in nu_dict
		]
		return sum(map(lambda wts, nu: np.dot(wts, nu), weights, nus))

	def exercise_problem_scaling_dual(self, exact_clustered_problem, beams=False):
		ecp = exact_clustered_problem
		if beams:
			nu_clu = nu_full = {
				s.label: np.random.rand(s.size) for s in ecp.reference_anatomy}
		else:
			nu_clu = {
				s.label: np.random.rand(s.size) for s in ecp.clustered_anatomy}
			fm = ecp.case.physics.retrieve_frame_mapping(
					ecp.reference_frame, ecp.clustered_frame)
			map_ = fm.voxel_map
			nu_full = map_.upsample(nu_clu)

		ecp.methods.apply_joint_scaling(ecp.reference_anatomy)
		ecp.methods.apply_joint_scaling(ecp.clustered_anatomy)

		obj_clu = ecp.dose_objective_dual_eval(ecp.clustered_anatomy, nu_clu)
		obj_full = ecp.dose_objective_dual_eval(ecp.reference_anatomy, nu_full)

		print('OBJ CLUSTERED == OBJ EXACT ? (DUAL)')
		print('{} == {}'.format(obj_clu, obj_full))
		self.assert_scalar_equal( obj_clu, obj_full )

		# TEST BEAM DUALS
		mu_full = ecp.methods.beam_prices(ecp.reference_anatomy, nu_full)
		mu_clu = ecp.methods.beam_prices(ecp.clustered_anatomy, nu_clu)
		if beams:
			fm = ecp.case.physics.retrieve_frame_mapping(
					ecp.reference_frame, ecp.clustered_frame)
			map_ = fm.beam_map
			mu_clu = map_.upsample(mu_clu)
			self.assert_vector_equal( mu_full, mu_clu )

		if OPTKIT_INSTALLED:
			obj_full_pogs = self.default_dual_dict_eval(
					ecp, nu_full, ref_frame=True)

			obj_clu_pogs = self.default_dual_dict_eval(
					ecp, nu_clu, ref_frame=False)

			print('POGS: OBJ CLUSTERED == OBJ EXACT ? (DUAL)')
			print('{} == {}'.format(obj_clu_pogs, obj_full_pogs))
			self.assert_scalar_equal( obj_clu_pogs, obj_full_pogs )

			print('POGS vs. EVAL AGREEMENT (DUAL)')
			print('{} == {}'.format(obj_full_pogs, obj_full))
			self.assert_scalar_equal( obj_full_pogs, obj_full )

	def exercise_full_problem(self, clustered_problem):
		cp = clustered_problem
		_, run = cp.case.plan(frame=cp.reference_frame, **self.SOLVER_OPTIONS)

		primal = cp.dose_objective_primal_eval(cp.reference_anatomy)
		dual = cp.dose_objective_dual_eval_vec(
				cp.reference_anatomy, run.output.optimal_variables['nu'])
		print('PRIMAL {}'.format(primal))
		print('DUAL {}'.format(dual))
		print('{} <= {} {}'.format(dual, primal, dual <= primal))

		A = cp.case.problem.solver.A_current
		nu = cp.case.problem.solver.y_dual
		mu = A.T.dot(nu)
		self.assertTrue(
				dual <= primal or np.abs(primal - dual) < 5e-2 + 1e-3 * primal )

	def exercise_clustered_problem(self, clustered_problem,
								   exact_clustered_problem=None):
		cp = clustered_problem
		_, run = cp.case.plan(frame=cp.reference_frame, **self.SOLVER_OPTIONS)
		obj = cp.dose_objective_primal_eval(cp.reference_anatomy)

		if exact_clustered_problem:
			ecp = exact_clustered_problem
			_, runx = ecp.case.plan(frame=ecp.reference_frame, **self.SOLVER_OPTIONS)
			objx = ecp.dose_objective_primal_eval(ecp.reference_anatomy)

			_, runx_clu = ecp.case.plan(frame=ecp.clustered_frame, **self.SOLVER_OPTIONS)
			objx_clu = ecp.dose_objective_primal_eval(ecp.clustered_anatomy)

			_, run_clu = cp.case.plan(frame=cp.clustered_frame, **self.SOLVER_OPTIONS)
			obj_clu = cp.dose_objective_primal_eval(cp.clustered_anatomy)

			# print "-----------------"
			# print "COMPARE SOLUTIONS"
			# print "-----------------"

			# print "OBJ, OBJ EXACT CLU", obj, objx_clu

			# print "CROSSVAL EXACT"
			# print "OBJ EXACT, SOL EXACT", ecp.dose_objective_primal_eval(
			# 		ecp.reference_anatomy, beam_intensities=runx.output.x)
			# print "OBJ EXACT, SOL X CLU", ecp.dose_objective_primal_eval(
			# 		ecp.reference_anatomy, beam_intensities=runx_clu.output.x)
			# print "OBJ X CLU, SOL EXACT", ecp.dose_objective_primal_eval(
			# 		ecp.clustered_anatomy, beam_intensities=runx.output.x)
			# print "OBJ X CLU, SOL X CLU", ecp.dose_objective_primal_eval(
			# 		ecp.clustered_anatomy, beam_intensities=runx_clu.output.x)
			# print "OBJ, OBJ CLU", obj, obj_clu

			# print "CROSSVAL FULL"
			# print "OBJ FULL, SOL FULL", cp.dose_objective_primal_eval(
			# 		cp.reference_anatomy, beam_intensities=run.output.x)
			# print "OBJ FULL, SOL CLU", cp.dose_objective_primal_eval(
			# 		cp.reference_anatomy, beam_intensities=run_clu.output.x)
			# print "OBJ CLU, SOL FULL", cp.dose_objective_primal_eval(
			# 		cp.clustered_anatomy, beam_intensities=run.output.x)
			# print "OBJ CLU, SOL CLU", cp.dose_objective_primal_eval(
			# 		cp.clustered_anatomy, beam_intensities=run_clu.output.x)

			# print "-----------------"
			# print "COMPARE PROBLEM MATRICES"
			# print "-----------------"

			# Axclu = ecp.clustered_problem.solver.A_current
			# Ax = ecp.reference_problem.solver.A_current
			# Aclu = cp.clustered_problem.solver.A_current
			# A = cp.reference_problem.solver.A_current

			# print "ULTIMATE ROW EQUAL"
			# print "full v clu", np.sum(A[-1, :] - Aclu[-1, :])
			# print "full v xclu", np.sum(A[-1, :] - Axclu[-1, :])
			# print "full v x", np.sum(A[-1, :] - Ax[-1, :])

			# print "PENULTIMATE ROW EQUAL"
			# print "full v clu", np.sum(A[-2, :] - Aclu[-2, :])
			# print "full v xclu", np.sum(A[-2, :] - Axclu[-2, :])
			# print "full v x", np.sum(A[-2, :] - Ax[-2, :])

			# print "TARGETS APPROPRIATELY RELATED"
			# map_ = cp.cluster_mapping[0]
			# map_x = ecp.cluster_mapping[0]
			# print "clu v xclu", np.linalg.norm(Aclu[:-2, :] - Axclu[:-2, :])
			# print "full v clu", np.linalg.norm(
			# 		map_.downsample(A[:-2, :], rescale_output=True) -
			# 		Aclu[:-2, :])
			# print "x v xclu", np.linalg.norm(
			# 		Ax[:-2, :] - map_x.upsample(Axclu[:-2, :]))

			# print "-----------------"
			# print "COMPARE FUNCTIONS"
			# print "-----------------"

			# x = np.random.rand(np.size(run.output.x))

			# f = cp.reference_problem.solver.objective_voxels
			# fx = ecp.reference_problem.solver.objective_voxels
			# fclu = cp.clustered_problem.solver.objective_voxels
			# fxclu = ecp.clustered_problem.solver.objective_voxels

			# y = np.dot(A, x)
			# yx = np.dot(Ax, x)
			# yclu = np.dot(Aclu, x)
			# yxclu = np.dot(Axclu, x)

			# def eval_fy(f, y):
			# 	return f.c.dot(np.abs(f.a * y - f.b)) + f.d.dot(f.a * y - f.b)

			# def eval_fAx(f, A, x):
			# 	return eval_fy(f, np.dot(A, x))

			# print "f(Ax_random) full", eval_fy(f, y)
			# print "f(Ax_random) x", eval_fy(fx, yx)
			# print "f(Ax_random) clu", eval_fy(fclu, yclu)
			# print "f(Ax_random) xclu", eval_fy(fxclu, yxclu)


			# print "f(Ax_full)", eval_fAx(f, A, run.output.x)
			# print "f(Ax_x)", eval_fAx(fx, Ax, runx.output.x)
			# print "f(Ax_clu)", eval_fAx(fclu, Aclu, run_clu.output.x)
			# print "f(Ax_xclu)", eval_fAx(fxclu, Axclu, runx_clu.output.x)

			# print "ULTIMATE ROW EQUAL a"
			# print "full v clu", np.sum(f.a[-1] - fclu.a[-1])
			# print "full v xclu", np.sum(f.a[-1] - fxclu.a[-1])
			# print "full v x", np.sum(f.a[-1] - fx.a[-1])

			# print "ULTIMATE ROW EQUAL b"
			# print "full v clu", np.sum(f.b[-1] - fclu.b[-1])
			# print "full v xclu", np.sum(f.b[-1] - fxclu.b[-1])
			# print "full v x", np.sum(f.b[-1] - fx.b[-1])

			# print "ULTIMATE ROW EQUAL c"
			# print "full v clu", np.sum(f.c[-1] - fclu.c[-1])
			# print "full v xclu", np.sum(f.c[-1] - fxclu.c[-1])
			# print "full v x", np.sum(f.c[-1] - fx.c[-1])

			# print "ULTIMATE ROW EQUAL d"
			# print "full v clu", np.sum(f.d[-1] - fclu.d[-1])
			# print "full v xclu", np.sum(f.d[-1] - fxclu.d[-1])
			# print "full v x", np.sum(f.d[-1] - fx.d[-1])

			# print "PENULTIMATE ROW EQUAL a"
			# print "full v clu", np.sum(f.a[-2] - fclu.a[-2])
			# print "full v xclu", np.sum(f.a[-2] - fxclu.a[-2])
			# print "full v x", np.sum(f.a[-2] - fx.a[-2])

			# print "PENULTIMATE ROW EQUAL b"
			# print "full v clu", np.sum(f.b[-2] - fclu.b[-2])
			# print "full v xclu", np.sum(f.b[-2] - fxclu.b[-2])
			# print "full v x", np.sum(f.b[-2] - fx.b[-2])

			# print "PENULTIMATE ROW EQUAL c"
			# print "full v clu", np.sum(f.c[-2] - fclu.c[-2])
			# print "full v xclu", np.sum(f.c[-2] - fxclu.c[-2])
			# print "full v x", np.sum(f.c[-2] - fx.c[-2])

			# print "PENULTIMATE ROW EQUAL d"
			# print "full v clu", np.sum(f.d[-2] - fclu.d[-2])
			# print "full v xclu", np.sum(f.d[-2] - fxclu.d[-2])
			# print "full v x", np.sum(f.d[-2] - fx.d[-2])

			# map_ = cp.cluster_mapping[0]
			# map_x = ecp.cluster_mapping[0]
			# print "TARGETS APPROPRIATELY RELATED a"
			# print "clu v xclu", np.linalg.norm(fclu.a[:-2] - fxclu.a[:-2])
			# print "full v clu", np.linalg.norm(
			# 		map_.downsample(f.a[:-2], rescale_output=True) -
			# 		fclu.a[:-2])
			# print "x v xclu", np.linalg.norm(
			# 		fx.a[:-2] - map_x.upsample(fxclu.a[:-2]))

			# print "TARGETS APPROPRIATELY RELATED b"
			# print "clu v xclu", np.linalg.norm(fclu.b[:-2] - fxclu.b[:-2])
			# print "full v clu", np.linalg.norm(
			# 		map_.downsample(f.b[:-2], rescale_output=True) -
			# 		fclu.b[:-2])
			# print "x v xclu", np.linalg.norm(
			# 		fx.b[:-2] - map_x.upsample(fxclu.b[:-2]))

			# print "TARGETS APPROPRIATELY RELATED c"
			# print "clu v xclu", np.linalg.norm(fclu.c[:-2] - fxclu.c[:-2])
			# print "full v clu", np.linalg.norm(
			# 		map_.downsample(f.c[:-2], rescale_output=False) -
			# 		fclu.c[:-2])
			# print "x v xclu", np.linalg.norm(
			# 		fx.c[:-2] -
			# 		map_x.upsample(fxclu.c[:-2], rescale_output=True))

			# print "TARGETS APPROPRIATELY RELATED d"
			# print "clu v xclu", np.linalg.norm(fclu.d[:-2] - fxclu.d[:-2])
			# print "full v clu", np.linalg.norm(
			# 		map_.downsample(f.d[:-2], rescale_output=False) -
			# 		fclu.d[:-2])
			# print "x v xclu", np.linalg.norm(
			# 		fx.d[:-2] -
			# 		map_x.upsample(fxclu.d[:-2], rescale_output=True))

		run_clu = cp.plan(**self.SOLVER_OPTIONS)
		print('LB {}'.format(run_clu.output.solver_info['lower_bound']))
		print('OBJ {}'.format(obj))
		print('UB {}'.format(run_clu.output.solver_info['upper_bound']))

		lb = run_clu.output.solver_info['lower_bound']
		ub = run_clu.output.solver_info['upper_bound']
		self.assertTrue( ub >= obj )

		# TODO: MAKE THIS TEST PASS FOR ALL CONDITIONS
		if isinstance(cp, UnconstrainedBeamClusteredProblem):
			self.assertTrue( lb <= obj )

class ClusteredProblemMock(ClusteredProblem):
	def solve_and_bound_clustered_problem(self, **solver_options):
		raise NotImplementedError

class BeamClusteredProblemMock(BeamClusteredProblem):
	def solve_and_bound_clustered_problem(self, **solver_options):
		raise NotImplementedError

class ClusteredProblemTestCase(ClusteredProblemGenericTestCase):
	def test_clustered_problem(self):
		case = self.build_simple_case()
		cp = ClusteredProblemMock(case, 'full', 'vclu')
		self.assertIs( cp.case, case )
		for s in (0, 1, 2, 'Target', 'Avoid1', 'Avoid2'):
			s in cp.reference_anatomy

		x_star = np.random.rand(10)

		cp.reference_anatomy.calculate_doses(x_star)
		obj_reference = cp.dose_objective_primal_eval(cp.reference_anatomy)

		cp.clustered_anatomy.calculate_doses(x_star)
		obj_clustered = cp.dose_objective_primal_eval(cp.clustered_anatomy)

		self.assert_scalar_equal( obj_reference, obj_clustered )

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
		obj_dual = cp.dose_objective_dual_eval(
				cp.reference_anatomy, price_dict)
		self.assertIsInstance(obj_dual, float)

		obj_dual_v = cp.dose_objective_dual_eval_vec(
				cp.reference_anatomy, price_vec)
		self.assert_scalar_equal( obj_dual, obj_dual_v )

		# update prices
		update_dict = {i: np.ones(10) for i in xrange(3)}
		cp.update_voxel_prices(price_dict, update_dict)
		for i in xrange(3):
			self.assert_vector_equal(
					price_dict[i], pv_recon[i * 10: (i+1) * 10] + 1 )

		# test dual <= primal
		self.exercise_full_problem(cp)

class VoxelClusteredProblemTestCase(ClusteredProblemGenericTestCase):
	pass

class UnconstrainedVoxClusProblemTestCase(ClusteredProblemGenericTestCase):
	@classmethod
	def setUpClass(self):
		self.case = self.build_fancy_case()
		self.vcp = UnconstrainedVoxelClusteredProblem(
				self.case, 'full', 'vclu')
		self.vcp_exact = UnconstrainedVoxelClusteredProblem(
				self.case, 'vclu_recon', 'vclu')

	def test_unconstrained_voxel_clustered_problem(self):
		# self.exercise_problem_scaling_primal(self.vcp_exact)
		# self.exercise_problem_scaling_dual(self.vcp_exact)
		self.exercise_full_problem(self.vcp)
		self.exercise_clustered_problem(self.vcp, self.vcp_exact)

class BeamClusteredProblemTestCase(ClusteredProblemGenericTestCase):
	def test_beam_clustered_problem(self):
		case = self.build_simple_case()
		cp = BeamClusteredProblemMock(case, 'full', 'bclu')

		# init
		self.assertIsInstance( cp.cluster_mapping, ClusterMapping )

class UnconstrainedBeamClusProblemTestCase(ClusteredProblemGenericTestCase):
	@classmethod
	def setUpClass(self):
		self.case = self.build_fancy_case()
		self.bcp = UnconstrainedBeamClusteredProblem(self.case, 'full', 'bclu')
		self.bcp_exact = UnconstrainedBeamClusteredProblem(
				self.case, 'bclu_recon', 'bclu')

	def test_build_A_infeas(self):
		dim_beams = self.bcp.cluster_mapping.n_points
		dim_vox = self.bcp.reference_anatomy.total_working_size
		prices = np.random.rand(dim_beams)
		prices[[0, dim_beams - 2, dim_beams - 1]] -= 10
		A_infeas = self.bcp.build_A_infeas(prices, 1e-3, k=4)
		self.assertEqual( A_infeas.shape, (dim_vox, 3) )
		A_infeas = self.bcp.build_A_infeas(prices, 1e-3, k=2)
		self.assertEqual( A_infeas.shape, (dim_vox, 2) )

	def test_build_A_infeas_augmented(self):
		dim_beams = self.bcp.cluster_mapping.n_points
		dim_vox = self.bcp.reference_anatomy.total_working_size

		prices = np.random.rand(dim_beams)
		voxel_prices = np.random.rand(dim_vox)
		prices[[0, dim_beams - 2, dim_beams - 1]] -= 10
		A_infeas = self.bcp.build_A_infeas(prices, 1e-3)
		A_aug = self.bcp.build_A_infeas_augmented(A_infeas, voxel_prices)
		m, n = A_infeas.shape
		A_aug_expect = np.zeros((n, m + 1))
		A_aug_expect[:, :m] += A_infeas.T
		A_aug_expect[:, -1] += A_infeas.T.dot(voxel_prices)

		self.assertEqual( A_aug.shape, A_aug_expect.shape )
		x = np.random.rand(A_aug.shape[1])
		self.assert_vector_equal( A_aug.dot(x), A_aug_expect.dot(x) )

	def build_feasible_dual(self, anatomy):
		dual_dict = {}
		for s in anatomy.list:
			self.bcp.methods.in_dual_domain(s, np.ones(s.working_size))
			for bound, method in s.objective.dual_constraint_queue:
				if 'greater' in method.__name__ or 'equals' in method.__name__:
					dual_dict[s.label] = bound * np.ones(s.working_size)
					break
		return dual_dict

	def assert_dual_in_domain(self, anatomy, dual_dict, tol=1e-3):
		for label in dual_dict:
			self.assertTrue( self.bcp.methods.in_dual_domain(
					anatomy[label], dual_dict[label]) )

	def assert_dual_in_cone(self, anatomy, dual_dict, tol=5e-2):
		mu = self.bcp.methods.beam_prices(anatomy, dual_dict)
		self.assertTrue( np.min(mu) > -tol * np.sqrt(mu.size) )

	def assert_update_nonnegative(self, dual_dict, tol=1e-3):
		self.assertTrue( all([np.min(nu) >= -tol for nu in dual_dict.values()]) )

	def test_solve_clustered(self):
		_, _, _, nu_bclu_dict = self.bcp.solve_clustered(**self.SOLVER_OPTIONS)
		self.assert_dual_in_domain(self.bcp.clustered_anatomy, nu_bclu_dict)
		self.assert_dual_in_cone(self.bcp.clustered_anatomy, nu_bclu_dict)

	def test_solve_dual_infeas_pogs(self):
		if not OPTKIT_INSTALLED:
			return

		k = self.bcp.cluster_mapping.n_clusters
		TOL = 1e-3

		voxel_prices = self.build_feasible_dual(self.bcp.reference_anatomy)
		self.assert_dual_in_domain(self.bcp.reference_anatomy, voxel_prices)
		with self.assertRaises(AssertionError):
			self.assert_dual_in_cone(self.bcp.reference_anatomy, voxel_prices)

		n_beams = self.bcp.cluster_mapping.n_points
		beam_prices = self.bcp.methods.beam_prices(
				self.bcp.reference_anatomy, voxel_prices)
		n_infeas = np.sum(beam_prices <= 0)

		self.assertEqual( n_infeas, n_beams )

		A_infeas = self.bcp.build_A_infeas(beam_prices, 1e-3)
		voxel_price_update, _ = self.bcp.solve_dual_infeas_pogs(
				A_infeas, voxel_prices, **self.SOLVER_OPTIONS)

		voxel_price_update = self.bcp.split_voxel_prices(
					voxel_price_update,
					self.bcp.reference_anatomy.label_order,
					self.bcp.reference_anatomy.working_sizes)

		self.assert_update_nonnegative(voxel_price_update)
		self.bcp.update_voxel_prices(voxel_prices, voxel_price_update)
		self.assert_dual_in_domain(self.bcp.reference_anatomy, voxel_prices)

		beam_prices = self.bcp.methods.beam_prices(
				self.bcp.reference_anatomy, voxel_prices)
		n_infeas_updated = np.sum(beam_prices <= -TOL)

		# make sure # infeasible decreased by >= k each time
		self.assertTrue( n_infeas_updated <= n_infeas - k )

	def test_dual_iterative(self):
		if not OPTKIT_INSTALLED:
			return

		k = self.bcp.cluster_mapping.n_clusters
		n_beams = self.bcp.cluster_mapping.n_points
		TOL = 1e-3

		voxel_prices = self.build_feasible_dual(self.bcp.reference_anatomy)

		voxel_prices_feasible, _, epochs = self.bcp.dual_iterative(
				voxel_prices, 1e-3, test_feasibility=True,
				**self.SOLVER_OPTIONS)

		self.assert_dual_in_domain(
				self.bcp.reference_anatomy, voxel_prices_feasible)
		self.assert_dual_in_cone(
				self.bcp.reference_anatomy, voxel_prices_feasible)

		beam_prices = self.bcp.methods.beam_prices(
				self.bcp.reference_anatomy, voxel_prices_feasible)
		n_infeas = np.sum(beam_prices <= -TOL)
		self.assertTrue( n_infeas == 0 )
		self.assertTrue( epochs * k <= n_beams )

	def test_unconstrained_beam_clustered_problem(self):
		if not OPTKIT_INSTALLED:
			return
		# self.exercise_problem_scaling_primal(self.bcp_exact, beams=True)
		# self.exercise_problem_scaling_dual(self.bcp_exact, beams=True)
		# self.exercise_full_problem(self.bcp)
		# self.exercise_clustered_problem(self.bcp)


