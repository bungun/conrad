"""
Unit tests for :mod:`conrad.optimization.solver_optkit`.
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
from conrad.medicine import Structure, D
from conrad.physics import Gy
from conrad.optimization.solver_optkit import *
from conrad.tests.base import *
from conrad.tests.test_solver import SolverGenericTestCase

class SolverOptkitTestCase(SolverGenericTestCase):
	""" TODO: docstring"""
	def test_solver_optkit_init(self):
		s = SolverOptkit()
		if s is None:
			return

		self.assertIsNone( s.objective_voxels )
		self.assertIsNone( s.objective_beams )
		self.assertIsNone( s.pogs_solver )
		self.assertIsNone( s.n_beams )
		self.assertIsNone( s._SolverOptkit__A_current )

	def test_solver_build_matrix(self):
		s = SolverOptkit()

		# -structure 1 not collapsable (reason: target)
		# -structure 2 collapsable
		# expected matrix size: {m0 + 1 \times n}
		A = s._SolverOptkit__build_matrix(self.anatomy.list)
		self.assertEqual( A.shape, (self.m_target + 1, self.n) )
		self.assert_vector_equal( A[:self.m_target, :], self.A_targ )
		self.assert_vector_equal(
				A[self.m_target, :], self.A_oar.sum(0) / self.m_oar )

		self.anatomy[1].constraints += D('mean') < 5 * Gy
		# -structure 1 not collapsable (reason: target)
		# -structure 2 collapsable (reason: only mean constraint)
		# expected matrix size: {m0 + 1 \times n}
		A = s._SolverOptkit__build_matrix(self.anatomy.list)
		self.assertEqual( A.shape, (self.m_target + 1, self.n) )
		self.assert_vector_equal( A[:self.m_target, :], self.A_targ )
		self.assert_vector_equal(
				A[self.m_target, :], self.A_oar.sum(0) / self.m_oar )

		self.anatomy[1].constraints += D(30) < 10 * Gy
		# -structure 1 not collapsable (reason: target)
		# -structure 2 not collapsable (reason: DVH constraint)
		# expected matrix size: {m0 + m1 \times n}
		A = s._SolverOptkit__build_matrix(self.anatomy.list)
		self.assertEqual( A.shape, (self.m_target + self.m_oar, self.n) )
		self.assert_vector_equal( A[:self.m_target, :], self.A_targ )
		self.assert_vector_equal( A[self.m_target:, :], self.A_oar )

	def __assert_voxel_objective_default(self, v_objective, compressed=True,
										 weight_targ=1., weight_oar=1.):
		size_expect = self.m_target
		size_expect += 1 if compressed else self.m_oar
		self.assertEqual( v_objective.size, size_expect )

		w_over = self.anatomy[0].objective.weight_over_raw
		w_under = self.anatomy[0].objective.w_under_raw
		w_abs = 0.5 * (w_over + w_under) / self.anatomy[0].weighted_size
		w_lin = 0.5 * (w_over - w_under) / self.anatomy[0].weighted_size
		w = self.anatomy[1].objective.weight_raw
		w_norm = w / self.anatomy[1].weighted_size

		equality_assertion = self.assert_scalar_equal if compressed else \
							 self.assert_vector_equal

		# param 'a' == 1
		self.assert_vector_equal( v_objective.a, 1)

		# param 'b' == 	dose if target,
		#				0 otherwise
		self.assert_vector_equal(
				v_objective.b[:self.m_target],
				float(self.anatomy[0].objective.target_dose) )
		equality_assertion( v_objective.b[self.m_target:], 0 )

		# param 'c' == 	w_abs if target,
		#				0 otherwise
		self.assert_vector_equal(
				v_objective.c[:self.m_target], w_abs * weight_targ )
		equality_assertion( v_objective.c[self.m_target:], 0 )

		# param 'd' == 	w_abs if target,
		#				w otherwise
		self.assert_vector_equal(
				v_objective.d[:self.m_target], w_lin * weight_targ )
		lhs = v_objective.d[self.m_target:]
		rhs = w if compressed else w_norm * weight_oar
		equality_assertion( lhs, rhs )

		# param 'e' == 0
		self.assert_vector_equal( v_objective.e, 0)

	def test_solver_build_voxel_objective(self):
		s = SolverOptkit()
		self.assertTrue( s.objective_voxels is None )

		# -structure 1 not collapsable (reason: target)
		# -structure 2 collapsable
		# expected matrix size: {m0 + 1 \times n}
		s._SolverOptkit__build_voxel_objective(self.anatomy.list)
		self.__assert_voxel_objective_default(s.objective_voxels)

		self.anatomy[1].constraints += D('mean') < 5 * Gy
		# -structure 1 not collapsable (reason: target)
		# -structure 2 collapsable (reason: only mean constraint)
		# expected matrix size: {m0 + 1 \times n}
		s._SolverOptkit__build_voxel_objective(self.anatomy.list)
		self.assertTrue( s.objective_voxels.size == self.m_target + 1 )
		self.__assert_voxel_objective_default(s.objective_voxels)

		self.anatomy[1].constraints += D(30) < 10 * Gy
		# -structure 1 not collapsable (reason: target)
		# -structure 2 not collapsable (reason: DVH constraint)
		# expected matrix size: {m0 + m1 \times n}
		s._SolverOptkit__build_voxel_objective(self.anatomy.list)
		self.assertTrue(
				s.objective_voxels.size == self.m_target + self.m_oar )
		self.__assert_voxel_objective_default(
				s.objective_voxels, compressed=False)

		# test incorporation of voxel weights into objective term weights,
		# both full and collapsed matrix cases
		voxel_weights0 = (1 + 10 * rand(self.m_target)).astype(int)
		voxel_weights1 = (1 + 10 * rand(self.m_oar)).astype(int)
		self.anatomy[0].voxel_weights = voxel_weights0
		self.anatomy[1].voxel_weights = voxel_weights1
		s._SolverOptkit__build_voxel_objective(self.anatomy.list)
		self.__assert_voxel_objective_default(
				s.objective_voxels, compressed=False,
				weight_targ=voxel_weights0, weight_oar=voxel_weights1)
		self.anatomy[1].constraints.clear()
		s._SolverOptkit__build_voxel_objective(self.anatomy.list)
		self.__assert_voxel_objective_default(
				s.objective_voxels, compressed=True,
				weight_targ=voxel_weights0, weight_oar=voxel_weights1)

	def test_solver_build_beam_objective(self):
		s = SolverOptkit()
		s._SolverOptkit__build_beam_objective(self.anatomy.list)
		self.assertTrue( s.objective_beams.size == self.n )

	def test_solver_optkit_problem_init(self):
		s = SolverOptkit()
		if s is None:
			return

		# this call doesn't do much, unlike the SolverCVXPY case
		s.init_problem(self.n)
		self.assertTrue( s.n_beams == self.n )

	def test_constraint_handling(self):
		s = SolverOptkit()
		if s is None:
			return

		struc = self.anatomy['tumor']
		struc.constraints += D(70) > 10 * Gy
		struc.calculate_dose(rand(self.n))
		cid = struc.constraints.last_key
		for slack_flag in [True, False]:
			with self.assertRaises(ValueError):
				s._SolverOptkit__percentile_constraint_restricted(
					struc.A, struc.constraints[cid], slack_flag)
			with self.assertRaises(ValueError):
				s._SolverOptkit__percentile_constraint_exact(
					struc.A, struc.y, struc.constraints[cid], slack_flag)
		for exact_flag in [True, False]:
			with self.assertRaises(ValueError):
				s._SolverOptkit__add_constraints(struc, exact_flag)

		self.assert_nan( s.get_slack_value(cid) )
		self.assert_nan( s.get_dual_value(cid) )
		self.assert_nan( s.get_dvh_slope(cid) )

	def test_can_solve(self):
		s = SolverOptkit()
		if s is None:
			return

		# unconstrained is solvable
		self.assertTrue( s.can_solve(self.anatomy.list) )

		# mean constrained: not solvable
		self.anatomy['tumor'].constraints += D('mean') > 10 * Gy
		self.assertFalse( s.can_solve(self.anatomy.list) )
		self.anatomy['tumor'].constraints -= self.anatomy[
			'tumor'].constraints.last_key

		# min constrained: not solvable
		self.anatomy['tumor'].constraints += D('min') > 10 * Gy
		self.assertFalse( s.can_solve(self.anatomy.list) )
		self.anatomy['tumor'].constraints -= self.anatomy[
			'tumor'].constraints.last_key

		# max constrained: not solvable
		self.anatomy['tumor'].constraints += D('max') < 10 * Gy
		self.assertFalse( s.can_solve(self.anatomy.list) )
		self.anatomy['tumor'].constraints -= self.anatomy[
			'tumor'].constraints.last_key

		# percentile constrained: not solvable
		self.anatomy['tumor'].constraints += D(10) > 10 * Gy
		self.assertFalse( s.can_solve(self.anatomy.list) )
		self.anatomy['tumor'].constraints -= self.anatomy[
			'tumor'].constraints.last_key

	def test_build(self):
		s = SolverOptkit()
		if s is None:
			return

		with self.assertRaises(ValueError):
			s.x
		with self.assertRaises(ValueError):
			s.x_dual
		with self.assertRaises(ValueError):
			s.y_dual
		with self.assertRaises(ValueError):
			s.solvetime
		with self.assertRaises(ValueError):
			s.status
		with self.assertRaises(ValueError):
			s.objective_value
		with self.assertRaises(ValueError):
			s.solveiters

		s.build(self.anatomy.list)
		size_A = 0
		for struc in self.anatomy:
			if struc.is_target:
				size_A += struc.size
			else:
				size_A += 1
		self.assertEqual( s.objective_voxels.size, size_A )
		self.assertEqual( s.objective_beams.size, self.n )

		self.assertEqual( s.x.size, self.n )
		self.assertEqual( s.x_dual.size, self.n )
		self.assertEqual( s.y_dual.size, size_A )
		self.assert_nan( s.solvetime )
		self.assertEqual( s.status, 0 )
		self.assert_nan( s.objective_value )
		self.assertEqual( s.solveiters, 0 )

		# exception when not solvable
		self.anatomy['tumor'].constraints += D('mean') > 10 * Gy
		with self.assertRaises(ValueError):
			s.build(self.anatomy.list)

	def test_solve(self):
		s = SolverOptkit()
		if s is None:
			return

		s.build(self.anatomy.list)
		converged = s.solve(verbose=0)
		self.assertTrue( converged or s.pogs_solver.info.status != 0 )
		self.assertIsInstance( s.solvetime, float )
		self.assertIsInstance( s.status, int )
		self.assertIsInstance( s.objective_value, float )
		self.assertIsInstance( s.solveiters, int )