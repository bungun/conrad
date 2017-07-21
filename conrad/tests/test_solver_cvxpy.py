"""
Unit tests for :mod:`conrad.optimization.solver_cvxpy`.
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

from conrad.defs import module_installed
from conrad.medicine import Structure, D
from conrad.physics import Gy
from conrad.optimization.solvers.solver_cvxpy import *
from conrad.tests.base import *
from conrad.tests.test_solver import SolverGenericTestCase

class SolverCVXPYTestCase(SolverGenericTestCase):
	""" TODO: docstring"""
	def test_solver_cvxpy_init(self):
		s = SolverCVXPY()
		if s is None:
			return

		self.assertIsNone( s.problem )
		self.assertIsInstance( s._SolverCVXPY__x, cvxpy.Variable )
		self.assertIsInstance( s._SolverCVXPY__constraint_indices, dict )
		self.assertEqual( len(s._SolverCVXPY__constraint_indices), 0 )
		self.assertIsInstance( s.constraint_dual_vars, dict )
		self.assertEqual( len(s.constraint_dual_vars), 0 )
		self.assertEqual( s.n_beams, 0 )

	def test_solver_cvxpy_problem_init(self):
		n_beams = self.n
		s = SolverCVXPY()
		if s is None:
			return

		s.init_problem(n_beams)
		self.assertTrue( s.use_slack )
		self.assertFalse( s.use_2pass )
		self.assertEqual( s.n_beams, n_beams )
		self.assertIsNotNone( s.problem  )
		self.assertIsNotNone( s.problem.objective  )
		self.assertEqual( len(s.problem.constraints), 1 )

		for slack_flag in (True, False):
			for two_pass_flag in (True, False):
				s.init_problem(n_beams, use_slack=slack_flag,
							   use_2pass=two_pass_flag)
				self.assertEqual( s.use_slack, slack_flag )
				self.assertEqual( s.use_2pass, two_pass_flag )

		s.init_problem(n_beams, gamma=1.2e-3)
		self.assert_scalar_equal( s.gamma, 1.2e-3 )

	def assert_problems_equivalent(self, p1, p2):
		pd1 = p1.get_problem_data('ECOS')
		pd2 = p2.get_problem_data('ECOS')

		self.assert_vector_equal( pd1['c'], pd2['c'] )
		# self.assertTrue( sum(pd1['c'] - pd2['c']) == 0 )
		self.assertEqual( len((pd1['G'] - pd2['G']).data), 0 )

		return pd1['c'].shape, pd1['G'].shape

	def test_percentile_constraint_restricted(self):
		n_beams = self.n
		s = SolverCVXPY()
		if s is None:
			return

		beta = cvxpy.Variable(1)
		x = cvxpy.Variable(n_beams)
		A = self.A_targ

		# lower dose limit:
		#
		#	NONCONVEX CONSTRAINT:
		#
		#	D(percentile) >= dose
		#			---> dose received by P% of tissue >= dose
		#			---> at most (100 - P)% of tissue below dose
		#
		#	sum 1_{ y < dose } <= (100 - percentile)% of structure
		#	sum 1_{ y < dose } <= (1 - fraction) * structure size
		#
		#
		#	CONVEX RESTRICTION:
		#	slope * { y < (dose + 1 / slope)}_- <= (1 - fraction) * size
		#			---> let beta = 1 / slope
		#
		#	(1 / beta) * \sum { y - dose + beta }_- <= (1 - fraction) * size
		#	\sum {beta - y + dose}_+ <= beta * (1 - fraction) * size
		#
		constr = D(80) >= 10 * Gy
		dose = constr.dose.value

		objective = cvxpy.Minimize(0)

		theta = (1 - constr.percentile.fraction) * self.m_target
		c = s._SolverCVXPY__percentile_constraint_restricted(
				A, x, constr, beta)
		c_direct = cvxpy.sum_entries(
				cvxpy.pos(beta + (-1) * (A*x - dose))) <= beta * theta

		obj_shape, mat_shape = self.assert_problems_equivalent(
				cvxpy.Problem(cvxpy.Minimize(0), [c]),
				cvxpy.Problem(cvxpy.Minimize(0), [c_direct]) )

		# m voxels, n beams, 1 slope variable (beta), 0 slack variables
		var_count = self.m_target + self.n + 1
		constr_count = 2 * self.m_target + 1

		self.assertEqual( obj_shape[0], mat_shape[1] )
		self.assertEqual( obj_shape[0], var_count )
		self.assertEqual( mat_shape[0], constr_count )

		# upper dose limit
		#
		#	NONCONVEX CONSTRAINT:
		#
		#	D(percentile) <= dose
		#			---> dose received by (100-P)% of tissue <= dose
		#			---> at most P% of tissue above dose
		#
		#	sum 1_{ y > dose } <= (percentile)% of structure
		#	sum 1_{ y > dose } <= fraction * structure size
		#
		#
		#	CONVEX RESTRICTION:
		#	slope * { y > (dose - 1 / slope)}_+ <= fraction * size
		#			---> let beta = 1 / slope
		#
		#	(1 / beta) * \sum { y - dose + beta }_+ <= fraction * size
		#	\sum {beta + y - dose}_+ <= beta * fraction * size
		#
		constr = D(80) <= 10 * Gy
		dose = constr.dose.value

		objective = cvxpy.Minimize(0)

		theta = constr.percentile.fraction * self.m_target
		c = s._SolverCVXPY__percentile_constraint_restricted(
				A, x, constr, beta)
		c_direct = cvxpy.sum_entries(
				cvxpy.pos(beta + (A*x - dose))) <= beta * theta

		obj_shape, mat_shape = self.assert_problems_equivalent(
				cvxpy.Problem(cvxpy.Minimize(0), [c]),
				cvxpy.Problem(cvxpy.Minimize(0), [c_direct]) )

		# m voxels, n beams, 1 slope variable (beta), 0 slack variables
		var_count = self.m_target + self.n + 1
		constr_count = 2 * self.m_target + 1

		self.assertEqual( obj_shape[0], mat_shape[1] )
		self.assertEqual( obj_shape[0], var_count )
		self.assertEqual( mat_shape[0], constr_count )

		# lower dose limit, with slack allowing dose threshold to be lower
		#
		#	\sum {beta - y + (dose - slack)}_+ <= beta * (1 - fraction) * size
		#
		slack = cvxpy.Variable(1)

		constr = D(80) >= 10 * Gy
		dose = constr.dose.value

		objective = cvxpy.Minimize(0)

		theta = (1 - constr.percentile.fraction) * self.m_target
		c = s._SolverCVXPY__percentile_constraint_restricted(
				A, x, constr, beta, slack=slack)
		c_direct = cvxpy.sum_entries(cvxpy.pos(
				beta + (-1) * (A * x - (dose - slack)))) <= beta * theta

		obj_shape, mat_shape = self.assert_problems_equivalent(
				cvxpy.Problem(cvxpy.Minimize(0), [c]),
				cvxpy.Problem(cvxpy.Minimize(0), [c_direct]) )

		# m voxels, n beams, 1 slope variable (beta), 1 slack variable
		var_count = self.m_target + self.n + 2
		constr_count = 2 * self.m_target + 1

		self.assertEqual( obj_shape[0], mat_shape[1] )
		self.assertEqual( obj_shape[0], var_count )
		self.assertEqual( mat_shape[0], constr_count )

		# upper dose limit, with slack allowing dose theshold to be higher
		#
		#	\sum {beta + y - (dose + slack)}_+ <= beta * fraction * size
		#
		constr = D(80) <= 10 * Gy
		dose = constr.dose.value

		objective = cvxpy.Minimize(0)

		theta = constr.percentile.fraction * self.m_target
		c = s._SolverCVXPY__percentile_constraint_restricted(
				A, x, constr, beta, slack=slack)
		c_direct = cvxpy.sum_entries(
				cvxpy.pos(beta + (A * x - (dose + slack)))) <= beta * theta

		obj_shape, mat_shape = self.assert_problems_equivalent(
				cvxpy.Problem(cvxpy.Minimize(0), [c]),
				cvxpy.Problem(cvxpy.Minimize(0), [c_direct]) )

		# m voxels, n beams, 1 slope variable (beta), 1 slack variable
		var_count = self.m_target + self.n + 2
		constr_count = 2 * self.m_target + 1

		self.assertEqual( obj_shape[0], mat_shape[1] )
		self.assertEqual( obj_shape[0], var_count )
		self.assertEqual( mat_shape[0], constr_count )

	def test_percentile_constraint_exact(self):
		n_beams = self.n
		s = SolverCVXPY()
		if s is None:
			return

		beta = cvxpy.Variable(1)
		x = cvxpy.Variable(n_beams)
		A = self.A_targ
		constr_size = (self.m_target, 1)

		# lower dose limit, exact
		#
		#	y[chosen indices] > dose
		#
		constr = D(80) >= 10 * Gy
		dose = constr.dose.value
		y = np.random.rand(self.m_target)

		m_exact = int(np.ceil(self.m_target * constr.percentile.fraction))
		# ensure constraint is met by vector y
		y[:m_exact] += 10
		A_exact = A[constr.get_maxmargin_fulfillers(y), :]

		c = s._SolverCVXPY__percentile_constraint_exact(A, x, y, constr,
														had_slack=False)
		c_direct = A_exact * x >= dose
		obj_shape, mat_shape = self.assert_problems_equivalent(
				cvxpy.Problem(cvxpy.Minimize(0), [c]),
				cvxpy.Problem(cvxpy.Minimize(0), [c_direct]) )
		self.assertEqual( obj_shape[0], mat_shape[1] )
		self.assertEqual( obj_shape[0], self.n )
		self.assertEqual( mat_shape[0], m_exact )

		# upper dose limit, exact
		#
		#	y[chosen indices] < dose
		#
		constr = D(80) <= 10 * Gy
		dose = constr.dose.value
		y = np.random.rand(self.m_target)

		m_exact = int(np.ceil(self.m_target * (1 - constr.percentile.fraction)))
		# ensure constraint is met by vector y
		y[m_exact:] += 10
		A_exact = A[constr.get_maxmargin_fulfillers(y), :]

		c = s._SolverCVXPY__percentile_constraint_exact(A, x, y, constr,
														had_slack=False)
		c_direct = A_exact * x <= dose
		obj_shape, mat_shape = self.assert_problems_equivalent(
				cvxpy.Problem(cvxpy.Minimize(0), [c]),
				cvxpy.Problem(cvxpy.Minimize(0), [c_direct]) )
		self.assertEqual( obj_shape[0], mat_shape[1] )
		self.assertEqual( obj_shape[0], self.n )
		self.assertEqual( mat_shape[0], m_exact )

	def test_add_constraints(self):
		s = SolverCVXPY()
		if s is None:
			return

		x = cvxpy.Variable(self.n)
		p = cvxpy.Problem(cvxpy.Minimize(0), [x >= 0])

		s.init_problem(self.n, use_slack=False, use_2pass=False)
		self.assert_problems_equivalent( p, s.problem )

		# no constraints
		s._SolverCVXPY__add_constraints(self.anatomy['tumor'])
		self.assert_problems_equivalent( p, s.problem )

		# add mean constraint
		constr = D('mean') <= 10 * Gy
		constr_cvxpy = self.anatomy['tumor'].A_mean * x <= constr.dose.value
		self.anatomy['tumor'].constraints += constr
		p.constraints = [x>=0, constr_cvxpy]
		s._SolverCVXPY__add_constraints(self.anatomy['tumor'])
		self.assert_problems_equivalent( p, s.problem )

		s.clear()

		# add mean constraint with slack (upper)
		for priority in xrange(1, 4):
			slack = cvxpy.Variable(1)
			s.use_slack = True
			constr = D('mean') <= 10 * Gy
			constr.priority = priority
			dose = constr.dose.value
			constr_cvxpy = self.anatomy['tumor'].A_mean * x - slack <= dose
			self.anatomy['tumor'].constraints.clear()
			self.anatomy['tumor'].constraints += constr
			p.objective += cvxpy.Minimize(
					s.gamma_prioritized(constr.priority) * slack)
			p.constraints = [x>=0, slack >= 0, constr_cvxpy]
			s._SolverCVXPY__add_constraints(self.anatomy['tumor'])
			self.assert_problems_equivalent( p, s.problem )

			p.objective = cvxpy.Minimize(0)
			s.clear()
			s.use_slack = False

			# add mean constraint with slack (lower)
			slack = cvxpy.Variable(1)
			s.use_slack = True
			constr = D('mean') >= 10 * Gy
			constr.priority = priority
			dose = constr.dose.value
			constr_cvxpy = self.anatomy['tumor'].A_mean * x + slack >= dose
			self.anatomy['tumor'].constraints.clear()
			self.anatomy['tumor'].constraints += constr
			p.objective += cvxpy.Minimize(
					s.gamma_prioritized(constr.priority) * slack)
			p.constraints = [x>=0, slack >= 0, slack <= dose, constr_cvxpy]
			s._SolverCVXPY__add_constraints(self.anatomy['tumor'])
			self.assert_problems_equivalent( p, s.problem )

			p.objective = cvxpy.Minimize(0)
			s.clear()
			s.use_slack = False

		# exact constraint flag=True not allowed when structure.y is None
		self.assertIsNone( self.anatomy['tumor'].y )
		with self.assertRaises(ValueError):
			s._SolverCVXPY__add_constraints(self.anatomy['tumor'], True)
		self.anatomy['tumor'].calculate_dose(np.random.rand(self.n))
		self.assertIsNotNone( self.anatomy['tumor'].y )
		s._SolverCVXPY__add_constraints(self.anatomy['tumor'])
		s.clear()

		# exact constraint flag=True not allowed when use_2pass flag not set
		s.use_2pass = False
		with self.assertRaises(ValueError):
			s._SolverCVXPY__add_constraints(self.anatomy['tumor'], True)
		s.clear()

		# set slack flag, but set conditions that don't allow for slack
		constr = D('mean') <= 10 * Gy
		self.anatomy['tumor'].constraints.clear()
		self.anatomy['tumor'].constraints += constr
		constr_cvxpy = self.anatomy['tumor'].A_mean * x <= constr.dose.value
		dose = constr.dose.value

		# exact=True cancels out use_slack=True, no slack problem built
		s.use_slack = True
		s.use_2pass = True # (required for exact=True)
		s._SolverCVXPY__add_constraints(self.anatomy['tumor'], exact=True)

		# no slack problem
		p.objective = cvxpy.Minimize(0)
		p.constraints = [x>=0, constr_cvxpy]

		self.assert_problems_equivalent( p, s.problem )
		s.use_slack = False

		# constraint.priority=0 (force no slack) cancels out use_slack=True
		# for *THAT* constraint
		s.clear()
		self.anatomy['tumor'].constraints.clear()
		constr = D('mean') <= 10 * Gy
		constr.priority = 0
		self.anatomy['tumor'].constraints += constr
		# (no slack)
		constr_cvxpy = self.anatomy['tumor'].A_mean * x <= constr.dose.value

		constr2 = D('mean') <= 8 * Gy
		slack = cvxpy.Variable(1)
		self.anatomy['tumor'].constraints += constr2
		# (yes slack)
		constr_cvxpy2 = self.anatomy['tumor'].A_mean * x - slack <= constr2.dose.value

		s.use_slack = True
		s._SolverCVXPY__add_constraints(self.anatomy['tumor'])
		p.objective += cvxpy.Minimize(
				s.gamma_prioritized(constr2.priority) * slack)

		# try permuting constraints if problem equivalence fails
		try:
			p.constraints = [x>=0, constr_cvxpy, slack>=0, constr_cvxpy2]
			self.assert_problems_equivalent( p, s.problem )
		except:
			p.constraints = [x>=0, slack>=0, constr_cvxpy2, constr_cvxpy]
			self.assert_problems_equivalent( p, s.problem )

		s.use_slack = False

		# add max constraint
		s.clear()
		self.anatomy['tumor'].constraints.clear()
		constr = D('max') <= 30 * Gy
		self.anatomy['tumor'].constraints += constr
		constr_cvxpy = self.anatomy['tumor'].A * x  <= constr.dose.value
		p.objective = cvxpy.Minimize(0)
		p.constraints = [x>=0, constr_cvxpy]
		s._SolverCVXPY__add_constraints(self.anatomy['tumor'])
		self.assert_problems_equivalent( p, s.problem )

		# add min constraint
		s.clear()
		self.anatomy['tumor'].constraints.clear()
		constr = D('min') >= 25 * Gy
		self.anatomy['tumor'].constraints += constr
		constr_cvxpy = self.anatomy['tumor'].A * x  >= constr.dose.value
		p.objective = cvxpy.Minimize(0)
		p.constraints = [x>=0, constr_cvxpy]
		s._SolverCVXPY__add_constraints(self.anatomy['tumor'])
		self.assert_problems_equivalent( p, s.problem )

		# add percentile constraint, restricted.
		# - tested above in test_percentile_constraint_restricted()

		# add percentile constraint, exact
		# - tested above in test_percentile_constraint_exact()

	def test_build(self):
		s = SolverCVXPY()
		if s is None:
			return
		s.init_problem(self.n)

		self.anatomy['tumor'].constraints.clear()
		self.anatomy['oar'].constraints.clear()
		structure_list = self.anatomy.list

		A = self.anatomy['tumor'].A
		weight_abs = self.anatomy['tumor'].objective.weight_abs
		weight_lin = self.anatomy['tumor'].objective.weight_linear
		dose = float(self.anatomy['tumor'].dose)
		A_mean = self.anatomy['oar'].A_mean
		weight_oar = self.anatomy['oar'].objective.weight_raw

		x = cvxpy.Variable(self.n)
		p = cvxpy.Problem(cvxpy.Minimize(0), [])
		p.constraints += [ x >= 0 ]
		p.objective += cvxpy.Minimize(
				weight_abs * cvxpy.norm(A * x - dose, 1) +
				weight_lin * cvxpy.sum_entries(A * x - dose) )
		p.objective += cvxpy.Minimize(
				weight_oar * cvxpy.sum_entries(A_mean.T * x))

		s.use_slack = False
		s.build(structure_list, exact=False)
		self.assertEqual( len(s.slack_vars), 0 )
		self.assertEqual( len(s.dvh_vars), 0 )
		self.assertEqual( len(s._SolverCVXPY__constraint_indices), 0 )
		self.assert_problems_equivalent( p, s.problem )

		structure_list[0].constraints += D('mean') >= 10 * Gy
		cid = structure_list[0].constraints.last_key

		s.use_slack = False
		s.build(structure_list, exact=False)
		self.assertEqual( len(s.slack_vars), 1 )
		self.assertIn( cid, s.slack_vars )
		self.assertEqual( s.get_slack_value(cid), 0 )
		self.assertEqual( len(s.dvh_vars), 0 )
		self.assertEqual( len(s._SolverCVXPY__constraint_indices), 1 )
		self.assertIn( cid, s._SolverCVXPY__constraint_indices )
		# constraint 0: x >= 0; constraint 1: this
		self.assertEqual( s._SolverCVXPY__constraint_indices[cid], 1 )
		# dual is none since unsolved, not populated by cvxpy
		self.assertIsNone( s.get_dual_value(cid) )

		s.use_slack = True
		s.build(structure_list, exact=False)
		self.assertEqual( len(s.slack_vars), 1 )
		self.assertIn( cid, s.slack_vars )
		# slack is None since unsolved, not populated by cvxpy
		self.assertIsNone( s.get_slack_value(cid) )
		# set slack value, as if cvxpy solve called
		s.slack_vars[cid].value = 1
		self.assertEqual( s.get_slack_value(cid), 1. )
		self.assertEqual( len(s.dvh_vars), 0 )
		self.assertEqual( len(s._SolverCVXPY__constraint_indices), 1 )
		self.assertIn( cid, s._SolverCVXPY__constraint_indices )
		# constraint 0: x >= 0;
		# constraint 1: slack >= 0;
		# constraint 2: slack <= dose; (omitted if upper constraint w/slack)
		# constraint 3: this
		self.assertEqual( s._SolverCVXPY__constraint_indices[cid], 3 )

		# add percentile constraint, test slope retrieval
		structure_list[0].constraints += D(20) >= 10 * Gy
		cid2 = structure_list[0].constraints.last_key

		s.use_slack = False
		s.build(structure_list, exact=False)
		self.assertNotIn( cid, s.dvh_vars )
		self.assertIn( cid2, s.dvh_vars )
		self.assertIsNone( s.dvh_vars[cid2].value, None )
		self.assert_nan( s.get_dvh_slope(cid2) )

		# artificially set value of DVH slope variable
		BETA = 2.
		s.dvh_vars[cid2].value = BETA
		self.assert_scalar_equal( s.get_dvh_slope(cid2), 1. / BETA )

	def test_solve(self):
		s = SolverCVXPY()
		if s is None:
			return
		s.init_problem(self.n, use_slack=False, use_2pass=False)

		# solve variants:

		#	(1) unconstrained
		self.anatomy['tumor'].constraints.clear()
		self.anatomy['oar'].constraints.clear()
		structure_list = self.anatomy.list
		s.build(structure_list, exact=False)

		self.assertEqual( s.x.size, 1 )
		self.assertEqual( s.x_dual.size, 1 )
		self.assertIsNone( s.status )
		self.assertEqual( s.solveiters, 'n/a' )
		self.assert_nan( s.solvetime )
		self.assertEqual( len(s.slack_vars), 0 )
		self.assertEqual( len(s.dvh_vars), 0 )
		self.assertEqual( len(s._SolverCVXPY__constraint_indices), 0 )

		solver_status = s.solve(verbose=0)
		self.assertEqual( s.x.size, self.n )
		self.assertEqual( s.x_dual.size, self.n )
		self.assertEqual( s.status, 'optimal' )
		self.assertEqual( s.solveiters, 'n/a' )
		self.assertIsInstance( s.solvetime, float )
		self.assertEqual( len(s.slack_vars), 0 )
		self.assertEqual( len(s.dvh_vars), 0 )
		self.assertEqual( len(s._SolverCVXPY__constraint_indices), 0 )

		#	(2) mean-constrained
		# test constraint dual value retrieval
		structure_list[0].constraints += D('mean') >= 20 * Gy
		cid = structure_list[0].constraints.last_key
		s.build(structure_list, exact=False)
		solver_status = s.solve(verbose=0)
		self.assertTrue( solver_status )
		self.assertEqual( s.get_slack_value(cid), 0 )

		# constraint is active:
		self.assertGreater( s.get_dual_value(cid), 0 )

		# redundant constraint
		structure_list[0].constraints += D('mean') >= 10 * Gy
		cid2 = structure_list[0].constraints.last_key
		s.build(structure_list, exact=False)
		solver_status = s.solve(verbose=0)
		self.assertTrue( solver_status )
		self.assertEqual( s.status, 'optimal' )
		self.assertEqual( s.get_slack_value(cid), 0 )

		# constraint is active:
		self.assertGreater( s.get_dual_value(cid), 0 )
		# constraint is inactive:
		self.assert_scalar_equal( s.get_dual_value(cid2), 0 )

		# infeasible constraint
		structure_list[0].constraints += D('mean') <= 10 * Gy
		cid3 = structure_list[0].constraints.last_key
		s.build(structure_list, exact=False)
		solver_status = s.solve(verbose=0)
		self.assertFalse( solver_status )
		self.assertEqual( s.status, 'infeasible' )

		# remove infeasible constraint from structure
		structure_list[0].constraints -= cid3

		#	(3) +min constrained
		structure_list[0].constraints += D('min') >= 1 * Gy
		cid_min = structure_list[0].constraints.last_key
		s.build(structure_list, exact=False)
		solver_status = s.solve(verbose=0)
		self.assertTrue( solver_status )
		# assert dual variable is vector
		self.assertEqual(
				len(s.get_dual_value(cid_min)),
				structure_list[0].A_full.shape[0]  )

		# #	(4) +max constrained
		structure_list[0].constraints += D('max') <= 50 * Gy
		cid_max = structure_list[0].constraints.last_key
		s.build(structure_list, exact=False)
		solver_status = s.solve(verbose=0)
		self.assertTrue( solver_status )
		# assert dual variable is vector
		self.assertEqual(
				len(s.get_dual_value(cid_max)),
				structure_list[0].A_full.shape[0]  )

		#	(5) +percentile constrained
		structure_list[0].constraints += D(10) >= 0.1 * Gy
		cid_dvh = structure_list[0].constraints.last_key
		s.build(structure_list, exact=False)
		solver_status = s.solve(verbose=0)
		self.assertTrue( solver_status )

		# retrieve percentile constraint slope
		self.assertIsInstance( s.get_dvh_slope(cid_dvh), float )
		self.assertGreater( s.get_dvh_slope(cid_dvh), 0 )
		self.assertIsInstance( s.get_dual_value(cid_dvh), float )

		#	(6) percentile constrained, two-pass
		for structure in structure_list:
			# (calculated doses needed for 2nd pass)
			structure.calculate_dose(s.x)

		s.use_2pass = True # (flag needed to allow for exact below)
		s.build(structure_list, exact=True)
		solver_status = s.solve(verbose=0)
		self.assertTrue( solver_status )
		idx = s._SolverCVXPY__constraint_indices[cid_dvh]

		frac = structure_list[0].constraints[cid_dvh].percentile.fraction
		if structure_list[0].constraints[cid_dvh].upper:
			frac = 1. - frac
		constr_size = int(np.ceil(structure_list[0].size * frac))
		if constr_size > 1:
			self.assertEqual( len(s.get_dual_value(cid_dvh)), constr_size )
		else:
			self.assertIsInstance( s.get_dual_value(cid_dvh), float )
		s.use_2pass = False

		#	(7) with slack
		# infeasible constraint
		structure_list[0].constraints += D('mean') <= 10 * Gy
		cid4 = structure_list[0].constraints.last_key
		s.build(structure_list, exact=False)
		solver_status = s.solve(verbose=0)
		self.assertFalse( solver_status )

		# feasible with slack
		s.use_slack = True
		structure_list[0].constraints[cid].priority = 0
		structure_list[0].constraints[cid4].priority = 1
		s.build(structure_list, exact=False)
		solver_status = s.solve(verbose=0)
		self.assertTrue( solver_status )
		self.assertEqual( s.get_slack_value(cid), 0 )
		self.assertGreater( s.get_slack_value(cid4), 0 )

	def test_solver_options(self):
		s = SolverCVXPY()
		if s is None:
			return
		s.init_problem(self.n, use_slack=False, use_2pass=False)

		# solve variants:

		self.anatomy['tumor'].constraints.clear()
		self.anatomy['oar'].constraints.clear()
		structure_list = self.anatomy.list
		structure_list[0].constraints += D('mean') >= 20 * Gy
		structure_list[0].constraints += D(10) >= 0.1 * Gy
		s.build(structure_list, exact=False)

		if module_installed('scs'):
			for INDIRECT in [True, False]:
				for GPU in [True, False]:
					solver_status = s.solve(
							solver=cvxpy.SCS, verbose=0,
							use_indirect=INDIRECT, use_gpu=GPU)
					self.assertTrue( solver_status )

		if module_installed('ecos'):
			solver_status = s.solve(
					solver=cvxpy.ECOS, verbose=0, use_indirect=INDIRECT)
			self.assertTrue( solver_status )