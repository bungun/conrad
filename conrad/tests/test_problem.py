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
import numpy as np

from conrad.compat import *
from conrad.defs import module_installed
from conrad.physics.units import Gy
from conrad.medicine import Structure, Anatomy
from conrad.medicine.dose import D
from conrad.optimization.problem import *
from conrad.optimization.history import *
from conrad.tests.base import *

class PlanningProblemTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.m, self.n = m, n = 500, 100
		self.m_target = m_target = 100
		self.m_oar = m_oar = m - m_target
		self.anatomy = Anatomy()
		self.anatomy += Structure(0, 'tumor', True, A=rand(m_target, n))
		self.anatomy += Structure(1, 'oar', True, A=rand(m_oar, n))

	def tearDown(self):
		for struc in self.anatomy:
			struc.constraints.clear()

	def test_problem_init(self):
		p = PlanningProblem()

		if module_installed('cvxpy'):
			self.assertTrue( p.solver_cvxpy is not None )
		else:
			self.assertTrue( p.solver_cvxpy is None )

		if module_installed('optkit'):
			self.assertTrue( p.solver_pogs is not None )
		else:
			self.assertTrue( p.solver_pogs is None )

		self.assertTrue( p._PlanningProblem__solver is None )
		self.assertTrue( p.solver is not None )

	def test_updates(self):
		p = PlanningProblem()
		struc = self.anatomy['tumor']
		struc.constraints += D(30) < 30 * Gy

		# CVXPY
		if p.solver_cvxpy is not None:
			p._PlanningProblem__solver = p.solver_cvxpy
			p.solver_cvxpy.init_problem(self.n)
			p.solver_cvxpy.build(self.anatomy.list)
			p.solver_cvxpy.solve(verbose=0)
			x_optimal = p.solver_cvxpy.x
			p._PlanningProblem__update_constraints(struc)
			for key in struc.constraints:
				self.assertTrue( struc.constraints[key].slack >= 0 )
			p._PlanningProblem__update_structure(struc)
			self.assert_vector_equal( struc.A.dot(x_optimal), struc.y )

		# OPKIT
		if p.solver_pogs is not None:
			# don't call build/solve since constrained structure will
			# raise exception
			p._PlanningProblem__solver = p.solver_pogs
			x_rand = rand(self.n)
			p.solver_pogs.build(self.anatomy.list)
			p.solver_pogs.pogs_solver.output.x[:] = x_rand
			p._PlanningProblem__update_constraints(struc)
			for key in struc.constraints:
				self.assertTrue( struc.constraints[key].slack == 0 )
			p._PlanningProblem__update_structure(struc)
			self.assert_vector_equal( struc.A.dot(x_rand), struc.y )

	def test_gather_solver_data(self):
		p = PlanningProblem()
		# CVXPY
		if p.solver_cvxpy is not None:
			p._PlanningProblem__solver = p.solver_cvxpy


			self.anatomy['tumor'].constraints += D(90) > 0.5 * Gy
			p.solver_cvxpy.init_problem(self.n)
			p.solver_cvxpy.build(self.anatomy.list)
			p.solver_cvxpy.solve(verbose=0)
			ro = RunOutput()

			# gather solver info
			p._PlanningProblem__gather_solver_info(ro)
			self.assertTrue( ro.solver_info['status'] == p.solver.status )
			self.assertTrue( ro.solver_info['time'] == p.solver.solvetime )
			self.assertTrue(
					ro.solver_info['objective'] == p.solver.objective_value )
			self.assertTrue( ro.solver_info['iters'] == p.solver.solveiters )

			# gather solver info, exact
			p._PlanningProblem__gather_solver_info(ro, exact=True)
			self.assertTrue(
					ro.solver_info['status_exact'] == p.solver.status )
			self.assertTrue(
					ro.solver_info['time_exact'] == p.solver.solvetime )
			self.assertTrue(
					ro.solver_info['objective_exact'] ==
					p.solver.objective_value )
			self.assertTrue(
					ro.solver_info['iters_exact'] == p.solver.solveiters )

			# gather solver variables
			p._PlanningProblem__gather_solver_vars(ro)
			self.assert_vector_equal( ro.optimal_variables['x'], p.solver.x )
			self.assert_vector_equal(
					ro.optimal_variables['mu'], p.solver.x_dual )
			self.assertTrue( ro.optimal_variables['nu'] is None )

			# gather solver variables, exact
			p._PlanningProblem__gather_solver_vars(ro, exact=True)
			self.assert_vector_equal(
					ro.optimal_variables['x_exact'],p.solver.x )
			self.assert_vector_equal(
					ro.optimal_variables['mu_exact'], p.solver.x_dual )
			self.assertTrue( ro.optimal_variables['nu_exact'] is None )

			# gather DVH slopes
			p._PlanningProblem__gather_dvh_slopes(ro, self.anatomy.list)
			self.assertTrue( all([
					slope > 0 for slope in ro.optimal_dvh_slopes.values()]) )
		# OPKIT
		if p.solver_pogs is not None:
			p._PlanningProblem__solver = p.solver_pogs
			p.solver.build(self.anatomy.list)
			p.solver.solve(verbose=0)
			ro = RunOutput()

			# gather solver info
			p._PlanningProblem__gather_solver_info(ro)
			self.assertTrue( ro.solver_info['status'] == p.solver.status )
			self.assertTrue( ro.solver_info['time'] == p.solver.solvetime )
			self.assertTrue(
					ro.solver_info['objective'] == p.solver.objective_value )
			self.assertTrue( ro.solver_info['iters'] == p.solver.solveiters )

			# gather solver info, exact
			p._PlanningProblem__gather_solver_info(ro, exact=True)
			self.assertTrue(
					ro.solver_info['status_exact'] == p.solver.status )
			self.assertTrue(
					ro.solver_info['time_exact'] == p.solver.solvetime )
			self.assertTrue(
					ro.solver_info['objective_exact'] ==
					p.solver.objective_value )
			self.assertTrue(
					ro.solver_info['iters_exact'] == p.solver.solveiters )

			# gather solver variables
			p._PlanningProblem__gather_solver_vars(ro)
			self.assert_vector_equal(
					ro.optimal_variables['x'], p.solver.x )
			self.assert_vector_equal(
					ro.optimal_variables['mu'], p.solver.x_dual )
			self.assert_vector_equal(
					ro.optimal_variables['nu'], p.solver.y_dual )

			# gather solver variables, exact
			p._PlanningProblem__gather_solver_vars(ro, exact=True)
			self.assert_vector_equal(
					ro.optimal_variables['x_exact'], p.solver.x )
			self.assert_vector_equal(
					ro.optimal_variables['mu_exact'], p.solver.x_dual )
			self.assert_vector_equal(
					ro.optimal_variables['nu_exact'], p.solver.y_dual )

			# gather DVH slopes
			p._PlanningProblem__gather_dvh_slopes(ro, self.anatomy.list)
			self.assertTrue( all([
					slope is nan for slope in ro.optimal_dvh_slopes.values()]) )

	def test_fastest_solver(self):
		p = PlanningProblem()

		# unconstrained problem: OPTKIT is fastest, if available
		p._PlanningProblem__set_solver_fastest_available(self.anatomy.list)
		if p.solver_pogs is not None:
			self.assertTrue( p.solver == p.solver_pogs )
		elif p.solver_cvxpy is not None:
			self.assertTrue( p.solver == p.solver_cvxpy )

		self.anatomy['tumor'].constraints += D('mean') < 15 * Gy
		p._PlanningProblem__set_solver_fastest_available(self.anatomy.list)
		if p.solver_cvxpy is not None:
			self.assertTrue( p.solver == p.solver_cvxpy )

		# constrained problem: CVXPY is only solver that can handle it
		self.anatomy['tumor'].constraints -= self.anatomy[
				'tumor'].constraints.last_key
		self.anatomy['tumor'].constraints += D('min') > 5 * Gy
		p._PlanningProblem__set_solver_fastest_available(self.anatomy.list)
		if p.solver_cvxpy is not None:
			self.assertTrue( p.solver == p.solver_cvxpy )
		self.anatomy['tumor'].constraints -= self.anatomy[
				'tumor'].constraints.last_key

		self.anatomy['tumor'].constraints += D('max') < 20 * Gy
		p._PlanningProblem__set_solver_fastest_available(self.anatomy.list)
		if p.solver_cvxpy is not None:
			self.assertTrue( p.solver == p.solver_cvxpy )
		self.anatomy['tumor'].constraints -= self.anatomy[
				'tumor'].constraints.last_key

		self.anatomy['tumor'].constraints += D(2) < 20 * Gy
		p._PlanningProblem__set_solver_fastest_available(self.anatomy.list)
		if p.solver_cvxpy is not None:
			self.assertTrue( p.solver == p.solver_cvxpy )
		self.anatomy['tumor'].constraints -= self.anatomy[
				'tumor'].constraints.last_key

	def test_verify_2pass(self):
		p = PlanningProblem()

		self.anatomy['tumor'].constraints += D('mean') > 10 * Gy
		self.assertFalse( p._PlanningProblem__verify_2pass_applicable(
				self.anatomy.list) )

		self.anatomy['tumor'].constraints += D('mean') < 15 * Gy
		self.assertFalse( p._PlanningProblem__verify_2pass_applicable(
				self.anatomy.list) )

		self.anatomy['tumor'].constraints += D('min') > 5 * Gy
		self.assertFalse( p._PlanningProblem__verify_2pass_applicable(
				self.anatomy.list) )

		self.anatomy['tumor'].constraints += D('max') < 20 * Gy
		self.assertFalse( p._PlanningProblem__verify_2pass_applicable(
				self.anatomy.list) )

		self.anatomy['tumor'].constraints += D(2) < 20 * Gy
		self.assertTrue( p._PlanningProblem__verify_2pass_applicable(
				self.anatomy.list) )

	def test_solve(self):
		p = PlanningProblem()

		# force exception
		p.solver_cvxpy = None
		p.solver_pogs = None
		ro = RunOutput()
		self.assert_exception( call=p.solve, args=[self.anatomy.list, ro] )

		p = PlanningProblem()

		# unconstrained, no slack (slack irrelevant)
		#	- return code = 1 (1 feasible problem solved)
		ro = RunOutput()
		feasible = p.solve(self.anatomy.list, ro, slack=False, verbose=0)
		self.assertTrue( feasible == 1 )
		self.assertTrue( ro.solvetime > 0 )
		self.assert_nan( ro.solvetime_exact )

		# constrained, no slack
		#	- return code = 1
		ro = RunOutput()
		self.anatomy['tumor'].constraints += D('mean') > 10 * Gy
		cid1 = self.anatomy['tumor'].constraints.last_key
		feasible = p.solve(self.anatomy.list, ro, slack=False, verbose=0)
		self.assertTrue( feasible == 1 )
		self.assertTrue( ro.solvetime > 0 )
		self.assert_nan( ro.solvetime_exact )

		# add infeasible constraint
		#	- return code = 0 (0 feasible problems solved)
		ro = RunOutput()
		self.anatomy['tumor'].constraints += D('mean') < 8 * Gy
		feasible = p.solve(self.anatomy.list, ro, slack=False, verbose=0)
		self.assertTrue( feasible == 0 )
		self.assertTrue( ro.solvetime > 0 )
		self.assert_nan( ro.solvetime_exact )
		self.anatomy['tumor'].constraints -= self.anatomy[
			'tumor'].constraints.last_key

		# no slack, 2-pass
		# 	- no DVH constraints: request but don't perform 2-pass
		#	- return code = 1
		ro = RunOutput()
		feasible = p.solve(self.anatomy.list, ro, slack=False, verbose=0,
						   exact_constraints=True)
		self.assertTrue( feasible == 1 )
		self.assertTrue( ro.solvetime > 0 )
		self.assert_nan( ro.solvetime_exact )

		# 	- +DVH constraint: request *and* perform 2-pass method
		#	- return code = 2 (2 feasible problems solved)
		ro = RunOutput()
		self.anatomy['tumor'].constraints += D(10) < 20 * Gy
		cid2 = self.anatomy['tumor'].constraints.last_key
		feasible = p.solve(self.anatomy.list, ro, slack=False, verbose=0,
						   exact_constraints=True)
		self.assertTrue( feasible == 2 )
		self.assertTrue( ro.solvetime > 0 )
		self.assertTrue( ro.solvetime_exact > 0 )

		# slack
		#	- infeasible constraint, problem feasible with slack
		#	(on percentile constraints)
		#	- return code = 1
		ro = RunOutput()
		self.anatomy['tumor'].constraints += D(50) < 8 * Gy
		cid3 = self.anatomy['tumor'].constraints.last_key
		self.anatomy['tumor'].constraints[cid1].priority = 0
		self.anatomy['tumor'].constraints[cid2].priority = 1
		self.anatomy['tumor'].constraints[cid3].priority = 1

		feasible = p.solve(self.anatomy.list, ro, slack=True, verbose=0)
		self.assertTrue( feasible == 1 )
		self.assertTrue( ro.solvetime > 0 )
		self.assert_nan( ro.solvetime_exact )

		# slack, 2-pass
		#	- return code = 2
		ro = RunOutput()
		feasible = p.solve(self.anatomy.list, ro, slack=True, verbose=0,
						   exact_constraints=True)
		self.assertTrue( feasible == 2 )
		self.assertTrue( ro.solvetime > 0 )
		self.assertTrue( ro.solvetime_exact > 0 )