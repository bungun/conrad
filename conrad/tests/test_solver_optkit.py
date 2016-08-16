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

		self.assertTrue( s.objective_voxels is None )
		self.assertTrue( s.objective_beams is None )
		self.assertTrue( s.pogs_solver is None )
		self.assertTrue( s.n_beams is None )
		self.assertTrue( s._SolverOptkit__A_current is None )

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
			self.assert_exception(
					call=s._SolverOptkit__percentile_constraint_restricted,
					args=[struc.A, struc.constraints[cid], slack_flag] )
			self.assert_exception(
					call=s._SolverOptkit__percentile_constraint_exact,
					args=[struc.A, struc.y, struc.constraints[cid], slack_flag] )
		for exact_flag in [True, False]:
			self.assert_exception(
					call=s._SolverOptkit__add_constraints,
					args=[struc, exact_flag] )

		self.assertTrue( s.get_slack_value(cid) is nan )
		self.assertTrue( s.get_dual_value(cid) is nan )
		self.assertTrue( s.get_dvh_slope(cid) is nan )

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

		self.assert_property_exception( obj=s, property_name='x' )
		self.assert_property_exception( obj=s, property_name='x_dual' )
		self.assert_property_exception( obj=s, property_name='y_dual' )
		self.assert_property_exception( obj=s, property_name='solvetime' )
		self.assert_property_exception( obj=s, property_name='status' )
		self.assert_property_exception( obj=s, property_name='objective_value' )
		self.assert_property_exception( obj=s, property_name='solveiters' )

		s.build(self.anatomy.list)
		size_A = 0
		for struc in self.anatomy:
			if struc.is_target:
				size_A += struc.size
			else:
				size_A += 1
		self.assertTrue( s.objective_voxels.size == size_A )
		self.assertTrue( s.objective_beams.size == self.n )

		self.assertTrue( s.x.size == self.n )
		self.assertTrue( s.x_dual.size == self.n )
		self.assertTrue( s.y_dual.size == size_A )
		self.assert_nan( s.solvetime )
		self.assertTrue( s.status == 0 )
		self.assert_nan( s.objective_value )
		self.assertTrue( s.solveiters == 0 )

		# exception when not solvable
		self.anatomy['tumor'].constraints += D('mean') > 10 * Gy
		self.assert_exception( call=s.build, args=[self.anatomy.list] )

	def test_solve(self):
		s = SolverOptkit()
		if s is None:
			return

		s.build(self.anatomy.list)
		converged = s.solve(verbose=0)
		self.assertTrue( converged or s.pogs_solver.info.status != 0 )
		self.assertTrue( isinstance(s.solvetime, float) )
		self.assertTrue( isinstance(s.status, int) )
		self.assertTrue( isinstance(s.objective_value, float) )
		self.assertTrue( isinstance(s.solveiters, int) )