"""
Unit tests for :mod:`conrad.optimization.objectives`.
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
import cvxpy

from conrad import Gy
from conrad.optimization.objectives import *
from conrad.optimization.objectives.default_weights import *
from conrad.tests.base import *

class TreatmentObjectiveTestImplementation(TreatmentObjective):
	@property
	def objective_type(self):
		return 'dummy objective'
	@property
	def is_target_objective(self):
		raise NotImplementedError
	@property
	def is_nontarget_objective(self):
		raise NotImplementedError
	def primal_eval(self, var, weights):
		raise NotImplementedError
	def dual_eval(self, var, weights):
		raise NotImplementedError
	def primal_expr(self, var, weights):
		raise NotImplementedError
	def primal_expr_Ax(self, mat, var, weights):
		raise NotImplementedError
	def dual_expr(self, var, weights):
		raise NotImplementedError
	def dual_domain_constraints(self, var, weights):
		raise NotImplementedError
	def primal_expr_pogs(self, size, voxel_weights=None):
		raise NotImplementedError
	def dual_expr_pogs(self, size, voxel_weights=None):
		raise NotImplementedError
	def dual_domain_constraints_pogs(self, size, voxel_weights=None):
		raise NotImplementedError

class ObjectivesTestCase(ConradTestCase):
	def test_treatment_objective(self):
		obj = TreatmentObjectiveTestImplementation(test_weight=2)
		self.assertEqual( obj.test_weight, 2 )
		with self.assertRaises(AttributeError):
			obj.linear_weight
		with self.assertRaises(AttributeError):
			obj.rx_dose
		self.assertEqual(
				obj.dict['type'],
				TreatmentObjectiveTestImplementation().objective_type)
		self.assertEqual(obj.dict['parameters']['test_weight'], 2)
		self.assertIsInstance( obj.parameters, dict )
		self.assertIn( 'test_weight', obj.parameters )

		with self.assertRaises(ValueError):
			obj2 = TreatmentObjectiveTestImplementation(test_weight=-2)
		with self.assertRaises(TypeError):
			obj2 = TreatmentObjectiveTestImplementation(test_dose=-2)
		obj3 = TreatmentObjectiveTestImplementation(test_dose=3 * Gy)
		self.assertEqual( obj3.test_dose, 3 * Gy)
		obj4 = TreatmentObjectiveTestImplementation(test_dose='300 cGy')
		self.assertEqual( obj4.test_dose, 3 * Gy)
		with self.assertRaises(TypeError):
			obj5 = TreatmentObjectiveTestImplementation(test_dose=3)

		obj = TreatmentObjectiveTestImplementation(dose='3Gy', weight=2.)
		self.assertEqual( obj.dose, 3 * Gy )
		self.assertEqual( obj.weight, 2 )
		obj.scale(2)
		self.assertEqual( obj.dose, 3 * Gy )
		self.assertEqual( obj.weight, 4 )

		with self.assertRaises(ValueError):
			obj.scale(-2)
		obj *= 3
		self.assertEqual( obj.weight, 12. )
		obj = obj * 4.
		self.assertEqual( obj.weight, 48. )
		obj = 0.5 * obj
		self.assertEqual( obj.weight, 24. )

		# test normalization
		self.assertEqual( obj.normalization, 1. )
		obj.normalization = 1e-3
		self.assertEqual( obj.weight, 24 * 1e-3 )
		self.assertEqual( obj.weight_raw, 24 )

		# test aliases
		obj._TreatmentObjective__add_aliases('weight', 'weight_linear', 'wt')
		self.assertEqual( obj.weight_linear, obj.weight )
		self.assertEqual( obj.wt, obj.weight )

		# test change parameters
		obj.change_parameters(weight=3.)
		self.assertEqual( obj.weight_raw, 3. )

	def __exercise_cvxpy_expressions(self, objective):
		# primal expression
		y = cvxpy.Variable(3)
		x = cvxpy.Variable(5)
		A = np.random.rand(3, 5)
		x.save_value(np.random.rand(5))
		y.save_value(np.dot(A, x.value))
		weights = np.random.rand(3)

		# unweighted, f(y) and f(Ax)
		f = objective.eval(y.value)
		fy = cvxpy.Minimize(objective.expr(y))
		fAx = cvxpy.Minimize(objective.expr_Ax(A, x))
		self.assert_scalar_equal( fy.value, f )
		self.assert_scalar_equal( fAx.value, f )

		# weighted, f(y) and f(Ax)
		f = objective.eval(y.value, weights)
		fy = cvxpy.Minimize(objective.expr(y, weights))
		fAx = cvxpy.Minimize(objective.expr_Ax(A, x, weights))
		self.assert_scalar_equal( fy.value, f )
		self.assert_scalar_equal( fAx.value, f )

		# dual expression
		nu = cvxpy.Variable(3)
		nu.save_value(np.random.rand(3))

		# unweighted, f^*(nu)
		ff = objective.dual_eval(nu.value)
		ffnu = cvxpy.Maximize(objective.dual_expr(nu))
		self.assert_scalar_equal( ffnu.value, ff )

		# weighted, f^*(nu)
		ff = objective.dual_eval(nu.value, weights)
		ffnu = cvxpy.Maximize(objective.dual_expr(nu, weights))
		self.assert_scalar_equal( ffnu.value, ff )

	def test_nontarget_objective_linear(self):
		obj = NontargetObjectiveLinear()
		self.assertEqual( obj.weight, WEIGHT_LIN_NONTARGET_DEFAULT )

		self.assertFalse( obj.is_target_objective )
		self.assertTrue( obj.is_nontarget_objective )

		obj = NontargetObjectiveLinear(3.)
		self.assertEqual( obj.weight, 3. )

		self.assertIsInstance( obj.dict, dict )
		obj2 = dictionary_to_objective(**obj.dict)
		self.assertEqual( type(obj2), type(obj) )

		# primal and dual eval
		self.assertEqual( obj.eval([3, 4, 5]), obj.weight * sum([3, 4, 5]) )
		self.assertEqual(
				obj.eval([3, 4, 5], [1, 1, 2]),
				obj.weight * np.dot([3, 4, 5], [1, 1, 2]) )
		self.assertEqual( obj.dual_eval([3, 4, 5]), 0 )
		self.assertEqual( obj.dual_eval([3, 4, 5], [1, 1, 2]), 0 )

		# primal and dual expressions
		self.__exercise_cvxpy_expressions(obj)

		# dual domain constraints
		constr = obj.dual_domain_constraints(cvxpy.Variable(3))
		self.assertIsInstance(
				constr, cvxpy.constraints.eq_constraint.EqConstraint )

	def test_target_objective_pwl(self):
		obj = TargetObjectivePWL()
		self.assertEqual( obj.weight_overdose, WEIGHT_PWL_OVER_DEFAULT )
		self.assertEqual( obj.weight_underdose, WEIGHT_PWL_UNDER_DEFAULT )
		self.assertEqual( obj.target_dose, 1 * Gy )

		self.assertTrue( obj.is_target_objective )
		self.assertFalse( obj.is_nontarget_objective )

		self.assertIsInstance( obj.dict, dict )
		obj2 = dictionary_to_objective(**obj.dict)
		self.assertEqual( type(obj2), type(obj) )

		obj = TargetObjectivePWL('2 Gy', 0.5, 1)
		self.assertEqual( obj.weight_overdose, 1 )
		self.assertEqual( obj.weight_underdose, 0.5 )
		self.assertEqual( obj.target_dose, 2 * Gy )
		self.assertEqual( obj.weight_abs, 0.5 * (1 + 0.5) )
		self.assertEqual( obj.weight_linear, 0.5 * (1 - 0.5) )

		# primal eval unweighted
		val = -0.5 * (1 - 2) + 1 * (3 - 2)
		self.assertEqual( obj.eval([1, 2, 3]), val )

		# primal eval weighted
		weights = [1, 2, 3, 4, 5]
		doses = [3, 2, 1, 4, 3]
		val = 0
		for i, d in enumerate(doses):
			val += obj.weight_overdose * weights[i] * max(
					d - float(obj.target_dose), 0)
			val += obj.weight_underdose * weights[i] * max(
					float(obj.target_dose) - d, 0)
		self.assertEqual( obj.eval(doses, weights), val )

		# dual_eval
		prices = [1, 2, 3, 4, 5]
		weights = np.random.rand(5)
		self.assertEqual(
				obj.dual_eval(prices),
				-float(obj.target_dose) * sum(prices) )

		# dual_eval_weighted
		self.assertEqual(
				obj.dual_eval(prices, weights),
				-float(obj.target_dose) * np.dot(prices, weights) )

		# primal and dual expressions
		self.__exercise_cvxpy_expressions(obj)

		# dual_domain_constraints
		weights = np.random.rand(3)
		for wt in [None, weights]:
			constr = obj.dual_domain_constraints(cvxpy.Variable(3), wt)
			self.assertIsInstance( constr, list )
			self.assertEqual( len(constr), 2 )
			for c in constr:
				self.assertIsInstance(
						c, cvxpy.constraints.leq_constraint.LeqConstraint )

		# test change parameters with aliases
		obj.change_parameters(weight_underdose=3.)
		obj.change_parameters(w_under=3.5)

	def test_objective_hinge(self):
		obj = ObjectiveHinge()
		self.assertEqual( obj.weight, WEIGHT_HINGE_DEFAULT )
		self.assertEqual( obj.deadzone_dose, 1 * Gy )

		self.assertTrue( obj.is_nontarget_objective )
		self.assertTrue( obj.is_target_objective )

		self.assertIsInstance( obj.dict, dict )
		obj2 = dictionary_to_objective(**obj.dict)
		self.assertEqual( type(obj2), type(obj) )

		obj = ObjectiveHinge('2 Gy', 0.5)
		self.assertEqual( obj.weight, 0.5 )
		self.assertEqual( obj.deadzone_dose, 2 * Gy )

		# primal eval unweighted
		val = 0.5 * (3 - 2)
		self.assertEqual( obj.eval([1, 2, 3]), val )

		# primal eval weighted
		weights = [1, 2, 3, 4, 5]
		doses = [3, 2, 1, 4, 3]
		val = 0
		for i, d in enumerate(doses):
			if d > float(obj.deadzone_dose):
				val += obj.weight * weights[i] * max(
						d - float(obj.deadzone_dose), 0)
		self.assertEqual( obj.eval(doses, weights), val )

		# dual_eval
		prices = [1, 2, 3, 4, 5]
		weights = np.random.rand(5)
		self.assertEqual(
				obj.dual_eval(prices),
				-float(obj.deadzone_dose) * sum(prices) )

		# dual_eval_weighted
		self.assertEqual(
				obj.dual_eval(prices, weights),
				-float(obj.deadzone_dose) * np.dot(prices, weights) )

		# primal and dual expressions
		self.__exercise_cvxpy_expressions(obj)

		# dual_domain_constraints
		weights = np.random.rand(3)
		for wt in [None, weights]:
			constr = obj.dual_domain_constraints(cvxpy.Variable(3), wt)
			self.assertIsInstance( constr, list )
			self.assertEqual( len(constr), 2 )
			for c in constr:
				self.assertIsInstance(
						c, cvxpy.constraints.leq_constraint.LeqConstraint )
