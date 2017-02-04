"""
Unit tests for :mod:`conrad.optimization.preprocessing`.
"""
"""
Copyright 2016--2017 Baris Ungun, Anqi Fu

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

from conrad.medicine.prescription import eval_constraint
from conrad.optimization.objectives import OPTKIT_INSTALLED
from conrad.optimization.preprocessing import *
from conrad.tests.base import *

class ObjectiveMethodsTestCase(ConradTestCase):
	def setUp(self):
		self.m, self.n = 100, 40
		self.A = np.random.rand(self.m, self.n)
		self.x = np.random.rand(self.n)
		self.y = np.dot(self.A, self.x)
		self.nu = np.random.rand(self.m)

		self.size = self.m
		self.target = Structure(0, 'TARGET', True, size=self.size)
		self.oar = Structure(1, 'OAR', False, size=self.size)
		self.voxel_weights = (1 + 10 * np.random.rand(self.size)).astype(int)
		self.constraints = [
				eval_constraint('Dmean < 20Gy'),
				eval_constraint('D90 < 25Gy')
		]

	def test_normalize(self):
		target = self.target
		ObjectiveMethods.normalize(target)
		self.assertEqual( 1./target.size, target.objective.normalization )

		norm = np.sum(self.voxel_weights)
		target.voxel_weights = self.voxel_weights
		ObjectiveMethods.normalize(target)
		self.assertEqual( 1./norm, target.objective.normalization )

		oar = self.oar
		ObjectiveMethods.normalize(oar)
		self.assertEqual( 1, oar.objective.normalization )

		oar.constraints += self.constraints[0]
		ObjectiveMethods.normalize(oar)
		self.assertEqual( 1, oar.objective.normalization )

		oar.constraints += self.constraints[1]
		ObjectiveMethods.normalize(oar)
		self.assertEqual( 1./oar.size, oar.objective.normalization )

	def test_primal_eval(self):
		# target:
		# assign dose from voxel data

		self.target.A_full = self.A
		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.target.voxel_weights = wt

			f = ObjectiveMethods.eval(self.target, self.y)
			expect = self.target.objective.eval(self.y, wt)
			self.assert_scalar_equal( f, expect )

			# calculate dose from beam intensities
			f = ObjectiveMethods.eval(self.target, x=self.x)
			self.assert_scalar_equal( f, expect )

		# oar:
		# unweighted, unconstrained
		self.oar.A_full = self.A
		self.oar.assign_dose(self.y)
		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.oar.voxel_weights = wt

			self.oar.constraints.clear()
			f_collapse = ObjectiveMethods.eval(self.oar)

			self.oar.constraints += self.constraints[1]
			f_full = ObjectiveMethods.eval(self.oar)

			self.assert_scalar_equal( f_collapse, f_full, 5e-3, 5e-3 )

	def test_dual_eval(self):
		# unweighted
		ff = ObjectiveMethods.dual_eval(self.target, self.nu)
		self.assert_scalar_equal( ff, self.target.objective.dual_eval(self.nu) )

		# weighted
		self.target.voxel_weights = self.voxel_weights
		ff_wt = ObjectiveMethods.dual_eval(self.target, self.nu)
		self.assert_scalar_equal(
				ff_wt,
				self.target.objective.dual_eval(self.nu, self.voxel_weights) )

		# oar:
		# unweighted, unconstrained
		self.oar.A_full = self.A
		self.oar.assign_dose(self.y)
		ff_collapse = ObjectiveMethods.dual_eval(self.oar, self.nu)

		# unweighted, constrained
		self.oar.constraints += self.constraints[1]
		ff_full = ObjectiveMethods.dual_eval(self.oar, self.nu)

		# compare: collapsed vs. uncollapsed
		self.assert_scalar_equal( ff_collapse, ff_full )

		# weighted, unconstrained
		self.oar.voxel_weights = self.voxel_weights
		self.oar.constraints.clear()
		ff_collapse = ObjectiveMethods.dual_eval(self.oar, self.nu)

		# weighted, constrained
		self.oar.constraints += self.constraints[1]
		ff_full = ObjectiveMethods.dual_eval(self.oar, self.nu)

		# compare: collapsed vs. uncollapsed
		self.assertEqual( ff_collapse, ff_full )

	def test_primal_expr(self):
		y_var = cvxpy.Variable(self.m)
		x_var = cvxpy.Variable(self.n)
		y_var.save_value(self.y)
		x_var.save_value(self.x)

		# target
		# unweighted, weighted
		self.target.A_full = self.A

		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.target.voxel_weights = wt

			f_y = cvxpy.Minimize(ObjectiveMethods.expr(self.target, y_var))
			expect = cvxpy.Minimize(self.target.objective.expr(y_var, wt))
			self.assert_scalar_equal( f_y.value, expect.value )

			f_Ax = cvxpy.Minimize(ObjectiveMethods.expr(self.target, x_var))
			self.assert_scalar_equal( f_Ax.value, expect.value )

		# nontarget
		# unweighted, weighted: collapsed vs. full
		y_mean_var = cvxpy.Variable(1)
		self.oar.A_full = self.A

		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.oar.voxel_weights = wt
				y_mean_var.save_value(np.dot(wt, self.y) / np.sum(wt))
			else:
				y_mean_var.save_value(np.mean(self.y))

			self.oar.constraints.clear()
			f_col_y = cvxpy.Minimize(ObjectiveMethods.expr(self.oar, y_mean_var))
			f_col_Ax = cvxpy.Minimize(ObjectiveMethods.expr(self.oar, x_var))
			self.oar.constraints += self.constraints[1]
			f_full = cvxpy.Minimize(ObjectiveMethods.expr(self.oar, y_var))
			self.assert_scalar_equal( f_col_y.value, f_col_Ax.value )
			self.assert_scalar_equal( f_col_y.value, f_full.value, 1e-3, 1e-3 )
			# TODO: WHY IS THIS LINE A PROBLEM???????

	def test_dual_expr(self):
		nu_var = cvxpy.Variable(self.m)
		nu_var.save_value(np.random.rand(self.m))

		# target
		# unweighted, weighted
		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.target.voxel_weights = wt

			ff_y = cvxpy.Maximize(ObjectiveMethods.dual_expr(
					self.target, nu_var))
			expect = cvxpy.Maximize(self.target.objective.dual_expr(
					nu_var, wt))
			self.assert_scalar_equal( ff_y.value, expect.value )

		# nontarget
		nu_mean_var = cvxpy.Variable(1)
		nu_mean_var.save_value(np.mean(nu_var.value))

		# unweighted, weighted: collapsed vs. full
		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.oar.voxel_weights = wt

			self.oar.constraints.clear()
			f_col = cvxpy.Maximize(ObjectiveMethods.dual_expr(
					self.oar, nu_mean_var))
			self.oar.constraints += self.constraints[1]
			f_full = cvxpy.Maximize(ObjectiveMethods.dual_expr(
					self.oar, nu_var))
			self.assert_scalar_equal( f_col.value, f_full.value )

	def __assert_pogs_function_vectors_equal(self, first, second):
		self.assert_vector_equal( first.h, second.h )
		self.assert_vector_equal( first.a, second.a )
		self.assert_vector_equal( first.b, second.b )
		self.assert_vector_equal( first.c, second.c )
		self.assert_vector_equal( first.d, second.d )
		self.assert_vector_equal( first.e, second.e )

	def test_primal_expr_pogs(self):
		if not OPTKIT_INSTALLED:
			with self.assertRaises(NotImplementedError):
				f.ObjectiveMethods.primal_expr_pogs(self.target)
			return

		# target
		# unweighted, weighted
		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.target.voxel_weights = wt

			f = ObjectiveMethods.primal_expr_pogs(self.target)
			expect = self.target.objective.primal_expr_pogs(
					self.target.size, wt)
			self.__assert_pogs_function_vectors_equal( f, expect )

		# oar
		# unweighted, weighted: collapsed vs. full
		self.oar.assign_dose(self.y)

		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.oar.voxel_weights = wt

			self.oar.constraints.clear()
			f_collapse = ObjectiveMethods.primal_expr_pogs(self.oar)
			self.oar.constraints += self.constraints[1]
			f_full = ObjectiveMethods.primal_expr_pogs(self.oar)
			self.assert_scalar_equal(
					f_collapse.eval(np.array([self.oar.y_mean])),
					f_full.eval(self.oar.y), 1e-3, 1e-3 )

	def test_dual_expr_pogs(self):
		if not OPTKIT_INSTALLED:
			with self.assertRaises(NotImplementedError):
				f.ObjectiveMethods.primal_expr_pogs(self.target)
			return

		# target
		# unweighted, weighted
		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.target.voxel_weights = wt

			ff = ObjectiveMethods.dual_expr_pogs(self.target)
			expect = self.target.objective.dual_expr_pogs(
					self.target.size, wt)
			self.__assert_pogs_function_vectors_equal( ff, expect )

		# oar
		# unweighted, weighted: collapsed vs. full
		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.oar.voxel_weights = wt

			self.oar.constraints.clear()
			ff_collapse = ObjectiveMethods.dual_expr_pogs(self.oar)
			self.oar.constraints += self.constraints[1]
			ff_full = ObjectiveMethods.dual_expr_pogs(self.oar)
			if ff_collapse.size == 0:
				self.assertEqual( ff_full.size, 0 )
			else:
				self.assert_scalar_equal(
						ff_collapse.eval(np.array([self.oar.y_mean])),
						ff_full.eval(self.oar.y) )

	def test_dual_domain_constraints(self):
		nu_var = cvxpy.Variable(self.m)
		nu_var.save_value(np.random.rand(self.m))

		# target
		# unweighted, weighted
		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.target.voxel_weights = wt

			constr = ObjectiveMethods.dual_domain_constraints(
					self.target, nu_var)

		# oar
		# unweighted, weighted: compressed vs. full
		for wt in [None, self.voxel_weights]:
			if wt is not None:
				self.oar.voxel_weights = wt

			self.oar.constraints.clear()
			constr_col = ObjectiveMethods.dual_domain_constraints(
					self.target, nu_var)
			self.oar.constraints += self.constraints[1]
			constr_full = ObjectiveMethods.dual_domain_constraints(
					self.target, nu_var)
			for i, c in enumerate(constr_full):
				self.assertTrue( c.value == constr_col[i].value )

	def test_dual_domain_constraints_pogs(self):
		with self.assertRaises(NotImplementedError):
			ObjectiveMethods.dual_domain_constraints_pogs(self.target)
		with self.assertRaises(NotImplementedError):
			ObjectiveMethods.dual_domain_constraints_pogs(self.oar)

		# in the future:
		# -- branch 0: not installed
		# if not OPTKIT_INSTALLED:
		# 	with self.assertRaises(NotImplementedError):
		# 		f.ObjectiveMethods.primal_expr_pogs(self.target)
		# 	return
		# -- branch 1: installed
			# unweighted
			# weighted
			# collapsed, unweighted
			# full, unweighted
			# compare: collapsed vs. uncollapsed
			# collapsed, weighted
			# full, weighted
			# compare: collapsed vs. uncollapsed
