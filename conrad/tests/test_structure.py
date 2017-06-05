"""
Unit tests for :mod:`conrad.medicine.structure`.
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
import scipy.sparse as sp

from conrad.defs import CONRAD_DEBUG_PRINT
from conrad.medicine.structure import *
from conrad.medicine.dose import D, Gy, PercentileConstraint
from conrad.tests.base import *

class StructureTestCase(ConradTestCase):
	def test_create_structure_minimal(self):
		# declare target structure
		s1 = Structure('LABEL', 'STRUCTURE NAME', True)
		self.assertEqual( s1.label, 'LABEL')
		self.assertEqual( s1.name, 'STRUCTURE NAME' )
		self.assertTrue( s1.is_target )
		self.assertIsInstance( s1.objective, TargetObjectivePWL )

		self.assertIsNone( s1.size )
		self.assertFalse( s1.collapsable )
		self.assertIsNone( s1.A_full )
		self.assertIsNone( s1.A_mean )
		self.assertIsNone( s1.A )
		self.assertEqual( s1.dose_rx, 1. * Gy )
		self.assertEqual( s1.dose, 1. * Gy )
		self.assertIsNone( s1.y )
		self.assert_nan( s1.mean_dose.value )
		self.assert_nan( s1.min_dose.value )
		self.assert_nan( s1.max_dose.value )
		self.assertIsNone( s1.dvh )
		self.assertIsInstance( s1.constraints, ConstraintList )
		self.assertFalse( s1.plannable )

		# declare non-target structure
		s2 = Structure('NEXT LABEL', 'NEXT NAME', False)
		self.assertEqual( s2.label, 'NEXT LABEL')
		self.assertEqual( s2.name, 'NEXT NAME' )
		self.assertFalse( s2.is_target )
		self.assertIsInstance( s2.objective, NontargetObjectiveLinear )

		self.assertIsNone( s2.size )
		self.assertTrue( s2.collapsable )
		self.assertIsNone( s2.A_full )
		self.assertIsNone( s2.A_mean )
		self.assertIsNone( s2.A )
		self.assertEqual( s2.dose_rx.value, 0. )
		self.assertEqual( s2.dose.value, 0. )
		self.assertIsNone( s2.y )
		self.assert_nan( s2.mean_dose.value )
		self.assert_nan( s2.min_dose.value )
		self.assert_nan( s2.max_dose.value )
		self.assertIsNone( s2.dvh )
		self.assertIsInstance( s2.constraints, ConstraintList )
		self.assertFalse( s2.plannable )

	def test_set_size(self):
		s = Structure('LABEL', 'STRUCTURE NAME', True)
		self.assertIsNone( s.size )
		s.size = 500
		self.assertEqual( s.size, 500 )
		self.assert_scalar_notequal(
				s.objective.w_under, s.objective.w_under_raw, 1e-3, 1e-3)
		self.assert_scalar_notequal(
				s.objective.w_over, s.objective.w_over_raw, 1e-3, 1e-3)
		self.assertFalse( s.plannable )
		self.assertIsInstance( s.dvh, DVH )

		# test exception handling
		with self.assertRaises(ValueError):
			s.size = 0

		with self.assertRaises(ValueError):
			s.size = -3

		with self.assertRaises(ValueError):
			s.size = 'random_string'

	def test_change_weights(self):
		s = Structure('LABEL', 'STRUCTURE NAME', True)
		s.objective.w_under = 3.
		s.objective.w_over = 5

		self.assertEqual( s.objective.w_under_raw, 3. )
		self.assertEqual( s.objective.w_over_raw, 5. )

		# test exception handling
		with self.assertRaises(ValueError):
			s.objective.w_under = -3

		with self.assertRaises(ValueError):
			s.objective.w_over = -3

		with self.assertRaises(ValueError):
			s.objective.w_under = 'random_string'

		with self.assertRaises(ValueError):
			s.objective.w_over = 'random_string'

	def test_change_dose(self):
		target = Structure('LABEL', 'STRUCTURE NAME', True)
		target.dose = 17. * Gy
		self.assertEqual( target.dose.value, 17. )
		self.assertEqual( target.dose_rx.value, 1. )
		target.dose_rx = 17. * Gy
		self.assertEqual( target.dose.value, 17. )
		self.assertEqual( target.dose_rx.value, 17. )
		self.assertEqual( target.dose, target.objective.dose )

		nontarget = Structure('LABEL', 'STRUCTURE NAME', False)
		nontarget.dose = 17. * Gy
		self.assertEqual( nontarget.dose.value, 0. )
		self.assertEqual( nontarget.dose_rx.value, 0. )
		nontarget.dose_rx = 17. * Gy
		self.assertEqual( nontarget.dose.value, 0. )
		self.assertEqual( nontarget.dose_rx.value, 0. )

		# test exception handling
		with self.assertRaises(ValueError):
			target.dose = 0 * Gy

		with self.assertRaises(ValueError):
			target.dose = -3 * Gy

		with self.assertRaises(TypeError):
			target.dose = 3

		with self.assertRaises(TypeError):
			target.dose = 'random_string'

	def test_add_matrix(self):
		s = Structure('LABEL', 'STRUCTURE NAME', True)
		self.assertFalse( s.plannable )
		s.A_full = np.random.rand(100, 300)
		self.assertEqual( s.size, 100 )
		self.assertIsInstance( s.dvh, DVH )
		self.assertTrue( s.plannable )
		self.assertIsNotNone( s.A_mean )
		self.assertEqual( len(s.A_mean), 300 )
		s.size = 120 # inconsistent size, no longer plannable
		self.assertFalse( s.plannable )
		s.size = 100
		self.assertTrue( s.plannable )

		# set CSR matrix

		s.reset_matrices()
		s.A_full = sp.rand(100, 50, 0.3, 'csr')
		self.assertIsNotNone( s.A_mean )
		self.assertEqual( len(s.A_mean), 50 )

		s.reset_matrices()
		s.A_full = sp.rand(100, 70, 0.3, 'csc')
		self.assertIsNotNone( s.A_mean )
		self.assertEqual( len(s.A_mean), 70 )

		with self.assertRaises(TypeError):
			s.reset_matrices()
			s.A_full = np.random.rand()

		with self.assertRaises(TypeError):
			s.reset_matrices()
			s.A_full = 'random_string'

		with self.assertRaises(ValueError):
			s.reset_matrices()
			s.A_full = np.random.rand(50, 300)

	def test_add_mean_matrix(self):
		s = Structure('LABEL', 'STRUCTURE NAME', True)
		self.assertFalse( s.plannable )
		s.A_mean = np.random.rand(40)
		self.assertFalse( s.plannable )
		s.size = 500
		# TARGET STRUCTURES ARE NOT COLLAPSABLE, MEAN MATRIX INSUFFICIENT
		self.assertFalse( s.plannable )

		s = Structure('LABEL', 'STRUCTURE NAME', False)
		self.assertFalse( s.plannable )
		s.A_mean = np.random.rand(40)
		self.assertFalse( s.plannable )
		s.size = 500
		# NON-TARGETS ARE COLLAPSABLE, MEAN MATRIX SUFFICES FOR PLANNING
		self.assertTrue( s.plannable )

	# 	# test exception handling
		with self.assertRaises(TypeError):
			s.A_mean = np.random.rand()

		with self.assertRaises(TypeError):
			s.A_mean = 'random_string'

	def test_reset_matrices(self):
		s = Structure('LABEL', 'STRUCTURE NAME', True)
		s.A_full = np.random.rand(50, 300)
		self.assertIsNotNone( s.A_full )
		self.assertIsNotNone( s.A_mean )
		s.reset_matrices()
		self.assertIsNone( s.A_full )
		self.assertIsNone( s.A_mean )

	def test_create_structure_options(self):
		# dense
		s = Structure('LABEL', 'NAME', True, size=400, dose=17 * Gy,
					  A=np.random.rand(400, 50), w_under=12, w_over=2)
		self.assertEqual( s.size, 400 )
		self.assertEqual( s.dose.value, 17. )
		self.assertEqual( s.objective.dose, s.dose )
		self.assertEqual( s.dose_rx.value, 17. )
		self.assertIsNotNone( s.A_full )
		self.assertIsNotNone( s.A_mean )
		self.assertEqual( s.objective.w_under_raw, 12. )
		self.assertEqual( s.objective.w_over_raw, 2. )
		self.assertTrue( s.plannable )
		del s

		# sparse CSR
		s = Structure('LABEL', 'NAME', True, size=400, dose=16 * Gy,
					  A=sp.rand(400, 50, 0.3, 'csr'), w_under=13, w_over=3)
		self.assertEqual( s.size, 400 )
		self.assertEqual( s.dose.value, 16. )
		self.assertEqual( s.objective.dose, s.dose )
		self.assertEqual( s.dose_rx.value, 16. )
		self.assertIsNotNone( s.A_full )
		self.assertIsNotNone( s.A_mean )
		self.assertEqual( s.objective.w_under_raw, 13. )
		self.assertEqual( s.objective.w_over_raw, 3. )
		self.assertTrue( s.plannable )
		del s

		# sparse CSC
		s = Structure('LABEL', 'NAME', True, size=400, dose=15 * Gy,
					  A=sp.rand(400, 50, 0.3, 'csc'), w_under=14, w_over=4)
		self.assertEqual( s.size, 400 )
		self.assertEqual( s.dose.value, 15. )
		self.assertEqual( s.objective.dose, s.dose )
		self.assertEqual( s.dose_rx.value, 15. )
		self.assertIsNotNone( s.A_full )
		self.assertIsNotNone( s.A_mean )
		self.assertEqual( s.objective.w_under_raw, 14. )
		self.assertEqual( s.objective.w_over_raw, 4. )
		self.assertTrue( s.plannable )
		del s

		# mean only, target
		s = Structure('LABEL', 'NAME', True, size=400, dose=14 * Gy,
					  A_mean=np.random.rand(50), w_under=15, w_over=5)
		self.assertEqual( s.size, 400 )
		self.assertEqual( s.dose.value, 14. )
		self.assertEqual( s.objective.dose, s.dose )
		self.assertEqual( s.dose_rx.value, 14. )
		self.assertIsNone( s.A_full )
		self.assertIsNotNone( s.A_mean )
		self.assertEqual( s.objective.w_under_raw, 15. )
		self.assertEqual( s.objective.w_over_raw, 5. )
		self.assertFalse( s.plannable )
		del s

		# mean only, non-target
		s = Structure('LABEL', 'NAME', False, size=400, dose=13 * Gy,
					  A_mean=np.random.rand(50), w_over=5)
		self.assertEqual( s.size, 400 )
		self.assertEqual( s.dose.value, 0. )
		self.assertEqual( s.dose_rx.value, 0. )
		self.assertIsNone( s.A_full )
		self.assertIsNotNone( s.A_mean )
		self.assertEqual( s.objective.w_over_raw, 5. )
		self.assertTrue( s.plannable )
		del s

		# compatible full and mean
		s = Structure('LABEL', 'NAME', True, size=400, dose=12 * Gy,
					  A=np.random.rand(400, 50), A_mean=np.random.rand(50),
					  w_under=16, w_over=6)
		self.assertEqual( s.size, 400 )
		self.assertEqual( s.dose.value, 12. )
		self.assertEqual( s.objective.dose, s.dose )
		self.assertEqual( s.dose_rx.value, 12. )
		self.assertIsNotNone( s.A_full )
		self.assertIsNotNone( s.A_mean )
		self.assertEqual( s.objective.w_under_raw, 16. )
		self.assertEqual( s.objective.w_over_raw, 6. )
		self.assertTrue( s.plannable )
		del s

		# test exception handling:
		# incompatible size and matrix row #
		with self.assertRaises(ValueError):
			s = Structure('LABEL', 'NAME', True, size=401, dose=17 * Gy,
						  A=np.random.rand(400, 50), w_under=12, w_over=2)

		# incompatible compatible full and mean
		with self.assertRaises(ValueError):
			s = Structure('LABEL', 'NAME', True, size=400, dose=13 * Gy,
						  A=np.random.rand(400, 50), A_mean=np.random.rand(40),
						  w_under=16, w_over=6)

	def test_calculate_dose(self):
		m, n = 400, 50
		A = np.random.rand(m, n)
		s = Structure('LABEL', 'NAME', True, A=A)

		x = np.random.rand(n)
		Ax = A.dot(x)

		s.calc_y(x)
		self.assert_vector_equal(Ax, s.y, 1e-7, 1e-7)
		self.assert_scalar_equal(Ax.mean(), s.mean_dose.value, 1e-7, 1e-7)
		self.assert_scalar_equal(Ax.max(), s.max_dose.value, 1e-7, 1e-7)
		self.assert_scalar_equal(Ax.min(), s.min_dose.value, 1e-7, 1e-7)

	def test_assign_dose(self):
		m, n = 400, 50
		y = np.random.rand(m)
		s = Structure('LABEL', 'NAME', True, A=np.random.rand(m, n))

		s.assign_dose(y)
		self.assert_vector_equal( y, s.y, 1e-7, 1e-7 )

		with self.assertRaises(ValueError):
			s.assign_dose(np.random.rand(m + 2))

	def test_add_constraints(self):
		s = Structure('LABEL', 'NAME', True)
		s.constraints += D(80) > 30 * Gy
		cid = s.constraints.last_key
		self.assertIsInstance( s.constraints[cid], PercentileConstraint )
		self.assertEqual( s.constraints[cid].threshold.value, 80 )
		self.assertEqual( s.constraints[cid].dose.value, 30 )
		self.assertEqual( s.constraints[cid].relop, RELOPS.GEQ )

		# indirect syntax for constraint modification
		s.set_constraint(cid, dose=40 * Gy)
		self.assertEqual( s.constraints[cid].dose.value, 40 )

		s.set_constraint(cid, relop='<')
		self.assertEqual( s.constraints[cid].relop, RELOPS.LEQ )

		s.set_constraint(cid, threshold=70)
		self.assertEqual( s.constraints[cid].threshold.value, 70 )

		# operator overloaded (i.e., direct) syntax
		s.constraints[cid] > 30 * Gy
		self.assertEqual( s.constraints[cid].dose.value, 30 )
		self.assertEqual( s.constraints[cid].relop, RELOPS.GEQ )

	def test_constraint_satisfaction(self):
		m = 500
		y = np.zeros(m)
		y[:int(0.8 * m)] = 10 * np.random.rand(int(0.8 * m))
		y[int(0.8 * m):] = 20 * np.random.rand(m - int(0.8 * m))

		s = Structure('LABEL', 'NAME', True, size=500)

		# load dose directly from vector "y" by assigning identity matrix
		# as dose matrix
		s.A_full = np.eye(m)
		s.calc_y(y)

		s.constraints += D(20) < 10 * Gy
		cid = s.constraints.last_key

		self.assertTrue( s.satisfies(s.constraints[cid])[0] )

		y += 10.
		s.calc_y(y)

		self.assertFalse( s.satisfies(s.constraints[cid])[0] )

		y -= 10
		s.calc_y(y)

		s.constraints += D(20) < 10 * Gy
		s.constraints += D(20) < 10 * Gy

		# multiple constraint check for a ConstraintList
		self.assertTrue( s.satisfies_all(s.constraints) )

	def test_plotting_data(self):
		m, n = 500, 30
		x = np.random.rand(n)
		A = np.random.rand(m, n)
		s = Structure('LABEL', 'Name', True, A=A)
		s.calc_y(x)

		s.constraints += D(40) >= 30 * Gy
		s.dose_rx = 42 * Gy
		self.assertIsInstance( s.plotting_data(), dict )
		self.assertIn( 'curve', s.plotting_data() )
		self.assertIn( 'constraints', s.plotting_data() )
		self.assertIn( 'rx', s.plotting_data() )
		self.assertIn( 'target', s.plotting_data() )

		maxlength = 25
		self.assertLessEqual(
				len(s.plotting_data(maxlength=25)['curve']['dose']),
				maxlength + 2 )

	def test_print_objective(self):
		s = Structure('LABEL', 'NAME', True)
		self.assertIn( 'Structure: NAME (label = LABEL)', s.objective_string )
		self.assertIn( 'target? True', s.objective_string )
		self.assertIn( 'rx dose: 1.0 Gy', s.objective_string )
		self.assertIn( 'weight_underdose: 1.0', s.objective_string )
		self.assertIn( 'weight_overdose: 0.05', s.objective_string )

		s = Structure('LABEL', 'NAME', False)
		self.assertIn( 'target? False', s.objective_string )
		self.assertIn( 'rx dose: 0.0 Gy', s.objective_string )
		self.assertIn( 'weight: 0.03', s.objective_string )

	def test_print_constraints(self):
		s = Structure('LABEL', 'NAME', True)
		s.constraints += D(80) > 30 * Gy
		s.constraints += D(99) < 50 * Gy

		self.assertIn(
				'Structure: NAME (label = LABEL)', s.constraints_string )
		for k in s.constraints:
			self.assertIn( str(s.constraints[k]), s.constraints_string )

	def test_print_summary(self):
		m, n = 500, 30
		s = Structure('LABEL', 'NAME', True)
		x = np.random.rand(n)
		A = np.random.rand(m, n)
		s = Structure('LABEL', 'Name', True, A=A)
		s.calc_y(x)
		self.assertIn( 'mean', s.summary_string )
		self.assertIn( 'min', s.summary_string )
		self.assertIn( 'max', s.summary_string )
		self.assertIn( 'D98', s.summary_string )
		self.assertIn( 'D75', s.summary_string )
		self.assertIn( 'D25', s.summary_string )
		self.assertIn( 'D2', s.summary_string )