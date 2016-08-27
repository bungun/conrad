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
		self.assertTrue( s1.label == 'LABEL')
		self.assertTrue( s1.name == 'STRUCTURE NAME' )
		self.assertTrue( s1.is_target )

		self.assertTrue( s1.size is None )
		self.assertFalse( s1.collapsable )
		self.assertTrue( s1.A_full is None )
		self.assertTrue( s1.A_mean is None )
		self.assertTrue( s1.A is None )
		self.assertTrue( s1.dose_rx == 1. * Gy )
		self.assertTrue( s1.dose == 1. * Gy )
		self.assert_nan( s1.w_under )
		self.assert_nan( s1.w_over )
		self.assertTrue( s1.w_under_raw == W_UNDER_DEFAULT )
		self.assertTrue( s1.w_over_raw == W_OVER_DEFAULT )
		self.assertTrue( s1.y is None )
		self.assert_nan( s1.mean_dose.value )
		self.assert_nan( s1.min_dose.value )
		self.assert_nan( s1.max_dose.value )
		self.assertTrue( s1.dvh is None )
		self.assertTrue( isinstance(s1.constraints, ConstraintList) )
		self.assertFalse( s1.plannable )

		# declare non-target structure
		s2 = Structure('NEXT LABEL', 'NEXT NAME', False)
		self.assertTrue( s2.label == 'NEXT LABEL')
		self.assertTrue( s2.name == 'NEXT NAME' )
		self.assertFalse( s2.is_target )

		self.assertTrue( s2.size is None )
		self.assertTrue( s2.collapsable )
		self.assertTrue( s2.A_full is None )
		self.assertTrue( s2.A_mean is None )
		self.assertTrue( s2.A is None )
		self.assertTrue( s2.dose_rx.value == 0. )
		self.assertTrue( s2.dose.value == 0. )
		self.assert_nan( s2.w_under )
		self.assert_nan( s2.w_over )
		self.assertTrue( s2.w_under_raw == 0. )
		self.assertTrue( s2.w_over_raw == W_NONTARG_DEFAULT )
		self.assertTrue( s2.y is None )
		self.assert_nan( s2.mean_dose.value )
		self.assert_nan( s2.min_dose.value )
		self.assert_nan( s2.max_dose.value )
		self.assertTrue( s2.dvh is None )
		self.assertTrue( isinstance(s2.constraints, ConstraintList) )
		self.assertFalse( s2.plannable )

	def test_set_size(self):
		s = Structure('LABEL', 'STRUCTURE NAME', True)
		self.assertTrue( s.size is None )
		s.size = 500
		self.assertTrue( s.size == 500 )
		self.assert_scalar_notequal(s.w_under, s.w_under_raw, 1e-3, 1e-3)
		self.assert_scalar_notequal(s.w_over, s.w_over_raw, 1e-3, 1e-3)
		self.assertFalse( s.plannable )
		self.assertTrue( isinstance(s.dvh, DVH) )

		# test exception handling
		try:
			s.size = 0
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			s.size = -3
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			s.size = 'random_string'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

	def test_change_weights(self):
		s = Structure('LABEL', 'STRUCTURE NAME', True)
		s.w_under = 3.
		s.w_over = 5
		self.assertTrue( s.w_under_raw == 3. )
		self.assertTrue( s.w_over_raw == 5. )

		# test exception handling
		try:
			s.w_under = -3
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			s.w_over = -3
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			s.w_under = 'random_string'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			s.w_over = 'random_string'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

	def test_change_dose(self):
		target = Structure('LABEL', 'STRUCTURE NAME', True)
		target.dose = 17. * Gy
		self.assertTrue( target.dose.value == 17. )
		self.assertTrue( target.dose_rx.value == 1. )
		target.dose_rx = 17. * Gy
		self.assertTrue( target.dose.value == 17. )
		self.assertTrue( target.dose_rx.value == 17. )

		nontarget = Structure('LABEL', 'STRUCTURE NAME', False)
		nontarget.dose = 17. * Gy
		self.assertTrue( nontarget.dose.value == 0. )
		self.assertTrue( nontarget.dose_rx.value == 0. )
		nontarget.dose_rx = 17. * Gy
		self.assertTrue( nontarget.dose.value == 0. )
		self.assertTrue( nontarget.dose_rx.value == 0. )

		# test exception handling
		try:
			target.dose = 0 * Gy
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			target.dose = -3 * Gy
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			target.dose = 3
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			target.dose = 'random_string'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

	def test_add_matrix(self):
		s = Structure('LABEL', 'STRUCTURE NAME', True)
		self.assertFalse( s.plannable )
		s.A_full = np.random.rand(100, 300)
		self.assertTrue( s.size == 100 )
		self.assertTrue( isinstance(s.dvh, DVH) )
		self.assertTrue( s.plannable )
		self.assertTrue( s.A_mean is not None )
		self.assertTrue( len(s.A_mean) == 300 )
		s.size = 120 # inconsistent size, no longer plannable
		self.assertFalse( s.plannable )
		s.size = 100
		self.assertTrue( s.plannable )

		# set CSR matrix
		try:
			s.reset_matrices()
			s.A_full = sp.rand(100, 50, 0.3, 'csr')
			self.assertTrue( s.A_mean is not None )
			self.assertTrue( len(s.A_mean) == 50 )
		except:
			self.assertTrue( False )

		# set CSC matrix
		try:
			s.reset_matrices()
			s.A_full = sp.rand(100, 70, 0.3, 'csc')
			self.assertTrue( s.A_mean is not None )
			self.assertTrue( len(s.A_mean) == 70 )
		except:
			self.assertTrue( False )

		# test exception handling
		try:
			s.reset_matrices()
			s.A_full = np.random.rand()
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			s.reset_matrices()
			s.A_full = 'random_string'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			s.reset_matrices()
			s.A_full = np.random.rand(50, 300)
			self.assertTrue( False )
		except:
			self.assertTrue( True )

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

		# test exception handling
		try:
			s.A_mean = np.random.rand()
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			s.A_mean = 'random_string'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

	def test_reset_matrices(self):
		s = Structure('LABEL', 'STRUCTURE NAME', True)
		s.A_full = np.random.rand(50, 300)
		self.assertTrue( s.A_full is not None )
		self.assertTrue( s.A_mean is not None )
		s.reset_matrices()
		self.assertTrue( s.A_full is None )
		self.assertTrue( s.A_mean is None )

	def test_create_structure_options(self):
		# dense
		s = Structure('LABEL', 'NAME', True, size=400, dose=17 * Gy,
					  A=np.random.rand(400, 50), w_under=12, w_over=2)
		self.assertTrue( s.size == 400 )
		self.assertTrue( s.dose.value == 17. )
		self.assertTrue( s.dose_rx.value == 17. )
		self.assertTrue( s.A_full is not None )
		self.assertTrue( s.A_mean is not None )
		self.assertTrue( s.w_under_raw == 12. )
		self.assertTrue( s.w_over_raw == 2. )
		self.assertTrue( s.plannable )
		del s

		# sparse CSR
		s = Structure('LABEL', 'NAME', True, size=400, dose=16 * Gy,
					  A=sp.rand(400, 50, 0.3, 'csr'), w_under=13, w_over=3)
		self.assertTrue( s.size == 400 )
		self.assertTrue( s.dose.value == 16. )
		self.assertTrue( s.dose_rx.value == 16. )
		self.assertTrue( s.A_full is not None )
		self.assertTrue( s.A_mean is not None )
		self.assertTrue( s.w_under_raw == 13. )
		self.assertTrue( s.w_over_raw == 3. )
		self.assertTrue( s.plannable )
		del s

		# sparse CSC
		s = Structure('LABEL', 'NAME', True, size=400, dose=15 * Gy,
					  A=sp.rand(400, 50, 0.3, 'csc'), w_under=14, w_over=4)
		self.assertTrue( s.size == 400 )
		self.assertTrue( s.dose.value == 15. )
		self.assertTrue( s.dose_rx.value == 15. )
		self.assertTrue( s.A_full is not None )
		self.assertTrue( s.A_mean is not None )
		self.assertTrue( s.w_under_raw == 14. )
		self.assertTrue( s.w_over_raw == 4. )
		self.assertTrue( s.plannable )
		del s

		# mean only, target
		s = Structure('LABEL', 'NAME', True, size=400, dose=14 * Gy,
					  A_mean=np.random.rand(50), w_under=15, w_over=5)
		self.assertTrue( s.size == 400 )
		self.assertTrue( s.dose.value == 14. )
		self.assertTrue( s.dose_rx.value == 14. )
		self.assertTrue( s.A_full is None )
		self.assertTrue( s.A_mean is not None )
		self.assertTrue( s.w_under_raw == 15. )
		self.assertTrue( s.w_over_raw == 5. )
		self.assertFalse( s.plannable )
		del s

		# mean only, non-target
		s = Structure('LABEL', 'NAME', False, size=400, dose=13 * Gy,
					  A_mean=np.random.rand(50), w_under=15, w_over=5)
		self.assertTrue( s.size == 400 )
		self.assertTrue( s.dose.value == 0. )
		self.assertTrue( s.dose_rx.value == 0. )
		self.assertTrue( s.A_full is None )
		self.assertTrue( s.A_mean is not None )
		self.assertTrue( s.w_under_raw == 0. )
		self.assertTrue( s.w_over_raw == 5. )
		self.assertTrue( s.plannable )
		del s

		# compatible full and mean
		s = Structure('LABEL', 'NAME', True, size=400, dose=12 * Gy,
					  A=np.random.rand(400, 50), A_mean=np.random.rand(50),
					  w_under=16, w_over=6)
		self.assertTrue( s.size == 400 )
		self.assertTrue( s.dose.value == 12. )
		self.assertTrue( s.dose_rx.value == 12. )
		self.assertTrue( s.A_full is not None )
		self.assertTrue( s.A_mean is not None )
		self.assertTrue( s.w_under_raw == 16. )
		self.assertTrue( s.w_over_raw == 6. )
		self.assertTrue( s.plannable )
		del s

		# test exception handling:
		# incompatible size and matrix row #
		try:
			s = Structure('LABEL', 'NAME', True, size=401, dose=17 * Gy,
						  A=np.random.rand(400, 50), w_under=12, w_over=2)
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		# incompatible compatible full and mean
		try:
			s = Structure('LABEL', 'NAME', True, size=400, dose=13 * Gy,
						  A=np.random.rand(400, 50), A_mean=np.random.rand(40),
						  w_under=16, w_over=6)
			self.assertTrue( False )
		except:
			self.assertTrue( True )

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

	def test_add_constraints(self):
		s = Structure('LABEL', 'NAME', True)
		s.constraints += D(80) > 30 * Gy
		cid = s.constraints.last_key
		self.assertTrue( isinstance(s.constraints[cid], PercentileConstraint) )
		self.assertTrue( s.constraints[cid].threshold.value == 80 )
		self.assertTrue( s.constraints[cid].dose.value == 30 )
		self.assertTrue( s.constraints[cid].relop == RELOPS.GEQ )

		# indirect syntax for constraint modification
		s.set_constraint(cid, dose=40 * Gy)
		self.assertTrue( s.constraints[cid].dose.value == 40 )

		s.set_constraint(cid, relop='<')
		self.assertTrue( s.constraints[cid].relop == RELOPS.LEQ )

		s.set_constraint(cid, threshold=70)
		self.assertTrue( s.constraints[cid].threshold.value == 70 )

		# operator overloaded (i.e., direct) syntax
		s.constraints[cid] > 30 * Gy
		self.assertTrue( s.constraints[cid].dose.value == 30 )
		self.assertTrue( s.constraints[cid].relop == RELOPS.GEQ )

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

	def test_plotting_data(self):
		m, n = 500, 30
		x = np.random.rand(n)
		A = np.random.rand(m, n)
		s = Structure('LABEL', 'Name', True, A=A)
		s.calc_y(x)

		s.constraints += D(40) >= 30 * Gy
		s.dose_rx = 42 * Gy
		self.assertTrue( isinstance(s.plotting_data, dict) )
		self.assertTrue( 'curve' in s.plotting_data )
		self.assertTrue( 'constraints' in s.plotting_data )
		self.assertTrue( 'rx' in s.plotting_data )
		self.assertTrue( 'target' in s.plotting_data )

	def test_print_objective(self):
		s = Structure('LABEL', 'NAME', True)
		self.assertTrue(
				'Structure: NAME (label = LABEL)' in s.objective_string )
		self.assertTrue(
				'target? True' in s.objective_string )
		self.assertTrue(
				'rx dose: 1.0 Gy' in s.objective_string )
		self.assertTrue(
				'weight_under: 1.0' in s.objective_string )
		self.assertTrue(
				'weight_over: 0.05' in s.objective_string )

		s = Structure('LABEL', 'NAME', False)
		self.assertTrue(
				'target? False' in s.objective_string )
		self.assertTrue(
				'rx dose: 0.0 Gy' in s.objective_string )
		self.assertTrue(
				'weight: 0.025' in s.objective_string )

	def test_print_constraints(self):
		s = Structure('LABEL', 'NAME', True)
		s.constraints += D(80) > 30 * Gy
		s.constraints += D(99) < 50 * Gy

		self.assertTrue(
				'Structure: NAME (label = LABEL)' in s.constraints_string )
		for k in s.constraints:
			self.assertTrue(str(s.constraints[k]) in s.constraints_string )

	def test_print_summary(self):
		m, n = 500, 30
		s = Structure('LABEL', 'NAME', True)
		x = np.random.rand(n)
		A = np.random.rand(m, n)
		s = Structure('LABEL', 'Name', True, A=A)
		s.calc_y(x)
		self.assertTrue( 'mean' in s.summary_string )
		self.assertTrue( 'min' in s.summary_string )
		self.assertTrue( 'max' in s.summary_string )
		self.assertTrue( 'D98' in s.summary_string )
		self.assertTrue( 'D75' in s.summary_string )
		self.assertTrue( 'D25' in s.summary_string )
		self.assertTrue( 'D2' in s.summary_string )