"""
Unit tests for :mod:`conrad.optimization.solver_base`.
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
from conrad.medicine import Structure, Anatomy
from conrad.medicine.dose import D, Gy
from conrad.optimization.solver_base import *
from conrad.tests.base import *

class SolverTestCase(ConradTestCase):
	def test_solver_init(self):
		s = Solver()
		self.assertFalse( s.use_2pass )
		self.assertTrue( s.use_slack )
		self.assertIsNone( s._Solver__x )
		self.assert_scalar_equal( s.gamma, GAMMA_DEFAULT )
		self.assertIsInstance( s.dvh_vars, dict )
		self.assertEqual( len(s.dvh_vars), 0 )
		self.assertIsInstance( s.slack_vars, dict )
		self.assertEqual( len(s.slack_vars), 0 )
		self.assertFalse( s.feasible )

		s.gamma = 1e-4
		self.assert_scalar_equal( s.gamma, 1e-4 )
		with self.assertRaises(ValueError):
			s.gamma = 'string input'

		self.assert_scalar_equal( s.gamma_prioritized(1), 9e-4 )
		self.assert_scalar_equal( s.gamma_prioritized(2), 4e-4 )
		self.assert_scalar_equal( s.gamma_prioritized(3), 1e-4 )
		with self.assertRaises(ValueError):
			s.gamma_prioritized(0)
		with self.assertRaises(ValueError):
			s.gamma_prioritized('string input')

		c, d = s.get_cd_from_wts(1, 0.05)
		self.assert_scalar_equal( c, 1.05 / 2. )
		self.assert_scalar_equal( d, -0.95 / 2. )

	def test_solver_dimcheck(self):
		m0 = 100
		m1 = 150
		n = 50
		n_mismatch = 52

		s = Solver()
		structures = []
		structures.append(Structure(0, 'tumor', True, A=rand(m0, n)))
		structures.append(Structure(1, 'OAR', False, A=rand(m1, n)))

		self.assertEqual( s._Solver__check_dimensions(structures), n )

		structures[0] = Structure(0, 'tumor', True, A=rand(m0, n_mismatch))

		with self.assertRaises(ValueError):
			s._Solver__check_dimensions(structures)

class SolverGenericTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.m_target = 100
		self.m_oar = 400
		self.m = self.m_target + self.m_oar
		self.n = 207

		# Structure labels
		self.label_tumor = 0
		self.label_oar = 1

		# Voxel labels on beam matrix
		self.labelel_order = [self.label_tumor, self.label_oar]
		self.voxel_labels = [self.label_tumor] * self.m_target
		self.voxel_labels += [self.label_oar] * self.m_oar

		self.anatomy = Anatomy()
		self.anatomy += Structure(self.label_tumor, 'tumor', True)
		self.anatomy += Structure(self.label_oar, 'oar', False)

	def setUp(self):
		# Construct dose matrix
		self.A_targ = 1.2 * rand(self.m_target, self.n)
		self.A_oar = 0.3 * rand(self.m_oar, self.n)
		self.anatomy['tumor'].A_full = self.A_targ
		self.anatomy['oar'].A_full = self.A_oar

	def tearDown(self):
		self.anatomy['tumor'].constraints.clear()
		self.anatomy['oar'].constraints.clear()
