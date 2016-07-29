from numpy import vstack

from conrad.compat import *
from conrad.optimization.solver_base import *
from conrad.optimization.solver_cvxpy import *
from conrad.optimization.solver_optkit import *
from conrad.tests.base import *

class SolverTestCase(ConradTestCase):
	def test_solver_init(self):
		s = Solver()
		self.assertFalse( s.use_2pass )
		self.assertTrue( s.use_slack )
		self.assertTrue( s._Solver__x is None )
		self.assert_scalar_equal( s.gamma, GAMMA_DEFAULT )
		self.assertTrue( isinstance(s.dvh_vars, dict) )
		self.assertTrue( len(s.dvh_vars) == 0 )
		self.assertTrue( isinstance(s.slack_vars, dict) )
		self.assertTrue( len(s.slack_vars == 0) )
		self.assertFalse( s.feasible )

		s.gamma = 1e-4
		self.assert_scalar_equal( s.gamma, 1e-4 )
		try:
			s.gamma = 'string input'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		self.assert_scalar_equal( s.gamma_prioritized(1), 9e-4 )
		self.assert_scalar_equal( s.gamma_prioritized(2), 4e-4 )
		self.assert_scalar_equal( s.gamma_prioritized(3), 1e-4 )
		try:
			s.gamma_prioritized(0)
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			s.gamma_prioritized('string input')
			self.assertTrue( False )
		except:
			self.assertTrue( True )

class SolverGenericTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.m_targ = 100
		self.m_oar = 400
		self.m = self.m_targ + self.m_oar
		self.n = 200

		# Structure labels
		self.lab_tum = 0
		self.lab_oar = 1

		# Voxel labels on beam matrix
		self.label_order = [self.lab_tum, self.lab_oar]
		self.voxel_labels = [self.lab_tum] * self.m_targ
		self.voxel_labels += [self.lab_oar] * self.m_oar

		self.anatomy = Anatomy()
		self.anatomy += Structure(self.lab_tum, 'tumor', True)
		self.anatomy += Structure(self.lab_oar, 'oar', False)

	def setUp(self):
		# Construct dose matrix
		A_targ = 1.2 * np.random.rand(self.m_targ, self.n)
		A_oar = 0.3 * np.random.rand(self.m_oar, self.n)
		self.A = np.vstack((A_targ, A_oar))
		self.physics = Physics(
				dose_matrix=self.A, voxel_labels=self.voxel_labels)

class SolverCVXPYTestCase(SolverGenericTestCase):
	""" TODO: docstring"""
	def test_solver_cvxpy_init(self):
		s = SolverCVXPY()
		if s is None:
			return

		self.assertTrue( s.objective is None )
		self.assertTrue( s.constraints is None )
		self.assertTrue( s.problem is None )
		self.assertTrue( isinstance(s._SolverCVXPY__x, Variable) )
		self.assertTrue( isinstance(s._SolverCVXPY__constraint_indices, dict) )
		self.assertTrue( len(s._SolverCVXPY__constraint_indices) == 0 )
		self.assertTrue( isinstance(s.constraint_dual_vars, dict) )
		self.assertTrue( len(s.constraint_dual_vars) == 0 )
		self.assertTrue( s.n_beams == 0 )

	def test_solver_cvxpy_problem_init(self):
		n_beams = 100
		s = SolverCVXPY()
		if s is None:
			return

		s.init_problem(n_beams)
		self.assertTrue( s.slack )
		self.assertFalse( s.use_2pass )
		self.assertTrue( s.n_beams == n_beams )
		self.assertTrue( s.objective is not None )
		self.assertTrue( len(s.constraints) == 1 )
		self.assertTrue( s.problem is not None )

		for slack_flag in (True, False):
			for two_pass_flag in (True, False):
				s.init_problem(n_beams, use_slack=slack_flag,
							   use_2pass=two_pass_flag)
				self.assertTrue( s.use_slack == slack_flag )
				self.assertTrue( s.use_2pass == two_pass_flag )

		s.init_problem(n_beams, gamma=1.2e-3)
		self.assert_scalar_equal( s.gamma, 1.2e-3 )

	def test_solver_cvxpy_dimcheck(self):
		s = SolverCVXPY(100)
		structures = []
		structures.append(Structure(0, 'tumor', True))
		structures.append(Structure(1, 'OAR', False))


