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
		self.assertTrue( s._Solver__x is None )
		self.assert_scalar_equal( s.gamma, GAMMA_DEFAULT )
		self.assertTrue( isinstance(s.dvh_vars, dict) )
		self.assertTrue( len(s.dvh_vars) == 0 )
		self.assertTrue( isinstance(s.slack_vars, dict) )
		self.assertTrue( len(s.slack_vars) == 0 )
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

		self.assertTrue( s._Solver__check_dimensions(structures) == n )

		structures[0] = Structure(0, 'tumor', True, A=rand(m0, n_mismatch))

		self.assert_exception( call=s._Solver__check_dimensions,
							   args=[structures] )

	def test_solver_matrix_gather(self):
		m0 = 100
		m1 = 150
		n = 50

		A0 = rand(m0, n)
		A1 = rand(m1, n)

		rx = 30 * Gy

		s = Solver()
		structures = []
		structures.append(Structure(0, 'tumor', True, dose=rx, A=A0))
		structures.append(Structure(1, 'OAR', False, A=A1))

		wo = structures[0].w_over
		wu = structures[0].w_under
		wabs, wlin = s.get_cd_from_wts(wu, wo)
		w = structures[1].w_over

		# -structure 1 not collapsable (reason: target)
		# -structure 2 collapsable
		# expected matrix size: {m0 + 1 \times n}
		A, dose, weight_abs, weight_lin = \
				s._Solver__gather_matrix_and_coefficients(structures)

		self.assertTrue( A.shape == (m0 + 1, n) )
		self.assert_vector_equal( A[:-1, :], A0 )
		self.assert_vector_equal( A[-1, :], A1.sum(0) / m1 )
		self.assertTrue( dose.size == m0 + 1 )
		self.assertTrue( sum(dose[:m0] - rx.value) == 0 )
		self.assertTrue( sum(dose[m0:]) == 0 )
		self.assertTrue( sum(weight_abs[:m0] - wabs) == 0 )
		self.assertTrue( sum(weight_abs[m0:] - w * m1) == 0 )
		self.assertTrue( sum(weight_lin[:m0] - wlin) == 0 )
		self.assertTrue( sum(weight_lin[m0:] - 0) == 0 )

		structures[1].constraints += D('mean') < 5 * Gy
		# -structure 1 not collapsable (reason: target)
		# -structure 2 collapsable (reason: only mean constraint)
		# expected matrix size: {m0 + 1 \times n}
		A, dose, weight_abs, weight_lin = \
				s._Solver__gather_matrix_and_coefficients(structures)

		self.assertTrue( A.shape == (m0 + 1, n) )
		self.assert_vector_equal( A[:-1, :], A0 )
		self.assert_vector_equal( A[-1, :], A1.sum(0) / m1 )
		self.assertTrue( dose.size == m0 + 1 )
		self.assertTrue( sum(dose[:m0] - rx.value) == 0 )
		self.assertTrue( sum(dose[m0:]) == 0 )
		self.assertTrue( sum(weight_abs[:m0] - wabs) == 0 )
		self.assertTrue( sum(weight_abs[m0:] - w * m1) == 0 )
		self.assertTrue( sum(weight_lin[:m0] - wlin) == 0 )
		self.assertTrue( sum(weight_lin[m0:] - 0) == 0 )

		structures[1].constraints += D(30) < 10 * Gy
		# -structure 1 not collapsable (reason: target)
		# -structure 2 not collapsable (reason: DVH constraint)
		# expected matrix size: {m0 + 1 \times n}
		A, dose, weight_abs, weight_lin = \
				s._Solver__gather_matrix_and_coefficients(structures)

		self.assertTrue( A.shape == (m0 + m1, n) )
		self.assert_vector_equal( A[:m0, :], A0 )
		self.assert_vector_equal( A[m0:, :], A1 )
		self.assertTrue( dose.size == m0 + m1 )
		self.assertTrue( sum(dose[:m0] - rx.value) == 0 )
		self.assertTrue( sum(dose[m0:]) == 0 )
		self.assertTrue( sum(weight_abs[:m0] - wabs) == 0 )
		self.assertTrue( sum(weight_abs[m0:] - w) == 0 )
		self.assertTrue( sum(weight_lin[:m0] - wlin) == 0 )
		self.assertTrue( sum(weight_lin[m0:] - 0) == 0 )

		# test incorporation of voxel weights into objective term weights,
		# both full and collapsed matrix cases
		voxel_weights0 = (1 + 10 * rand(m0)).astype(int)
		voxel_weights1 = (1 + 10 * rand(m1)).astype(int)
		structures[0].voxel_weights = voxel_weights0
		structures[1].voxel_weights = voxel_weights1

		structures[1].constraints.clear()
		A, dose, weight_abs, weight_lin = \
				s._Solver__gather_matrix_and_coefficients(structures)

		self.assertTrue( A.shape == (m0 + 1, n) )
		self.assert_vector_equal( weight_abs[:m0], wabs * voxel_weights0 )
		self.assert_vector_equal( weight_abs[m0:], w * sum(voxel_weights1) )
		self.assert_vector_equal( weight_lin[:m0], wlin * voxel_weights0 )
		self.assert_vector_equal( weight_lin[m0:], 0 )

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
