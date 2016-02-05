import conrad
import numpy as np
import unittest
import cvxpy

from conrad import Case

# some tests should be here
class TestExamples(unittest.TestCase):
	""" Unit tests using example problems. """
	
	def test_basic(self):
		# Construct dose matrix
		m_targ = 100
		m_oar = 400
		m = m_targ + m_oar
		n = 200
		
		lab_targ = 0
		lab_oar = 1

		A_targ = np.random.rand(m_targ, n)
		A_oar = 0.5 * np.random.rand(m_oar, n)
		A = np.vstack((A_targ, A_oar))
		
		# Prescription for each structure
		rx = [{'label': lab_targ, 'name': 'tumor', 'is_target': True,  'dose': 1., 'constraints': None},
			  {'label': lab_oar,  'name': 'oar',   'is_target': False, 'dose': 0., 'constraints': None}]
		
		# Construct and solve case
		cs = Case(A, [lab_targ] * m_targ + [lab_oar] * m_oar, [lab_targ, lab_oar], rx)
		cs.plan("ECOS", verbose = 1)
