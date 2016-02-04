import conrad
import numpy as np
import unittest
import cvxpy

from conrad import Structure, Prescription

# some tests should be here
class TestExamples(unittest.TestCase):
	""" Unit tests using example problems. """
	
	def test_basic(self):
		# dose matrix
		m_targ = 100
		m_oar = 400
		m = m_targ + m_oar
		n = 200

		A_targ = np.random.rand(m_targ, n)
		A_oar = 0.5 * np.random.rand(m_oar, n)
		A = np.vstack((A_targ, A_oar))
		
		# rx = Prescription([{'label' = 'tumor_label', 'name' = 'tumor', 'is_target' = True,  'dose' = 1., 'constraints' = None},
		#				   {'label' = 'oar_label',   'name' = 'oar',   'is_target' = False, 'dose' = 0., 'constraints' = None}])
