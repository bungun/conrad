import numpy as np
import unittest
from os import path, remove as os_remove
from warnings import warn
from conrad import *

class TestConstruction(unittest.TestCase):
	""" TODO: docstring"""
	def setUp(self):
		# Construct dose matrix
		A_targ = 1.2 * np.random.rand(self.m_targ, self.n)
		A_oar = 0.3 * np.random.rand(self.m_oar, self.n)
		self.A = np.vstack((A_targ, A_oar))

		# Prescription for each structure
		self.rx = [{'label': self.lab_tum, 'name': 'tumor', 'is_target': True,  'dose': 1., 'constraints': None},
			  {'label': self.lab_oar, 'name': 'oar',   'is_target': False, 'dose': 0., 'constraints': None}]

	# Runs once before all unit tests
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
		self.voxel_labels = [self.lab_tum] * self.m_targ + [self.lab_oar] * self.m_oar


	def test_cvxpy_matrix_size(self):
		"""TODO: Docstring"""

		# TODO: this test takes advantage of the fact after Case.plan() has been
		# called with a CVXPY solver, Case.problem.solver.problem happens to be
		# the CVXPY Problem() object, which has a get_problem_data(<solver_type>)
		# method that can be called to retrieve the data passed from CVXPY to the
		# solver. In the future there should be something less hacky.


		# Construct unconstrained case
		cs = Case(self.A, self.voxel_labels, self.label_order, self.rx)

		# Add DVH constraints to PTV
		cs.structures[self.lab_tum].constraints += D(20) <= 1.15
		cs.structures[self.lab_tum].constraints += D(80) >= 0.95

		cs.plan(solver='ECOS')

		data_small = cs.problem.solver.problem.get_problem_data('ECOS')


		#
		# TODO: TEST SOMETHING HERE WITH THE DATA EXTRACTED FROM CVXPY
		#
		#
		#

		# Add DVH constraints to OAR
		cs.structures[self.lab_oar].constraints += D('mean') <= 0.5

		cs.plan(solver='ECOS')
		data_also_small = cs.problem.solver.problem.get_problem_data('ECOS')

		#
		# TODO: TEST THAT MEAN CONSTRAINT DOESN'T RESULT IN MUCH LARGER PROBLEM
		#
		#
		#

		cs.structures[self.lab_oar].constraints += D(80) <= 0.2

		cs.plan(solver='ECOS')
		data_larger = cs.problem.solver.problem.get_problem_data('ECOS')

		#
		# TODO: TEST THAT PERCENTILE CONSTRAINT DOES RESULT IN MUCH LARGER PROBLEM
		#
		#
		#