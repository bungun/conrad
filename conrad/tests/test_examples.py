import numpy as np
import unittest
import cvxpy
from os import path, remove as os_remove
from warnings import warn
from conrad import *

class TestExamples(unittest.TestCase):
	""" Unit tests using example problems. """
	def setUp(self):
		# Construct dose matrix
		A_targ = 1.2 * np.random.rand(self.m_targ, self.n)
		A_oar = 0.3 * np.random.rand(self.m_oar, self.n)
		self.A = np.vstack((A_targ, A_oar))

		# Prescription for each structure
		self.rx = [{'label': self.lab_tum, 'name': 'tumor', 'is_target': True,  'dose': 1., 'constraints': None},
			  {'label': self.lab_oar, 'name': 'oar',   'is_target': False, 'dose': 0., 'constraints': None}]

	# Runs once before all unit tests
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

	# Runs once after all unit tests
	def tearDownClass(self):
		files_to_delete = ['test_plotting.pdf']
		for fname in files_to_delete:
			fpath = path.join(path.abspath(path.dirname(__file__)), fname)
			if path.isfile(fpath): os_remove(fpath)

	setUpClass = classmethod(setUpClass)
	tearDownClass = classmethod(tearDownClass)

	def test_basic(self):
		# Construct unconstrained case
		cs = Case(self.A, self.voxel_labels, self.label_order, self.rx)

		# Add DVH constraints and solve
		cs.structures[self.lab_tum].constraints += D(20) <= 1.15
		cs.structures[self.lab_tum].constraints += D(80) >= 0.95
		cs.structures[self.lab_oar].constraints += D(50) < 0.30
		cs.structures[self.lab_oar].constraints += D(10) < 0.55
		cs.plan()
		print 'solution found in {} seconds\n'.format(cs.solvetime)
		print 'dose summary:\n', cs.dose_summary_string
		print cs.x
		print self.A.dot(cs.x)

	def test_2pass_no_constr(self):
		# Construct unconstrained case
		cs = Case(self.A, self.voxel_labels, self.label_order, self.rx)

		# Solve with slack in single pass
		cs.plan(solver = 'ECOS')
		res_x = cs.x
		res_obj = cs.solver_info['objective']

		# Check results from 2-pass identical if no DVH constraints
		cs.plan(solver = 'ECOS', dvh_exact = True)
		res_x_2pass = cs.x
		res_obj_2pass = cs.solver_info['objective']
		self.assertItemsEqual(res_x, res_x_2pass)
		self.assertEqual(res_obj, res_obj_2pass)

	def test_2pass_noslack(self):
		# Construct unconstrained case
		cs = Case(self.A, self.voxel_labels, self.label_order, self.rx)

		# Add DVH constraints and solve
		cs.structures[self.lab_tum].constraints += D(20) <= 1.15
		cs.structures[self.lab_tum].constraints += D(80) >= 0.95
		cs.structures[self.lab_oar].constraints += D(50) < 0.30
		cs.plan(solver = 'ECOS', dvh_slack = False)
		res_obj = cs.problem.solver.objective.value

		# Check objective from 2nd pass <= 1st pass (since 1st constraints more restrictive)
		cs.plan(solver = 'ECOS', dvh_slack = False, dvh_exact = True)
		res_obj_2pass = cs.problem.solver.objective.value
		self.assertTrue(res_obj_2pass <= res_obj)

	def test_plotting(self):
		# Construct unconstrained case
		cs = Case(self.A, self.voxel_labels, self.label_order, self.rx)
		p = CasePlotter(cs)

		# Add DVH constraints
		cs.structures[self.lab_tum].constraints += D(20) <= 1.15
		cs.structures[self.lab_tum].constraints += D(80) >= 0.95
		cs.structures[self.lab_oar].constraints += D(50) < 0.30

		# This constraint makes no-slack problem infeasible
		cs.structures[self.lab_oar].constraints += D(99) < 0.05
		
		# Add a DVH mean constraint
		cs.structures[self.lab_tum].constraints += D('mean') <= 1.0

		# Solve and plot resulting DVH curves
		if cs.plan(solver = 'ECOS'):
			p.plot(cs, show = False, file = 'test_plotting.pdf')
		else:
			warn(Warning('plan infeasible, no plotting performed'))
