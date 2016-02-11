import conrad
import numpy as np
import unittest
import cvxpy

from os import path, remove as os_remove
from conrad import Case

class TestExamples(unittest.TestCase):
	""" Unit tests using example problems. """
	def setUp(self):
		# Construct dose matrix
		A_targ = np.random.rand(self.m_targ, self.n)
		A_oar = 0.5 * np.random.rand(self.m_oar, self.n)
		self.A = np.vstack((A_targ, A_oar))
	
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
		files_to_delete = ['test_plotting.pdf', 'test_plotting_density.pdf', 'test_plotting_2pass.pdf']
		for fname in files_to_delete:
			fpath = path.join(path.abspath(path.dirname(__file__)), fname)
			if path.isfile(fpath): os_remove(fpath)
	
	setUpClass = classmethod(setUpClass)
	tearDownClass = classmethod(tearDownClass)

	def test_basic(self):
		# Prescription for each structure
		rx = [{'label': self.lab_tum, 'name': 'tumor', 'is_target': True,  'dose': 1., 'constraints': None},
			  {'label': self.lab_oar, 'name': 'oar',   'is_target': False, 'dose': 0., 'constraints': None}]
		
		# Construct unconstrained case
		cs = Case(self.A, self.voxel_labels, self.label_order, rx)
				
		# Add DVH constraints and solve
		cs.add_dvh_constraint(self.lab_tum, 1.05, 0.3, '<')
		cs.add_dvh_constraint(self.lab_tum, 0.8, 0.2, '>')
		cs.add_dvh_constraint(self.lab_oar, 0.5, 0.5, '<')
		cs.add_dvh_constraint(self.lab_oar, 0.55, 0.1, '>')   # This constraint makes no-slack problem infeasible
		# cs.plan("ECOS", verbose = 1)
		cs.plan("ECOS", "dvh_2pass", verbose = 1)
		cs.summary()
	
	def test_2pass_no_constr(self):
		# Prescription for each structure
		rx = [{'label': self.lab_tum, 'name': 'tumor', 'is_target': True,  'dose': 1., 'constraints': None},
			  {'label': self.lab_oar, 'name': 'oar',   'is_target': False, 'dose': 0., 'constraints': None}]
		
		# Construct unconstrained case
		cs = Case(self.A, self.voxel_labels, self.label_order, rx)
		
		# Solve with slack in single pass
		cs.plan("ECOS")
		res_x = cs.problem.solver._x.value
		res_obj = cs.problem.solver.objective.value
		
		# Check results from 2-pass identical if no DVH constraints
		cs.plan("ECOS", "dvh_2pass")
		res_x_2pass = cs.problem.solver._x.value
		res_obj_2pass = cs.problem.solver.objective.value
		self.assertItemsEqual(res_x, res_x_2pass)
		self.assertEqual(res_obj, res_obj_2pass)
	
	def test_2pass_noslack(self):
		# Prescription for each structure
		rx = [{'label': self.lab_tum, 'name': 'tumor', 'is_target': True,  'dose': 1., 'constraints': None},
			  {'label': self.lab_oar, 'name': 'oar',   'is_target': False, 'dose': 0., 'constraints': None}]
		
		# Construct unconstrained case
		cs = Case(self.A, self.voxel_labels, self.label_order, rx)
		
		# Add DVH constraints and solve
		cs.add_dvh_constraint(self.lab_tum, 1.05, 0.3, '<')
		cs.add_dvh_constraint(self.lab_tum, 0.8, 0.2, '>')
		cs.add_dvh_constraint(self.lab_oar, 0.5, 0.5, '<')
		cs.plan("ECOS", "dvh_no_slack")
		res_obj = cs.problem.solver.objective.value
		
		# Check objective from 2nd pass <= 1st pass (since 1st constraints more restrictive)
		cs.plan("ECOS", "dvh_no_slack", "dvh_2pass")
		res_obj_2pass = cs.problem.solver.objective.value
		self.assertTrue(res_obj_2pass <= res_obj)
	
	def test_plotting(self):
	 	# Prescription for each structure
		rx = [{'label': self.lab_tum, 'name': 'tumor', 'is_target': True,  'dose': 1., 'constraints': None},
			  {'label': self.lab_oar, 'name': 'oar',   'is_target': False, 'dose': 0., 'constraints': None}]
		
		# Construct unconstrained case
		cs = Case(self.A, self.voxel_labels, self.label_order, rx)
		
		# Add DVH constraints and solve
		cs.add_dvh_constraint(self.lab_tum, 1.05, 0.3, '<')
		cs.add_dvh_constraint(self.lab_tum, 0.8, 0.2, '>')
		cs.add_dvh_constraint(self.lab_oar, 0.5, 0.5, '<')
		cs.add_dvh_constraint(self.lab_oar, 0.55, 0.1, '>')   # This constraint makes no-slack problem infeasible
		
		# Solve and plot resulting DVH curves
		cs.plan("ECOS", plot = True, show = False)
		cs.plan("ECOS", plot = True, show = False, plotfile = "test_plotting.pdf")
		cs.plot_density(show = False, plotfile = "test_plotting_density.pdf")
		
		# Solve with 2-pass and plot DVH curves for both passes
		cs.plan("ECOS", "dvh_2pass", plot = False)
		cs.plot(show = False, plot_2pass = True, plotfile = "test_plotting_2pass.pdf")
