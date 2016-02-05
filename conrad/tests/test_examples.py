import conrad
import numpy as np
import unittest
import cvxpy
from os import path

from conrad import Case

# some tests should be here
class TestExamples(unittest.TestCase):
	""" Unit tests using example problems. """
	def setup(self):
		self.m_targ = 100
		self.m_oar = 400
		self.m = self.m_targ + self.m_oar
		self.n = 200
		
		# Construct dose matrix
		A_targ = np.random.rand(self.m_targ, self.n)
		A_oar = 0.5 * np.random.rand(self.m_oar, self.n)
		self.A = np.vstack((A_targ, A_oar))

	
	def setup_yaml_json(self):
		# beams
		n_beams = 200

		# organ sizes
		sizes = {'target' : 100, 'oar1' : 400, 'oar2' : 200}

		# submatrices
		A = {}
		A['target'] = np.random.rand(sizes['target'], n_beams)
		A['oar1'] = 0.5 * np.random.rand(sizes['oar2'], n_beams)
		A['oar2'] = 0.3 * np.random.rand(sizes['oar1'], n_beams)

		# labels
		labels = {'target' : 1, 'oar1' : 4, 'oar2' : 7}


		self.A = np.vstack((A['target'], A['oar1'], A['oar2']))
		self.voxel_labels = [ sizes['target'] * [labels['target'] + 
			sizes['oar1'] * [labels['oar1'] + sizes['oar2'] * [labels['oar2'] ]
		self.label_order = [labels['target'], labels['oar1'], labels['oar2']]


	def test_basic(self):
		# Prescription for each structure
		lab_tum = 0
		lab_oar = 1
		rx = [{'label': lab_tum, 'name': 'tumor', 'is_target': True,  'dose': 1., 'constraints': None},
			  {'label': lab_oar,  'name': 'oar',   'is_target': False, 'dose': 0., 'constraints': None}]
		
		# Construct and solve case
		label_order = [lab_tum, lab_oar]
		voxel_labels = [lab_tum] * self.m_targ + [lab_oar] * self.m_oar
		cs = Case(self.A, voxel_labels, label_order, rx)
		cs.plan("ECOS", verbose = 1)
		
		# Add DVH constraints and re-solve
		cs.add_dvh_constraint(lab_tum, 1.05, 30., '<')
		cs.add_dvh_constraint(lab_tum, 0.8, 20., '>')
		cs.add_dvh_constraint(lab_oar, 0.5, 50, '<')
		cs.add_dvh_constraint(lab_oar, 0.55, 10, '>')
		cs.plan("SCS", verbose = 1)

		# try plotting:
		# try plotting + save:

	 def test_rx_from_JSON(self):
	 	"""TODO: docstring"""
	 	self.setup_yaml_json()
	 	input_file = path.abspath('json_rx.json')
	 	cs = Case(self.A, self.voxel_labels, self.label_order, input_file)
	 	cs.plan("ECOS", verbose = 1)

	 def test_rx_from_YAML(self):
	 	"""TODO: docstring"""
	 	self.setup_yaml_json()
	 	input_file = path.abspath('yaml_rx.yml')
	 	cs = Case(self.A, self.voxel_labels, self.label_order, input_file)
	 	cs.plan("ECOS", verbose = 1)

	 def test_plotting(self, case):
	 	"""TODO: docstring"""
	 	pass	 	