import conrad
import numpy as np
import unittest
import cvxpy
from os import path

from conrad import Case

class TestYaml(unittest.TestCase):
	""" Unit tests using external data import. """
	def setUp(self):
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
		self.label_order = [labels['target'] + labels['oar1'] + labels['oar2']]
		self.voxel_labels = [sizes['target'] * labels['target'] + sizes['oar1'] * labels['oar1'] + sizes['oar2'] * labels['oar2']]

	def test_rx_from_JSON(self):
	 	"""TODO: docstring"""
	 	self.setup_yaml_json()
	 	input_file = path.abspath('json_rx.json')
	 	cs = Case(self.A, self.voxel_labels, self.label_order, input_file)

	 	print "prescription loaded from JSON:\n", cs.prescription 

	 	cs.plan("ECOS", verbose = 1)

	 def test_rx_from_YAML(self):
	 	"""TODO: docstring"""
	 	self.setup_yaml_json()
	 	input_file = path.abspath('yaml_rx.yml')
	 	cs = Case(self.A, self.voxel_labels, self.label_order, input_file)

	 	print "prescription loaded from YAML:\n", cs.prescription 

	 	cs.plan("ECOS", verbose = 1)
