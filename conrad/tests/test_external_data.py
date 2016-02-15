import conrad
import numpy as np
import unittest
import cvxpy
from os import path, remove as os_remove
from numpy import ones
from conrad import *

class TestExternalData(unittest.TestCase):
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
		size_by_label = {}
		for k, l in labels.iteritems():
			size_by_label[l] = sizes[k]

		self.A = np.vstack((A['target'], A['oar1'], A['oar2']))
		self.label_order = [labels['target'], labels['oar1'], labels['oar2']]
		self.voxel_labels = ones(self.A.shape[0], dtype = int)
		idx1 = idx2 = 0
		for l in self.label_order:
			idx2 += size_by_label[l]
			self.voxel_labels[idx1 : idx2] = l
			idx1 = idx2
	
	# Runs once after all unit tests
	def tearDownClass(self):
		files_to_delete = ['yaml_test_plot.pdf', 'json_test_plot.pdf']
		for fname in files_to_delete:
			fpath = path.join(path.abspath(path.dirname(__file__)), fname)
			if path.isfile(fpath): os_remove(fpath)
	
	tearDownClass = classmethod(tearDownClass)

	def test_rx_from_JSON(self):
	 	"""TODO: docstring"""
	 	input_file = path.join(path.abspath(path.dirname(__file__)), 'json_rx.json')
	 	cs = Case(self.A, self.voxel_labels, self.label_order, input_file)
	 	p = CasePlotter(cs)

	 	print "prescription loaded from JSON:\n", cs.prescription 

	 	if cs.plan(solver = 'ECOS', verbose = 1):
		 	p.plot(cs, show = False, file = 'json_test_plot.pdf')

	 	print "complete"

	def test_rx_from_YAML(self):
	 	"""TODO: docstring"""
	 	input_file = path.join(path.abspath(path.dirname(__file__)), 'yaml_rx.yml')
	 	cs = Case(self.A, self.voxel_labels, self.label_order, input_file)
	 	p = CasePlotter(cs)

	 	print "prescription loaded from YAML:\n", cs.prescription 

	 	if cs.plan(solver = 'ECOS', verbose = 1):
		 	p.plot(cs, show = False, file = 'yaml_test_plot.pdf')

	 	print "complete"