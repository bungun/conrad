# import conrad
# import numpy as np
# import unittest
# import cvxpy
# from os import path, remove as os_remove
# from operator import add
# from numpy import ones

# from conrad.compat import *
# from conrad import *

# class TestRxInput(unittest.TestCase):
# 	""" Unit tests using external data import. """
# 	@classmethod
# 	def setUpClass(self):
# 		# beams
# 		self.n = self.n_beams = 200
# 		mx = 10
# 		my = 10
# 		mz = 7
# 		self.m = mx * my * mz
# 		self.voxels = VoxelGrid(mx, my, mz)
# 		self.beams = BeamSet(n_beams=self.n)
# 		self.physics = Physics(self.voxels, self.beams)

# 		# labels
# 		self.sizes = {'target': 100, 'oar1': 400, 'oar2': 200}
# 		label_size_pairs = {
# 				'target' : (1, 100),
# 				'oar1' : (5, 400),
# 				'oar2' : (7, 200)
# 			}

# 		self.anatomy = Anatomy()
# 		for name, pair in label_size_pairs.items():
# 			self.anatomy += Structure(pair[0], name, name == 'target',
# 									  dose=int(name=='target'),
# 									  size=pair[1])

# 		self.voxel_labels = reduce(add, listmap(
# 				lambda ls :[ls[0]]*ls[1], label_size_pairs.values()))
# 		self.anatomy.labels = self.voxel_labels

# 	def setUp(self):
# 		# submatrices
# 		A = {}
# 		A['target'] = np.random.rand(self.sizes['target'], self.n_beams)
# 		A['oar1'] = 0.5 * np.random.rand(self.sizes['oar2'], self.n_beams)
# 		A['oar2'] = 0.3 * np.random.rand(self.sizes['oar1'], self.n_beams)
# 		self.A = np.vstack((A['target'], A['oar1'], A['oar2']))

# 	# Runs once after all unit tests
# 	@classmethod
# 	def tearDownClass(self):
# 		files_to_delete = ['yaml_test_plot.pdf', 'json_test_plot.pdf']
# 		for fname in files_to_delete:
# 			fpath = path.join(path.abspath(path.dirname(__file__)), fname)
# 			if path.isfile(fpath): os_remove(fpath)


# 	def test_rx_from_JSON(self):
# 	 	"""test for reading in prescription from JSON file"""
# 	 	input_file = path.join(
# 	 			path.abspath(path.dirname(__file__)), 'prescriptions',
# 	 			'json_rx.json')

# 	 	self.physics.dose_matrix = self.A
# 	 	self.rx = Prescription(input_file)
# 	 	cs = Case(self.physics, self.anatomy, self.rx)
# 	 	p = CasePlotter(cs)

# 	 	print "prescription loaded from JSON:\n", cs.prescription

# 	 	if cs.plan(solver='ECOS', verbose=1):
# 		 	p.plot(cs, show=False, file='json_test_plot.pdf')
# 			print 'prescription report:\n', cs.prescription_report_string

# 	 	print "complete"

# 	def test_rx_from_YAML(self):
# 	 	"""test for reading in prescription from YAML file"""
# 	 	input_file = path.join(
# 	 			path.abspath(path.dirname(__file__)),
# 	 			'prescriptions', 'yaml_rx.yml')
# 	 	cs = Case(self.A, self.voxel_labels, self.label_order, input_file)
# 	 	p = CasePlotter(cs)

# 	 	print "prescription loaded from YAML:\n", cs.prescription

# 	 	if cs.plan(solver='ECOS', verbose=1):
# 		 	p.plot(cs, show=False, file='yaml_test_plot.pdf')
# 		 	print 'prescription report:\n', cs.prescription_report_string

# 	 	print "complete"
