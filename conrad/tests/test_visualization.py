from conrad.compat import *
from conrad.visualization.plot import *
from conrad.tests.base import *

class VisualizationTestCase(ConradTestCase):
	def test_dvh_plot_init(self):
		pass

	def test_case_plotter(self):
		pass


# import numpy as np
# import cvxpy
# from os import path, remove as os_remove
# from warnings import warn

# from conrad.compat import *
# from conrad import *
# from conrad.test.base import ConradTestCase

# class TestExamples(ConradTestCase):
# 	""" Unit tests using example problems. """
# 	def setUp(self):
# 		# Construct dose matrix
# 		A_targ = 1.2 * np.random.rand(self.m_targ, self.n)
# 		A_oar = 0.3 * np.random.rand(self.m_oar, self.n)
# 		self.A = np.vstack((A_targ, A_oar))

# 	# Runs once before all unit tests
# 	def setUpClass(self):
# 		mx = 10
# 		my = 10
# 		mz = 5
# 		self.m = mx * my * mz
# 		self.m_targ = 100
# 		self.m_oar = self.m - self.m_targ
# 		self.n = 200

# 		self.voxels = VoxelGrid(mx, my, mz)
# 		self.beams = BeamSet(n_beams=self.n)
# 		self.physics = Physics(beams=self.beams, dose_grid=self.voxels)

# 		# Structure labels
# 		self.lab_tum = 0
# 		self.lab_oar = 1

# 		# Prescription for each structure
# 		self.rx = [
# 				{
# 					'label': self.lab_tum,
# 					'name': 'tumor',
# 					'is_target': True,
# 					'dose': 1.,
# 					'constraints': None
# 				},{
# 					'label': self.lab_oar,
# 					'name': 'oar',
# 					'is_target': False,
# 					'dose': 0.,
# 					'constraints': None
# 				}]

# 		# Voxel labels on beam matrix
# 		self.label_order = [self.lab_tum, self.lab_oar]
# 		self.voxel_labels = [self.lab_tum] * self.m_targ + \
# 							[self.lab_oar] * self.m_oar

# 		self.anatomy = Anatomy()
# 		for s in self.rx:
# 			self.anatomy += Structure(s['label'], s['name'], s['is_target'],
# 									  dose=s['dose'])

# 		self.physics.voxel_labels = self.voxel_labels

# 	# Runs once after all unit tests
# 	def tearDownClass(self):
# 		files_to_delete = ['test_plotting.pdf']
# 		for fname in files_to_delete:
# 			fpath = path.join(path.abspath(path.dirname(__file__)), fname)
# 			if path.isfile(fpath): os_remove(fpath)

# 	setUpClass = classmethod(setUpClass)
# 	tearDownClass = classmethod(tearDownClass)

# 	def test_plotting(self):
# 		# Construct unconstrained case
# 		case = Case(physics=self.physics, anatomy=self.anatomy,
# 					prescription=self.rx)

# 		p = CasePlotter(case)

# 		# Add DVH constraints
# 		case.structures['tumor'].constraints += D(20) <= 1.15
# 		case.structures['tumor'].constraints += D(80) >= 0.95
# 		case.structures['oar'].constraints += D(50) < 0.30

# 		# This constraint makes no-slack problem infeasible
# 		case.structures['oar'].constraints += D(99) < 0.05

# 		# Add a DVH mean constraint
# 		case.structures['tumor'].constraints += D('mean') <= 1.0

# 		# Solve and plot resulting DVH curves
# 		if case.plan(solver='ECOS'):
# 			p.plot(case, show=False, file='test_plotting.pdf')
# 		else:
# 			warn(Warning('plan infeasible, no plotting performed'))