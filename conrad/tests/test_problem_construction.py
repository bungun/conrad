# import numpy as np
# import unittest
# from os import path, remove as os_remove
# from warnings import warn

# from conrad.compat import *
# from conrad import *

# class TestConstruction(unittest.TestCase):
# 	""" TODO: docstring"""
# 	def setUp(self):
# 		# Construct dose matrix
# 		A_targ = 1.2 * np.random.rand(self.m_targ, self.n)
# 		A_oar = 0.3 * np.random.rand(self.m_oar, self.n)
# 		self.A = np.vstack((A_targ, A_oar))

# 		# Prescription for each structure
# 		self.rx = [
# 				{'label': self.lab_tum, 'name': 'tumor', 'is_target': True,
# 				 'dose': 1., 'constraints': None},
# 				{'label': self.lab_oar, 'name': 'oar', 'is_target': False,
# 				 'dose': 0., 'constraints': None}]

# 		def print_dims(data, solver='ECOS'):
# 			# min c'*x
# 			# s.t. A*x = b
# 			#      G*x <= _K h
# 			# See https://github.com/embotech/ecos
# 			print 'c has length {}'.format(len(data['c']))
# 			print 'A has {} rows and {} columns'.format(
# 					data['A'].shape[0], data['A'].shape[1])
# 			print 'G has {} rows and {} columns\n'.format(
# 					data['G'].shape[0], data['G'].shape[1])

# 		self.print_dims = print_dims

# 	# Runs once before all unit tests
# 	@classmethod
# 	def setUpClass(self):
# 		self.m_targ = 100
# 		self.m_oar = 400
# 		self.m = self.m_targ + self.m_oar
# 		self.n = 200

# 		# Structure labels
# 		self.lab_tum = 0
# 		self.lab_oar = 1

# 		# Voxel labels on beam matrix
# 		self.label_order = [self.lab_tum, self.lab_oar]
# 		self.voxel_labels = [self.lab_tum] * self.m_targ + [self.lab_oar] * self.m_oar

# 		self.anatomy = Anatomy()
# 		self.anatomy += Structure(self.lab_tum, 'tumor', True)
# 		self.anatomy += Structure(self.lab_oar, 'oar', False)
# 		self.anatomy.labels = self.voxel_labels

# 		self.physics = Physics(VoxelGrid(self.m, 1, 1), BeamSet(n_beams=self.n))

# 	def test_cvxpy_matrix_size(self):
# 		"""TODO: Docstring"""

# 		# TODO: this test takes advantage of the fact after Case.plan() has been
# 		# called with a CVXPY solver, Case.problem.solver.problem happens to be
# 		# the CVXPY Problem() object, which has a get_problem_data(<solver_type>)
# 		# method that can be called to retrieve the data passed from CVXPY to the
# 		# solver. In the future there should be something less hacky.

# 		# Construct unconstrained case
# 		self.physics.dose_matrix = self.A
# 		cs = Case(self.physics, self.anatomy)

# 		# Add DVH constraints to PTV
# 		cs.structures[self.lab_tum].constraints += D(20) <= 1.15
# 		cs.structures[self.lab_tum].constraints += D(80) >= 0.95

# 		cs.plan(solver='ECOS')
# 		data_small = cs.problem.solver.problem.get_problem_data('ECOS')
# 		data_small_solvetime = cs.solvetime
# 		data_small_c_len = len(data_small['c'])
# 		data_small_G_rows = data_small['G'].shape[0]

# 		self.print_dims(data_small)
# 		print 'solution found in {} seconds\n'.format(data_small_solvetime)

# 		# Test mathematical coherence of problem matrix
# 		self.assertEqual(data_small_c_len, data_small['A'].shape[1])   # len(c) == ncol(A)
# 		self.assertEqual(data_small_c_len, data_small['G'].shape[1])   # len(c) == ncol(G)
# 		self.assertEqual(data_small['A'].shape[0], 0)   # No equality constraints
# 		self.assertTrue(data_small_G_rows > 0)          # Since we added DVH inequality constraints

# 		# Add DVH constraints to OAR
# 		cs.structures[self.lab_oar].constraints += D('mean') <= 0.5

# 		cs.plan(solver='ECOS')
# 		data_also_small = cs.problem.solver.problem.get_problem_data('ECOS')
# 		data_also_small_solvetime = cs.solvetime
# 		data_also_small_c_len = len(data_also_small['c'])
# 		data_also_small_G_rows = data_also_small['G'].shape[0]

# 		self.print_dims(data_also_small)
# 		print 'solution found in {} seconds\n'.format(data_also_small_solvetime)

# 		# Test mean constraint doesn't result in much larger problem
# 		self.assertAlmostEqual(data_also_small_solvetime, data_small_solvetime, places = 2)   # Solve time should be roughly the same
# 		self.assertEqual(data_also_small_c_len, data_small_c_len + 1)   # Add one more term to objective
# 		self.assertEqual(data_also_small_G_rows, data_small_G_rows + 2)   # TODO: Why are 2 rows added to end of G instead of 1?

# 		cs.structures[self.lab_oar].constraints += D(80) <= 0.2

# 		cs.plan(solver='ECOS')
# 		data_larger = cs.problem.solver.problem.get_problem_data('ECOS')
# 		data_larger_solvetime = cs.solvetime
# 		data_larger_c_len = len(data_larger['c'])
# 		data_larger_G_rows = data_larger['G'].shape[0]

# 		self.print_dims(data_larger)
# 		print 'solution found in {} seconds\n'.format(data_larger_solvetime)

# 		# Test percentile constraint on OAR results in larger problem
# 		self.assertTrue(data_larger_solvetime > data_also_small_solvetime)
# 		self.assertTrue(data_larger_c_len > data_also_small_c_len)
# 		self.assertTrue(data_larger_G_rows > data_also_small_G_rows)
