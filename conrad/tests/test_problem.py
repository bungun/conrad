import numpy as np

from conrad.compat import *
from conrad.medicine import Structure
from conrad.optimization.problem import *
from conrad.tests.base import *

class ProblemTestCase(ConradTestCase):

	def print_dims(data, solver='ECOS'):
		# min c'*x
		# s.t. A*x = b
		#      G*x <= _K h
		# See https://github.com/embotech/ecos
		print('c has length {}'.format(len(data['c']))
		print('A has {} rows and {} columns'.format(*data['A'].shape))
		print('G has {} rows and {} columns\n'.format(*data['G'].shape))

	def test_cvxpy_matrix_size(self):
		"""TODO: Docstring"""

		# TODO: this test takes advantage of the fact after Case.plan()
		# has been called with a CVXPY solver, Case.problem.solver.problem
		# happens to be the CVXPY Problem() object, which has a
		# get_problem_data(<solver_type>) method that can be called to
		# retrieve the data passed from CVXPY to the solver.
		# In the future there should be something less hacky.

		# Construct unconstrained case
		cs = Case(anatomy=self.anatomy, physics=self.physics)

		# Add DVH constraints to PTV
		cs.anatomy['tumor'].constraints += D(20) <= 1.15
		cs.anatomy['tumor'].constraints += D(80) >= 0.95

		cs.plan(solver='ECOS')
		data_small = cs.problem.solver.problem.get_problem_data('ECOS')
		data_small_solvetime = cs.solvetime
		data_small_c_len = len(data_small['c'])
		data_small_G_rows = data_small['G'].shape[0]

		self.print_dims(data_small)
		print('solution found in {} seconds\n'.format(data_small_solvetime))

		# Test mathematical coherence of problem matrix:
		#	len(c) == ncol(A)
		#	len(c) == ncol(G)
		self.assertTrue( data_small_c_len == data_small['A'].shape[1] )
		self.assertTrue( data_small_c_len == data_small['G'].shape[1] )

		# No equality constraints
		self.assertTrue( data_small['A'].shape[0] == 0 )

		# Since we added DVH inequality constraints
		self.assertTrue( data_small_G_rows > 0 )

		# Add DVH constraints to OAR
		cs.anatomy['tumor'].constraints += D('mean') <= 0.5

		cs.plan(solver='ECOS')
		data_also_small = cs.problem.solver.problem.get_problem_data('ECOS')
		data_also_small_solvetime = cs.solvetime
		data_also_small_c_len = len(data_also_small['c'])
		data_also_small_G_rows = data_also_small['G'].shape[0]

		self.print_dims(data_also_small)
		print('solution found in {} seconds\n'.format(
				data_also_small_solvetime))

		# Test mean constraint doesn't result in much larger problem
		# solve time within (relative) 10% or (absolute) 0.1s
		self.assert_scalar_equal(
				data_also_small_solvetime, data_small_solvetime, 1e-1, 1e-1)

		# objective is one term longer
		self.assertTrue( data_also_small_c_len == data_small_c_len + 1 )
		# TODO: Why are 2 rows added to end of G instead of 1?
		self.assertTrue( data_also_small_G_rows == data_small_G_rows + 2 )

		cs.anatomy['oar'].constraints += D(80) <= 0.2

		cs.plan(solver='ECOS')
		data_larger = cs.problem.solver.problem.get_problem_data('ECOS')
		data_larger_solvetime = cs.solvetime
		data_larger_c_len = len(data_larger['c'])
		data_larger_G_rows = data_larger['G'].shape[0]

		self.print_dims(data_larger)
		print('solution found in {} seconds\n'.format(data_larger_solvetime))

		# Test percentile constraint on OAR results in larger problem
		self.assertTrue(data_larger_solvetime > data_also_small_solvetime)
		self.assertTrue(data_larger_c_len > data_also_small_c_len)
		self.assertTrue(data_larger_G_rows > data_also_small_G_rows)
