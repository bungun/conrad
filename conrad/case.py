"""
case.py docstring
"""

import os
import sys
import numpy as np
import matplotlib
import cxvpy

class Case(object):
	"""TODO:

	initialize a case with:
		- dose matrix, A, matrix \in R^m x n, m = # voxels, n = # beams
		- voxel labels, (int) vector in \R^m
		- prescription: dictionary where each element has the fields:
			label (int, also the key)
			dose (float)
			is_target (bool)
			dose_constraints (dict, or probably string/tuple list)
	"""

	# def __init__(self):
	# 	"""Case.__init__() docstring"""
	# 	self.problem = None
	# 	self.clinical_spec = None
	# 	self.voxel_labels = None
	# 	self.structures = {}

	# 	# dose matrix
	# 	self.A = None
	# 	# self.shape = (self.voxels, self.beams) = A.shape

	# 	# (most recent) beam intensity design
	# 	self.x = None
	# 	self.run_records = {}

	# def add_dvh_constraint(self):
	# 	pass
	# 	# return constr id?

	# def drop_dvh_constraint(self, constr_id):
	# 	pass

	# def form_objective(self):
	# 	# do something with self.structures, self.problem
	# 	pass

	# def form_constraints(self):
	# 	pass

	# def form_problem(self):
	# 	pass

	# def plan(self):
	# 	# should emit run record with run id
	# 	pass

	# def get_design(self):
	# 	pass

	
	def __init__(self, structures, prescription, num_beams, dvh_constrs_by_struct = None):
		if constraints is None:
			constraints = []
		self.num_beams = num_beams		   # TODO: Extract from A matrix
		self.prescription = prescription
		self.structures = structures
		self.dvh_constrs_by_struct = dvh_constrs_by_struct
	
	def num_beams(self):
		return self.num_beams
	
	def num_dvh_constr(self):
		return sum([dc.count for dc in self.dvh_constrs_by_struct])
	
	def plan(self, wt_under = 1., wt_over = 0.05, wt_oar = 0.2, solver = ECOS, flex_constrs = False, second_pass = False):
		b = self.prescription
		n = self.num_beams()
		n_constr = self.num_dvh_constr()
		
		# Compute weights in objective function
		alpha = wt_over / wt_under
		c_ = (alpha + 1)/2
		d_ = (alpha - 1)/2
		c = c_ * (b > 0) + wt_oar * (b == 0)
		d = d_ * (b > 0)
		
		# Define variables
		x = Variable(n)
		beta = Variable(n_constr)
		
		# Define objective and constraints
		obj = Minimize( c.T * abs(A*x - b) + d.T * (A*x - b) )
		constraints = [x >= 0]
		
		if flex_constrs:
			b_slack = Variable(n_constr)
			obj += Minimize( wt_slack * sum_entries(b_slack) )
			constraints += [b_slack >= 0]
		constraints += self._prob_dvh_constrs(A, b, beta, b_slack, flex_constrs)
		
		prob = Problem(obj, constraints)
		prob.solve(solver = solver)
		if not second_pass:     # TODO: Return beta and b_slack as well?
			return (prob, x)
		
		# Second pass with exact voxel DVH constraints
		x_exact = Variable(n)
		constraints_exact = [x_exact >= 0]
		constraints_exact += self._prob_exact_constrs(A, b, x, x_exact)
		prob_exact = Problem(obj, constraints_exact)
		prob_exact.solve(solver = solver)
		return (prob_exact, x_exact)
	
	# Upper bound: \sum max(beta + (Ax - (b + b_slack)), 0) <= beta * p
	# Lower bound: \sum max(beta - (Ax - (b - b_slack)), 0) <= beta * p
	@staticmethod
	def dvh_restriction(A, x, b, p, beta, upper = True, slack = 0):
		sign = 1 if upper else -1
		return sum_entries(pos( beta + sign * (A * x - (b + sign * slack)) )) <= beta * p
	
	# Constrain only p voxels that satisfy DVH constraint by largest margin
	@staticmethod
	def dvh_exact_constrs(A, x, b, p, x_exact, upper = True):
		sign = 1 if upper else -1
		constr_diff = sign * (A.dot(x.value) - b)
		idx_sort_diff = np.argsort(constr_diff, axis = 0)
		idx_sub = idx_sort_diff[0:floor(p)]
		return sign * (A[idx_sub, :] * x_exact - b) <= 0
	
	# Restrict DVH constraints using convex approximation
	def _prob_dvh_constrs(self, A, x, b, beta, b_slack, flex_constrs = False):
		constr_idx = 0
		constr_solver = []
		n_structures = len(self.structures)
		
		for s in xrange(n_structures):
			i_start = self.structures[s].pointer
			i_end = self.structures[s + 1].pointer - 1
			A_sub = A[i_start : i_end, :]
			b_sub = b[i_start : i_end, :]
			
			for dvh_constr in self.dvh_constrs_by_struct[s].constraints:
				p = self.structures[s].size * (dvh_constr.percentile / 100.)
				# b_ = dvh_constr.dose
				# sign = -1 + 2 * dvh_constr.upper_bound
				# if flex_constrs:
				#	b_ += sign * b_slack[constr_idx]
				# constr = sum_entries(pos(beta[constr_idx] + sign * (A_sub * x - b_) )) <= beta[constr_idx] * p
				
				slack = b_slack[constr_idx] if flex_constrs else 0
				constr = dvh_restriction(A_sub, x, dvh_constr.dose, p, beta[constr_idx], dvh_constr.upper_bound, slack)
				constr_solver.append(constr)
				constr_idx += 1
		return constr_solver
	
	# Determine exact voxels to constrain
	def _prob_exact_constrs(self, A, b, x, x_exact):
		constr_idx = 0
		constr_exact = []
		n_structures = len(self.structures)
	
		for s in xrange(n_structures):
			i_start = self.structures[s].pointer
			i_end = self.structures[s + 1].pointer - 1
			A_sub = A[i_start : i_end, :]
			b_sub = b[i_start : i_end, :]
			
			for dvh_constr in self.dvh_constrs_by_struct[s].constraints:
				p = self.structures[s].size * (dvh_constr.percentile / 100.)
				# b_ = dvh_constr.dose
				# sign = -1 + 2 * dvh_constr.upper_bound
				
				constr = dvh_exact_constrs(A_sub, x, dvh_constr.dose, p, x_exact, dvh_constr.upper_bound)
				constr_exact.append(constr)
				constr_idx += 1
		return constr_exact
	
