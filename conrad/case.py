"""
case.py docstring
"""

import os
import sys
import numpy as np
import matplotlib
import cxvpy

class Case(object):
	"""Case object docstring"""
	
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
		
		# Compute weights in objective function
		alpha = wt_over / wt_under
		c_ = (alpha + 1)/2
		d_ = (alpha - 1)/2
		c = c_ * (b > 0) + wt_oar * (b == 0)
		d = d_ * (b > 0)
		
		# Define variables
		n = self.num_dvh_constr()
		x = Variable(self.num_beams())
		beta = Variable(n)
		
		# Define objective and constraints
		obj = Minimize( c.T * abs(A*x - b) + d.T * (A*x - b) )
		constraints = [x >= 0]
		
		if flex_constrs:
			b_slack = Variable(n)
			obj += Minimize( wt_slack * sum_entries(b_slack) )
			constraints += [b_slack >= 0]
		
		constraints += _prob_dvh_constrs(A, b, beta, b_slack, flex_constrs)
		
		prob = Problem(obj, constraints)
		prob.solve(solver = solver)
		return(prob, x)
	
	def _prob_dvh_constrs(self, A, b, b_slack, beta, flex_constrs = False):
		constr_idx = 0
		prob_constraints = []
		n_structures = len(self.structures)
		
		for s in xrange(n_structures):
			i_start = self.structures[s].pointer
			i_end = self.structures[s + 1].pointer - 1
			A_sub = A[i_start : i_end, :]
			b_sub = b[i_start : i_end, :]
			
			for dvh_constr in self.dvh_constrs_by_struct[s].constraints:
				p = self.structures[s].size * (dvh_constr.percentile / 100.)
				sign = -1 + 2 * dvh_constr.upper_bound
				b_ = dvh_constr.dose
				if flex_constrs:
					b_ += sign * b_slack[constr_idx]
				constr = sum_entries(pos(beta[constr_idx] + sign * (A_sub * x - b_) )) <= beta[constr_idx] * p
				prob_constraints.append(constr)
				constr_idx += 1
		
		return prob_constraints
