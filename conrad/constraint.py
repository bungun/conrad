import os
import sys
import numpy as np
import matplotlib

class DVHConstraint(object):
	def __init__(self, dose, percentile, bound_type):
		if bound_type not in ('upper', 'lower'):
			ValueError("argument `bound_type` must either be `upper` or `lower`")
		self.dose = dose
		self.percentile = percentile
		self.upper_bound = bound_type == 'upper'
		
	@property
	def tuple(self):
		return (self.dose, self.percentile, self.upper_bound)
	
	@property	
	def is_upper(self):
		return self.upper_bound

class DVHConstraintList(object):
	def __init__(self, *constraints, **options):
		self.constraints = []
		self.upper_constraints = [] 
		self.lower_constraints = []
		self.add(*constraints)
		self.options = options

	@property
	def count(self):
	    return len(self.upper_constraints) + len(self.lower_constraints)
	
	def add(self, *constraints):
		if constraints is None: return
		for c in constraints:
			if not isinstance(c, DVHConstraint):
				TypeError("input `constraints` must be "
					"a tuple of Constraint objects.\n"
					"Provided: {}".format(type(c)))

			self.constraints.append(c)
			if c.upper_bound:
				self.upper_constraints.append(c.tuple)
			else:
				self.lower_constraints.append(c.tuple)

	def plot(self, canvas, **options):
		for key in self.options:
			if not key in options:
				options[key]=self.options[key]

		uc_dose = [c[0] for c in self.upper_constraints]
		uc_pct = [c[1] for c in self.upper_constraints]
		lc_dose = [c[0] for c in self.lower_constraints]
		lc_pct = [100 - c[1] for c in self.lower_constraints]

		canvas.plot(uc_dose, uc_pct, '<', **options)
		canvas.plot(lc_dose, lc_pct, '>', **options)
