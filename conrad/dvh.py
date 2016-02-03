"""
dvh.py docstring
"""


class DoseConstraint(object):
	"""
	
	"""

	def __init__(self, dose, percentile, direction):
		self.id = None
		self.dose_requested = dose
		self.percentile = percentile
		self.direction = direction # this could also be called constr_type or something
		self.dose_actual = None

		# TODO: figure out how to specify constraint direction (ge/le, upper/lower, ...)

	def set_actual_dose(self, slack):
		if self.direction == "upper":
			self.dose_actual = self.dose_requested + slack
		else:
			self.dose_actual = self.dose_requested - slack


	@property
	def constr_id(self):
		""" TODO: return constraint UUID ? """
		pass

	@property
	def plot_data(self):
		""" TODO: return plottable data """
	    pass
	

class DVHCurve(object):
	def __init__(self):
		pass

	@property
	def plot_data(self):
		""" TODO: return plottable data """
		pass

