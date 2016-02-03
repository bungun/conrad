import os
import sys
import numpy as np
import matplotlib

class DVH(object):
	def __init__(self, dose, percentile, **options):
		self.dose = dose
		self.percentile = percentile
		self.options = options
		self.maxdose = None
		
	def plot(self, canvas, **options):
		for key in self.options:
			if not key in options:
				options[key]=self.options[key]
		canvas.plot(self.dose, self.percentile, **options)
		if self.maxdose is not None:
			canvas.xlim(0, 1.1 * self.maxdose)
		else:
			canvas.xlim(0, 1.1 * np.max(self.dose))
		canvas.ylim(0, 100)
