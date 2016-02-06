import matplotlib
from os import path
from math import ceil 
from numpy import linspace
from os import getenv

if getenv('DISPLAY') is not None: 
	import matplotlib.pyplot as plt
	SHOW = plt.show
else:
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	SHOW = lambda : None

from matplotlib.pyplot import get_cmap
from matplotlib.colors import LinearSegmentedColormap

# TODO: unit test

"""
TODO: plot.py docstring
"""
# plotting logic

def panels_to_cols(n_panels):
	n_cols = 1
	if n_panels > 1:
		n_cols += 1
	if n_panels > 4:
		n_cols += 1
	if n_panels > 7:
		panels += 1
	return n_cols

class DVHPlot(object):
	""" TODO: docstring """

	def __init__(self, panels_by_structure, names_by_structure):
		""" TODO: docstring """
		self.fig = plt.figure()
		self.n_structures = len(panels_by_structure.keys())
		self.panels_by_structure = panels_by_structure
		self.n_panels = max(panels_by_structure.itervalues())
		self.cols = panels_to_cols(self.n_panels)
		self.rows = int(ceil(float(self.n_panels) / self.cols))
		self.names_by_structure = names_by_structure
		self.colors_by_structure = {}

		# set colors to something to start
		self.autoset_series_colors()


	def set_series_names(self, names_by_structure):
		""" TODO: docstring """
		self.names_by_structure = names_by_structure
		

	def set_series_colors(self, colors_by_structure):
		""" TODO: docstring """
		self.colors_by_structure = colors_by_structure

	def autoset_series_colors(self, structure_order_dict = None, colormap = None):
		""" TODO: docstring """
		if isinstance(colormap, LinearSegmentedColormap):
			colors = map(colormap, linspace(0.1, 0.9, self.n_structures))
		else:
			cmap = get_cmap('rainbow')
			colors = map(cmap, linspace(0.9, 0.1, self.n_structures))
		
		for idx, label in enumerate(self.panels_by_structure.keys()):
			if structure_order_dict is not None:
				self.colors_by_structure[label] = colors[structure_order_dict[label]]
			else:
				self.colors_by_structure[label] = colors[idx]


	def plot(self, plot_data, **options):
		""" TODO: docstring """
		self.fig.clf()

		max_dose = max([data['curve']['dose'].max() for data in plot_data.itervalues()])


		for label, data in plot_data.iteritems():
			plt.subplot(self.rows, self.cols, self.panels_by_structure[label])
			
			color = self.colors_by_structure[label]
			name = self.names_by_structure[label]
			plt.plot(data['curve']['dose'], data['curve']['percentile'],
				color = color, label = name, **options)
			plt.xlim(0, 1.1 * max_dose)
			plt.ylim(0, 100)

			for constraint in data['constraints']:
				plt.plot(constraint['dose'][0], constraint['percentile'][0], 
					constraint['symbol'], color = color, **options)
				plt.plot(constraint['dose'][1], constraint['percentile'][1], 
					constraint['symbol'], alpha  = 0.7, color = color, **options)
				plt.plot(constraint['dose'], constraint['percentile'], 
					'-', color = color, **options)

		SHOW()

	def save(self, filepath):
		try:
			plt.savefig(filepath, bbox_inches = 'tight')
		except:
			print str('could not save plot to file: {}'.format(filepath))

	def __del__(self):
		""" TODO: docstring """
		plt.close(self.fig)
