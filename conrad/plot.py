import matplotlib
from matplotlib import pyplot as plt
from os import path
from math import ceil 
from numpy import linspace

# TODO: unit test

"""
TODO: plot.py docstring
"""
#plotting logic

def panels_to_cols(n_panels):
	n_cols = 1
	if n_panels > 1:
		n_cols += 1
	if n_panels > 4:
		n_cols += 1
	if n_panels > 7
		panels += 1

class DVHPlot(object):
	""" TODO: docstring """

	def __init__(self, panels_by_structure, names_by_structure):
		""" TODO: docstring """
		self.fig = plt.figure()
		self.n_structures = len(panels_by_structure.keys())
		self.panels_by_structure = panels_by_structure
		self.n_panels = n_panels
		self.cols = panels_to_cols(n_panels)
		self.rows = int(ceil(float(n_panels) / self.cols))
		self.names_by_structure = names_by_structure
		self.colors_by_structure = None

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
			colors = map(colormap, linspace(0.1, 0.9, n_structures))
		else:
			cmap = get_cmap('rainbow')
			colors = map(cmap, np.linspace(0.9,0.1,len(dvhlist)))
		
		for idx, label in enumerate(self.panels_by_structure.keys()):
			if structure_order_dict is not None:
				self.colors_by_structure[label] = colors[structure_order_dict[label]]
			else:
				self.colors_by_structure[label] = colors[idx]


	def plot(self, plot_data, **options):
		""" TODO: docstring """
		plt.clear(self.fig)


		max_dose = reduce(max, 
			data['curve']['dose'].max() for data in plot_data.itervals())


		for label, data in plot_data:
			plt.subplot(self.rows, self.cols, self.panels_by_structure[label])
			
			color = self.colors_by_structure[label]
			name = self.names_by_structure[label]
			plt.plot(data['curve']['percentile'], data['curve']['dose'],
				color = color, label = name, **options)
			plt.xlim(0, 1.1 * maxdose)
			plt.ylim(0, 100)

			for constraint in data['constraints']:
				plt.plot(constraint['percentile'][0], 
					constraint['symbol'][0], **options)
				plt.plot(constraint['percentile'][0], 
					constraint['symbol'][0], alpha  = 0.7, **options)

		plt.show()

	def save(self, filepath):
		try:
			plt.savefig(filepath, bbox_inches = 'tight')
		except:
			print str('could not save plot to file: {}'.format(filepath))

	def __del__(self):
		""" TODO: docstring """
		plt.close(self.fig)