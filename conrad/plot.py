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
from conrad.case import Case

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
		n_cols += 1
	return n_cols

class DVHPlot(object):
	""" TODO: docstring """

	def __init__(self, panels_by_structure, names_by_structure):
		""" TODO: docstring """
		self.fig = plt.figure()
		self.n_structures = len(panels_by_structure.keys())
		self.panels_by_structure = panels_by_structure
		self.n_panels = max(panels_by_structure.values())
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
		for label, color in colors_by_structure.items():
			self.colors_by_structure[label] = color

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


	def plot(self, plot_data, show = False, clear = True, **options):
		""" TODO: docstring """
		if clear: self.fig.clf()

		max_dose = max([data['curve']['dose'].max() for data in plot_data.values()])
		xmax = options.pop('xmax', None)
		legend = options.pop('legend', False)

		for label, data in plot_data.items():
			plt.subplot(self.rows, self.cols, self.panels_by_structure[label])

			color = self.colors_by_structure[label]
			name = self.names_by_structure[label]
			plt.plot(data['curve']['dose'], data['curve']['percentile'],
				color=color, label=name, **options)
			plt.title(name)

			for constraint in data['constraints']:
				# TODO: What should we plot for other constraints like mean, min, max, etc?
				if constraint[1]['type'] is 'percentile':
					plt.plot(
							constraint[1]['dose'][0],
							constraint[1]['percentile'][0],
							constraint[1]['symbol'],
							alpha=0.7, color=color, label=None, **options)
					plt.plot(
							constraint[1]['dose'][1],
							constraint[1]['percentile'][1],
							constraint[1]['symbol'], label=None, color=color,
							**options)
					slack = abs(constraint[1]['dose'][1] -
								constraint[1]['dose'][0])
					if slack > 0.1:
						plt.plot(constraint[1]['dose'],
								 constraint[1]['percentile'], ls='--',
								 label=None, color=color, **options)

					# So we don't cut off DVH constraint labels
					max_dose = max(max_dose, constraint[1]['dose'][0])

			xlim_upper = xmax if xmax is not None else 1.1 * max_dose

		plt.xlim(0, xlim_upper)
		plt.ylim(0, 103)
		if legend:
			labels = [self.names_by_structure[label] for label in plot_data]
			plt.legend(labels, ncol=1, loc='upper right',
					   bbox_to_anchor=[1.1, 0.9], columnspacing=1.0,
					   labelspacing=0.0, handletextpad=0.0,
           			   handlelength=1.5, fancybox=True, shadow=True)

		if show: SHOW()

	def save(self, filepath, overwrite=True):
		filepath = path.abspath(filepath)
		directory = path.dirname(filepath)
		if not path.isdir(path.dirname(filepath)):
			raise ValueError('argument "filepath" specified with invalid'
							 'directory')
		elif not overwrite and path.exists(filepath):
			raise ValueError('argument "filepath" specifies an existing file'
							 'and argument "overwrite" is set to False')
		else:
			try:
				print "SAVING TO ", filepath
				plt.savefig(filepath, bbox_inches = 'tight')
			except:
				raise RuntimeError('could not save plot to file: {}'.format(
								   filepath))

	def __del__(self):
		""" TODO: docstring """
		plt.close(self.fig)

class CasePlotter(object):
	def __init__(self, case):
		"""
		Initializes :class:`CasePlotter` object.

		Uses structure information from argument "case" to initialize a
		:class:`DVHPlot` object with the names and labels of each
		structure associated with the case.

		Args:
			case : :class:`Case`

		Returns:
			None

		Raises:
			None
		"""
		if not isinstance(case, Case):
			TypeError('argument "case" must be of type conrad.Case')

		# plot setup
		panels_by_structure = {}
		names_by_structure = {}
		for idx, label in enumerate(case.label_order):
			# place all curves on one panel by default (clinicians' preference)
			panels_by_structure[label] = 1
			names_by_structure[label] = case.structures[label].name
		self.dvh_plot = DVHPlot(panels_by_structure, names_by_structure)
		self.valid_labels = case.structures.keys()


	def set_display_groups(grouping='together', group_list=None):
		"""
		Specifies structure-to-panel assignments for display.

		Args:
			grouping : {'together', 'separate', 'list'}
					if 'together', all curves plotted on single panel
					if 'separate', each curve plotton on its own panel
					if 'list', curves grouped according to "group_list"
			group_list : list of tuple
					if provided as a :class:`list` of :class:`tuple`s,
					each element of the i-th :class:`tuple` is assumed
					to be a valid structure label, and the DVH curve for
					the corresponding structure is assigned to panel i

		Returns:
			None

		Raises:
			Raises ValueError if "grouping" is not one of
					('together', separate', 'list').
			Raises ValueError if members of "group_list" are not each a
					:class:`tuple`.
			Raises ValueError if each label in each :class:`tuple` in
				   "group_list" does not correspond to a structure label
				   from the case used to initializes this
				   :class:`CasePlotter` object.
		"""

		if not isinstance(grouping, str):
			raise TypeError('argument "grouping" must be of type {}'.format(
							str))
		if grouping not in ('together', 'separate', 'list'):
			raise ValueError('argument "grouping" must be one of the '
							 'following: ("together", "separate", '
							 'or "list")')

		if grouping == 'together':
			for k in self.dvh_plot.panels_by_structure:
				self.dvh_plot.panels_by_structure[k] = 1
		elif grouping == 'separate':
			for i, k in enumerate(self.dvh_plot.panels_by_structure):
				self.dvh_plot.panels_by_structure[k] = i + 1
		elif grouping == 'list':
			valid &= isinstance(group_list, list)
			valid &= all(map(lambda x: isinstance(x, tuple), group_list))
			if valid:
				for i, group in enumerate(group_list):
					for label in group:
						if label in self.valid_labels:
							self.dvh_plot.panels_by_structure[label] = i + 1
						else:
							raise ValueError('specified label {} in tuple {} '
											 'does not correspond to any known'
											 'structure labels in the current'
											 'case'.format(label))
			else:
				raise TypeError('')





	def plot(self, case, show=False, clear=True, subset=None, plotfile=None,
		  	 **options):
		"""
		Plots dose data given current state of argument "case".


		Args:
			case: A :class:`Case` instance. case.plotting_data is used
				to build the DVH curves.
			show: bool, toggles whether plot is displayed.
			clear: bool, toggles whether plot is cleared before data
				from "case" is appended.
			subset: list or tuple, optional. specifies structure labels
				of DVH curves to be plotted, other are suppressed. All
				structures' DVH curves are plotted by default.
			plotfile: str, optional. When provided, passed to to the
				:class:`DVHPlot` object to as a target filepath to save
				the drawn plot.
			options: Optional keyword arguments to be passed to
				matplotlib

		Returns:
			None

		Raises:
			raises TypeError if "subset" is specified but not a list or
				tuple
			raises KeyError if "subset" is specified but contains items
				that are not recognized as valid structure labels given
				the Case object used to initialize this CasePlotter
				instance.
		"""
		plotfile = options.pop('file', None)
		data_ = case.plotting_data

		# filter data to only plot DVH for structures with requested labels
		if subset is None:
			data = data_
		else:
			if not isinstance(subset, (list, tuple)):
				raise TypeError('argument "subset" must be of type {} or '
								'{}'.format(list, tuple))
			if not all([label in data_.keys() for label in subset]):
				raise KeyError('argument "subset" specifies an invalid '
							   'structure label')
			data = {}
			for label in subset:
				data[label] = data_[label]

		self.dvh_plot.plot(
				data,
				show=show,
				clear=clear,
				**options)
		if plotfile is not None:
			self.dvh_plot.save(plotfile)