"""
Copyright 2016 Baris Ungun, Anqi Fu

This file is part of CONRAD.

CONRAD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CONRAD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CONRAD.  If not, see <http://www.gnu.org/licenses/>.
"""
from os import path
from math import ceil
from numpy import linspace
from os import getenv

from conrad.compat import *
from conrad.defs import module_installed
from conrad.optimization.history import RunRecord
from conrad.case import Case

# allow for CONRAD use without plotting by making visualization types
# optional
if module_installed('matplotlib'):
	PLOTTING_INSTALLED = True

	import matplotlib
	if getenv('DISPLAY') is not None:
		import matplotlib.pyplot as plt
		SHOW = plt.show
	else:
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		SHOW = lambda : None

	from matplotlib.pyplot import get_cmap
	from matplotlib.colors import LinearSegmentedColormap
else:
	PLOTTING_INSTALLED = False

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
	if n_panels > 6:
		n_cols += 1
	return n_cols

if not PLOTTING_INSTALLED:
	DVHPlot = lambda arg1, arg2: None
	CasePlotter = lambda arg1: None
else:
	class DVHPlot(object):
		""" TODO: docstring """

		def __init__(self, panels_by_structure, names_by_structure):
			""" TODO: docstring """
			self.fig = plt.figure()
			self.n_structures = len(panels_by_structure)
			self.__panels_by_structure = panels_by_structure
			self.n_panels = max(panels_by_structure.values())
			self.cols = panels_to_cols(self.n_panels)
			self.rows = int(ceil(float(self.n_panels) / self.cols))
			self.__names_by_structure = names_by_structure
			self.__colors_by_structure = {}

			# set colors to something to start
			self.autoset_series_colors()

		@property
		def series_panels(self):
			return self.__panels_by_structure

		@property
		def series_names(self):
			return self.__names_by_structure

		@series_names.setter
		def series_names(self, names_by_structure):
			""" TODO: docstring """
			for label, name in names_by_structure.items():
				self.__names_by_structure[label] = color

		@property
		def series_colors(self):
			return self.__colors_by_structure

		@series_colors.setter
		def set_series_colors(self, colors_by_structure):
			""" TODO: docstring """
			for label, color in colors_by_structure.items():
				self.__colors_by_structure[label] = color

		def autoset_series_colors(self, structure_order_dict=None, colormap=None):
			""" TODO: docstring """
			if isinstance(colormap, LinearSegmentedColormap):
				colors = listmap(colormap, linspace(0.1, 0.9, self.n_structures))
			else:
				cmap = get_cmap('rainbow')
				colors = listmap(cmap, linspace(0.9, 0.1, self.n_structures))

			for idx, label in enumerate(self.series_panels.keys()):
				if structure_order_dict is not None:
					self.series_colors[label] = colors[structure_order_dict[label]]
				else:
					self.series_colors[label] = colors[idx]


		def plot(self, plot_data, show=False, clear=True, xmax=None, legend=False,
				 title=None, self_title=False, large_markers=False,
				 suppress_constraints=False, suppress_xticks=False,
				 suppress_yticks=False, suppress_xlabel=True, suppress_ylabel=True,
				 **options):
			""" TODO: docstring """
			if clear:
				self.fig.clf()

			max_dose = max([data['curve']['dose'].max() for data in plot_data.values()])
			marker_size = 16 if large_markers else 12

			for label, data in plot_data.items():
				plt.subplot(self.rows, self.cols, self.series_panels[label])

				color = self.series_colors[label]
				name = self.series_names[label] if legend else '_nolegend_'
				plt.plot(data['curve']['dose'], data['curve']['percentile'],
					color=color, label=name, **options)
				if self_title:
					plt.title(name)
				elif title is not None:
					plt.title(title)

				if data['rx'] > 0:
					plt.axvline(x=data['rx'], linewidth=1, color=color,
								linestyle='dotted', label='_nolegend_')

				if suppress_constraints:
					continue

				for constraint in data['constraints']:
					# TODO: What should we plot for other constraints like mean, min, max, etc?
					if constraint[1]['type'] is 'percentile':
						plt.plot(
								constraint[1]['dose'][0],
								constraint[1]['percentile'][0],
								constraint[1]['symbol'], alpha=0.55, color=color,
								markersize=marker_size, label='_nolegend_',
								**options)
						plt.plot(
								constraint[1]['dose'][1],
								constraint[1]['percentile'][1],
								constraint[1]['symbol'], label='_nolegend_',
								color=color, markersize=marker_size, **options)
						slack = abs(constraint[1]['dose'][1] -
									constraint[1]['dose'][0])
						if slack > 0.1:
							plt.plot(constraint[1]['dose'],
									 constraint[1]['percentile'], ls='-',
									 alpha=0.6, label='_nolegend_', color=color)

						# So we don't cut off DVH constraint labels
						max_dose = max(max_dose, constraint[1]['dose'][0])

			xlim_upper = xmax if xmax is not None else 1.1 * max_dose

			plt.xlim(0, xlim_upper)
			plt.ylim(0, 103)
			if suppress_yticks:
				plt.yticks([])
			else:
				plt.yticks(fontsize=14)
			if suppress_xticks:
				plt.suppress_xticks([])
			else:
				plt.yticks(fontsize=14)
			if not suppress_xlabel:
				plt.xlabel('Dose (Gy)', fontsize=16)
			if not suppress_ylabel:
				plt.ylabel('Percentile')
			if legend:
				plt.legend(ncol=1, loc='upper right', columnspacing=1.0,
						   labelspacing=0.0, handletextpad=0.0, handlelength=1.5,
						   fancybox=True, shadow=True)

			if show:
				SHOW()

		def save(self, filepath, overwrite=True, verbose=False):
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
					if verbose:
						print("SAVING TO ", filepath)
					plt.savefig(filepath, bbox_inches='tight')
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
			panels_by_structure = {label: 1 for label in case.anatomy.label_order}
			names_by_structure = {
					label: case.anatomy[label].name for
					label in case.anatomy.label_order}
			self.dvh_plot = DVHPlot(panels_by_structure, names_by_structure)
			self.__labels = {}
			for s in case.anatomy:
				self.__labels[s.label] = s.label
				self.__labels[s.name] = s.label

		def label_is_valid(self, label):
			return label in self.__labels

		def set_display_groups(self, grouping='together', group_list=None):
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
				for k in self.dvh_plot.series_panels:
					self.dvh_plot.series_panels[k] = 1
			elif grouping == 'separate':
				for i, k in enumerate(self.dvh_plot.series_panels):
					self.dvh_plot.series_panels[k] = i + 1
			elif grouping == 'list':
				valid = isinstance(group_list, list)
				valid &= all(map(lambda x: isinstance(x, tuple), group_list))
				if valid:
					for i, group in enumerate(group_list):
						for label in group:
							if label in self.__labels:
								self.dvh_plot.series_panels[
										self.__labels[label]] = i + 1
							else:
								raise ValueError('specified label {} in tuple {} '
												 'does not correspond to any known'
												 'structure labels in the current'
												 'case'.format(label, group))
				else:
					raise TypeError('TODO: explain error')


		def plot(self, data, second_pass=False, show=False, clear=True,
				 subset=None, plotfile=None, **options):
			"""
			Plots dose data given current state of argument "case".


			Args:
				data: dict, used to build the DVH curves.
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
			if isinstance(data, RunRecord):
				if second_pass and data.plotting_data['exact'] is not None:
					data = data.plotting_data['exact']
				else:
					data = data.plotting_data[0]
			data_ = data

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