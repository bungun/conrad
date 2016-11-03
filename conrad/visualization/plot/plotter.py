"""
TODO: DOCSTRING
"""
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
from conrad.compat import *

from numpy import linspace

from conrad.case import Case
from conrad.optimization.history import RunRecord
from conrad.visualization.plot.elements import LineAesthetic
from conrad.visualization.plot.collections import PlanGraphBase, PlanDVHGraph
from conrad.visualization.plot.canvases import DVHPlot

# import matplotlib into namespace as mpl, with lines, axes, figure,
# color and pyplot modules available. also imports constant
# PLOTTING_INSTALLED and DISPLAY AVAILABLE.
from conrad.visualization.plot.mpl import *

if not PLOTTING_INSTALLED:
	CasePlotter = lambda arg1: None
else:
	class CasePlotter(object):
		"""
		Wrap :class:`DVHPlot` for visualizing treatment plan data.

		Attributes:
			dvh_plot (:class:`DVHPlot`): Dose volume histogram plot.

		Examples:
			>>> # intialize based on an existing :class:`Case` object "case"
			>>> plotter = CasePlotter(case)

			>>> # form treatment plan with case
			>>> _, run = case.plan(**args)

			>>> # plot the output emitted by the case.plan() call
			>>> plotter.plot(run, **options)
		"""
		def __init__(self, case, subset=None):
			"""
			Initialize :class:`CasePlotter`.

			Use structure information from ``case`` to initialize a
			:class:`DVHPlot` object with the names and labels of each
			structure associated with the case.

			Args:
				case (:class:`Case`): Treatment planning case to use as
					basis for configuring object's :class:`DVHPlot`
				subset (:obj:`list`, optional): List of labels of
					structures to use in a restricted plotting context.

			Raises:
				TypeError: If argument is not of type :class:`Case`
			"""
			if not isinstance(case, Case):
				raise TypeError('argument "case" must be of type conrad.Case')

			# plot setup
			self.__dvh_plot = None
			self.__dvh_set = None
			self.__structure_subset = case.anatomy.label_order
			self.__structure_colors = {}
			self.__tag2label = {s.label: s.label for s in case.anatomy}
			self.__tag2label.update({s.name: s.label for s in case.anatomy})
			self.__grouping = 'together'

			if subset is not None:
				self.structure_subset = subset
			self.__dvh_plot = DVHPlot({label: 0 for label in self.structure_subset})

			self.autoset_structure_colors()

		@property
		def dvh_plot(self):
			return self.__dvh_plot

		@property
		def dvh_set(self):
			return self.__dvh_set

		@dvh_set.setter
		def dvh_set(self, data):
			if isinstance(data, PlanDVHGraph):
				data = data.structure_DVHs

			self.__dvh_set = PlanDVHGraph(self.filter_data(data))
			self.dvh_set.structure_colors = self.structure_colors

		@property
		def n_structures(self):
			return len(self.structure_subset)

		@property
		def structure_colors(self):
			return self.__structure_colors

		@structure_colors.setter
		def structure_colors(self, color_dict):
			self.__structure_colors.update(self.filter_data(color_dict, None))

		def autoset_structure_colors(self, structure_order=None,
								  colormap='viridis'):
			"""
			Set series colors with a (possibly default)
			:class:`matplotlib.colors.LinearSegmentedColormap`.

			Args:
				structure_order (:obj:`list`, optional): Permuted list
					of structure labels/names with an entry for each
					structure in :attr:`CasePlotter.structure_subset`;
					enumeration order determines order that
					automatically generated colors are applied to the
					structures.
				colormap (:obj:`str`, optional) Assumed to be a valid
					:mod:`matplotlib.pyplot` colormap name.

			Returns:
				None

			Raises:
				ValueError: if ``structure_order`` provided and does not
					yield a valid permutation of
					:attr:`CasePlotter.structure_subset` or ``colormap``
					provided and is not a valid
					:mod:`matplotlib.pyplot` colormap name
			"""
			if structure_order is not None:
				order = self.filter_labels(structure_order)
			else:
				order = self.structure_subset

			if len(order) != self.n_structures:
				raise ValueError(
						'if provided, `structure_order` must have '
						'an entry for each structure enumerated in '
						'{}.structure_subset'.format(CasePlotter))

			if not isinstance(colormap, mpl.colors.LinearSegmentedColormap):
				cmap = mpl.pyplot.get_cmap(colormap)
			colors = listmap(cmap, linspace(1., 0., self.n_structures))
			self.structure_colors.update({label: colors[idx] for idx, label in
										  enumerate(order)})

		@property
		def structure_subset(self):
			return self.__structure_subset

		@structure_subset.setter
		def structure_subset(self, subset):
			if subset is not None:
				if not all(t in self.__tag2label for t in subset):
					raise ValueError(
							'invalid structure label/name listed in '
							'argument `subset`')
				self.__structure_subset = list({
						self.__tag2label[t] for t in subset})

		def filter_labels(self, subset):
			filtered = []
			subset = (self.__tag2label[tag] for tag in subset
					  if tag in self.__tag2label)
			for label in subset:
				if label not in filtered:
					filtered.append(label)
			return filtered

		def filter_data(self, data, subset=None):
			if subset is None:
				subset = self.structure_subset
			else:
				subset = self.filter_labels(subset)

			iterator = data if isinstance(data, PlanGraphBase) else data.items()
			output = {}
			for tag, value in iterator:
				if tag not in self.__tag2label:
					continue
				label = self.__tag2label[tag]
				if label in subset and label not in output:
					output[label] = value
			return output

		@property
		def grouping(self):
			"""
			Specify structure-to-panel assignments for display.

			Args:
				grouping (:obj:`str`, optional): Should be one of
					'together', 'separate', or 'list'. If 'together',
					all curves plotted on single panel. If 'separate',
					each curve plotton on its own panel. If 'list',
					curves grouped according to ``group_list``.
				group_list (:obj:`list` of :obj:`tuple`, optional): If
						provided, each element of the i-th :obj:`tuple`
						is assumed to be a valid structure label, and
						the DVH curve for the corresponding structure is
						assigned to panel i.

			Returns:
				None
			"""
			return self.__grouping

		@grouping.setter
		def grouping(self, grouping):
			label2subplot = {}
			if isinstance(grouping, str):
				if grouping == 'separate':
					label2subplot.update({
							label: idx for idx, label in
							enumerate(self.structure_subset)})
				elif grouping == 'together':
					label2subplot.update({
							label: 0 for label in self.structure_subset})
				else:
					raise ValueError(
							'if `grouping` provided as {}, it must '
							'be one of {}'
							''.format(str, ['separate', 'together']))

			elif '__iter__' in dir(grouping):
				for subplot_idx, g in enumerate(grouping):
					if isinstance(g, str):
						label2subplot[g] = subplot_idx
					elif '__iter__' in dir(g):
						for label in g:
							label2subplot[label] = subplot_idx
					else:
						label2subplot[g] = subplot_idx
				label2subplot = self.filter_data(label2subplot)
				grouping = 'list'
				if set(self.structure_subset) != set(label2subplot.keys()):
					raise ValueError(
							'each structure in `{}.structure_subset` '
							'must be represented when `gropuing` is '
							'specified as an iterable collection of '
							'structure labels/structure label iterables'
							''.format(CasePlotter))
			else:
				raise TypeError(
						'argument `grouping` must be of type {} or '
						'be an iterable collection of structure labels '
						'and/or iterables containing structure labels')
			self.__grouping = grouping
			self.dvh_plot.subplot_assignments = label2subplot
			# if self.dvh_set is not None:
				# self.dvh_set.undraw()
			# self.dvh_plot.build()

		def plot(self, data, second_pass=True, show=False, clear=True,
				 subset=None, plotfile=None, aesthetic=None, **options):
			"""
			Plot dose volume histograms from argument `data`.

			Args:
				data (:obj:`dict`, or :class:`RunRecord`): Used to build
					the DVH curves. Assumed to be compatible with the
					`Case` used to initialize this object.
				second_pass (:obj:`bool`, optional): Plot data from
					second planning pass when ``True`` and ``data`` is a
					:class:`RunRecord`.
				show (:obj:`bool`, optional): Show figure after drawing.
				clear (:obj:`bool`, optional): Clear figure before
					rendering data in ``data``.
				subset (:obj:`list` or :obj:`tuple`, optional): Iterable
					collection of labels of DVH curves to be plotted;
					others are suppressed. All structures' DVH curves
					are plotted by default.
				plotfile (:obj:`str`, optional): Passed to to the
					:class:`DVHPlot` as a target filepath to save the
					drawn plot.
				aesthetic (:class:`LineAesthetic`, optional): Passed to
					:meth:`~DVHPlot.plot` to apply to the DVH curves.
				**options: Arbitrary keyword arguments passed through to
					:meth:`~DVHPlot.plot`.

			Returns:
				None
			"""
			if isinstance(data, RunRecord):
				if second_pass and data['exact'] is not None:
					data = data['exact']
				else:
					data = data[0]

			if subset is not None:
				self.subset = subset

			self.dvh_set = data
			self.dvh_plot.plot(self.dvh_set, clear=clear, aesthetic=aesthetic,
							   **options)
			if show:
				self.dvh_plot.show()

			if plotfile is None:
				plotfile = options.pop('file', None)
			if plotfile is not None:
				self.dvh_plot.save(plotfile)

		def plot_twopass(self, data, show=False, clear=True, subset=None,
						 plotfile=None, aesthetics=None, **options):
			if isinstance(data, (dict, RunRecord)):

				data = [data[0], data['exact']]
			else:
				if len(data) != 2:
					raise ValueError('argument `data` must be of type '
									 '{}, or be an iterable collection '
									 'with an entry for each solver '
									 'pass'.format(RunRecord))

			if aesthetics is None:
				aesthetics = [LineAesthetic(style='-'), LineAesthetic(style='--')]
			else:
				if len(aesthetics) != 2:
					raise ValueError('argument `aesthetics`, if '
									 'provided, must specify line '
									 'aesthetics for first and second '
									 'solver passes')

			self.plot(data[0], show=False, clear=clear, subset=subset,
					  plotfile=None, aesthetic=aesthetics[0], **options)
			self.plot(data[1], show=show, clear=False, subset=subset,
					  plotfile=plotfile, aesthetic=aesthetics[1], **options)

		def plot_multi(self, run_data, run_names, reference_data=None,
					   reference_name='reference', show=False, clear=True,
				 	   subset=None, plotfile=None, layout='auto',
				 	   vary_markers=True, vary_marker_sizes=False,
				 	   marker_size_increasing=True,
				 	   universal_marker=None, vary_line_weights=False,
				 	   vary_line_colors=False, vary_line_styles=False,
				 	   darken_reference=True, series_labels_on_plot=False,
				 	   series_label_coordinates=None, **options):
			"""
			Plot data from multiple runs.

			Args:
				run_data (:obj:`list` of :class:`RunRecord` or :obj:`dict`):
					List of plans to be plotted.
				run_names (:obj:`list` of :obj:`str`):
					List of names to associate with each compared plan.
				reference_data (:class:`RunRecord` or :obj:`dict`, optional):
					Reference plan.
				reference_name (:obj:`str`, optional): Name of reference
					plan.
				show (:obj:`bool`, optional): If ``True``, display
					resulting plot in GUI.
				clear (:obj:`bool`, optional): If ``True``, clear canvas
					before drawing DVH curves for specified plans.
				subset (:obj:`list`, optional): If provided, should be a
					list of labels/names of structures in the case; DVH
					curves will be plotted for only the requested subset.
				plotfile (:obj:`str`, optional): If provided, result
					will be saved to the specified path.
				layout (:obj:`str`, optional): Set layout of DVH plot to
					be one of ``'auto'``, ``'horizontal'``, or
					``'vertical'`` to arrange subplots.
				vary_markers (:obj:`bool`, optional): If ``True``,
					marker styles for DVH curves of each non-reference
					plan will be cycled among 6 styles.
				vary_marker_sizes (:obj:`bool`, optional): If ``True``,
					plot DVH curves with different marker sizes for each
					non-reference plan.
				marker_size_increasing (:obj:`bool`, optional): If
					``True``, the DVH curves for non-reference plans
					will be drawn with markers of increasing size. If
					``False``, this is reversed. Increasing/decreasing
					size is applied according to the plans' list order
					in ``run_data``.
				universal_marker (:obj:`str`, optional): If set to a
					valid :mod:`matplotlib` marker string, each
					non-reference plan will be plotted with that marker
					(increasing marker size will be enabled to
					differentiate plans).
				vary_line_weights (:obj:`bool`, optional): If ``True``,
					the DVH curves for non-reference plans will be drawn
					with lines of increasing weight. Increasing weight
					is applied according to the plans' list order in
					``run_data``.
				vary_line_colors (:obj:`bool`, optional): If ``True``,
					attenuate color assigned to each structure's DVH
					curve (use weighted average of specified color and
					black) when plotting each non-reference data series.
				vary_line_styles (:obj:`bool`, optional): If ``True``,
					line styles for DVH curves of each non-reference
					plan will be cycled among 4 styles.
				darken_reference (:obj:`bool`, optional): If ``True``,
					attenuate color assigned to each structure's DVH
					curve (use weighted average of specified color and
					black) when plotting reference series.
				series_labels_on_plot(:obj:`bool`, optional): If ``True``,
					structure names will be printed on the corresponding
					subplots.
				series_label_coordinates(:obj:`dict`, optional): If
					provided, each key is expected to be a structure
					label and each value is expected to be an (x, y)
					coordinate at which to write the structure label.
				**options: Keyword arguments passed to
					:meth:`CasePlotter.plot`
			"""
			n_compared = len(run_data) + int(reference_data is not None)
			run_aesthetics = [LineAesthetic() for i in xrange(n_compared)]
			run_data = [PlanDVHGraph(d) for d in run_data]

			line_styles = ['-', '--', '-.', ':']
			if len(run_data) > 4 and vary_line_styles:
				vary_markers = True

			vary_markers &= universal_marker is None
			vary_marker_sizes |= universal_marker is not None
			marker_styles = ['o', 's', '^']
			fill_styles = ['none', 'full']
			if len(run_data) > 6 and vary_markers:
				vary_line_styles = True

			weight_step = 0.5
			min_weight = 0.5
			max_weight = 2.0
			if vary_line_weights:
				max_weight = max(
						max_weight, min_weight + weight_step * n_compared)

			max_attenuation = 0.7
			min_attentuation = 1.0
			attenuation_step = -0.5 / n_compared

			# set layout
			self.dvh_plot.layout = layout

			# set aesthetics for each data series
			for i, aesthetic in enumerate(run_aesthetics):
				if vary_line_styles:
					aesthetic.style = line_styles[i % 4]
				if universal_marker is not None:
					aesthetic.marker = universal_marker
				if vary_markers:
					aesthetic.marker = marker_styles[i % 3]
					aesthetic.fill = fill_styles[i % 2]
				if vary_marker_sizes:
					if marker_size_increasing:
						aesthetic.markersize = 4 + 2 * i
					else:
						aesthetic.markersize = 4 + 2 * (n_compared - 1 - i)
				if vary_line_weights:
					aesthetic.weight = min_weight + i * weight_step
				if vary_line_colors:
					aesthetic.color_attenuation = 1.0 - i * attenuation_step

			# set reference aesthetics
			if reference_data is not None:
				run_data.append(PlanDVHGraph(reference_data))
				run_names.append(str(reference_name))

				run_aesthetics[-1].weight = max_weight
				if darken_reference:
					run_aesthetics[-1].color_attenuation = max_attenuation

			# get single x-window appropriate across all compared series
			options['xmax'] = options.pop(
					'xmax', max([d.maxdose(d) for d in run_data]))

			if clear:
				self.dvh_plot.clear()

			# plot each data series
			for i in xrange(n_compared):
				if i == 0:
					options['self_title_subplots'] = True

				self.plot(run_data[i], clear=False, subset=subset, legend=False,
						  aesthetic=run_aesthetics[i], **options)

			# plot legend
			self.dvh_plot.plot_virtual(run_names, run_aesthetics, **options)

			# write any requested series labels on plot
			if series_labels_on_plot and isinstance(series_label_coordinates, dict):
				self.dvh_plot.plot_labels(
						self.filter_data(series_label_coordinates, subset),
						self.dvh_set, **options)

			if show:
				self.dvh_plot.show()

			if plotfile is not None:
				self.dvh_plot.save(plotfile)