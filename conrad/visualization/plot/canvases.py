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

from math import ceil
import os.path

from conrad.visualization.plot.collections import \
		PlanDVHGraph, PlanConstraintsGraph
# import matplotlib into namespace as mpl, with lines, axes, figure,
# color and pyplot modules available. also imports constant
# PLOTTING_INSTALLED and DISPLAY AVAILABLE.
from conrad.visualization.plot.mpl import *

if not PLOTTING_INSTALLED:
	DVHSubplot = lambda arg1: None
	DVHPlot = lambda arg1, arg2: None
else:
	class DVHSubplot(object):
		def __init__(self, subplot_axes):
			if not isinstance(subplot_axes, mpl.axes.Subplot):
				raise TypeError('argument `subplot_axes` must be of type'
								''.format(mpl.axes.Subplot))
			self.__subplot_axes = subplot_axes
			self.__title_location = 'left'
			self.__title_fontsize = 12
			self.__title_fontweight = 'bold'
			self.__title_font_dictionary = {
					'fontsize': self.__title_fontsize,
					'fontweight': self.__title_fontweight}
			self.__title = self.axes.set_title(
					'', loc='left', fontdict=self.__title_font_dictionary)


		@property
		def axes(self):
			return self.__subplot_axes

		@property
		def left(self):
			return self.axes.is_first_col()

		@property
		def bottom(self):
			return self.axes.is_last_row()

		@property
		def title(self):
			return self.__title.get_text()

		@title.setter
		def title(self, title):
			self.__title.set_text(title)

		@property
		def legend(self):
			return self.axes.get_legend()

		@legend.setter
		def legend(self, legend_visible):
			if bool(self.legend is None and legend_visible and
					len(self.axes.lines) > 0):
				self.axes.legend(loc='best')

			if self.legend is not None:
				self.legend.set_visible(legend_visible)
				if legend_visible:
					frame = self.legend.get_frame()
					frame.set_facecolor('1.0')
					frame.set_edgecolor('1.0')

		@property
		def xmax(self):
			return self.axes.get_xlim()[1]

		@xmax.setter
		def xmax(self, xmax):
			self.axes.set_xlim(0, xmax)

		@staticmethod
		def label_size(label):
			return 16 - 2 * (len(label) > 10) - 2 * (len(label) > 25)

		@property
		def xlabel(self):
			return self.axes.get_xlabel()

		@xlabel.setter
		def xlabel(self, xlabel):
			self.axes.set_xlabel(xlabel, fontsize=self.label_size(xlabel))

		@property
		def ylabel(self):
			return self.axes.get_ylabel()

		@ylabel.setter
		def ylabel(self, ylabel):
			self.axes.set_ylabel(ylabel, fontsize=self.label_size(ylabel))

		@property
		def xaxis(self):
			return self.axes.spines['bottom'].get_visible()

		@xaxis.setter
		def xaxis(self, visible):
			self.axes.tick_params(
					axis='x', bottom=visible, labelbottom=visible,
					top=False, labeltop=False, direction='out')
			self.axes.spines['bottom'].set_visible(visible)

			if visible:
				# keep x axis ticks light
				ticks = self.axes.get_xticks()
				if len(ticks) > 7:
					self.axes.set_xticks(ticks[::2])

		@property
		def yaxis(self):
			return self.axes.spines['left'].get_visible()

		@yaxis.setter
		def yaxis(self, visible):
			self.axes.tick_params(
					axis='y', left=visible, labelleft=visible,
					right=False, labelright=False, direction='out')
			self.axes.spines['left'].set_visible(visible)

		def format(self, xlimit, xlabel, ylabel, minimal_axes=True):

			self.xmax = xlimit
			self.axes.set_ylim(0, 103)

			# top & right spines invisible
			self.axes.spines['top'].set_visible(False)
			self.axes.spines['right'].set_visible(False)

			# offset spines
			for spine in self.axes.spines.values():
				spine.set_position(('outward', 5))

			# horizontal gridlines
			self.axes.grid(axis='y', color='0.9', linestyle='-', linewidth=1)

			# put ticks and gridlines behind plotted curves
			self.axes.set_axisbelow(True)

			# conditional (bottom) x axis formatting
			self.xlabel = xlabel if self.bottom else ''
			self.xaxis = self.bottom

			# conditional (left) y axis formatting
			self.ylabel = ylabel if self.left else ''
			self.yaxis = self.left or not minimal_axes

	class DVHPlot(object):
		"""
		Tool for visualizing dose volume histograms.

		Figure contains :attr:`~DVHPlot.n_structures` dose volume
		histograms distributed among (:attr:`~DVHPlot.cols` by
		:attr:`~DVHPlot.rows`) subplots. This can be adjusted
		dynamically by changing the subplot indices assigned to each
		series.
		"""

		def __init__(self, subplot_assignments, layout='auto'):
			"""
			Initialize :class:`DVHPlot`.

			Initialize a :class:`matplotlib.Figure` as a blank canvas.
			Private dictionaries track panel (subplot index), series
			names, and series color assignments, all keyed by structure
			labels. Arguments set series names and subplot indices.

			Args:
				 subplot_assignments (:obj:`dict`): Dictionary of series
				 	subplot indices keyed by series (structure) labels.
				 layout: (:obj:`str`): Layout for subplots, used to set
				 	property attr:`DVHPlot.layout`.
			"""
			self.__figure = None
			self.__subplots_by_structure = {}
			self.__subplot_list = []
			self.__subplot_assignments_by_structure = {}
			self.__cols = 1
			self.__rows = 1
			self.__layout = 'auto'

			self.n_structures = len(subplot_assignments)
			self.subplot_assignments = subplot_assignments
			self.layout = layout

		@staticmethod
		def subplots_to_cols(n_subplots):
			"""
			Convert number of subplots to number of subplot columns.

			Used to standardize and balance subplot layout when using
			multiple structures. Prioritizes horizontal expansion over
			vertical expansion up to a maximum of 4 columns.

			Args:
				n_subplots (:obj:`int`): number of subplot panels.

			Returns:
				:obj:`int`: number of subplot columns.

			Raises:
				None
			"""
			n_cols = 1
			if n_subplots > 1:
				n_cols += 1
			if n_subplots > 4:
				n_cols += 1
			if n_subplots > 6:
				n_cols += 1
			return n_cols

		@staticmethod
		def sift_options(**options):
			plot_options = {}
			legend_options = {}
			for o in options:
				if 'legend_' in o:
					legend_options[o.replace('legend_','')] = options[o]
				else:
					plot_options[o] = options[o]
			return plot_options, legend_options

		@property
		def figure(self):
			"""
			:class:`matplotlib.Figure` for rendering dose volume histograms.
			"""
			return self.__figure

		@figure.setter
		def figure(self, figure):
			if not isinstance(figure, mpl.figure.Figure):
				raise TypeError('figure must be of type {}'
								''.format(mpl.figure.Figure))
			self.__figure = figure

		@property
		def subplots(self):
			"""
			Dictionary of subplots in plot, keyed by structure label.
			"""
			return self.__subplots_by_structure

		@property
		def panels(self):
			""" List of subplots. """
			return self.__subplot_list

		@property
		def rows(self):
			""" Number of subplot rows. """
			return self.__rows

		@property
		def cols(self):
			""" Number of subplot columns. """
			return self.__cols

		@property
		def n_subplots(self):
			""" Total number of suplots. """
			return len(set(self.subplot_assignments.values()))

		@property
		def layout(self):
			"""
			Subplot layout: ``'auto'``, ``'vertical'``, or ``'horizontal'``.

			Raises:
				ValueError: If argument to setter is not one of the
					three accepted layout strings.
			"""
			return self.__layout

		@layout.setter
		def layout(self, layout):
			layout = str(layout)
			if layout not in ('auto', 'vertical', 'horizontal'):
				raise ValueError('argument `layout` must be one of:\n'
								 '-"auto"\n-"vertical"\n-"horizontal"')
			if layout != self.__layout or self.figure is None:
				self.__layout = layout
				self.build()

		@property
		def subplot_assignments(self):
			"""
			Dictionary of series subplot indices keyed by series labels.
			"""
			return self.__subplot_assignments_by_structure

		@subplot_assignments.setter
		def subplot_assignments(self, subplot_assignments):
			self.__subplot_assignments_by_structure = {}
			for label in subplot_assignments:
				self.__subplot_assignments_by_structure[label] = max(
						0, int(subplot_assignments[label]))

			self.calculate_panels()

		def __getitem__(self, key):
			""" Overload operator []. """
			if key in self.subplots:
				return self.subplots[key]
			elif key in ('upper right', 'upper_right'):
				for s in self.subplots.values():
					if s.axes.is_first_row() and s.axes.is_last_col():
						return s
				raise KeyError('upper right subplot not found')
			else:
				raise KeyError('key "{}" does not correspond to a '
							   'structure known to the {}'
							   ''.format(key, DVHPlot))

		def calculate_panels(self):
			"""
			Calculate number of subplot rows and columns in DVHPlot,
			given the series->panel assignements and layout
			specification.
			"""
			if self.layout == 'vertical':
				self.__cols = 1
				self.__rows = self.n_subplots
			elif self.layout == 'horizontal':
				self.__cols = self.n_subplots
				self.__rows = 1
			else:  # if self.layout == 'auto':
				self.__cols = self.subplots_to_cols(self.n_subplots)
				self.__rows = 1 + max(0, self.n_subplots - 1) // self.__cols

		def build(self):
			"""
			Build :attr:`DVHPlot.figure` with subplots laid out in a
			:attr:`DVHPlot.rows` by :attr:`DVHPlot.cols` grid.

			Handles to the subplots are stored as a dictionary (keyed by
			structure label) in the field :attr:`DVHPlot.subplots`.

			Returns:
				None
			"""
			if self.figure is not None:
				self.clear()

			self.calculate_panels()
			self.figure, subplots = mpl.pyplot.subplots(
					self.rows, self.cols, sharex='col', sharey='row')
			self.figure.set_size_inches(3.25 * self.cols, 3.25 * self.rows)
			self.figure.subplots_adjust(left=0.09, bottom=0.1, right=0.99,
										top=0.99, wspace=0.1, hspace=0.15)

			# build list of subplots
			for panel_idx in xrange(self.n_subplots):
				row = panel_idx // self.cols
				col = panel_idx % self.cols

				if self.n_subplots == 1:
					subplot_axes = subplots
				elif self.rows == 1:
					subplot_axes = subplots[col]
				elif self.cols == 1:
					subplot_axes = subplots[row]
				else:
					subplot_axes = subplots[row, col]

				self.panels.append(DVHSubplot(subplot_axes))

			# build structure label->subplot dictionary
			for label, panel_idx in self.subplot_assignments.items():
				self.subplots[label] = self.panels[panel_idx]

		def clear(self):
			"""
			Closes the :mod:`matplotlib` figure at `DVHPlot.figure`,
			clears the dictionary of subplot handles `DVHPlot.subplots`.

			Returns:
				None
			"""
			if self.figure is not None:
				mpl.pyplot.close(self.figure)
				self.__figure = None
				self.__subplots_by_structure = {}
				self.__subplot_list = []

		def draw_legend(self, series, names, alignment=None,
						coordinates=None, **options):
			"""
			Draw figure (not subplot) legend comprising the specified
			series, labeled with the specified names.

			Args:
				series(:obj:`list`): List of :mod:`matplotlib` series
					(i.e., ``artist`` objects) to render in legend.
				names (:obj:`list` of :obj:`str`): Names of series in
					legend.
				alignment (:obj:`str`, optional): Legend location
					relative to legend anchor position. Should conform
					to specification for keyword argument ``loc`` of
					:meth:``matplotlib.figure.Figure.legend``.
				coordinates (optional): Legend anchor position,
					in (x,y)-coordinates, relative to lower left corner
					of figure (upper right corner is ``(1,1)``).
				options: Keyword arguments passed to
					:meth:`~matplotlib.figures.Figure.legend`

			Returns:
				None
			"""
			legend_args = {
					'ncol':1,
					'loc':'upper right',
					'columnspacing':1.0,
					'labelspacing': options.pop('label_spacing', 0.5),
					'handletextpad': options.pop('handle_textpad', 0.5),
					'handlelength':2.0,
					'fontsize': options.pop('fontsize', 10),
					'fancybox': options.pop('box_rounded', True),
					'shadow': options.pop('shadow', False),
			}
			if alignment is not None:
				legend_args['loc'] = alignment
			if coordinates is not None:
				legend_args['bbox_to_anchor'] = coordinates
			legend = self.figure.legend(series, names, **legend_args)
			if not options.pop('border', True):
				frame = legend.get_frame()
				frame.set_edgecolor('1.0')
				frame.set_facecolor('1.0')

		def draw_title(self, title):
			if title is not None:
				self.figure.suptitle(title)

		def format_subplots(self, xlim_upper, x_label='Dose (Gy)',
							y_label='Percentile', minimal_axes=False):
			for subplot in self.panels:
				subplot.format(xlim_upper, x_label, y_label,
							   minimal_axes=minimal_axes)

		def check_figure(self):
			if self.figure is None:
				raise AttributeError('no figure. call `DVHPlot.build()` '
									 'before plotting')
			return True

		def plot_virtual(self, series_names, series_aesthetics,
						 legend_alignment=None, legend_coordinates=None,
						 **legend_options):
			"""
			Add series to DVH Plot that only appear in legend.

			Enable figure's overall legend (i.e., not a subplot legend).

			Args:
				series_names (): DESCRIPTION
				series_aesthetics (): DESCRIPTION
				legend_alignment (optional): Legend location relative to
					legend anchor position.
				legend_coordinates (optional): Legend anchor position,
					in (x,y)-coordinates, relative to lower left corner
					of figure (upper right corner is ``(1,1)``).
				legend_options: Keyword arguments passed to
					:meth:`~matplotlib.figures.Figure.legend`

			Returns:
				None
			"""
			self.check_figure()

			line_generator = lambda a: mpl.lines.Line2D(
					[], [], **a.plot_args('#222222'))
			series = listmap(line_generator, series_aesthetics)
			self.draw_legend(series, series_names, alignment=legend_alignment,
							 coordinates=legend_coordinates, **legend_options)

		def plot_labels(self, coordinate_dict, dvh_set, **options):
			self.check_figure()
			if not isinstance(dvh_set, PlanDVHGraph):
				dvh_set = PlanDVHGraph(dvh_set)

			fontsize = int(options.pop('text_fontsize', 12))
			fontweight = str(options.pop('text_weight', 'normal'))
			for label in coordinate_dict:
				subplot = self.subplots[label]
				name = dvh_set[label].name
				color = dvh_set[label].color
				xpos, ypos = coordinate_dict[label]
				subplot.axes.text(xpos, ypos, name, fontdict={
						'fontsize': fontsize, 'fontweight': fontweight,
						'color': color})

		def plot_constraints(self, plan_constraints, structure_colors=None,
							 **options):
			self.check_figure()
			# interpret data, if necessary
			if not isinstance(plan_constraints, PlanConstraintsGraph):
				plan_constraints = PlanConstraintsGraph(
						plan_constraints, structure_colors)
			else:
				if structure_colors is not None:
					plan_constraints.structure_colors = structure_colors

			for label in plan_constraints.structure_labels:
				plan_constraints[label].draw(self.subplots[label].axes,
											 **options)

		def plot(self, plan_dvh, show=False, clear=True, xmax=None,
				 legend=True, title=None, self_title_subplots=False,
				 suppress_rx=False, suppress_constraints=False,
				 x_label='Dose (Gy)', y_label='Percentile',
				 legend_coordinates=None, legend_alignment=None,
				 aesthetic=None, minimal_axes=True, **options):
			"""
			Plot ``plot_data`` to the object's :class:`matplotlib.Figure`.

			Args:
				plot_data (:obj:`dict`) Collection of DVH curves, keyed
					by structure/series label, with format specified
					above.
				show (:obj:`bool`, optional): Show
					:class:`matplotlib.Figure` canvas after
					``plot_data`` elements are drawn.
				clear (:obj:`bool`, optional): Clear
					:class:`matplotlib.Figure` before ``plot_data``
					elements are drawn.
				xmax (:obj:`float`, optional): Upper limit for x-axis
					set to this value if provided. Otherwise, upper
					limit set to 110% of largest dose encountered in
					``plot_data``.
				legend (optional): Enable legend in
					:class:`matplotlib.Figure`. Set overall legend if
					value is ``True``, set legend per subplot if value
					is ``'each'``, set legend in upper-right-most
					subplot if value is ``'upper_right'``.
				title (:obj:`str`, optional): Contents drawn as title of
					:class:`matplotlib.Figure`.
				suppress_constraints (:obj:`bool`, optional): Suppress
					rendering of dose volume constraints.
				x_label (:obj:`str`, optional): x-axis label.
				y_label (:obj:`str`, optional): y-axis label.
				legend_coordinates (:obj:`list`, optional): Position, as
				 	(x,y)-coordinates, of legend anchor relative to
				 	figure; passed as kewyword argument ``bbox_to_anchor``
				 	in  :meth:`matplotlib.Figure.legend`.
				legend_alignment (:obj:`str`, optional): String defining
					alignment of legend relative to anchor, passed as
					keyword argument ``loc`` in
					:meth:`matplotlib.Figure.legend`.
				**options: Arbitrary keyword arguments, passed to
					:meth:`matplotlib.Figure.plot`.

			Returns:
				None
			"""
			plot_options, legend_options = self.sift_options(**options)

			if clear:
				self.clear()

			# get figure, subplot axes,
			if self.figure is None:
				self.build()

			# interpret data, if necessary
			if not isinstance(plan_dvh, PlanDVHGraph):
				plan_dvh = PlanDVHGraph(plan_dvh)

			# get x-axis limits
			max_dose = plan_dvh.maxdose(suppress_constraints)
			xlim_upper = xmax if xmax is not None else 1.1 * max_dose

			# plot title
			self.draw_title(title)

			# format subplots:
			self.format_subplots(xlim_upper, x_label, y_label, minimal_axes)

			# plot data
			for label, dvh in plan_dvh:
				dvh.draw(self.subplots[label].axes, aesthetic=aesthetic,
						 constraints=not suppress_constraints,
						 rx=not suppress_rx, **plot_options)

			# title subplots
			if self_title_subplots:
				# compile subplot titles
				subplot_names = {}
				for label, dvh in plan_dvh:
					panel_idx = self.subplot_assignments[label]
					if panel_idx in subplot_names:
						subplot_names[panel_idx] += ', {}'.format(dvh.name)
					else:
						subplot_names[panel_idx] = str(dvh.name)

				# set subplot titles
				for panel_idx in xrange(self.n_subplots):
					self.panels[panel_idx].title = subplot_names[panel_idx]

			# legend: overall, subplots, or none
			if legend is 'each':
				for p in self.panels:
					p.legend = True
			elif legend is True:
				series = []
				names = []
				for _, dvh in plan_dvh:
					series.append(dvh.curve.graph[0])
					names.append(dvh.name)
				self.draw_legend(series, names, legend_alignment,
								 legend_coordinates, **legend_options)

			if show:
				self.show()

		def show(self):
			"""
			If matplotlib is in communication with a display, render
			plot onscreen. "
			"""
			self.check_figure()
			if DISPLAY_AVAILABLE:
				self.figure.show()

		def save(self, filepath, overwrite=True, verbose=False):
			"""
			Save the object's current plot to ``filepath``.

			Args:
				filepath (:obj:`str`): Specify path to save plot.
				overwrite (bool):, Allow overwrite of file at
					``filepath``if ``True``.
				verbose (:obj:`bool`): Print confirmation of save if
					``True``.

			Returns:
				None

			Raises:
				ValueError: If ``filepath`` does not exist *or* is an
					existing file and flag ``overwrite`` is ``False``.
				RuntimeError: If save fails for any other reason.
			"""
			if self.figure is None:
				raise AttributeError('no figure to save')

			filepath = os.path.abspath(filepath)
			directory = os.path.dirname(filepath)
			if not os.path.isdir(os.path.dirname(filepath)):
				raise ValueError(
						'argument "filepath" specified with invalid'
						'directory')
			elif not overwrite and os.path.exists(filepath):
				raise ValueError(
						'argument "filepath" specifies an existing file'
						'and argument "overwrite" is set to False')
			else:
				try:
					if verbose:
						print("SAVING TO ", filepath)
					self.figure.savefig(filepath, dpi=600, bbox_inches='tight')
				except:
					raise RuntimeError(
							'could not save plot to file: {}'.format(filepath))

		def __del__(self):
			"""
			Close object's :class:`matplotlib.Figure` when out of scope.
			"""
			self.clear()