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

# import matplotlib into namespace as mpl, with lines, axes, figure,
# color and pyplot modules available. also imports constant
# PLOTTING_INSTALLED and DISPLAY AVAILABLE.
from conrad.visualization.plot.mpl import *

if not PLOTTING_INSTALLED:
	LineAesthetic = lambda: None
	DVHPlotElement = lambda: None
	DoseVolumeGraph = lambda arg1, arg2: None
	PercentileConstraintGraph = lambda arg1, arg2, arg3: None
	PrescriptionGraph = lambda arg1: None
else:
	class LineAesthetic(object):
		"""
		Abstraction of :mod:`matplotlib` line styling, intended for use with
		a family of lines with all style aspects shared *except* color.

		The :class:`LineAesthetic` does not specify a color; it must be
		paired with a color to create a full specification that can be
		applied to :mod:`matplotlib` plotting methods.

		Attributes:
			style (:obj:`str`): Line style. Must be compatible with styles
				specified in :mod:`matplotlib.lines`.
			weight (:obj:`float`): Line weight.
			marker (:obj:`str` or ``None``): Marker style to apply to line.
			fill (:obj:`str`): Fill style to apply to line's markers.
			num_markers (:obj:`int`): Number of markers to draw when
				rendering line.
			alpha (:obj:`float`): Value in [0, 1] specifying opacity of
				line.
			color_attenuation (:obj:`float`): Value in [0, 1] (recommended
				range: [0.7, 1]) specifying attenuation of colors paired
				with this :class:`LineAesthetic`. ``1.0`` yields original
				color; ``0.0`` yields black.
		"""
		__verification_line = mpl.lines.Line2D([],[])

		def __init__(self, aesthetic='dvh_curve', **kwargs):
			self.__style = '-'
			self.__weight = 1.0
			self.__marker = None
			self.__markersize = 5.0
			self.__fill = 'none'
			self.__num_markers = 20
			self.__alpha = 1.0
			self.__color_attenuation = 1.0

			if 'dvh_constraint' in aesthetic:
				self.style = ''
				self.markersize = 16
			elif 'slack' in aesthetic:
				self.alpha = 0.6
			elif 'rx' in aesthetic:
				self.style = ':'
				self.weight = 1.5

			if 'style' in kwargs:
				self.style = kwargs['style']
			if 'weight' in kwargs:
				self.weight = kwargs['weight']
			if 'marker' in kwargs:
				self.marker = kwargs['marker']
			if 'markersize' in kwargs:
				self.markersize = kwargs['markersize']
			if 'fill' in kwargs:
				self.fill = kwargs['fill']
			if 'num_markers' in kwargs:
				self.num_markers = kwargs['num_markers']
			if 'alpha' in kwargs:
				self.alpha = kwargs['alpha']
			if 'color_attenuation' in kwargs:
				self.color_attenuation = kwargs['color_attenuation']

		def __str__(self):
			return 	str('Line Aesthetic:' +
					   '\n\tline style: %s' % self.__style +
					   '\n\tline weight: %0.1f' % self.__weight +
					   '\n\tmarker: %s' % self.__marker +
					   '\n\tmarker size: %0.1f' % self.__markersize +
					   '\n\tmarker fill: %s' % self.__fill +
					   '\n\t# markers: %i' % self.__num_markers +
					   '\n\topacity: %s' % self.__alpha +
					   '\n\tcolor scaling: %s' % self.__color_attenuation)

		def __eq__(self, other):
			if not isinstance(other, LineAesthetic):
				raise TypeError('equality comparison only defined for '
								'compared object of type {}'.format(
								LineAesthetic))
			return bool(self.style == other.style and
						self.weight == other.weight and
						self.marker == other.marker and
						self.markersize == other.markersize and
						self.fill == other.fill and
						self.num_markers == other.num_markers and
						self.alpha == other.alpha and
						self.color_attenuation == other.color_attenuation )

		@property
		def style(self):
			return self.__style

		@style.setter
		def style(self, style):
			self.__verification_line.set_linestyle(style)
			self.__style = style

		@property
		def weight(self):
			return self.__weight

		@weight.setter
		def weight(self, weight):
			self.__verification_line.set_linewidth(weight)
			self.__weight = weight

		@property
		def marker(self):
			return self.__marker

		@marker.setter
		def marker(self, marker):
			self.__verification_line.set_marker(marker)
			self.__marker = marker

		@property
		def markersize(self):
			return self.__markersize

		@markersize.setter
		def markersize(self, markersize):
			self.__verification_line.set_markersize(markersize)
			self.__markersize = markersize

		@property
		def fill(self):
			return self.__fill

		@fill.setter
		def fill(self, fill):
			self.__verification_line.set_fillstyle(fill)
			self.__fill = fill

		@property
		def num_markers(self):
			return self.__num_markers

		@num_markers.setter
		def num_markers(self, num_markers):
			num_markers = int(num_markers)
			if num_markers < 0:
				raise ValueError('number of markers cannot be negative')
			self.__num_markers = num_markers

		@property
		def alpha(self):
			return self.__alpha

		@alpha.setter
		def alpha(self, alpha):
			alpha = float(alpha)
			if alpha < 0 or alpha > 1:
				raise ValueError('`alpha` must be in the interval [0, 1]')
			self.__alpha = alpha

		@property
		def color_attenuation(self):
			return self.__color_attenuation

		@color_attenuation.setter
		def color_attenuation(self, factor):
			factor = float(factor)
			if factor < 0 or factor > 1:
				raise ValueError('`factor` must be in the interval [0, 1]')
			self.__color_attenuation = factor

		def copy(self, other):
			self.__style = other.style
			self.__weight = other.weight
			self.__marker = other.marker
			self.__markersize = other.markersize
			self.__fill = other.fill
			self.__num_markers = other.num_markers
			self.__alpha = other.alpha
			self.__color_attenuation = other.color_attenuation

		def scale_rgb(self, color):
			"""
			Convert any :mod:`matplotlib` color to RGBA tuple and scale all
			color entries of tuple ``rgb`` by a factor of
			:attr:``LineAesthetic.color_attenuation``.

			Args:
				rgb (:obj:`tuple`): RGB or RGBA tuple.
				factor (:obj:`float`, optional): Scalar in [0, 1].

			Returns:
				:obj:`tuple`: Scaled version of input RGB or RGBA tuple.
			"""
			r"""
			Return a weighted average of black and specified color.

			Let :math:`\alpha` be the attenuation specified by
			attr:`LineAesthetic.color_attenuation`, :math:`c` be the vector
			representation of the input color ``color`` (e.g., in RGB),
			where :math:`0` is black. Then, the returned color is
			:math:`\alpha c`

			Args:
				color: Any :mod:`matplotlib`-compatible color description,
					base color to attenuate.
			"""
			rgba = mpl.colors.colorConverter.to_rgba(color)

			if self.color_attenuation == 1.0:
				return rgba

			rgba_scaled = tuple(val * self.color_attenuation for val in rgba[:3])
			rgba_scaled += (rgba[-1],)
			return rgba_scaled

		def get_sample_factor(self, series_length):
			if isinstance(series_length, int):
				sample_factor = max(1,  series_length // self.num_markers)
			else:
				sample_factor = 1
			return sample_factor

		def apply_color(self, line, color):
			line.set_color(self.scale_rgb(color))

		def apply(self, line, color):
			if not isinstance(line, mpl.lines.Line2D):
				raise TypeError('argument "line" must be of type {}'
								''.format(mpl.lines.Line2D))
			series_length = len(line.get_xdata())
			line.set_linestyle(self.style)
			line.set_linewidth(self.weight)
			line.set_marker(self.marker)
			line.set_markersize(self.markersize)
			line.set_fillstyle(self.fill)
			line.set_markevery((1, self.get_sample_factor(series_length)))
			line.set_alpha(self.alpha)
			self.apply_color(line, color)

		def plot_args(self, color, series_length=None):
			"""
			Generate dictionary of keyword arguments representing line
			aesthetic for :mod:`matplotlib` methods.

			Args:
				color: Any :mod:`matplotlib`-compatible color description,
					pairs with line aesthetic data to create style keywords.
				series_length (:obj:`int`, optional): If provided, used to
					calculate sampling frequency of markers, based on total
					number of desired markers as specified by
					attr:`LineAesthetic.num_markers`.

			Returns:
				Dictionary of keyword arguments compatible with
				:meth:`matplotlib.figure.Figure.plot()`
				or, equivalently, initializer of
				:class:`matplotlib.lines.Line2D`.
			"""
			return {'linestyle': self.style,
					'linewidth': self.weight,
					'marker': self.marker,
					'markersize': self.markersize,
					'markevery': (1, self.get_sample_factor(series_length)),
					'fillstyle': self.fill,
					'color': self.scale_rgb(color),
					'alpha': self.alpha,}

	class DVHPlotElement(object):
		def __init__(self, color=None, aesthetic=None):
			self.__axes = None
			self.__graph = []
			self.__color = 'black'
			self.__aesthetic = LineAesthetic()
			self.__label = '_nolabel_'

			if color is not None:
				self.color = color

			if aesthetic is not None:
				self.aesthetic = aesthetic

		def __iadd__(self, other):
			if isinstance(other, list):
				for o in other:
					self += o
			elif not isinstance(other, mpl.lines.Line2D):
				raise TypeError('argument to {} must be of type {}'
								''.format(DVHPlotElement.__iadd__,
										  mpl.lines.Line2D) )
			else:
				other.set_label(self.label)
				self.__graph.append(other)
			return self

		@property
		def axes(self):
			return self.__axes

		@axes.setter
		def axes(self, axes):
			if not isinstance(axes, mpl.axes.Subplot):
				raise TypeError('argument "axes" must be of type {}'.format(
								mpl.axes.Subplot))

			self.__axes = axes
			for g in self.graph:
				if g not in self.__axes.lines:
					self.__axes.add_line(g)

		@property
		def graph(self):
			return self.__graph

		@property
		def color(self):
			return self.__color

		@color.setter
		def color(self, color):
			self.__color = mpl.colors.colorConverter.to_rgba(color)
			for g in self.graph:
				self.aesthetic.apply_color(g, self.__color)

		@property
		def aesthetic(self):
			return self.__aesthetic

		@aesthetic.setter
		def aesthetic(self, aesthetic):
			if aesthetic is not None:
				if not isinstance(aesthetic, LineAesthetic):
					raise ValueError('argument "aesthetic" must be of '
									 'type {}'.format(LineAesthetic))
				self.__aesthetic.copy(aesthetic)

		@property
		def label(self):
			return self.__label

		@label.setter
		def label(self, label):
			if label in ('', ' ', None):
				self.__label = '_nolabel_'
			else:
				self.__label = str(label)
			for g in self.graph:
				g.set_label(self.label)

		def show(self):
			for g in self.graph:
				g.set_visible(True)

		def hide(self):
			for g in self.graph:
				g.set_visible(False)

		def draw(self, axes):
			return NotImplemented

		def undraw(self):
			if self.axes is not None:
				for graph in self.graph:
					# unpair line and axes
					if graph in self.axes.lines:
						self.axes.lines.remove(graph)
					graph.axes = None
				self.__axes = None

	class DoseVolumeGraph(DVHPlotElement):
		def __init__(self, doses, percentiles, color=None, aesthetic=None):
			DVHPlotElement.__init__(self, color, aesthetic)
			self.__maxdose = doses[-1]
			self.__curve = mpl.lines.Line2D(doses, percentiles)
			self += self.__curve

		def draw(self, axes, aesthetic=None):
			self.aesthetic = aesthetic
			self.aesthetic.apply(self.__curve, self.color)
			self.axes = axes

		@property
		def maxdose(self):
			return self.__maxdose

	class PercentileConstraintGraph(DVHPlotElement):
		def __init__(self, doses, percentiles, symbol, color=None):
			DVHPlotElement.__init__(self, color, LineAesthetic('dvh_constraint'))
			self.__maxdose = max(doses)
			self.__requested = mpl.lines.Line2D([doses[0]], [percentiles[0]])
			self.__achieved = mpl.lines.Line2D([doses[1]], [percentiles[1]])
			self.__slack = mpl.lines.Line2D(doses, percentiles)
			self.__slack_amount = abs(doses[1] - doses[0])
			self.__slack_aesthetic = LineAesthetic('slack')
			self += self.__achieved

		def draw(self, axes, size=16, slack_threshold=0.1):
			slack_threshold = float(slack_threshold)
			if slack_threshold < 0:
				raise ValueError('argument "slack_threshold" must be '
								 'nonnegative')

			self.undraw()

			self.aesthetic.markersize = size
			self.aesthetic.alpha = 1.0
			self.aesthetic.apply(self.__achieved, self.color)

			if self.__slack_amount > slack_threshold:
				self.aesthetic.alpha = 0.55
				self.aesthetic.apply(self.__requested, self.color)
				self.__slack_aesthetic.apply(self.__slack, self.color)
				if self.__requested not in self.graph:
					self += self.__requested
				if self.__slack not in self.graph:
					self += self.__slack
			else:
				if self.__requested in self.graph:
					self.graph.remove(self.__requested)
				if self.__slack in self.graph:
					self.graph.remove(self.__slack)

			self.axes = axes

		@property
		def maxdose(self):
			return self.__maxdose

	class PrescriptionGraph(DVHPlotElement):
		def __init__(self, rx_dose, color=None):
			DVHPlotElement.__init__(self, color, LineAesthetic('rx'))

			self.__rx = mpl.lines.Line2D([float(rx_dose)]*2, [0, 110])
			self += [self.__rx]

		def draw(self, axes):
			self.aesthetic.apply(self.__rx, self.color)
			self.axes = axes