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

from conrad.medicine import Structure, Anatomy
from conrad.visualization.plot.elements import \
		LineAesthetic, DoseVolumeGraph, PercentileConstraintGraph, \
		PrescriptionGraph
# import matplotlib into namespace as mpl, with lines, axes, figure,
# color and pyplot modules available. also imports constant
# PLOTTING_INSTALLED and DISPLAY AVAILABLE.
from conrad.visualization.plot.mpl import *

if not PLOTTING_INSTALLED:
	StructureDVHGraph = lambda arg1: None
	StructureConstraintsGraph = lambda arg1: None
	PlanGraphBase = lambda arg1: None
	PlanDVHGraph = lambda arg1: None
	PlanConstraintsGraph = lambda arg1: None
else:
	class StructureConstraintsGraph(object):
		def __init__(self, data, color=None):
			if isinstance(data, Structure):
				data = data.plotting_data(constraints_only=True)

			if color is None:
				color = 'black'

			self.__constraints = {}
			self.__color = LineAesthetic().scale_rgb(color)
			for constr_id, constraint in data:
				if constraint['type'] == 'percentile':
					self.__constraints[constr_id] = PercentileConstraintGraph(
							constraint['dose'], constraint['percentile'],
							constraint['symbol'], color)

		def __iter__(self):
			return self.__constraints.values().__iter__()

		@property
		def constraints(self):
			return self.__constraints

		@property
		def color(self):
			return self.__color

		@color.setter
		def color(self, color):
			for c in self.constraints.values():
				c.color = color

		@property
		def maxdose(self):
			if len(self.constraints) > 0:
				return max(map(lambda c: c.maxdose, self.constraints.values()))
			else:
				return 0.

		def draw(self, axes, markersize=16, slack_rendering_threshold=0.1):
			for constraint_graph in self:
				constraint_graph.draw(axes, markersize,
									  slack_rendering_threshold)

		def undraw(self):
			for constraint_graph in self:
				constraint_graph.undraw()

		def show(self):
			for constraint_graph in self:
				constraint_graph.show()

		def hide(self):
			for constraint_graph in self:
				constraint_graph.hide()

	class StructureDVHGraph(object):
		def __init__(self, data, color=None, aesthetic=None):
			if isinstance(data, Structure):
				data = data.plotting_data()

			self.__name = data['name']
			self.__curve = DoseVolumeGraph(data['curve']['dose'],
									data['curve']['percentile'], color,
									aesthetic)
			self.curve.label = self.name
			self.__constraints = StructureConstraintsGraph(
					data['constraints'], color=color)

			rx = data.pop('rx', 0)
			if rx > 0:
				self.__rx = PrescriptionGraph(rx, color)
			else:
				self.__rx = None

		@property
		def name(self):
			return self.__name

		@property
		def curve(self):
			return self.__curve

		@property
		def constraints(self):
			return self.__constraints

		@property
		def rx(self):
			return self.__rx

		@property
		def color(self):
			return self.curve.color

		@color.setter
		def color(self, color):
			self.curve.color = color
			if self.rx is not None:
				self.rx.color = color
			for c in self.constraints:
				c.color = color

		@property
		def aesthetic(self):
			return self.curve.aesthetic

		@aesthetic.setter
		def aesthetic(self, aesthetic):
			self.curve.aesthetic = aesthetic

		def maxdose(self, exclude_constraints=False):
			if exclude_constraints:
				return self.curve.maxdose
			else:
				return max(self.constraints.maxdose, self.curve.maxdose)

		def draw(self, axes, aesthetic=None, curve=True, rx=True,
				 constraints=True, constraint_markersize=16,
				 constraint_slack_rendering_threshold=0.1):

			# draw curve
			if curve:
				self.curve.draw(axes, aesthetic=aesthetic)
			else:
				self.curve.undraw()

			# draw prescription
			if self.rx is not None:
				if rx:
					self.rx.draw(axes)
				else:
					self.rx.undraw()

			# draw constraints
			if constraints:
				self.constraints.draw(axes, constraint_markersize,
									  constraint_slack_rendering_threshold)
			else:
				self.constraints.undraw()

		def undraw(self):
			self.curve.undraw()
			if self.rx is not None:
				self.rx.undraw()
			self.constraints.undraw()

		def show(self):
			self.curve.show()
			if self.rx is not None:
				self.rx.show()
			self.constraints.show()

		def show_curve_only(self):
			self.curve.show()
			if self.rx is not None:
				self.rx.hide()
			self.constraints.hide()

		def hide(self):
			self.curve.hide()
			if self.rx is not None:
				self.rx.hide()
			self.constraints.hide()

	class PlanGraphBase(object):
		def __init__(self, data, colors=None):
			self.__structure_graphs = {label: None for label in data}
			if colors is not None:
				self.__colors = {label: colors[label] for label in colors}
			else:
				self.__colors = {label: None for label in data}

		def __getitem__(self, key):
			""" Overload operator []. """
			return self.__structure_graphs[key]

		def __iter__(self):
			""" Python3-compatible iterator implementation. """
			return self.__structure_graphs.items().__iter__()

		@property
		def structure_labels(self):
			return self.__structure_graphs.keys()

		@property
		def structure_colors(self):
			""" Dictionary of structure colors keyed by labels. """
			return self.__colors

		@structure_colors.setter
		def structure_colors(self, colors_by_structure):
			for label, color in colors_by_structure.items():
				if label in self.__structure_graphs:
					self.__structure_graphs[label].color = color
					self.__colors[label] = color

		def __maxdose(self, get_max):
			return max(map(get_max, self.__structure_graphs.values()))

		def undraw(self):
			for sg in self.__structure_graphs.values():
				if isinstance(
						sg, (StructureConstraintsGraph, StructureDVHGraph)):
					sg.undraw()

	class PlanDVHGraph(PlanGraphBase):
		def __init__(self, data, colors=None, aesthetic=None):
			if isinstance(data, Anatomy):
				data = data.plotting_data()
			if isinstance(data, PlanDVHGraph):
				data = data.structure_DVHs
				if colors is None:
					colors = data.structure_colors

			PlanGraphBase.__init__(self, data, colors)

			for label in data:
				if isinstance(data[label], StructureDVHGraph):
					self.structure_DVHs[label] = data[label]
					self.structure_DVHs[label].aesthetic = aesthetic
				else:
					self.structure_DVHs[label] = StructureDVHGraph(
							data[label], aesthetic=aesthetic)
			if colors is not None:
				self.structure_colors = colors

		@property
		def structure_DVHs(self):
			return self._PlanGraphBase__structure_graphs

		def maxdose(self, exclude_constraints=False):
			return self._PlanGraphBase__maxdose(
					lambda s: s.maxdose(exclude_constraints))

	class PlanConstraintsGraph(PlanGraphBase):
		def __init__(self, data, colors=None):
			if isinstance(data, Anatomy):
				data = data.plotting_data(constraints_only=True)

			PlanGraphBase.__init__(self, data, colors)
			for label, constraints in data.items():
				self.structure_constraints[label] = StructureConstraintsGraph(
						constraints)

		@property
		def structure_constraints(self):
			return self._PlanGraphBase__structure_graphs

		@property
		def maxdose(self):
			return self._PlanGraphBase__maxdose(lambda c: c.maxdose)
