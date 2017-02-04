"""
Unit tests for :mod:`conrad.visualization.plot.canvases`
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

import os
import numpy as np

from conrad.physics.units import Gy
from conrad.medicine import Anatomy, Structure, D
from conrad.case import Case
from conrad.visualization.plot.elements import LineAesthetic
from conrad.visualization.plot.canvases import *
from conrad.tests.base import *

if not PLOTTING_INSTALLED:
	print('skipping tests for conrad.visualization.plot.canvases ')
else:
	class DVHSubplotTestCase(ConradTestCase):
		def test_dvhsubplot_init_properties(self):
			ax = mpl.figure.Figure().add_subplot(1, 1, 1)

			# test standard initialization and properties
			ds = DVHSubplot(ax)
			self.assertEqual( ds._DVHSubplot__title_location, 'left' )
			self.assertEqual( ds._DVHSubplot__title_fontsize, 12 )
			self.assertEqual( ds._DVHSubplot__title_fontweight, 'bold' )
			self.assertIsInstance(
					ds._DVHSubplot__title_font_dictionary, dict )
			self.assertIs( ds.axes, ax )
			self.assertTrue( ds.left )
			self.assertTrue( ds.bottom )
			self.assertEqual( ds.title, '' )

			ds.title = test_title = 'subplot title test'
			self.assertEqual( ds.title, test_title )

			self.assertIsNone( ds.legend )

			# test initialization with type other than mpl.axes.AxesSubplot
			with self.assertRaises(TypeError):
				ds = DVHSubplot(3)

			# test subplot positions
			f = mpl.figure.Figure()
			ax_NW = f.add_subplot(2, 2, 1)
			ax_NE = f.add_subplot(2, 2, 2)
			ax_SW = f.add_subplot(2, 2, 3)
			ax_SE = f.add_subplot(2, 2, 4)

			ds_NW = DVHSubplot(ax_NW)
			ds_NE = DVHSubplot(ax_NE)
			ds_SW = DVHSubplot(ax_SW)
			ds_SE = DVHSubplot(ax_SE)

			self.assertTrue( ds_NW.left )
			self.assertTrue( ds_SW.left )
			self.assertFalse( ds_NE.left )
			self.assertFalse( ds_SE.left )

			self.assertFalse( ds_NW.bottom )
			self.assertFalse( ds_NE.bottom )
			self.assertTrue( ds_SW.bottom )
			self.assertTrue( ds_SE.bottom )

		def test_dvhsubplot_legend(self):
			ds = DVHSubplot(mpl.figure.Figure().add_subplot(1, 1, 1))
			self.assertIsNone( ds.legend )

			# no data series
			ds.legend = True
			self.assertIsNone( ds.legend )

			ds.axes.lines.append(mpl.lines.Line2D([], [], label='test line'))
			ds.legend = True
			self.assertIsNotNone( ds.legend )
			self.assertTrue( ds.legend.get_visible() )

			ds.legend = False
			self.assertFalse( ds.legend.get_visible() )

			ds.legend = True
			self.assertTrue( ds.legend.get_visible() )

		def test_dvhsubplot_format(self):
			f = mpl.pyplot.figure()

			# test subplot format conditioned on position
			ax_NW = f.add_subplot(2, 2, 1)
			ax_NE = f.add_subplot(2, 2, 2)
			ax_SW = f.add_subplot(2, 2, 3)
			ax_SE = f.add_subplot(2, 2, 4)

			ds_NW = DVHSubplot(ax_NW)
			ds_NE = DVHSubplot(ax_NE)
			ds_SW = DVHSubplot(ax_SW)
			ds_SE = DVHSubplot(ax_SE)

			# minimal/explicit y-axes
			xlim = 22.0
			for minimal in [True, False]:
				for subplot in [ds_NW, ds_NE, ds_SW, ds_SE]:
					subplot.format(xlim, 'X LABEL', 'Y LABEL', minimal)
					self.assertEqual( subplot.axes.get_xlim(), (0.0, xlim) )
					self.assertEqual( subplot.axes.get_ylim(), (0.0, 103.0) )

					self.assertEqual(
							bool(subplot.xlabel == 'X LABEL'), subplot.bottom )
					self.assertEqual(
							bool(subplot.ylabel == 'Y LABEL'),
							bool(subplot.left or not minimal) )

					self.assertEqual( subplot.xaxis, subplot.bottom )
					self.assertEqual(
							subplot.yaxis,
							bool(subplot.left or not minimal) )

	class DVHPlotTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.beams = 50
			self.anatomy = Anatomy([
					Structure(
							0, 'PTV', True, A=np.random.rand(100, self.beams)),
					Structure(
							1, 'OAR1', False, A=np.random.rand(300, self.beams)),
					Structure(
							2, 'OAR2', False, A=np.random.rand(250, self.beams))
				])
			self.panel_assignments = {0: 0, 1: 0, 2: 0}
			self.names = {0: 'PTV', 1: 'OAR1', 2: 'OAR2'}
			self.case = Case(anatomy=self.anatomy)

		def setUp(self):
			self.build_call = DVHPlot.build

		def tearDown(self):
			DVHPlot.build = self.build_call

		def test_dvhplot_init_properties(self):
			DVHPlot.build = lambda arg_self: None

			d = DVHPlot(self.panel_assignments)
			self.assertNotIsInstance( d.figure, mpl.figure.Figure )
			self.assertEqual( d.n_structures, 3 )
			self.assertIsInstance( d.subplots, dict )
			self.assertEqual( len(d.subplots), 0 )
			self.assertIsInstance( d.panels, list )
			self.assertEqual( len(d.panels), 0 )
			for k, v in d.subplot_assignments.items():
				self.assertEqual( self.panel_assignments[k], v )

			self.assertEqual( d.cols, 1 )
			self.assertEqual( d.rows, 1 )
			self.assertEqual( d.layout, 'auto' )

		def test_dvhplot_panels_to_cols(self):
			DVHPlot.build = lambda arg_self: None
			d = DVHPlot(self.panel_assignments)

			self.assertEqual( d.subplots_to_cols(1), 1 )
			self.assertEqual( d.subplots_to_cols(2), 2 )
			self.assertEqual( d.subplots_to_cols(3), 2 )
			self.assertEqual( d.subplots_to_cols(4), 2 )
			self.assertEqual( d.subplots_to_cols(5), 3 )
			self.assertEqual( d.subplots_to_cols(6), 3 )
			self.assertEqual( d.subplots_to_cols(7), 4 )
			self.assertEqual( d.subplots_to_cols(8), 4 )
			self.assertEqual( d.subplots_to_cols(12), 4 )
			self.assertEqual( d.subplots_to_cols(15), 4 )
			self.assertEqual( d.subplots_to_cols(25), 4 )

		def test_dvhplot_sift_options(self):
			DVHPlot.build = lambda arg_self: None
			d = DVHPlot(self.panel_assignments)
			options = {'legend_opt1':1, 'legend_opt2':2, 'other_opt1':3,
					   'other_opt2':4}
			legend_options = {'opt1':1, 'opt2':2}
			other_options = {'other_opt1':3, 'other_opt2':4}

			o_opt, l_opt = d.sift_options(**options)
			for k, v in legend_options.items():
				self.assertEqual( l_opt[k], v )
			for k, v in other_options.items():
				self.assertEqual( o_opt[k], v )

		def test_dvhplot_build_clear(self):
			DVHPlot.build = lambda arg_self: None
			d = DVHPlot(self.panel_assignments)
			self.assertNotIsInstance( d.figure, mpl.figure.Figure )
			self.assertIsInstance( d.subplots, dict )
			self.assertEqual( len(d.subplots), 0 )
			self.assertIsInstance( d.panels, list )
			self.assertEqual( len(d.panels), 0 )

			# test calculate panels
			d.calculate_panels()
			self.assertEqual( d.n_subplots, 1 )
			self.assertEqual( d.rows, 1 )
			self.assertEqual( d.cols, 1 )

			d._DVHPlot__subplot_assignments_by_structure = {0: 0, 1: 1, 2: 2}
			d.calculate_panels()
			self.assertEqual( d.n_subplots, 3 )
			self.assertEqual( d.rows, 2 )
			self.assertEqual( d.cols, 2 )

			d._DVHPlot__layout = 'vertical'
			d.calculate_panels()
			self.assertEqual( d.n_subplots, 3 )
			self.assertEqual( d.rows, 3 )
			self.assertEqual( d.cols, 1 )

			d._DVHPlot__layout = 'horizontal'
			d.calculate_panels()
			self.assertEqual( d.n_subplots, 3 )
			self.assertEqual( d.rows, 1 )
			self.assertEqual( d.cols, 3 )

			# test build/clear
			DVHPlot.build = self.build_call
			d._DVHPlot__layout = 'auto'
			d.subplot_assignments = {0: 0, 1: 0, 2: 0}
			d.build()
			self.assertIsInstance( d.figure, mpl.figure.Figure )
			self.assertEqual( len(d.subplots), 3 )
			for sp in d.subplots.values():
				self.assertIsInstance( sp, DVHSubplot )
			self.assertIsInstance( d.panels, list )
			self.assertEqual( len(d.panels), 1 )
			for sp in d.panels:
				self.assertIsInstance( sp, DVHSubplot )

			d.clear()
			self.assertNotIsInstance( d.figure, mpl.figure.Figure )
			self.assertIsInstance( d.subplots, dict )
			self.assertEqual( len(d.subplots), 0 )
			self.assertIsInstance( d.panels, list )
			self.assertEqual( len(d.panels), 0 )

		def test_dvhplot_panels_layout(self):
			d = DVHPlot(self.panel_assignments)
			self.assertIsInstance( d.figure, mpl.figure.Figure )
			self.assertEqual( d.n_subplots, 1 )
			self.assertEqual( d.rows, 1 )
			self.assertEqual( d.cols, 1 )
			self.assertEqual( len(d.subplots), 3 )
			self.assertEqual( len(d.panels), 1 )
			for i in xrange(3):
				self.assertEqual( d.subplots[i], d.panels[0] )
				self.assertTrue( d.subplots[i].bottom )
				self.assertTrue( d.subplots[i].left )

			d.subplot_assignments = {0: 0, 1: 1, 2: 2}
			d.build()

			self.assertEqual( d.n_subplots, 3 )
			self.assertEqual( d.rows, 2 )
			self.assertEqual( d.cols, 2 )
			self.assertEqual( len(d.subplots), 3 )
			self.assertEqual( len(d.panels), 3 )
			for i in xrange(3):
				self.assertEqual( d.subplots[i], d.panels[i] )
				self.assertEqual( d.subplots[i].bottom, bool(i >= d.cols) )
				self.assertEqual( d.subplots[i].left, bool(i % d.cols == 0) )

			d.layout = 'vertical'
			self.assertEqual( d.n_subplots, 3 )
			self.assertEqual( d.rows, 3 )
			self.assertEqual( d.cols, 1 )
			self.assertEqual( len(d.subplots), 3 )
			self.assertEqual( len(d.panels), 3 )
			for i in xrange(3):
				self.assertEqual( d.subplots[i], d.panels[i] )
				self.assertEqual( d.subplots[i].bottom,
								 bool(i == d.n_subplots - 1 ) )
				self.assertTrue( d.subplots[i].left )

			d.layout = 'horizontal'
			self.assertEqual( d.n_subplots, 3 )
			self.assertEqual( d.rows, 1 )
			self.assertEqual( d.cols, 3 )
			self.assertEqual( len(d.subplots), 3 )
			self.assertEqual( len(d.panels), 3 )
			for i in xrange(3):
				self.assertEqual( d.subplots[i], d.panels[i] )
				self.assertTrue( d.subplots[i].bottom  )
				self.assertEqual( d.subplots[i].left, bool(i == 0) )

			with self.assertRaises(ValueError):
				d.layout = 'not a valid layout string'

		def test_dvhplot_getitem(self):
			d = DVHPlot(self.panel_assignments)

			for key in [0, 1, 2, 'upper right']:
				self.assertIsInstance( d[key], DVHSubplot )
				self.assertEqual( d[key], d.panels[0] )

			d.subplot_assignments = {0: 0, 1: 1, 2: 2}
			d.build()

			for key in [0, 1, 2, 'upper right']:
				self.assertIsInstance( d[key], DVHSubplot )
				if isinstance(key, int):
					self.assertEqual( d[key], d.panels[key] )
				else:
					self.assertEqual( d[key], d.panels[1] )

			d.layout = 'horizontal'
			self.assertEqual( d['upper right'], d.panels[-1] )

		def test_dvhplot_drawlegend(self):
			d = DVHPlot(self.panel_assignments)
			series = [mpl.lines.Line2D([],[]) for i in xrange(3)]
			names = ['first', 'second', 'third']

			self.assertEqual( len(d.figure.legends), 0 )
			d.draw_legend(series, names)
			self.assertEqual( len(d.figure.legends), 1 )

			self.assertEqual(
					len(d.figure.legends[-1].get_lines()), len(series) )
			for text in d.figure.legends[-1].get_texts():
				self.assertTrue( text.get_text() in names )

			# coordinates
			width = d.figure.legends[-1].get_bbox_to_anchor().x1
			height = d.figure.legends[-1].get_bbox_to_anchor().y1

			coords = [0.5, 0.6]
			d.draw_legend(series, names, coordinates=coords)

			origin_x = d.figure.legends[-1].get_bbox_to_anchor().x0
			origin_y = d.figure.legends[-1].get_bbox_to_anchor().y0
			self.assertTrue( coords[0] * width == origin_x )
			self.assertTrue( coords[1] * height == origin_y )

			# location
			d.draw_legend(series, names, coordinates=coords, alignment='right')
			d.draw_legend(series, names, coordinates=coords, alignment='center')
			self.assertNotEqual(
					d.figure.legends[-2]._loc, d.figure.legends[-1]._loc )

		def test_dvhplot_drawtitle(self):
			d = DVHPlot(self.panel_assignments)
			d.draw_title('test title')
			self.assertEqual( d.figure.texts[-1].get_text(), 'test title' )

		def test_dvhplot_format_subplots(self):
			d = DVHPlot(self.panel_assignments)

			xmax = 50.
			d.format_subplots(xmax)

			for subplot in d.panels:
				self.assertEqual( subplot.xmax, xmax )

		def test_check_figure(self):
			DVHPlot.build = lambda arg_self: None

			d = DVHPlot(self.panel_assignments)
			with self.assertRaises(AttributeError):
				d.check_figure()

			DVHPlot.build = self.build_call

			d = DVHPlot(self.panel_assignments)
			d.check_figure()

			d.clear()
			with self.assertRaises(AttributeError):
				d.check_figure()

		def test_dvhplot_plotvirtual(self):
			d = DVHPlot(self.panel_assignments)

			series_names = ['first', 'second', 'third']
			series_aesthetics = [LineAesthetic() for i in xrange(3)]

			d.plot_virtual(series_names, series_aesthetics)

			self.assertEqual( len(d.figure.legends), 1 )
			self.assertEqual( len(d.figure.legends[-1].get_lines()), 3 )
			for text in d.figure.legends[-1].get_texts():
				self.assertIn( text.get_text(), series_names )

		def test_dvhplot_plotlabels(self):
			d = DVHPlot(self.panel_assignments)

			coords = (0.5, 0.6)
			d.plot_labels({1: coords}, self.anatomy.plotting_data())

			self.assertEqual( len(d.subplots[1].axes.texts), 1 )
			t = d.subplots[1].axes.texts[-1]
			self.assertEqual( t.get_text(), self.anatomy[1].name )
			self.assertEqual( t.get_unitless_position(), coords )

			with self.assertRaises(KeyError):
				d.plot_labels({4: coords}, self.anatomy.plotting_data())

		def test_dvhplot_plotconstraints(self):
			d = DVHPlot(self.panel_assignments)

			LABEL = 0
			self.anatomy[LABEL].constraints += D(30) < 20 * Gy
			pc = PlanConstraintsGraph(self.anatomy)

			d.plot_constraints(pc)
			for constr in pc[LABEL]:
				self.assertIs( constr.axes, d.subplots[LABEL].axes )

		def test_dvhplot_plot(self):
			d = DVHPlot({i: i for i in xrange(3)})
			d.layout = 'horizontal'

			LABEL = 0
			self.anatomy.calculate_doses(np.random.rand(self.beams))
			self.anatomy[LABEL].constraints += D(30) < 20 * Gy
			pd = PlanDVHGraph(self.anatomy.plotting_data())

			# vanilla plot + self-title subplots
			d.plot(self.anatomy.plotting_data(), self_title_subplots=True)
			for structure in self.anatomy:
				self.assertEqual( d.subplots[structure.label].title,
								 structure.name  )

			# individual legends
			d.plot(self.anatomy.plotting_data(), legend='each')
			for structure in self.anatomy:
				self.assertIsNotNone( d.subplots[structure.label].legend )
				self.assertEqual(
						len(d.subplots[structure.label].legend.get_lines()),
						1 )
			self.assertEqual( len(d.figure.legends), 0 )

			# overall legend
			d.plot(self.anatomy.plotting_data(), legend=True)
			for structure in self.anatomy:
				self.assertIsNone( d.subplots[structure.label].legend )
			self.assertEqual( len(d.figure.legends), 1 )

			# self-title for multiple structures on single subplot
			d.subplot_assignments = {i: 0 for i in xrange(3)}
			d.build()
			d.plot(pd, self_title_subplots=True)
			title = ''
			for label, _ in pd:
				title += self.anatomy[label].name
				title += ', '

			title = title[:-2] # trim terminal ", "
			self.assertEqual( d.panels[0].title, title )

		# def test_dvhplot_show(self):
			# pass

		def test_dvhplot_save(self):
			d = DVHPlot(self.panel_assignments)

			filename = os.path.join(os.path.abspath(
					os.path.dirname(__file__)), 'test.pdf')

			self.assertFalse( os.path.exists(filename) )
			d.build()
			d.save(filename)
			self.assertTrue( os.path.exists(filename) )
			os.remove(filename)
			self.assertFalse( os.path.exists(filename) )
