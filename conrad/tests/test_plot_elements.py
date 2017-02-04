"""
Unit tests for :mod:`conrad.visualization.plot.elements`
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
from conrad.visualization.plot.elements import *
from conrad.tests.base import *

if not PLOTTING_INSTALLED:
	print('skipping tests for conrad.visualization.plot.elements ')
else:
	class LineAestheticTestCase(ConradTestCase):
		def test_aesthetic_init_properties(self):
			la = LineAesthetic()
			vline = la._LineAesthetic__verification_line
			self.assertIsInstance( vline, mpl.lines.Line2D )

			# default assignments
			self.assertEqual( la.style, '-' )
			self.assertEqual( la.weight, 1.0 )
			self.assertEqual( la.marker, None )
			self.assertEqual( la.markersize, 5.0 )
			self.assertEqual( la.fill, 'none' )
			self.assertEqual( la.num_markers, 20 )
			self.assertEqual( la.alpha, 1.0 )
			self.assertEqual( la.color_attenuation, 1.0 )

			for style in ['--', ':', '-.', 'solid', 'dashed', 'dotted',
						  'dashdot']:
				la.style = style
				self.assertEqual( la.style, style)
			with self.assertRaises(ValueError):
				la.style = 'not a style'

			for weight in [1, 2.3, '2']:
				la.weight = weight
				self.assertEqual( la.weight, weight)
			with self.assertRaises(ValueError):
				la.weight = 'not a weight'

			for marker in vline.filled_markers:
				la.marker = marker
				self.assertEqual( la.marker, marker )
			with self.assertRaises(ValueError):
				la.marker = 'not a marker'

			for markersize in [1, 2.3, '5']:
				la.markersize = markersize
				self.assertEqual( la.markersize, markersize )
			with self.assertRaises(ValueError):
				la.markersize = 'not a markersize'

			for fill in vline.fillStyles:
				la.fill = fill
				self.assertEqual( la.fill, fill )
			with self.assertRaises(ValueError):
				la.fill = 'not a fill'

			for number in [1, 5, 100, 2.3, '2']:
				la.num_markers = number
				self.assertEqual( la.num_markers, int(number) )
			with self.assertRaises(ValueError):
				la.fill = 'not a number'

			for alpha in [1, 0, 0.5, '0.76']:
				la.alpha = alpha
				self.assertEqual( la.alpha, float(alpha) )
			for alpha_fail  in [-0.2, 1.4, 'not an alpha']:
				with self.assertRaises(ValueError):
					la.alpha = alpha_fail

			for attenuation in [1, 0, 0.5, '0.76']:
				la.color_attenuation = attenuation
				self.assertEqual(
						la.color_attenuation, float(attenuation) )
			for attenuation_fail  in [-0.2, 1.4, 'not an attenuation']:
				with self.assertRaises(ValueError):
					la.color_attenuation = attenuation_fail

			# initialization with names aesthetics
			la = LineAesthetic('dvh_constraint')
			self.assertEqual( la.style, '' )
			self.assertEqual( la.markersize, 16 )

			la = LineAesthetic('slack')
			self.assertEqual( la.alpha, 0.6 )

			la = LineAesthetic('rx')
			self.assertEqual( la.style, ':' )
			self.assertEqual( la.weight, 1.5 )

			# initialization with keywords
			la = LineAesthetic(
					style='--', weight=1.5, marker='o', markersize=12,
					fill='left', num_markers=10, alpha=0.8,
					color_attenuation=0.8)
			self.assertEqual( la.style, '--' )
			self.assertEqual( la.weight, 1.5 )
			self.assertEqual( la.marker, 'o' )
			self.assertEqual( la.markersize, 12 )
			self.assertEqual( la.fill, 'left' )
			self.assertEqual( la.num_markers, 10 )
			self.assertEqual( la.alpha, 0.8 )
			self.assertEqual( la.color_attenuation, 0.8 )

		def test_aesthetic_copy(self):
			la_0 = LineAesthetic()
			la_1 = LineAesthetic()
			la_1.style = ''
			la_1.weight = 2.1
			la_1.marker = '<'
			la_1.markersize = 15
			la_1.fill = 'left'
			la_1.num_markers = 3
			la_1.alpha = 0.4
			la_1.color_attenuation = 0.3
			la_0.copy(la_1)
			self.assertEqual( la_0.style, la_1.style )
			self.assertEqual( la_0.weight, la_1.weight )
			self.assertEqual( la_0.marker, la_1.marker )
			self.assertEqual( la_0.markersize, la_1.markersize )
			self.assertEqual( la_0.fill, la_1.fill )
			self.assertEqual( la_0.num_markers, la_1.num_markers )
			self.assertEqual( la_0.alpha, la_1.alpha )
			self.assertEqual( la_0.color_attenuation,
							 la_1.color_attenuation )

			# test operator ==
			self.assertEqual( la_0, la_1 )

		def test_aesthetic_scale_rgb(self):
			la = LineAesthetic()

			color = mpl.colors.colorConverter.to_rgba('cornflowerblue')
			color2 = la.scale_rgb(color)
			self.assertEqual( color, color2 )

			la.color_attenuation = 0.7
			color3 = list(color)
			for i in range(3):
				color3[i] *= la.color_attenuation
			color3 = tuple(color3)
			color4 = la.scale_rgb(color)
			self.assertEqual( color3, color4 )

		def test_aesthetic_sample_factor(self):
			la = LineAesthetic()

			for series_length in [1, 10, 100, 1000]:
				quotient = la.num_markers
				sampling = max(1, series_length // quotient)
				self.assertEqual( la.get_sample_factor(series_length),
								 sampling )

		def test_aesthetic_apply(self):
			la = LineAesthetic()
			vline = la._LineAesthetic__verification_line

			la.style = ''
			la.weight = 2.1
			la.marker = '<'
			la.markersize = 15
			la.fill = 'left'
			la.num_markers = 3
			la.alpha = 0.4
			la.color_attenuation = 0.3

			line = mpl.lines.Line2D([], [])
			la.apply(line, 'blue')

			self.assertEqual(
					line.get_linestyle(), vline.get_linestyle() )
			self.assertEqual( line.get_linewidth(), la.weight )
			self.assertEqual( line.get_marker(), la.marker )
			self.assertEqual( line.get_markersize(), la.markersize )
			self.assertEqual( line.get_fillstyle(), la.fill )
			self.assertEqual(
					line.get_markevery(), (1, la.get_sample_factor(0)) )
			self.assertEqual( line.get_alpha(), la.alpha )
			self.assertEqual( line.get_color(), la.scale_rgb('blue') )

			la.apply_color(line, 'red')
			self.assertEqual( line.get_color(), la.scale_rgb('red') )

		def test_aesthetic_plot_args(self):
			la = LineAesthetic()
			args = la.plot_args('blue', 100)
			color = la.scale_rgb('blue')
			markevery = la.get_sample_factor(100)

			self.assertIn( 'linestyle', args )
			self.assertEqual( args['linestyle'], la.style )

			self.assertIn( 'linewidth', args )
			self.assertEqual( args['linewidth'], la.weight )

			self.assertIn( 'marker', args )
			self.assertEqual( args['marker'], la.marker )

			self.assertIn( 'markersize', args )
			self.assertEqual( args['markersize'], la.markersize )

			self.assertIn( 'markevery', args )
			self.assertEqual( args['markevery'], (1, markevery) )

			self.assertIn( 'fillstyle', args )
			self.assertEqual( args['fillstyle'], la.fill )

			self.assertIn( 'color', args )
			self.assertEqual( args['color'], color )

			self.assertIn( 'alpha', args )
			self.assertEqual( args['alpha'], la.alpha )

	class DVHPlotElementTestCase(ConradTestCase):
		def test_dvhplot_element_init_properties(self):
			dpe = DVHPlotElement()
			self.assertIsNone( dpe.axes )
			self.assertIsInstance( dpe.graph, list )
			self.assertEqual( len(dpe.graph), 0 )
			self.assertEqual( dpe.color, 'black' )
			self.assertIsInstance( dpe.aesthetic, LineAesthetic )
			self.assertEqual( dpe.label, '_nolabel_' )

			dpe.color = 'blue'
			self.assertEqual(
					dpe.color, mpl.colors.colorConverter.to_rgba('blue') )
			with self.assertRaises(ValueError):
				dpe.color = 'not a color'

			aes = LineAesthetic()
			aes.style = '--'
			aes.marker = '>'
			aes.weight = 3.5
			aes.markersize = 9
			aes.alpha = 0.5
			self.assertNotEqual( dpe.aesthetic, aes )
			dpe.aesthetic = aes
			self.assertEqual( dpe.aesthetic, aes )

			label = 'label string'
			dpe.label = label
			self.assertEqual( dpe.label, label )
			for label in ['', ' ', None]:
				dpe.label = label
				self.assertEqual( dpe.label, '_nolabel_' )

			ax = mpl.figure.Figure().add_subplot(1, 1, 1)
			dpe.axes = ax
			self.assertEqual( dpe.axes, ax )
			with self.assertRaises(TypeError):
				dpe.axes = '2'

		def test_dvhplot_element_iadd(self):
			dpe = DVHPlotElement()
			self.assertEqual( len(dpe.graph), 0 )

			# add single line
			dpe += mpl.lines.Line2D([], [])
			self.assertEqual( len(dpe.graph), 1 )

			# add list of lines
			dpe += [mpl.lines.Line2D([], []) for i in xrange(4)]
			self.assertEqual( len(dpe.graph), 5 )

		def test_dvhplot_element_show_hide(self):
			dpe = DVHPlotElement()

			dpe += [mpl.lines.Line2D([], []) for i in xrange(4)]

			dpe.hide()
			for g in dpe.graph:
				self.assertFalse( g.get_visible() )

			dpe.show()
			for g in dpe.graph:
				self.assertTrue( g.get_visible() )

		def test_dvhplot_element_draw_undraw(self):
			dpe = DVHPlotElement()
			dpe += [mpl.lines.Line2D([], []) for i in xrange(4)]

			ax = mpl.figure.Figure().add_subplot(1, 1, 1)
			dpe.axes = ax
			for g in dpe.graph:
				self.assertIs( g.axes, ax )
				self.assertIn( g, ax.lines )

			with self.assertRaises(NotImplementedError):
				dpe.draw(ax)

			dpe.undraw()
			for g in dpe.graph:
				self.assertIsNone( g.axes )
				self.assertNotIn( g, ax.lines )

	class DoseVolumeGraphTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.doses = [0., 1., 2., 3.]
			self.percentiles = [100, 72, 55, 3]
			self.color = 'blue'
			self.ax = mpl.figure.Figure().add_subplot(1, 1, 1)

		def test_dosevolume_graph_init_draw(self):
			dvg = DoseVolumeGraph(self.doses, self.percentiles, self.color)
			self.assertEqual( len(dvg.graph), 1 )

			dvg.draw(self.ax)
			self.assertIs( dvg.graph[0].axes, self.ax )
			self.assertIn( dvg.graph[0], self.ax.lines )

			dvg.draw(self.ax, LineAesthetic(style=':'))
			self.assertEqual( dvg.aesthetic.style, ':' )

		def test_dosevolume_graph_maxdose(self):
			dvg = DoseVolumeGraph(self.doses, self.percentiles, self.color)
			self.assertEqual( dvg.maxdose, self.doses[-1] )

	class PercentileConstraintGraphTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.doses = [30., 33.]
			self.percentiles = [55, 55]
			self.symbol = '>'
			self.color = 'blue'
			self.ax = mpl.figure.Figure().add_subplot(1, 1, 1)

		def test_percentileconstraint_graph_init(self):
			pcg = PercentileConstraintGraph(self.doses, self.percentiles,
											self.symbol, self.color)
			self.assertEqual( len(pcg.graph), 1 )
			self.assertEqual(
					pcg._PercentileConstraintGraph__slack_amount, 3. )


		def test_percentileconstraint_graph_draw(self):
			pcg = PercentileConstraintGraph(self.doses, self.percentiles,
											self.symbol, self.color)
			slack = pcg._PercentileConstraintGraph__slack
			achieved = pcg._PercentileConstraintGraph__achieved
			requested = pcg._PercentileConstraintGraph__requested

			# draw requested & achieved constraints and slack
			pcg.draw(self.ax)
			self.assertEqual( pcg.aesthetic.markersize, 16 )
			self.assertEqual( len(pcg.graph), 3 )
			self.assertIs( slack.axes, self.ax )
			self.assertIn( slack, self.ax.lines )
			self.assertIs( requested.axes, self.ax )
			self.assertIn( requested, self.ax.lines )
			self.assertIs( achieved.axes, self.ax )
			self.assertIn( achieved, self.ax.lines )

			# draw only achieved constraint
			pcg.draw(self.ax, size=12, slack_threshold=5)
			self.assertEqual( pcg.aesthetic.markersize, 12 )
			self.assertEqual( len(pcg.graph), 1 )
			self.assertIsNone( slack.axes, None )
			self.assertNotIn( slack, self.ax.lines )
			self.assertIs( requested.axes, None )
			self.assertNotIn( requested, self.ax.lines )
			self.assertIs( achieved.axes, self.ax )
			self.assertIn( achieved, self.ax.lines )

		def test_percentileconstraint_graph_maxdose(self):
			pcg = PercentileConstraintGraph(self.doses, self.percentiles,
											self.symbol, self.color)
			self.assertEqual( pcg.maxdose, max(self.doses) )

	class PrescriptionGraphTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.doses = [30., 30.]
			self.percentiles = [55, 52]
			self.symbol = '>'
			self.color = 'blue'
			self.ax = mpl.figure.Figure().add_subplot(1, 1, 1)

		def test_prescription_graph_init_draw(self):
			pg = PrescriptionGraph(30.0, self.color)
			self.assertEqual( len(pg.graph), 1 )

			pg.draw(self.ax)
			self.assertIs( pg.graph[0].axes, self.ax )
			self.assertIn( pg.graph[0], self.ax.lines )