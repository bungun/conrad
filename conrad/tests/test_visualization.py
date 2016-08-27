"""
Unit tests for :mod:`conrad.visualization.plot`
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

from os import remove as os_remove

from conrad.defs import vec
from conrad.medicine import Anatomy, Structure
from conrad.tests.base import *

try:
	from conrad.visualization.plot import *
	if PLOTTING_INSTALLED:
		from matplotlib.figure import Figure

	class VisualizationTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.beams = 50
			self.anatomy = Anatomy([
					Structure(0, 'PTV', True, A=rand(100, self.beams)),
					Structure(1, 'OAR1', False, A=rand(300, self.beams)),
					Structure(2, 'OAR2', False, A=rand(250, self.beams))
				])
			self.panels = {0: 1, 1: 1, 2: 1}
			self.names = {0: 'PTV', 1: 'OAR1', 2: 'OAR2'}
			self.case = Case(anatomy=self.anatomy)

		def test_panels_to_cols(self):
			self.assertTrue( panels_to_cols(1) == 1 )
			self.assertTrue( panels_to_cols(2) == 2 )
			self.assertTrue( panels_to_cols(3) == 2 )
			self.assertTrue( panels_to_cols(4) == 2 )
			self.assertTrue( panels_to_cols(5) == 3 )
			self.assertTrue( panels_to_cols(6) == 3 )
			self.assertTrue( panels_to_cols(7) == 4 )
			self.assertTrue( panels_to_cols(8) == 4 )
			self.assertTrue( panels_to_cols(12) == 4 )
			self.assertTrue( panels_to_cols(15) == 4 )
			self.assertTrue( panels_to_cols(25) == 4 )

		def test_dvh_plot_init(self):
			d = DVHPlot(self.panels, self.names)
			if d is None:
				return

			self.assertTrue( isinstance(d.fig, Figure) )
			self.assertTrue( d.n_structures == 3 )
			self.assertTrue( d._DVHPlot__panels_by_structure == self.panels )
			self.assertTrue( d.n_panels == 1 )
			self.assertTrue( d.cols == 1 )
			self.assertTrue( d.rows == 1 )
			self.assertTrue( d._DVHPlot__names_by_structure == self.names )
			self.assertTrue( isinstance(d._DVHPlot__colors_by_structure, dict) )
			self.assertTrue( len(d._DVHPlot__colors_by_structure) == 3 )

		def test_dvh_plot_plot(self):
			d = DVHPlot(self.panels, self.names)
			if d is None:
				return

			self.anatomy.calculate_doses(rand(self.beams))
			d.plot(self.anatomy.plotting_data)
			self.assertTrue( isinstance(d.fig, Figure) )

			# TODO: test options

		def test_dvh_plot_save(self):
			d = DVHPlot(self.panels, self.names)
			if d is None:
				return

			filename = path.join(path.abspath(path.dirname(__file__)), 'test.pdf')

			self.assertFalse( path.exists(filename) )
			d.save(filename)
			self.assertTrue( path.exists(filename) )
			os_remove(filename)
			self.assertFalse( path.exists(filename) )

		def test_case_plotter_init(self):
			cp = CasePlotter(self.case)
			if cp is None:
				return

			self.assertTrue( isinstance(cp.dvh_plot, DVHPlot) )

			label_list = [0, 1, 2, 'PTV', 'OAR1', 'OAR2']
			self.assertTrue( all(
				[cp.label_is_valid(label) for label in label_list]) )

		def test_case_plotter_set_display_groups(self):
			self.case.anatomy.label_order = [0, 1, 2]

			cp = CasePlotter(self.case)
			if cp is None:
				return

			cp.set_display_groups('together')
			self.assert_vector_equal(
					vec(list(cp.dvh_plot.series_panels.values())),
					vec([1, 1, 1]) )

			cp.set_display_groups('separate')
			self.assert_vector_equal(
					vec(list(cp.dvh_plot.series_panels.values())),
					vec([1, 2, 3]) )

			cp.set_display_groups('list', [('PTV',), ('OAR1', 'OAR2')])
			self.assert_vector_equal(
					vec(list(cp.dvh_plot.series_panels.values())),
					vec([1, 2, 2]) )


		def test_case_plotter_plot(self):
			cp = CasePlotter(self.case)
			if cp is None:
				return

			self.anatomy.calculate_doses(rand(self.beams))
			cp.plot(self.anatomy.plotting_data)
			self.assertTrue( isinstance(cp.dvh_plot.fig, Figure) )

			# TODO: test options
except:
	print('plotting tests threw exception (matplotlib); skipped')