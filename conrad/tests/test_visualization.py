from os import remove as os_remove

from conrad.compat import *
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
					vec(cp.dvh_plot.series_panels.values()), vec([1, 1, 1]) )

			cp.set_display_groups('separate')
			self.assert_vector_equal(
					vec(cp.dvh_plot.series_panels.values()), vec([1, 2, 3]) )

			cp.set_display_groups('list', [('PTV',), ('OAR1', 'OAR2')])
			self.assert_vector_equal(
					vec(cp.dvh_plot.series_panels.values()), vec([1, 2, 2]) )


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