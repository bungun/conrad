"""
Unit tests for :mod:`conrad.visualization.plot.plotter`
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

import numpy as np

from conrad.defs import vec
from conrad.physics.units import Gy
from conrad.medicine import Anatomy, Structure, D
from conrad.case import Case
from conrad.visualization.plot.plotter import *
from conrad.tests.base import *

if not PLOTTING_INSTALLED:
	print('skipping tests for conrad.visualization.plot.plotter ')
else:
	class CasePlotterTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.beams = 50
			self.anatomy = Anatomy([
					Structure(
							0, 'PTV', True, A=np.random.rand(100, self.beams)),
					Structure(
							1, 'OAR1', False,
							A=np.random.rand(300, self.beams)),
					Structure(
							2, 'OAR2', False,
							A=np.random.rand(250, self.beams))
				])
			self.anatomy['PTV'].constraints += D(80) > 25 * Gy
			self.anatomy['PTV'].constraints += D(30) < 28 * Gy
			self.anatomy['OAR1'].constraints += D(20) < 10 * Gy
			self.panels = {s.label: 0 for s in self.anatomy}
			self.names = {s.label: s.name for s in self.anatomy}
			self.case = Case(anatomy=self.anatomy)

		def setUp(self):
			self.anatomy.calculate_doses(np.random.rand(self.beams))

			# store function in cases when it is disabled
			self.autoset_colors = CasePlotter.autoset_structure_colors

		def tearDown(self):
			CasePlotter.autoset_structure_colors = self.autoset_colors

		def test_caseplotter_init_properties(self):
			# disable autoset colors in initialization
			CasePlotter.autoset_structure_colors = lambda arg_self: None

			with self.assertRaises(TypeError):
				cp = CasePlotter('not a case')

			# standard initialization
			cp = CasePlotter(self.case)
			n_struc = self.case.anatomy.n_structures
			self.assertIsInstance( cp.dvh_plot, DVHPlot )
			self.assertIsNone( cp.dvh_set )
			self.assertIsInstance( cp.structure_subset, list )
			self.assertEqual( len(cp.structure_subset), n_struc )
			self.assertIsInstance( cp.structure_colors, dict )
			self.assertEqual( len(cp.structure_colors), 0 )
			self.assertIsInstance( cp._CasePlotter__tag2label, dict )
			self.assertEqual( len(cp._CasePlotter__tag2label), 2 * n_struc )

			self.assertEqual( cp.grouping, 'together' )
			self.assertEqual( cp.n_structures, self.anatomy.n_structures )

			tags = [s.label for s in self.anatomy]
			tags += [s.name for s in self.anatomy]
			self.assertTrue( all(
					[t in cp._CasePlotter__tag2label for t in tags]) )

			# subset filtering
			test_list = [0, 1, 2, 3, 4, 'PTV', 'OAR', 'OAR1', 'some string']
			filtered_list = cp.filter_labels(test_list)
			self.assertSetEqual(
					set(filtered_list), {i for i in xrange(n_struc)} )

			test_dictionary = {item: item for item in test_list}
			filtered_dict = cp.filter_data(test_dictionary)
			self.assertDictEqual(
					filtered_dict, {i: i for i in xrange(n_struc)} )

			# add data
			cp.dvh_set = self.case.anatomy.plotting_data()
			self.assertIsInstance( cp.dvh_set, PlanDVHGraph )
			self.assertEqual( len(cp.dvh_set.structure_DVHs), n_struc )

			# subset initialization
			subset1 = [0, 1]
			n_subset = len(subset1)
			cp = CasePlotter(self.case, subset=subset1)
			self.assertEqual( len(cp.structure_subset), n_subset )
			self.assertEqual( len(cp._CasePlotter__tag2label), 2 * n_struc )
			self.assertEqual( cp.n_structures, n_subset )

			cp.dvh_set = self.case.anatomy.plotting_data()
			self.assertEqual( len(cp.dvh_set.structure_DVHs), n_subset )

			subset2 = [self.names[label] for label in subset1]
			cp = CasePlotter(self.case, subset=subset2)
			self.assertEqual( len(cp.structure_subset), n_subset )
			self.assertEqual( len(cp._CasePlotter__tag2label), 2 * n_struc )
			self.assertEqual( cp.n_structures, n_subset )

			cp.dvh_set = self.case.anatomy.plotting_data()
			self.assertEqual( len(cp.dvh_set.structure_DVHs), n_subset )

			# verify that redundant entries are included once
			subset3 = subset1 + subset2
			cp = CasePlotter(self.case, subset=subset3)
			self.assertEqual( len(cp.structure_subset), n_subset )
			self.assertEqual( len(cp._CasePlotter__tag2label), 2 * n_struc )
			self.assertEqual( cp.n_structures, n_subset )

			cp.dvh_set = self.case.anatomy.plotting_data()
			self.assertEqual( len(cp.dvh_set.structure_DVHs), n_subset )

		def test_caseplotter_colors(self):
			# disable autoset colors in initialization
			CasePlotter.autoset_structure_colors = lambda arg_self: None

			cp = CasePlotter(self.case)
			colors = {0: 'blue', 1: 'red', 2:'yellow'}
			colors_snames = {'PTV': 'blue', 'OAR1': 'red', 'OAR2':'yellow'}

			cp.structure_colors = colors
			self.assertEqual( cp.structure_colors, colors )

			cp.structure_colors = colors_snames
			self.assertEqual( cp.structure_colors, colors )

			colors[0] = 'chartreuse'
			cp.structure_colors = {0: 'chartreuse'}
			self.assertEqual( cp.structure_colors, colors )

			# restore autoset colors
			CasePlotter.autoset_structure_colors = self.autoset_colors

			# test autoset_structure_colors
			cp._CasePlotter__structure_colors = {}
			self.assertEqual( len(cp.structure_colors), 0 )
			cp.autoset_structure_colors()
			self.assertEqual( len(cp.structure_colors), cp.n_structures )

			# autoset with structure order permutation
			order0 = [0, 1, 2]
			order1 = [2, 0, 1]
			colors_permuted = {label: cp.structure_colors[order0[i]]
							   for i, label in enumerate(order1)}
			cp.autoset_structure_colors(structure_order=order1)
			self.assertEqual( colors_permuted, cp.structure_colors )

			# autoset with invalid structure order permutation
			with self.assertRaises(ValueError):
				cp.autoset_structure_colors(structure_order=[2, 0])

			# autoset with non-default colormap
			cp._CasePlotter__structure_colors = {}
			cp.autoset_structure_colors(colormap='rainbow')
			self.assertEqual( len(cp.structure_colors), cp.n_structures )

			# autoset with invalid colormap
			with self.assertRaises(ValueError):
				cp.autoset_structure_colors(colormap='garbage string')

		def test_caseplotter_grouping(self):
			self.case.anatomy.label_order = [0, 1, 2]
			cp = CasePlotter(self.case)

			cp.grouping = 'together'
			self.assertEqual( cp.grouping, 'together' )
			for i in xrange(3):
				self.assertEqual( cp.dvh_plot.subplot_assignments[i], 0 )

			cp.grouping = 'separate'
			self.assertEqual( cp.grouping, 'separate' )
			for i in xrange(3):
				self.assertEqual( cp.dvh_plot.subplot_assignments[i], i )

			group_alternates = [
					(0, (1, 2)), (0, [1, 2]), [0, (1, 2)], [0, [1, 2]]]
			group_alternates += [
					('PTV', ('OAR1', 'OAR2')), ('PTV', [1, 'OAR2'])]
			for g in group_alternates:
				cp.grouping = g
				for i in xrange(3):
					self.assertEqual( cp.grouping, 'list' )
					self.assertEqual(
							cp.dvh_plot.subplot_assignments[i], int(i > 0) )

			with self.assertRaises(TypeError):
				cp.grouping = 1
			with self.assertRaises(ValueError):
				cp.grouping = 'not a valid grouping'
			with self.assertRaises(ValueError):
				cp.grouping = [0, 1]

		def test_caseplotter_plot(self):
			cp = CasePlotter(self.case)
			plan_data = self.case.plotting_data(x=np.random.rand(self.beams))
			cp.plot(plan_data)

		# 	# TODO: test options

		def test_caseplotter_twopass(self):
			cp = CasePlotter(self.case)
			plan_data_pass1 = self.case.plotting_data(
					x=np.random.rand(self.beams))
			plan_data_pass2 = self.case.plotting_data(
					x=np.random.rand(self.beams))
			cp.plot_twopass([plan_data_pass1, plan_data_pass2])

		# 	# TODO: test options

		def test_caseplotter_plotmulti(self):
			cp = CasePlotter(self.case)

			plan_data_ref = self.case.plotting_data(
					x=np.random.rand(self.beams))
			plan_data_list = [
				self.case.plotting_data(x=np.random.rand(self.beams))
						for i in xrange(3)]
			plan_names = ['first', 'second', 'third']

			cp.plot_multi(plan_data_list, plan_names)
			cp.plot_multi(plan_data_list, plan_names,
						  reference_data=plan_data_ref)

		# 	# TODO: test options
