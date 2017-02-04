"""
Unit tests for :mod:`conrad.optimization.history`.
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

from conrad.medicine import Prescription
from conrad.optimization.history import *
from conrad.tests.base import *

class RunProfileTestCase(ConradTestCase):
	def setUp(self):
		rx_file = os.path.join(
				os.path.abspath(os.path.dirname(__file__)), 'yaml_rx.yml')
		self.structures = Prescription(rx_file).structure_dict.values()

	def test_run_profile_init(self):
		rp = RunProfile()
		self.assertTrue( rp.use_slack )
		self.assertFalse( rp.use_2pass )
		self.assertIsInstance( rp.objectives, dict )
		self.assertEqual( len(rp.objectives), 0 )
		self.assertIsInstance( rp.constraints, dict )
		self.assertEqual( len(rp.objectives), 0 )
		self.assertEqual( rp.gamma, 'default' )

	def test_pull_objectives(self):
		rp = RunProfile()
		rp.pull_objectives(self.structures)

		self.assertEqual( len(rp.objectives), len(self.structures) )
		for s in self.structures:
			self.assertIn( s.label, rp.objectives )

	def test_pull_constraints(self):
		rp = RunProfile()
		rp.pull_constraints(self.structures)

		n_constraints = 0
		for s in self.structures:
			n_constraints += s.constraints.size

		self.assertEqual( len(rp.constraints), n_constraints )
		for s in self.structures:
			for cid in s.constraints:
				self.assertIn( cid, rp.constraints )

class RunOutputTestCase(ConradTestCase):
	def test_run_output_init(self):
		ro = RunOutput()
		self.assertIsInstance( ro.optimal_variables, dict )
		self.assertIn( 'x', ro.optimal_variables )
		self.assertIsNone( ro.optimal_variables['x'] )
		self.assertIn( 'x_exact', ro.optimal_variables )
		self.assertIsNone( ro.optimal_variables['x_exact'] )
		self.assertIsInstance( ro.optimal_dvh_slopes, dict )
		self.assertIsInstance( ro.solver_info, dict )
		self.assertIn( 'time', ro.solver_info )
		self.assert_nan( ro.solver_info['time'] )
		self.assertIn( 'time_exact', ro.solver_info )
		self.assert_nan( ro.solver_info['time_exact'] )
		self.assertFalse( ro.feasible )

	def test_run_output_properties(self):
		ro = RunOutput()
		self.assertIsNone( ro.x )
		self.assertIsNone( ro.x_exact )
		self.assert_nan( ro.solvetime )
		self.assert_nan( ro.solvetime_exact )

		ro.optimal_variables['x'] = np.random.rand(100)
		ro.optimal_variables['x_exact'] = np.random.rand(100)
		self.assertIsInstance( ro.x, np.ndarray )
		self.assertIsInstance( ro.x_exact, np.ndarray )

		ro.solver_info['time'] = np.random.rand()
		ro.solver_info['time_exact'] = np.random.rand()
		self.assertIsInstance( ro.solvetime, float )
		self.assertIsInstance( ro.solvetime_exact, float )

class RunRecordTestCase(ConradTestCase):
	def test_run_record_init(self):
		rr = RunRecord()
		self.assertIsInstance( rr.profile, RunProfile )
		self.assertTrue( rr.profile.use_slack )
		self.assertFalse( rr.profile.use_2pass )
		self.assertEqual( rr.profile.gamma, 'default' )
		self.assertIsInstance( rr.output, RunOutput )
		self.assertIsInstance( rr.plotting_data, dict )
		self.assertIn( 0, rr.plotting_data )
		self.assertIsNone( rr.plotting_data[0] )
		self.assertIn( 'exact', rr.plotting_data )
		self.assertIsNone( rr.plotting_data['exact'] )

	def test_run_record_properties(self):
		rr = RunRecord()

		self.assertFalse( rr.feasible )
		self.assertIsInstance( rr.info, dict )
		self.assertIsNone( rr.x )
		self.assertIsNone( rr.x_exact )
		with self.assertRaises(ValueError):
			rr.nonzero_beam_count
		with self.assertRaises(ValueError):
			rr.nonzero_beam_count_exact
		self.assert_nan( rr.solvetime )
		self.assert_nan( rr.solvetime_exact )

		x_rand = np.random.rand(100)
		xe_rand = np.random.rand(100)

		x_rand = x_rand * (x_rand >= 0.5) + (1e-6 * x_rand) * (x_rand < 0.5)
		xe_rand = xe_rand * (xe_rand >= 0.5) + (1e-6 * xe_rand) * (xe_rand < 0.5)

		count = sum(x_rand > 1e-6)
		count_exact = sum(xe_rand > 1e-6)

		rr.output.optimal_variables['x'] = x_rand
		self.assertIsInstance( rr.x, np.ndarray )
		self.assertEqual( rr.nonzero_beam_count, count )

		rr.output.optimal_variables['x_exact'] = xe_rand
		self.assertIsInstance( rr.x_exact, np.ndarray )
		self.assertEqual( rr.nonzero_beam_count_exact, count_exact )

class PlanningHistoryTestCase(ConradTestCase):
	def test_planning_history_init(self):
		h = PlanningHistory()
		self.assertIsInstance( h.runs, list )
		self.assertEqual( len(h.runs), 0 )
		self.assertIsInstance( h.run_tags, dict )
		self.assertEqual( len(h.run_tags), 0 )

	def test_planning_history_add_retrieve(self):
		h = PlanningHistory()
		h += RunRecord()
		self.assertEqual( len(h.runs), 1 )
		self.assertEqual( len(h.run_tags), 0 )

	def test_planning_history_properties(self):
		h = PlanningHistory()
		with self.assertRaises(TypeError):
			h.no_run_check()
		with self.assertRaises(ValueError):
			h.no_run_check('last_feasible')
		with self.assertRaises(ValueError):
			h.last_feasible
		with self.assertRaises(ValueError):
			h.last_info
		with self.assertRaises(ValueError):
			h.last_x
		with self.assertRaises(ValueError):
			h.last_x_exact
		with self.assertRaises(ValueError):
			h.last_solvetime
		with self.assertRaises(ValueError):
			h.last_solvetime_exact

		h += RunRecord()
		self.assertFalse( h.last_feasible )
		self.assertIsNone( h.last_x )
		self.assertIsNone( h.last_x_exact )
		self.assert_nan( h.last_solvetime )
		self.assert_nan( h.last_solvetime_exact )

	def test_planning_history_tag_last(self):
		h = PlanningHistory()
		with self.assertRaises(ValueError):
			h.tag_last('my tag')
		h += RunRecord()
		self.assertEqual( len(h.runs), 1 )
		self.assertEqual( len(h.run_tags), 0 )
		h.tag_last('my tag')
		self.assertEqual( len(h.run_tags), 1 )
		self.assertEqual( h.run_tags['my tag'], 0 )
		self.assertIsInstance( h[0], RunRecord )
		self.assertIsInstance( h['my tag'], RunRecord )
		self.assertEqual( h[0], h['my tag'] )