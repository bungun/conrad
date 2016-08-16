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
from os import path
from numpy import ndarray

from conrad.compat import *
from conrad.medicine import Prescription
from conrad.optimization.history import *
from conrad.tests.base import *

class RunProfileTestCase(ConradTestCase):
	def setUp(self):
		rx_file = path.join(
				path.abspath(path.dirname(__file__)), 'yaml_rx.yml')
		self.structures = Prescription(rx_file).structure_dict.values()

	def test_run_profile_init(self):
		rp = RunProfile()
		self.assertTrue( rp.use_slack )
		self.assertFalse( rp.use_2pass )
		self.assertTrue( isinstance(rp.objectives, dict) )
		self.assertTrue( len(rp.objectives) == 0 )
		self.assertTrue( isinstance(rp.constraints, dict) )
		self.assertTrue( len(rp.objectives) == 0 )
		self.assertTrue( rp.gamma == 'default' )

	def test_pull_objectives(self):
		rp = RunProfile()
		rp.pull_objectives(self.structures)

		self.assertTrue( len(rp.objectives) == len(self.structures) )
		for s in self.structures:
			self.assertTrue( s.label in rp.objectives )

	def test_pull_constraints(self):
		rp = RunProfile()
		rp.pull_constraints(self.structures)

		n_constraints = 0
		for s in self.structures:
			n_constraints += s.constraints.size

		self.assertTrue( len(rp.constraints) == n_constraints )
		for s in self.structures:
			for cid in s.constraints:
				self.assertTrue( cid in rp.constraints )

class RunOutputTestCase(ConradTestCase):
	def test_run_output_init(self):
		ro = RunOutput()
		self.assertTrue( isinstance(ro.optimal_variables, dict) )
		self.assertTrue( 'x' in ro.optimal_variables )
		self.assertTrue( ro.optimal_variables['x'] is None )
		self.assertTrue( 'x_exact' in ro.optimal_variables )
		self.assertTrue( ro.optimal_variables['x_exact'] is None )
		self.assertTrue( isinstance(ro.optimal_dvh_slopes, dict) )
		self.assertTrue( isinstance(ro.solver_info, dict) )
		self.assertTrue( 'time' in ro.solver_info )
		self.assert_nan( ro.solver_info['time'] )
		self.assertTrue( 'time_exact' in ro.solver_info )
		self.assert_nan( ro.solver_info['time_exact'] )
		self.assertFalse( ro.feasible )

	def test_run_output_properties(self):
		ro = RunOutput()
		self.assertTrue( ro.x is None )
		self.assertTrue( ro.x_exact is None )
		self.assert_nan( ro.solvetime )
		self.assert_nan( ro.solvetime_exact )

		ro.optimal_variables['x'] = rand(100)
		ro.optimal_variables['x_exact'] = rand(100)
		self.assertTrue( isinstance(ro.x, ndarray) )
		self.assertTrue( isinstance(ro.x_exact, ndarray) )

		ro.solver_info['time'] = rand()
		ro.solver_info['time_exact'] = rand()
		self.assertTrue( isinstance(ro.solvetime, float) )
		self.assertTrue( isinstance(ro.solvetime_exact, float) )

class RunRecordTestCase(ConradTestCase):
	def test_run_record_init(self):
		rr = RunRecord()
		self.assertTrue( isinstance(rr.profile, RunProfile) )
		self.assertTrue( rr.profile.use_slack )
		self.assertFalse( rr.profile.use_2pass )
		self.assertTrue( rr.profile.gamma == 'default' )
		self.assertTrue( isinstance(rr.output, RunOutput) )
		self.assertTrue( isinstance(rr.plotting_data, dict) )
		self.assertTrue( 0 in rr.plotting_data )
		self.assertTrue( rr.plotting_data[0] is None )
		self.assertTrue( 'exact' in rr.plotting_data )
		self.assertTrue( rr.plotting_data['exact'] is None )

	def test_run_record_properties(self):
		rr = RunRecord()

		self.assertFalse( rr.feasible )
		self.assertTrue( isinstance(rr.info, dict) )
		self.assertTrue( rr.x is None )
		self.assertTrue( rr.x_exact is None )
		self.assert_property_exception(
				obj=rr, property_name='nonzero_beam_count' )
		self.assert_property_exception(
				obj=rr, property_name='nonzero_beam_count_exact' )
		self.assert_nan( rr.solvetime )
		self.assert_nan( rr.solvetime_exact )

		x_rand = rand(100)
		xe_rand = rand(100)

		x_rand = x_rand * (x_rand >= 0.5) + (1e-6 * x_rand) * (x_rand < 0.5)
		xe_rand = xe_rand * (xe_rand >= 0.5) + (1e-6 * xe_rand) * (xe_rand < 0.5)

		count = sum(x_rand > 1e-6)
		count_exact = sum(xe_rand > 1e-6)

		rr.output.optimal_variables['x'] = x_rand
		self.assertTrue( isinstance(rr.x, ndarray) )
		self.assertTrue( rr.nonzero_beam_count == count )

		rr.output.optimal_variables['x_exact'] = xe_rand
		self.assertTrue( isinstance(rr.x_exact, ndarray) )
		self.assertTrue( rr.nonzero_beam_count_exact == count_exact )

class PlanningHistoryTestCase(ConradTestCase):
	def test_planning_history_init(self):
		h = PlanningHistory()
		self.assertTrue( isinstance(h.runs, list) )
		self.assertTrue( len(h.runs) == 0 )
		self.assertTrue( isinstance(h.run_tags, dict) )
		self.assertTrue( len(h.run_tags) == 0 )

	def test_planning_history_add_retrieve(self):
		h = PlanningHistory()
		h += RunRecord()
		self.assertTrue( len(h.runs) == 1 )
		self.assertTrue( len(h.run_tags) == 0 )

	def test_planning_history_properties(self):
		h = PlanningHistory()
		self.assert_exception( call=h.no_run_check, args=[] )
		self.assert_property_exception( obj=h, property_name='last_feasible' )
		self.assert_property_exception( obj=h, property_name='last_info' )
		self.assert_property_exception( obj=h, property_name='last_x' )
		self.assert_property_exception( obj=h, property_name='last_x_exact' )
		self.assert_property_exception( obj=h, property_name='last_solvetime' )
		self.assert_property_exception(
				obj=h, property_name='last_solvetime_exact' )
		h += RunRecord()
		self.assertFalse( h.last_feasible )
		self.assertTrue( h.last_x is None )
		self.assertTrue( h.last_x_exact is None )
		self.assert_nan( h.last_solvetime )
		self.assert_nan( h.last_solvetime_exact )

	def test_planning_history_tag_last(self):
		h = PlanningHistory()
		self.assert_exception( call=h.tag_last, args=['my tag'] )
		h += RunRecord()
		self.assertTrue( len(h.runs) == 1 )
		self.assertTrue( len(h.run_tags) == 0 )
		h.tag_last('my tag')
		self.assertTrue( len(h.run_tags) == 1 )
		self.assertTrue( h.run_tags['my tag'] == 0 )
		self.assertTrue( isinstance(h[0], RunRecord) )
		self.assertTrue( isinstance(h['my tag'], RunRecord) )
		self.assertTrue( h[0] == h['my tag'] )