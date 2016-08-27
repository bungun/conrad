"""
Unit tests for :mod:`conrad.case`.
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

from os import path

from conrad.physics import Gy
from conrad.medicine import D, Structure
from conrad.case import *
from conrad.tests.base import *

class CaseTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		m, n = 100, 50
		self.anatomy = Anatomy([
				Structure(0, 'PTV', True),
				Structure(1, 'OAR1', False),
				Structure(2, 'OAR2', False)
			])
		self.physics = Physics(
				dose_matrix=rand(m, n),
				voxel_labels=(3 * rand(m)).astype(int)
			)

		# let T = average value of target voxel rows of dose matrix
		# 	  N = average value of non-target rows,
		# then
		#	target_snr = T / N
		target_snr = 4.
		for i, label in enumerate(self.physics.voxel_labels):
			if label == 0:
				self.physics.dose_matrix[i, :] *= target_snr

		self.prescription = Prescription([
			{
				'name' : 'PTV',
				'label' : 0,
				'is_target' : True,
				'dose' : '1 Gy',
				'constraints' : ['D85 > 0.8 rx', 'D1 < 1.2 Gy']
			},{
				'name' : 'OAR_RESILIENT',
				'label' : 1,
				'is_target' : False,
				'dose' : None,
				'constraints' : ['D50 < 0.6 Gy', 'D2 < 0.8 Gy']
			}])

	def test_case_init(self):
		m, n = voxels, beams = 100, 50

		# bare initialization
		c0 = Case()
		self.assertTrue( isinstance(c0._Case__physics, Physics) )
		self.assertTrue( isinstance(c0.physics, Physics) )
		self.assertTrue( isinstance(c0._Case__anatomy, Anatomy) )
		self.assertTrue( isinstance(c0.anatomy, Anatomy) )
		self.assertTrue( isinstance(c0._Case__prescription, Prescription) )
		self.assertTrue( isinstance(c0.prescription, Prescription) )
		self.assertTrue( isinstance(c0._Case__problem, PlanningProblem) )
		self.assertTrue( isinstance(c0.problem, PlanningProblem) )
		self.assertTrue( isinstance(c0.structures, dict) )
		self.assertTrue( len(c0.structures) == 0 )
		self.assertTrue( c0.A is None )
		self.assertTrue( c0.n_structures == 0 )
		self.assertTrue( c0.n_voxels is None )
		self.assertTrue( c0.n_beams is None )
		self.assertFalse( c0.physics.plannable )
		self.assertFalse( c0.anatomy.plannable )
		self.assertFalse( c0.plannable )
		self.assertTrue( isinstance(c0.plotting_data(), dict) )
		self.assertTrue( len(c0.plotting_data()) == 0 )

		# anatomy & physics-based intialization
		a = Anatomy()
		a += Structure(0, 'organ at risk', False)
		p = Physics(voxels, beams)

		c1 = Case(anatomy=a, physics=p)
		self.assertTrue( c1.n_structures == 1 )
		self.assertTrue( c1.n_voxels == voxels )
		self.assertTrue( c1.n_beams == beams )
		self.assertFalse( c1.physics.plannable )
		self.assertFalse( c1.anatomy.plannable )
		self.assertFalse( c1.plannable )

		# at least one target makes anatomy plannable (but needs dose matrix)
		c1.anatomy += Structure(1, 'target', True)
		self.assertTrue( c1.n_structures == 2 )
		self.assertFalse( c1.physics.plannable )
		self.assertFalse( c1.anatomy.plannable )
		self.assertFalse( c1.plannable )

		# dose matrix...
		p.dose_matrix = rand(voxels, beams)
		self.assertTrue( c1.A is not None )
		self.assertTrue( c1.A.shape == (voxels, beams) )
		self.assertFalse( c1.physics.plannable )
		self.assertFalse( c1.anatomy.plannable )
		self.assertFalse( c1.plannable )

		# ...and voxel labels makes physics plannable, rendering
		# the case object plannable upon data transfer from physics to
		# anatomy (invoked by checking property "plannable")
		p.voxel_labels = (2 * rand(m)).astype(int)
		self.assertTrue( c1.physics.plannable )
		self.assertFalse( c1.anatomy.plannable )
		self.assertTrue( c1.plannable )
		self.assertTrue( c1.anatomy.plannable )

		# inline version of anatomy & physics-based initialization
		c2 = Case(
				Anatomy([
						Structure(0, 'oar', False),
						Structure(1, 'tumor', True)
				]),
				Physics(
						dose_matrix=rand(voxels, beams),
						voxel_labels=(2 * rand(voxels)).astype(int)
				))

		self.assertTrue( c2.n_structures == 2 )
		self.assertTrue( c2.n_voxels == voxels )
		self.assertTrue( c2.n_beams == beams )
		self.assertTrue( c2.A is not None )
		self.assertTrue( c2.A.shape == (voxels, beams) )
		self.assertTrue( c2.physics.plannable )
		self.assertFalse( c2.anatomy.plannable )
		self.assertTrue( c2.plannable )
		self.assertTrue( c2.anatomy.plannable )

	def test_rx_to_anatomy(self):
		c = Case()
		self.assertTrue( c.n_structures == 0 )
		c.prescription = [{
			'name' : 'PTV',
			'label' : 1,
			'is_target' : True,
			'dose' : '1 Gy',
			'constraints' : ['D85 > 0.8 rx', 'D1 < 1.2 Gy']
		},{
			'name' : 'OAR_RESILIENT',
			'label' : 4,
			'is_target' : False,
			'dose' : None,
			'constraints' : ['D50 < 0.6 Gy', 'D2 < 0.8 Gy']
		}]
		self.assertTrue( c.n_structures == 2 )
		self.assertTrue( c.anatomy['PTV'] == c.anatomy[1] )
		self.assertTrue( c.anatomy['OAR_RESILIENT'] == c.anatomy[4] )
		self.assertTrue( c.anatomy[1].constraints.size == 0 )
		self.assertTrue( c.anatomy[4].constraints.size == 0 )
		c.transfer_rx_constraints_to_anatomy()
		self.assertTrue( c.anatomy[1].constraints.size == 2 )
		self.assertTrue( c.anatomy[1].constraints.contains(D(85) > 0.8 * Gy) )
		self.assertTrue( c.anatomy[1].constraints.contains(D(1) < 1.2 * Gy) )
		self.assertTrue( c.anatomy[4].constraints.size == 2 )
		self.assertTrue( c.anatomy[4].constraints.contains(D(50) < 0.6 * Gy) )
		self.assertTrue( c.anatomy[4].constraints.contains(D(2) < 0.8 * Gy) )

	def test_case_init_prescription(self):
		# prescription-based initialization (python object)
		rx = [{
			'name' : 'PTV',
			'label' : 1,
			'is_target' : True,
			'dose' : '1 Gy',
			'constraints' : ['D85 > 0.8 rx', 'D1 < 1.2 Gy']
		},{
			'name' : 'OAR_RESILIENT',
			'label' : 4,
			'is_target' : False,
			'dose' : None,
			'constraints' : ['D50 < 0.6 Gy', 'D2 < 0.8 Gy']
		}]

		c = Case(prescription=Prescription(rx))
		self.assertFalse( c.physics.plannable )
		self.assertFalse( c.anatomy.plannable )
		self.assertFalse( c.plannable )
		self.assertTrue( c.anatomy[1].constraints.size == 2 )
		self.assertTrue( c.anatomy[1].constraints.contains(D(85) > 0.8 * Gy) )
		self.assertTrue( c.anatomy[1].constraints.contains(D(1) < 1.2 * Gy) )
		self.assertTrue( c.anatomy[4].constraints.size == 2 )
		self.assertTrue( c.anatomy[4].constraints.contains(D(50) < 0.6 * Gy) )
		self.assertTrue( c.anatomy[4].constraints.contains(D(2) < 0.8 * Gy) )

		# intialize from prescription as above, but suppress constraints
		# from being passed to anatomy
		c = Case(prescription=Prescription(rx), suppress_rx_constraints=True)
		self.assertFalse( c.physics.plannable )
		self.assertFalse( c.anatomy.plannable )
		self.assertFalse( c.plannable )
		self.assertTrue( c.anatomy[1].constraints.size == 0 )
		self.assertTrue( c.anatomy[4].constraints.size == 0 )

		# prescription-based initialization (python list)
		c = Case(prescription=rx)
		self.assertFalse( c.physics.plannable )
		self.assertFalse( c.anatomy.plannable )
		self.assertFalse( c.plannable )
		self.assertTrue( c.anatomy[1].constraints.size == 2 )
		self.assertTrue( c.anatomy[1].constraints.contains(D(85) > 0.8 * Gy) )
		self.assertTrue( c.anatomy[1].constraints.contains(D(1) < 1.2 * Gy) )
		self.assertTrue( c.anatomy[4].constraints.size == 2 )
		self.assertTrue( c.anatomy[4].constraints.contains(D(50) < 0.6 * Gy) )
		self.assertTrue( c.anatomy[4].constraints.contains(D(2) < 0.8 * Gy) )

		# prescription-based initialization (yaml/json file)
		curr_dir = path.abspath(path.dirname(__file__))
		c = Case(prescription=path.join(curr_dir, 'json_rx.json'))
		self.assertFalse( c.physics.plannable )
		self.assertFalse( c.anatomy.plannable )
		self.assertFalse( c.plannable )

		c = Case(prescription=path.join(curr_dir, 'yaml_rx.yml'))
		self.assertFalse( c.physics.plannable )
		self.assertFalse( c.anatomy.plannable )
		self.assertFalse( c.plannable )

	def test_constraint_manipulation(self):
		case = Case(self.anatomy)

		# test add and remove
		cid = case.add_constraint('PTV', D('mean') > 30 * Gy)
		self.assertTrue(
				(D('mean') > 30 * Gy) in case.anatomy['PTV'].constraints.list )
		case.drop_constraint(cid)
		self.assertFalse(
				(D('mean') > 30 * Gy) in case.anatomy['PTV'].constraints.list )

		# test change relop
		cid = case.add_constraint('PTV', D('mean') > 30 * Gy)
		case.change_constraint(cid, 'mean', '<', 30 * Gy)
		self.assertFalse(
				(D('mean') > 30 * Gy) in case.anatomy['PTV'].constraints.list )
		self.assertTrue(
				(D('mean') < 30 * Gy) in case.anatomy['PTV'].constraints.list )

		# test change dose
		case.change_constraint(cid, 'mean', '<', 50 * Gy)
		self.assertFalse(
				(D('mean') < 30 * Gy) in case.anatomy['PTV'].constraints.list )
		self.assertTrue(
				(D('mean') < 50 * Gy) in case.anatomy['PTV'].constraints.list )

		# test change threshold
		case.change_constraint(cid, 'max', '<', 50 * Gy)
		self.assertTrue(
				(D('mean') < 50 * Gy) in case.anatomy['PTV'].constraints.list )
		self.assertFalse(
				(D('max') < 50 * Gy) in case.anatomy['PTV'].constraints.list )

		# test clear
		case.clear_constraints()
		self.assertTrue( all([s.constraints.size == 0] for s in case.anatomy) )

	def test_objective_manipulation(self):
		case = Case(self.anatomy)

		# manipulate target objective
		w_new = 2.
		w_old_raw = case.anatomy['PTV'].w_over_raw
		w_old_problem = case.anatomy['PTV'].w_over
		case.change_objective('PTV', w_over=w_new)
		w_new_problem = case.anatomy['PTV'].w_over

		self.assert_scalar_equal(
				w_new / w_old_raw, w_new_problem / w_old_problem )

		w_new = 2.
		w_old_raw = case.anatomy['PTV'].w_under_raw
		w_old_problem = case.anatomy['PTV'].w_under
		case.change_objective('PTV', w_under=w_new)
		w_new_problem = case.anatomy['PTV'].w_under

		self.assert_scalar_equal(
				w_new / w_old_raw, w_new_problem / w_old_problem )

		dose_new = 10. * Gy
		case.change_objective('PTV', dose=dose_new)
		self.assertTrue( case.anatomy['PTV'].dose == dose_new )

		# manipulate non-target objective
		w_new = 2.
		w_old_raw = case.anatomy['OAR1'].w_over_raw
		w_old_problem = case.anatomy['OAR1'].w_over
		case.change_objective('OAR1', w_over=w_new)
		w_new_problem = case.anatomy['OAR1'].w_over

		self.assert_scalar_equal(
				w_new / w_old_raw, w_new_problem / w_old_problem )

		# weight_underdose does not change for non-target
		w_new = 2.
		self.assertTrue( case.anatomy['OAR1'].w_under_raw == 0 )
		self.assertTrue( case.anatomy['OAR1'].w_under == 0 )
		case.change_objective('OAR1', w_under=w_new)
		self.assertTrue( case.anatomy['OAR1'].w_under == 0 )

		# dose does not change for non-target
		dose_new = 10. * Gy
		self.assertTrue( case.anatomy['OAR1'].dose.value == 0 )
		case.change_objective('OAR1', dose=dose_new)
		self.assertTrue( case.anatomy['OAR1'].dose.value == 0 )


	def test_physics_to_anatomy(self):
		case = Case(self.anatomy, self.physics)

		self.assertFalse( case.physics.data_loaded )
		case.load_physics_to_anatomy()
		self.assertTrue( case.physics.data_loaded )

		for structure in case.anatomy:
			A = case.physics.dose_matrix_by_label(structure.label)
			vw = case.physics.voxel_weights_by_label(structure.label)
			self.assert_vector_equal( structure.A, A )
			self.assert_vector_equal( structure.voxel_weights, vw )

		self.assert_exception( call=case.load_physics_to_anatomy, args=[] )

	def test_calculate_doses(self):
		case = Case(self.anatomy, self.physics)
		case.load_physics_to_anatomy()
		x = rand(case.n_beams)
		case.calculate_doses(x)

		for structure in case.anatomy:
			self.assert_vector_equal( structure.y, structure.A.dot(x) )

	def test_plotting_data(self):
		c = Case(self.anatomy, self.physics)
		plot_data = c.plotting_data()
		self.assertTrue( isinstance(plot_data, dict)  )
		self.assertTrue( len(plot_data) == c.n_structures )
		self.assertTrue( all(s.label in plot_data for s in c.anatomy) )

	def test_plan(self):
		# Exception if case unplannable
		case = Case()
		self.assert_exception( call=case.plan, args=[] )

		# Construct and solve unconstrained case
		case = Case(self.anatomy, self.physics)

		success, run0 = case.plan(verbose=0)
		self.assertTrue( success )
		self.assertTrue( isinstance(run0, RunRecord) )

		# Add DVH constraints and solve
		case.prescription = self.prescription
		case.transfer_rx_constraints_to_anatomy()

		for slack in [False, True]:
			for exact in [False, True]:
				success, run = case.plan(
						use_slack=slack, use_2pass=exact, verbose=0)
				self.assertTrue( success )
				self.assertTrue( isinstance(run, RunRecord) )
				self.assertTrue( isinstance(run.plotting_data, dict) )
				self.assertTrue( 0 in run.plotting_data )
				if exact:
					self.assertTrue( 'exact' in run.plotting_data )
