"""
Unit tests for :mod:`conrad.visualization.plot.collections`
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

from conrad.physics.units import Gy
from conrad.medicine import D, Anatomy, Structure
from conrad.case import Case
from conrad.visualization.plot.collections import *
from conrad.tests.base import *

if not PLOTTING_INSTALLED:
	print('skipping tests for conrad.visualization.plot.collections ')
else:
	class StructureConstraintsGraphTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.PTV = Structure(0, 'PTV', True)
			self.PTV.constraints += D(30) < 50 * Gy
			self.PTV.constraints += D(80) > 45 * Gy

		def test_structure_constraints_graph_init_properties(self):
			# initialization
			pd = self.PTV.plotting_data(constraints_only=True)
			scg = StructureConstraintsGraph(pd)

			# test constraints, color, maxdose
			self.assertEqual( len(scg.constraints), 2 )

			self.assertEqual( scg.color, LineAesthetic().scale_rgb('black') )
			self.assertEqual( scg.maxdose, 50.0 )

			# alternate initialization
			scg = StructureConstraintsGraph(self.PTV)
			self.assertEqual( len(scg.constraints), 2 )

		def test_structure_constraints_graph_iterator(self):
			scg = StructureConstraintsGraph(self.PTV)
			for constraint_graph in scg:
				self.assertIsInstance(
						constraint_graph, PercentileConstraintGraph )

		def test_structure_constraints_graph_draw_undraw(self):
			scg = StructureConstraintsGraph(self.PTV)
			ax = mpl.figure.Figure().add_subplot(1, 1, 1)

			# test draw
			scg.draw(ax, markersize=12)
			for constr in scg:
				self.assertIs( constr.axes, ax )
				self.assertTrue( all(map(
						lambda g: g in ax.lines, constr.graph)) )

			# test hide/show
			scg.hide()
			for constr in scg:
				self.assertFalse( any(map(
						lambda g: g.get_visible(), constr.graph)) )

			scg.show()
			for constr in scg:
				self.assertTrue( all(map(
						lambda g: g.get_visible(), constr.graph)) )

			# test undraw
			scg.undraw()
			self.assertTrue( constr.axes is None )
			self.assertTrue( all(map(
					lambda g: g not in ax.lines, constr.graph)) )

	class StructureDVHGraphTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.beams = 50
			self.anatomy = Anatomy([
					Structure(0, 'PTV', True, A=rand(100, self.beams)),
					Structure(1, 'OAR1', False, A=rand(300, self.beams)),
					Structure(2, 'OAR2', False, A=rand(250, self.beams))
				])

		def setUp(self):
			self.anatomy['PTV'].calculate_dose(rand(self.beams))
			self.anatomy['PTV'].constraints += D(90) > 10 * Gy

		def test_structure_dvh_graph_init_properties(self):
			pd = self.anatomy['PTV'].plotting_data()
			sdvh = StructureDVHGraph(pd, 'blue')

			# test name, curve, constraints and rx properties
			self.assertEqual( sdvh.name, 'PTV' )
			self.assertIsInstance(sdvh.curve, DoseVolumeGraph )
			for c in sdvh.constraints:
				self.assertIsInstance(c, PercentileConstraintGraph )
			self.assertIsInstance(sdvh.rx, PrescriptionGraph )

			# rx should be none for non-target
			pd2 = self.anatomy['OAR1'].plotting_data()
			sdvh2 = StructureDVHGraph(pd2, 'red')
			self.assertIsNone( sdvh2.rx )

			# test color, aesthetic, maxdose properties
			self.assertEqual( sdvh.color, sdvh.aesthetic.scale_rgb('blue') )
			color = sdvh.aesthetic.scale_rgb('red')
			sdvh.color = color
			self.assertEqual( sdvh.curve.color, color )
			self.assertEqual( sdvh.rx.color, color )
			self.assertEqual( sdvh.curve.color, color )
			for constr in sdvh.constraints:
				self.assertEqual( constr.color, color )

			# test alternate initialization syntax
			sdvh_alternate_init = StructureDVHGraph(self.anatomy['PTV'])
			self.assertIsInstance( sdvh_alternate_init, StructureDVHGraph )

		@staticmethod
		def gather_graphs(structure_dvh, curve=True, noncurve=True):
			graphs = []
			if curve:
				graphs += structure_dvh.curve.graph
			if noncurve:
				if structure_dvh.rx is not None:
					graphs += structure_dvh.rx.graph
				for constr in structure_dvh.constraints:
					graphs += constr.graph
			return graphs

		def test_structure_dvh_graph_draw_undraw(self):
			sdvh = StructureDVHGraph(self.anatomy['PTV'], 'blue')
			ax = mpl.figure.Figure().add_subplot(1, 1, 1)

			# test draw
			sdvh.draw(ax)
			self.assertIs( sdvh.curve.axes, ax )
			self.assertIn( sdvh.curve.graph[0], ax.lines )
			self.assertIs( sdvh.rx.axes, ax )
			self.assertIn( sdvh.rx.graph[0], ax.lines )
			for constr in sdvh.constraints:
				self.assertIs( constr.axes, ax )
				self.assertTrue(
						all(map(lambda g: g in ax.lines, constr.graph)) )

			# test undraw
			sdvh.undraw()
			self.assertIsNone( sdvh.curve.axes )
			self.assertNotIn( sdvh.curve, ax.lines )
			self.assertIsNone( sdvh.rx.axes )
			self.assertNotIn( sdvh.rx, ax.lines )
			for constr in sdvh.constraints:
				self.assertIsNone( constr.axes )
				self.assertTrue(
						all(map(lambda g: g not in ax.lines, constr.graph)) )

			# test draw options
			la = LineAesthetic(style=':', alpha=0.7)
			sdvh.draw(ax, curve=True, rx=False, constraints=False,
					  aesthetic=la)
			self.assertIs( sdvh.curve.axes, ax )
			self.assertIn( sdvh.curve.graph[0], ax.lines )
			self.assertIsNone( sdvh.rx.axes )
			self.assertNotIn( sdvh.rx.graph[0], ax.lines )
			for constr in sdvh.constraints:
				self.assertIsNone( constr.axes )
				self.assertTrue(
						all(map(lambda g: g not in ax.lines, constr.graph)) )
			self.assertEqual( sdvh.curve.aesthetic, la )

			sdvh.draw(ax, curve=False, rx=True, constraints=False)
			self.assertIsNone( sdvh.curve.axes, None )
			self.assertNotIn( sdvh.curve.graph[0], ax.lines )
			self.assertIs( sdvh.rx.axes, ax )
			self.assertIn( sdvh.rx.graph[0], ax.lines )
			for constr in sdvh.constraints:
				self.assertIsNone( constr.axes )
				self.assertTrue(
						all(map(lambda g: g not in ax.lines, constr.graph)) )

			sdvh.draw(ax, curve=False, rx=False, constraints=True,
					  constraint_markersize=12)
			self.assertIsNone( sdvh.curve.axes )
			self.assertNotIn( sdvh.curve.graph[0], ax.lines )
			self.assertIsNone( sdvh.rx.axes )
			self.assertNotIn( sdvh.rx.graph[0], ax.lines )
			for constr in sdvh.constraints:
				self.assertIs( constr.axes, ax )
				self.assertTrue(
						all(map(lambda g: g in ax.lines, constr.graph)) )
				self.assertEqual( constr.aesthetic.markersize, 12 )

			# test hide/show
			sdvh.hide()
			graphs = self.gather_graphs(sdvh)
			self.assertFalse( any(map(lambda g: g.get_visible(), graphs)) )

			sdvh.show()
			graphs = self.gather_graphs(sdvh)
			self.assertTrue( all(map(lambda g: g.get_visible(), graphs)) )

			sdvh.show_curve_only()
			g_visible = self.gather_graphs(sdvh, noncurve=False)
			self.assertTrue( all(map(lambda g: g.get_visible(), g_visible)) )

			g_invisible = self.gather_graphs(sdvh, curve=False)
			self.assertFalse( any(map(lambda g: g.get_visible(), g_invisible)) )

	class PlanGraphBaseTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.test_data = {'KEY1':'DATA1', 'KEY2':'DATA2', 'KEY3':'DATA3'}
			self.test_colors = {'KEY1':'red', 'KEY2':'blue', 'KEY3':'orange'}

		def test_plan_graph_base_init(self):
			# initialize with data dictionary
			pgb = PlanGraphBase(self.test_data)
			for label in self.test_data:
				self.assertIn( label, pgb.structure_labels )
				self.assertIn( label, pgb.structure_colors )
				self.assertIs( pgb.structure_colors[label], None )

			# initialize with data and color dictionaries
			pgb = PlanGraphBase(self.test_data, self.test_colors)
			for label in self.test_data:
				self.assertIn( label, pgb.structure_labels )
				self.assertIn( label, pgb.structure_colors )
				self.assertEqual(
						pgb.structure_colors[label], self.test_colors[label] )

		def test_plan_graph_base_accessor_iterator(self):
			pgb = PlanGraphBase(self.test_data)
			for key, datum in self.test_data.items():
				pgb._PlanGraphBase__structure_graphs[key] = datum

			# test operator []
			for key in self.test_data:
				self.assertEqual( pgb[key], self.test_data[key] )

			# test iterator
			counter = 0
			test_type = type(self.test_data['KEY1'])
			for _, structure_datum in pgb:
				self.assertIsInstance(structure_datum, test_type )
				counter += 1
			self.assertEqual( counter, len(self.test_data) )

		def test_plan_graph_base_colors(self):
			pgb = PlanGraphBase(self.test_data)

			# color setter fails
			with self.assertRaises(AttributeError):
				pgb.structure_colors = self.test_colors

			# color setter succeeds
			class Foo(object):
				def __init__(self):
					self.color = 'black'

			for key in self.test_data:
				pgb._PlanGraphBase__structure_graphs[key] = Foo()
			pgb.structure_colors = self.test_colors

		def test_plan_graph_base_maxdose(self):
			pgb = PlanGraphBase(self.test_data)

			# maxdose
			for key, datum in self.test_data.items():
				pgb._PlanGraphBase__structure_graphs[key] = datum
			self.assertEqual(
					3, pgb._PlanGraphBase__maxdose(
							lambda s: int(s.lstrip('DATA'))) )

	class PlanDVHGraphTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.beams = 50
			self.anatomy = Anatomy([
					Structure(0, 'PTV', True, A=rand(100, self.beams)),
					Structure(1, 'OAR1', False, A=rand(300, self.beams)),
					Structure(2, 'OAR2', False, A=rand(250, self.beams))
				])
			self.anatomy['PTV'].constraints += D(80) > 40 * Gy
			self.anatomy['PTV'].constraints += D(20) < 45 * Gy

		def test_plandvh_graph_init(self):
			# explicit initialization
			pd = self.anatomy.plotting_data()
			pdg = PlanDVHGraph(pd)
			self.assertIsInstance(pdg.structure_DVHs, dict )
			for structure in self.anatomy:
				self.assertIn( structure.label, pdg.structure_DVHs )
				self.assertIn( structure.label, pdg.structure_labels )
				self.assertIn( structure.label, pdg.structure_colors )
				self.assertIsInstance(
						pdg[structure.label], StructureDVHGraph )
			for _, s_dvh in pdg:
				self.assertIsInstance(s_dvh, StructureDVHGraph )

			# implicit initialization
			pdg = PlanDVHGraph(self.anatomy)
			for structure in self.anatomy:
				self.assertIsInstance( pdg[structure.label], StructureDVHGraph )
			for _, s_dvh in pdg:
				self.assertIsInstance(s_dvh, StructureDVHGraph )

		def test_plandvh_graph_maxdose(self):
			self.anatomy.calculate_doses(rand(self.beams))
			pd = self.anatomy.plotting_data()
			dmax = dmaxdc = 0
			for label in pd:
				sdata = pd[label]
				dmax = max(dmax, sdata['curve']['dose'][-1])
				dmaxdc = max(dmaxdc, dmax)
				for _, constr in sdata['constraints']:
					dmaxdc = max(dmaxdc, max(constr['dose']))

			pdg = PlanDVHGraph(pd)
			self.assertEqual( dmaxdc, pdg.maxdose() )
			self.assertEqual( dmax, pdg.maxdose(exclude_constraints=True) )

	class PlanConstraintsGraphTestCase(ConradTestCase):
		@classmethod
		def setUpClass(self):
			self.beams = 50
			self.anatomy = Anatomy([
					Structure(0, 'PTV', True, A=rand(100, self.beams)),
					Structure(1, 'OAR1', False, A=rand(300, self.beams)),
					Structure(2, 'OAR2', False, A=rand(250, self.beams))
				])
			self.anatomy['PTV'].constraints += D(80) > 40 * Gy
			self.anatomy['PTV'].constraints += D(20) < 45 * Gy

		def test_plan_constraints_graph_init(self):
			# # explicit initialization
			pd = self.anatomy.plotting_data(constraints_only=True)
			pcg = PlanConstraintsGraph(pd)
			self.assertIsInstance(pcg.structure_constraints, dict )
			for structure in self.anatomy:
				self.assertIn( structure.label, pcg.structure_constraints )
				self.assertIn( structure.label, pcg.structure_labels )
				self.assertIn( structure.label, pcg.structure_colors )
				self.assertIsInstance(pcg[structure.label],
								  	 		StructureConstraintsGraph )
			for _, s_constr in pcg:
				self.assertIsInstance(s_constr,
											StructureConstraintsGraph )

			# implicit initialization
			pcg = PlanConstraintsGraph(self.anatomy)
			for structure in self.anatomy:
				self.assertIsInstance(pcg[structure.label],
								  	 		StructureConstraintsGraph )
			for _, s_constr in pcg:
				self.assertIsInstance(s_constr,
											StructureConstraintsGraph )

		def test_plan_constraints_graph_maxdose(self):
			self.anatomy.calculate_doses(rand(self.beams))
			pd = self.anatomy.plotting_data(constraints_only=True)
			dmax = 0
			for label in pd:
				for _, constr in pd[label]:
					dmax = max(dmax, max(constr['dose']))

			pcg = PlanConstraintsGraph(pd)
			self.assertEqual( dmax, pcg.maxdose )
