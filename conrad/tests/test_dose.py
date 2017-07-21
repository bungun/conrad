"""
Unit tests for :mod:`conrad.medicine.dose`.
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
import scipy.sparse as sp

from conrad.defs import CONRAD_DEBUG_PRINT
from conrad.physics.units import DeliveredDose, Gy, cGy, percent, mm3, cm3
from conrad.medicine.dose import *
from conrad.tests.base import *

class ConstraintTestCase(ConradTestCase):
	def test_relops(self):
		""" test contents of __ConstraintRelops object """
		self.assertEqual( RELOPS.GEQ, '>=' )
		self.assertEqual( RELOPS.LEQ, '<=' )
		self.assertEqual( RELOPS.INDEFINITE, '<>' )

	def test_generic_constraint_init(self):
		""" test Constraint object intialization and properties """
		c = Constraint()
		self.assertFalse( c.resolved )
		self.assert_nan( c.dose.value )
		self.assertEqual( c.relop, RELOPS.INDEFINITE )
		self.assertIsNone( c.threshold )

		# expect exception, since constraint direction indefinite
		with self.assertRaises(ValueError):
			c.upper

		self.assertIsInstance( c.rx_dose, DeliveredDose )
		self.assert_nan( c.rx_dose.value )

		self.assert_nan( c.dual_value )
		self.assertEqual( c.slack, 0. )

		# expect exception, since constraint direction indefinite
		with self.assertRaises(ValueError):
			c.dose_achieved

		self.assertEqual( c.priority, 1 )
		self.assertEqual( c.symbol, '<>' )

	def test_generic_constraint_setters(self):
		""" test Constraint object setters """
		c = Constraint()
		c.dose = 30 * Gy
		self.assertEqual( c.dose.value, 30.)
		c.dose = 30 * cGy
		self.assertEqual( c.dose.to_Gy.value, 0.3 )
		c.dose = 40 * percent
		self.assert_nan( c.dose.value )
		c.rx_dose = 50 * Gy
		self.assertEqual( c.rx_dose.value, 50 )
		self.assertEqual( c.dose.value, 0.4 * 50 )

		# trigger exceptions with unitless input
		with self.assertRaises(TypeError):
			c.dose = 12.
		with self.assertRaises(TypeError):
			c.dose = 'string'
		with self.assertRaises(TypeError):
			c.rx_dose = 50
		with self.assertRaises(TypeError):
			c.rx_dose = 'string'

		c.relop = '>'
		self.assertEqual( c.relop, RELOPS.GEQ )
		self.assertFalse( c.upper )
		c.relop = '>='
		self.assertEqual( c.relop, RELOPS.GEQ )
		self.assertFalse( c.upper )
		c.relop = '<'
		self.assertEqual( c.relop, RELOPS.LEQ )
		self.assertTrue( c.upper )
		c.relop = '<='
		self.assertEqual( c.relop, RELOPS.LEQ )
		self.assertTrue( c.upper )

		c.dual_value = 0.
		self.assertEqual( c.dual_value, 0 )
		self.assertFalse( c.active )
		c.dual_value = np.random.rand()
		self.assertGreater( c.dual_value, 0 )
		self.assertTrue( c.active )


		c.slack = 12.
		self.assertEqual( c.slack, 12 )
		c.relop = '<'
		self.assertEqual( c.dose_achieved.value, c.dose.value + 12 )
		c.relop = '>'
		self.assertEqual( c.dose_achieved.value, c.dose.value - 12 )

		# trigger exceptions with bad input
		with self.assertRaises(ValueError):
			c.slack = -23
		with self.assertRaises(TypeError):
			c.slack = 'random string'

		for i in xrange(4):
			c.priority = i
			self.assertEqual( c.priority, i )

		with self.assertRaises(ValueError):
			c.priority = -1
		with self.assertRaises(ValueError):
			c.priority = 4
		with self.assertRaises(TypeError):
			c.priority = 'random string'

	def test_generic_constraint_operators(self):
		""" test Constraint object overloaded operators """
		c = Constraint()

		c > 30 * Gy
		self.assertEqual( c.relop, RELOPS.GEQ )
		self.assertEqual( c.dose.value, 30. )
		self.assertIsInstance( c.dose, type(Gy) )
		c < 20 * Gy
		self.assertEqual( c.relop, RELOPS.LEQ )
		self.assertEqual( c.dose.value, 20. )
		self.assertIsInstance( c.dose, type(Gy) )
		c >= 32 * Gy
		self.assertEqual( c.relop, RELOPS.GEQ )
		self.assertEqual( c.dose.value, 32. )
		self.assertIsInstance( c.dose, type(Gy) )
		c <= 25 * Gy
		self.assertEqual( c.relop, RELOPS.LEQ )
		self.assertEqual( c.dose.value, 25. )
		self.assertIsInstance( c.dose, type(Gy) )

		# defer test of Constraint.__eq__() method to DTestCase below

	def test_generic_constraint_print(self):
		""" test Constraint object __str__() function """
		c = Constraint()
		self.assertEqual( str(c), 'DNone <> nan Gy' )
		c > 30 * Gy
		self.assertEqual( str(c), 'DNone >= 30.0 Gy' )
		c < 25 * Gy
		self.assertEqual( str(c), 'DNone <= 25.0 Gy' )

class PercentileConstraintTestCase(ConradTestCase):
	def test_percentile_constraint(self):
		pc = PercentileConstraint()
		self.assertEqual( pc.relop, RELOPS.INDEFINITE )
		self.assert_nan( pc.dose.value )
		self.assertIsNone( pc.percentile )
		pc.percentile = 12
		self.assertIsInstance( pc.percentile, type(percent) )
		self.assertEqual( pc.percentile.value, 12 )
		pc.percentile = 43 * percent
		self.assertEqual( pc.percentile.value, 43 )
		pc > 22. * Gy
		self.assertEqual( pc.relop, RELOPS.GEQ )
		self.assertEqual( pc.dose.value, 22. )

		pc = PercentileConstraint(percentile=32., relop='<', dose=12 * Gy)
		self.assertEqual( pc.percentile.value, 32 )
		self.assertEqual( pc.relop, RELOPS.LEQ )
		self.assertEqual( pc.dose.value, 12 )

		d = pc.plotting_data
		self.assertEqual( d['type'], 'percentile' )
		self.assertEqual( d['percentile'][0], pc.percentile.value )
		self.assertEqual( d['percentile'][1], pc.percentile.value )
		self.assertEqual( d['dose'][0], pc.dose.value )
		self.assertEqual( d['dose'][1], pc.dose_achieved.value )
		self.assertEqual( d['symbol'], '<' )

	def test_maxmargin_fulfillers(self):
		pc = PercentileConstraint()
		pc.percentile = 12
		pc <= 5 * Gy
		pc.slack = 3

		self.assertEqual( pc.dose.value, 5 )
		self.assertEqual( pc.dose_achieved.value, 8 )

		dose_vec = 10 * np.random.rand(200)

		# without slack, upper limit
		n_fulfilled = int(np.ceil((1 - pc.percentile.fraction) * len(dose_vec)))
		maxmargin_fulfillers = pc.get_maxmargin_fulfillers(dose_vec)
		self.assertEqual( len(maxmargin_fulfillers), n_fulfilled )
		dose_sub = dose_vec[maxmargin_fulfillers]

		# confirm sorted
		self.assertEqual( sum(np.diff(dose_sub.argsort()) != 1), 0 )

		# without slack, lower limit
		pc >= 5 * Gy
		n_fulfilled = int(np.ceil(pc.percentile.fraction * len(dose_vec)))
		maxmargin_fulfillers = pc.get_maxmargin_fulfillers(dose_vec)
		self.assertEqual( len(maxmargin_fulfillers), n_fulfilled )
		dose_sub = dose_vec[maxmargin_fulfillers]

		# confirm sorted
		self.assertEqual( sum(np.diff(dose_sub.argsort()) != 1), 0 )

		# with slack, upper limit
		maxmargin_fulfillers_slack = pc.get_maxmargin_fulfillers(
				dose_vec, had_slack=True)
		self.assertEqual( len(maxmargin_fulfillers_slack), n_fulfilled )
		dose_sub_slack = dose_vec[maxmargin_fulfillers_slack]

		# confirm sorted
		self.assertEqual( sum(np.diff(dose_sub_slack.argsort()) != 1), 0 )

class MeanConstraintTestCase(ConradTestCase):
	def test_mean_constraint(self):
		mc = MeanConstraint()
		mc <= 5 * Gy
		self.assertEqual( mc.relop, RELOPS.LEQ )
		self.assertEqual( mc.dose.value, 5 )

		d = mc.plotting_data
		self.assertEqual( d['type'], 'mean' )
		self.assertEqual( d['dose'][0], mc.dose.value )
		self.assertEqual( d['dose'][1], mc.dose_achieved.value )
		self.assertEqual( d['symbol'], '<' )

class MaxConstraintTestCase(ConradTestCase):
	def test_max_constraint(self):
		maxc = MaxConstraint()
		maxc <= 5 * Gy
		self.assertEqual( maxc.relop, RELOPS.LEQ )
		self.assertEqual( maxc.dose.value, 5 )

		d = maxc.plotting_data
		self.assertEqual( d['type'], 'max' )
		self.assertEqual( d['dose'][0], maxc.dose.value )
		self.assertEqual( d['dose'][1], maxc.dose_achieved.value )
		self.assertEqual( d['symbol'], '<' )

		# test exception handling (Dmax > X not allowed)
		with self.assertRaises(ValueError):
			maxc >= 3 * Gy

class MinConstraintTestCase(ConradTestCase):
	def test_min_constraint(self):
		minc = MinConstraint()
		minc >= 2 * Gy
		self.assertEqual( minc.relop, RELOPS.GEQ )
		self.assertEqual( minc.dose.value, 2 )

		d = minc.plotting_data
		self.assertEqual( d['type'], 'min' )
		self.assertEqual( d['dose'][0], minc.dose.value )
		self.assertEqual( d['dose'][1], minc.dose_achieved.value )
		self.assertEqual( d['symbol'], '>' )

		# test exception handling (Dmax > X not allowed)
		with self.assertRaises(ValueError):
			minc <= 1 * Gy

class DTestCase(ConradTestCase):
	def test_D(self):
		""" test function D() """

		# percentile
		# - valid dose
		c = D(80) < 12 * Gy
		self.assertIsInstance( c, PercentileConstraint )
		self.assertEqual( c.dose.value, 12 )
		self.assertIsInstance( c.threshold, Percent )
		self.assertEqual( c.threshold.value, 80 )
		self.assertEqual( c.relop, RELOPS.LEQ )

		c = D(80) > 12 * Gy
		self.assertIsInstance( c, PercentileConstraint )
		self.assertEqual( c.dose.value, 12 )
		self.assertIsInstance( c.threshold, Percent )
		self.assertEqual( c.threshold.value, 80 )
		self.assertEqual( c.relop, RELOPS.GEQ )

		c = D(80) < 95 * percent
		self.assertIsInstance( c, PercentileConstraint )
		self.assert_nan( c.dose.value )
		self.assertIsInstance( c.threshold, Percent )
		self.assertEqual( c.threshold.value, 80 )
		self.assertEqual( c.relop, RELOPS.LEQ )
		c.rx_dose = 10 * Gy
		self.assertEqual( c.dose.value, 0.95 * 10 )

		c = D(80) > 95 * percent
		self.assertIsInstance( c, PercentileConstraint )
		self.assert_nan( c.dose.value )
		self.assertIsInstance( c.threshold, Percent )
		self.assertEqual( c.threshold.value, 80 )
		self.assertEqual( c.relop, RELOPS.GEQ )
		c.rx_dose = 10 * Gy
		self.assertEqual( c.dose.value, 0.95 * 10 )

		c = D(80.2) < 12 * Gy
		self.assertIsInstance( c, PercentileConstraint )
		self.assertEqual( c.dose.value, 12 )
		self.assertIsInstance( c.threshold, Percent )
		self.assertEqual( c.threshold.value, 80.2 )
		self.assertEqual( c.relop, RELOPS.LEQ )

		# - invalid dose
		with self.assertRaises(TypeError):
			c = D(80) < 12

		# mean
		# - valid dose
		c = D('mean') < 12 * Gy
		self.assertIsInstance( c, MeanConstraint )
		self.assertEqual( c.dose.value, 12 )
		self.assertEqual( c.threshold, 'mean' )
		self.assertEqual( c.relop, RELOPS.LEQ )

		c = D('mean') > 12 * Gy
		self.assertIsInstance( c, MeanConstraint )
		self.assertEqual( c.dose.value, 12 )
		self.assertEqual( c.threshold, 'mean' )
		self.assertEqual( c.relop, RELOPS.GEQ )

		c = D('mean') < 95 * percent
		self.assertIsInstance( c, MeanConstraint )
		self.assert_nan( c.dose.value )
		self.assertEqual( c.threshold, 'mean' )
		self.assertEqual( c.relop, RELOPS.LEQ )
		c.rx_dose = 10 * Gy
		self.assertEqual( c.dose.value, 0.95 * 10 )


		c = D('mean') > 95 * percent
		self.assertIsInstance( c, MeanConstraint )
		self.assert_nan( c.dose.value )
		self.assertEqual( c.threshold, 'mean' )
		self.assertEqual( c.relop, RELOPS.GEQ )
		c.rx_dose = 10 * Gy
		self.assertEqual( c.dose.value, 0.95 * 10 )

		# - invalid dose
		with self.assertRaises(TypeError):
			c = D('mean') > 12

		# min
		# - valid dose, relop
		c = D('min') > 10 * Gy
		self.assertIsInstance( c, MinConstraint )
		self.assertEqual( c.dose.value, 10 )
		self.assertEqual( c.threshold, 'min' )
		self.assertEqual( c.relop, RELOPS.GEQ )

		c = D('min') > 90 * percent
		self.assertIsInstance( c, MinConstraint )
		self.assert_nan( c.dose.value )
		self.assertEqual( c.threshold, 'min' )
		self.assertEqual( c.relop, RELOPS.GEQ )
		c.rx_dose = 10 * Gy
		self.assertEqual( c.dose.value, 0.9 * 10 )

		# - invalid dose
		with self.assertRaises(TypeError):
			c = D('min') > 90

		# - invalid relop
		with self.assertRaises(ValueError):
			c = D('min') < 90 * percent

		# max
		# - valid dose, relop
		c = D('max') < 32 * Gy
		self.assertIsInstance( c, MaxConstraint )
		self.assertEqual( c.dose.value, 32 )
		self.assertEqual( c.threshold, 'max' )
		self.assertEqual( c.relop, RELOPS.LEQ )

		c = D('max') < 110 * percent
		self.assertIsInstance( c, MaxConstraint )
		self.assert_nan( c.dose.value )
		self.assertEqual( c.threshold, 'max' )
		self.assertEqual( c.relop, RELOPS.LEQ )
		c.rx_dose = 10 * Gy
		self.assertEqual( c.dose.value, 1.1 * 10 )

		# - invalid dose
		with self.assertRaises(TypeError):
			c = D('max') < 32

		# - invalid relop
		with self.assertRaises(ValueError):
			c = D('max') > 32 * Gy

		# test Constraint.__eq__() method here, deferred from
		# ConstraintTestCase above
		self.assertEqual( D(80) >= 30 * Gy, D(80.0) > 30. * Gy )
		self.assertEqual( D('mean') >= 30 * Gy, D('mean') > 30. * Gy )
		self.assertEqual( D('min') >= 30 * Gy, D('min') > 30. * Gy )
		self.assertEqual( D('max') <= 30 * Gy, D('max') < 30. * Gy )


class AbsoluteVolumeConstraintTestCase(ConradTestCase):
	def test_absolutevolumeconstraint(self):
		avc = AbsoluteVolumeConstraint()
		self.assertIsNone( avc.volume )
		self.assertIsNone( avc.total_volume )
		self.assertTrue( avc.relop == RELOPS.INDEFINITE )
		self.assert_nan( avc.dose.value )

		avc.volume = 500 * mm3
		self.assertTrue( avc.volume.value == 500 )

		avc > 550 * mm3
		self.assertTrue( avc.volume.value == 550 )
		self.assertTrue( avc.relop == RELOPS.GEQ )
		avc < 520 * mm3
		self.assertTrue( avc.volume == 520 * mm3 )
		self.assertTrue( avc.relop == RELOPS.LEQ )

		avc.dose = 1 * Gy

		# conversion to percentile constraint should fail when total volume
		# not set
		with self.assertRaises(ValueError):
			pc_failed = avc.to_percentile_constraint()

		avc.total_volume = 40 * cm3

		# constraint resolution status is exception
		with self.assertRaises(ValueError):
			status = avc.resolved

		pc = avc.to_percentile_constraint()
		self.assertTrue( pc.resolved )

		# exception: constrained volume / total volume > 1
		with self.assertRaises(ValueError):
			avc.total_volume = 100 * mm3
			pc_failed = avc.to_percentile_constraint()

		with self.assertRaises(ValueError):
			avc.volume = 0 * mm3
			pc_failed = avc.to_percentile_constraint()

		avc = AbsoluteVolumeConstraint(1 * Gy)
		self.assertTrue( avc.dose == 1 * Gy )


class GenericVolumeConstraintTestCase(ConradTestCase):
	def test_genericvolumeconstraint(self):
		gvc = GenericVolumeConstraint()
		self.assertIsNone( gvc.volume )
		self.assertTrue( gvc.relop == RELOPS.INDEFINITE )
		self.assert_nan( gvc.dose.value )

		self.assertTrue(
				isinstance(gvc.specialize(), GenericVolumeConstraint) )

		gvc.volume = 500 * mm3
		self.assertTrue( gvc.volume.value == 500 )
		self.assertIsNone( gvc.threshold )
		self.assertTrue(
				isinstance(gvc.specialize(), AbsoluteVolumeConstraint) )

		gvc = GenericVolumeConstraint()
		gvc.volume = 50 * Percent()
		self.assertIsNone( gvc.volume )
		self.assertTrue( gvc.threshold.value == 50 )
		self.assertTrue(
				isinstance(gvc.specialize(), PercentileConstraint) )

		avc = gvc < 5 * cm3
		self.assertTrue( isinstance(avc, AbsoluteVolumeConstraint) )
		self.assertTrue( avc.relop == RELOPS.LEQ )
		self.assertTrue( avc.volume == 5 * cm3 )

		avc = gvc > 5 * cm3
		self.assertTrue( isinstance(avc, AbsoluteVolumeConstraint) )
		self.assertTrue( avc.relop == RELOPS.GEQ )
		self.assertTrue( avc.volume == 5 * cm3 )

		pc = gvc < 30 * Percent()
		self.assertTrue( isinstance(pc, PercentileConstraint) )
		self.assertTrue( pc.relop == RELOPS.LEQ )
		self.assertTrue( pc.threshold.value == 30 )

		pc = gvc > 30 * Percent()
		self.assertTrue( isinstance(pc, PercentileConstraint) )
		self.assertTrue( pc.relop == RELOPS.GEQ )
		self.assertTrue( pc.threshold.value == 30 )

class VTestCase(ConradTestCase):
	def test_V(self):
		""" test function V() """

		c = V(80 * Gy) < 12 * percent
		self.assertIsInstance( c, PercentileConstraint )
		self.assertEqual( c.dose.value, 80 )
		self.assertIsInstance( c.threshold, Percent )
		self.assertEqual( c.threshold.value, 12 )
		self.assertEqual( c.relop, RELOPS.LEQ )

		c = V(80 * Gy) > 12 * cm3
		self.assertIsInstance( c, AbsoluteVolumeConstraint )
		self.assertEqual( c.dose.value, 80 )
		self.assertIsNone( c.threshold )
		self.assertIsInstance( c.volume, Volume )
		self.assertEqual( c.volume.value, 12 )
		self.assertEqual( c.relop, RELOPS.GEQ )

		c = V(80 * percent) < 15 * percent
		self.assertIsInstance( c, PercentileConstraint )
		self.assert_nan( c.dose.value )
		self.assertIsInstance( c.threshold, Percent )
		self.assertEqual( c.threshold.value, 15 )
		self.assertEqual( c.relop, RELOPS.LEQ )
		c.rx_dose = 10 * Gy
		self.assertEqual( c.dose.value, 0.8 * 10 )

		c = V(80 * percent) > 20 * cm3
		self.assertIsInstance( c, AbsoluteVolumeConstraint )
		self.assert_nan( c.dose.value )
		self.assertIsNone( c.threshold )
		self.assertEqual( c.volume, 20 * cm3 )
		self.assertEqual( c.relop, RELOPS.GEQ )
		c.rx_dose = 10 * Gy
		self.assertEqual( c.dose.value, 0.8 * 10 )

		with self.assertRaises(TypeError):
			c = V(80) < 12 * Percent()

		with self.assertRaises(TypeError):
			c = V(80 * Gy) < 12

class ConstraintParsingTestCase(ConradTestCase):
	def test_eval_constraint(self):
		# PARSABLE:
		# - "min > x Gy"
		# - "mean < x Gy"
		# - "max < x Gy"
		# - "D__% < x Gy"
		# - "D__% > x Gy"
		# - "V__ Gy < p %" ( == "x Gy to < p %")
		# - "V__ Gy > p %" ( == "x Gy to > p %")

		# - "min > x Gy"
		s_variants = [
			'min > 10 Gy', 'min >= 10 Gy', 'min > 10.0 Gy',
			'Min > 10 Gy', 'min >= 10 Gy', 'Min > 10.0 Gy',
			'min > 10 Gray', 'min > 1000 cGy'
			]
		c = D('min') > 10 * Gy
		for s in s_variants:
			self.assertTrue( eval_constraint(s) == c )

		# - "max < x Gy"
		s_variants = [
			'max < 10 Gy', 'max <= 10 Gy', 'max < 10.0 Gy',
			'Max < 10 Gy', 'max <= 10 Gy', 'Max < 10.0 Gy',
			'max < 10 Gray', 'max < 1000 cGy'
			]
		c = D('max') < 10 * Gy
		for s in s_variants:
			self.assertTrue( eval_constraint(s) == c )

		# - "mean > x Gy"
		s_variants = [
			'mean > 10 Gy', 'mean >= 10 Gy', 'mean > 10.0 Gy',
			'Mean > 10 Gy', 'mean >= 10 Gy', 'Mean > 10.0 Gy',
			'mean > 10 Gray', 'mean > 1000 cGy'
			]
		c = D('mean') > 10 * Gy
		for s in s_variants:
			self.assertTrue( eval_constraint(s) == c )

		# - "mean < x Gy"
		s_variants = [
			'mean < 10 Gy', 'mean <= 10 Gy', 'mean < 10.0 Gy',
			'Mean < 10 Gy', 'mean <= 10 Gy', 'Mean < 10.0 Gy',
			'mean < 10 Gray', 'mean < 1000 cGy'
			]
		c = D('mean') < 10 * Gy
		for s in s_variants:
			self.assertTrue( eval_constraint(s) == c )

		# - "D__% < x Gy"
		s_variants = [
			'D20 < 10 Gy', 'D20 <= 10 Gy', 'D20 < 10.0 Gy',
			'D20% < 10 Gy', 'D20 <= 10 Gy', 'D20% < 10.0 Gy',
			'D20.0 < 10 Gy', 'D20 < 10 Gray', 'D20 < 1000 cGy'
			]
		c = D(20) < 10 * Gy
		for s in s_variants:
			self.assertTrue( eval_constraint(s) == c )

		# - "D__% > x Gy"
		s_variants = [
			'D80 > 10 Gy', 'D80 >= 10 Gy', 'D80 > 10.0 Gy',
			'D80% > 10 Gy', 'D80 >= 10 Gy', 'D80% > 10.0 Gy',
			'D80.0 > 10 Gy', 'D80 > 10 Gray', 'D80 > 1000 cGy'
			]
		c = D(80) > 10 * Gy
		for s in s_variants:
			self.assertTrue( eval_constraint(s) == c )

		# - "V__ Gy < p %" ( == "x Gy to < p %")
		s_variants = [
			'V10 Gy < 20%', 'V10 Gy < 20 %', 'V10 Gy < 20.0%',
			'V10.0 Gy < 20%', '10 Gy to < 20%'
			]
		c = V(10 * Gy) < 20 * percent
		cd = D(20) < 10 * Gy
		for s in s_variants:
			self.assertTrue( eval_constraint(s) == c )
			self.assertTrue( eval_constraint(s) == cd )

		# - "V__ Gy > p %" ( == "x Gy to > p %")
		s_variants = [
			'V10 Gy > 20%', 'V10 Gy > 20 %', 'V10 Gy > 20.0%',
			'V10 Gy >= 20', 'V10.0 Gy > 20%', '10 Gy to > 20%',
			'10 Gy to >= 20%'
			]
		c = V(10 * Gy) > 20 * percent
		cd = D(20) > 10 * Gy
		for s in s_variants:
			self.assertTrue( eval_constraint(s) == c )
			self.assertTrue( eval_constraint(s) == cd )

		# PARSABLE WITH RX_DOSE PROVIDED:
		rx_dose = 10 * Gy
		rx_dose_fail = 100

		# - "D__% < {frac} rx"
		s_variants = [
			'D20 < 1.05 rx', 'D20.0 < 1.05rx', 'D20% < 1.05rx',
			'D20.0% < 105% rx', 'D20 < 105% rx', 'D20 < 105.0% rx',
			'D20 <1.05rx', 'D20 <= 1.05rx', 'D20<=1.05rx'
			]
		c = D(20) < 1.05 * rx_dose
		for s in s_variants:
			self.assertEqual( eval_constraint(s, rx_dose=rx_dose), c )
			with self.assertRaises(ValueError):
				c1 = eval_constraint(s)
			with self.assertRaises(TypeError):
				c1 = eval_constraint(s, rx_dose=rx_dose_fail)

		# - "D__% > {frac} rx"
		s_variants = [
			'D80 > 0.95 rx', 'D80.0 > 0.95rx', 'D80% > 0.95rx',
			'D80.0% > 95% rx', 'D80 > 95% rx', 'D80 > 95.0% rx'
			]
		c = D(80) > 0.95 * rx_dose
		for s in s_variants:
			self.assertEqual( eval_constraint(s, rx_dose=rx_dose), c )
			with self.assertRaises(ValueError):
				c1 = eval_constraint(s)
			with self.assertRaises(TypeError):
				c1 = eval_constraint(s, rx_dose=rx_dose_fail)

		# - "V__ rx < p %"
		s_variants = [
			'V 0.1 rx < 80%', 'V 10% rx < 80%', 'V10.0% rx < 80%',
			'V 0.1 rx < 80.0%'
			]
		c = V(0.1 * rx_dose) < 80 * percent
		for s in s_variants:
			self.assertEqual( eval_constraint(s, rx_dose=rx_dose), c )
			with self.assertRaises(ValueError):
				c1 = eval_constraint(s)
			with self.assertRaises(TypeError):
				c1 = eval_constraint(s, rx_dose=rx_dose_fail)

		# - "V__ rx > p %"
		s_variants = [
			'V 0.9 rx > 80%', 'V 90% rx > 80%', 'V90.0% rx > 80%',
			'V 0.9 rx > 80.0%'
			]
		c = V(0.9 * rx_dose) > 80 * percent
		for s in s_variants:
			self.assertEqual( eval_constraint(s, rx_dose=rx_dose), c )
			with self.assertRaises(ValueError):
				c1 = eval_constraint(s)
			with self.assertRaises(TypeError):
				c1 = eval_constraint(s, rx_dose=rx_dose_fail)

		# PARSABLE, BUT CONSTRAINT.RESOLVED FAILURE WITHOUT TOTAL VOLUME:
		# - "V_ Gy > x cm3"
		# - "V_ Gy < x cm3"
		# - "V_ rx > x cm3"
		# - "V_ rx < x cm3"\

		s_variants = [
				'V20Gy > 10 cm3', 'V20Gy > 10.0 cm3', 'V20Gy > 10 CM3',
				'V20Gy > 10 cc', 'V20Gy > 10 CC']
		c = V(20 * Gy) > 50 * percent
		i = 0
		for relop in ('>', '<'):
			print("ITER", i)
			c.relop = relop
			for s in s_variants:
				s = s.replace('>', relop)
				c_eval = eval_constraint(s)
				with self.assertRaises(ValueError):
					c_eval.resolved
				cp = c_eval.to_percentile_constraint(20 * cm3)
				cp.resolved
				self.assertEqual( cp, c )

		s_variants = [
			'V 0.9rx > 10 cm3', 'V 0.9rx > 10.0 cm3', 'V90% rx > 10cm3',
			'V 90.0% rx > 10 CM3','V 0.9rx > 10 CM3',
			'V 0.9rx > 10 CM3', 'V 0.9rx > 10 cc', 'V 0.9rx > 10 CC']
		c = V(0.9 * rx_dose) > 10 * cm3

		for relop in ('>', '<'):
			c.relop = relop
			for s in s_variants:
				s = s.replace('>', relop)
				c_eval = eval_constraint(s, rx_dose=rx_dose)
				self.assertEqual( c_eval, c )
				with self.assertRaises(ValueError):
					c_eval.resolved
				with self.assertRaises(ValueError):
					c1 = eval_constraint(s)
				with self.assertRaises(TypeError):
					c1 = eval_constraint(s, rx_dose=rx_dose_fail)
				cp = c_eval.to_percentile_constraint(20 * cm3)
				cp.resolved
				self.assertIsInstance( cp.threshold, Percent )
				self.assertEqual( cp.threshold.value, 50. )

class ConstraintListTestCase(ConradTestCase):
	def test_constraint_list(self):
		""" test ConstraintList object initialization, methods and
			properties
		"""
		cl = ConstraintList()
		self.assertIsInstance( cl.items, dict )
		self.assertIsNone( cl.last_key )
		self.assertEqual( cl.size, 0 )
		self.assertTrue( cl.mean_only )

		# test constraint addition
		cl += D('mean') > 0.2 * Gy
		self.assertGreater( cl.size, 0 )
		last_key = cl.last_key
		self.assertIn( last_key, cl )
		self.assertIsInstance( cl[cl.last_key], Constraint )
		self.assertIsInstance( cl[cl.last_key], MeanConstraint )

		# test constraint removal by key
		cl -= last_key
		self.assertEqual( cl.size, 0 )

		# test constraint removal by info (in general won't work since
		#  constraint properties are mutable)
		cl += D('mean') > 0.2 * Gy
		self.assertGreater( cl.size, 0 )
		last_key_2 = cl.last_key
		cl -= D('mean') > 0.2 * Gy
		self.assertEqual( cl.size, 0 )

		# test that constraints are mean only
		self.assertTrue( cl.mean_only )
		cl += D('min') > 0.1 * Gy
		self.assertIsInstance( cl[cl.last_key], Constraint )
		self.assertIsInstance( cl[cl.last_key], MinConstraint )
		self.assertFalse( cl.mean_only )
		cl -= cl.last_key
		self.assertTrue( cl.mean_only )
		cl += D('max') < 1 * Gy
		self.assertIsInstance( cl[cl.last_key], Constraint )
		self.assertIsInstance( cl[cl.last_key], MaxConstraint )
		self.assertFalse( cl.mean_only )
		cl -= cl.last_key
		self.assertTrue( cl.mean_only )
		cl += D(80) < 1 * Gy
		self.assertIsInstance( cl[cl.last_key], Constraint )
		self.assertIsInstance( cl[cl.last_key], PercentileConstraint )
		self.assertFalse( cl.mean_only )

		cl.clear()
		self.assertEqual( cl.size, 0 )
		cl += D('min') > 0.1 * Gy
		cl += D('max') < 1 * Gy
		cl += D(80) < 1 * Gy

		def run_assertions(c_list):
			self.assertIn( D('min') > 0.1 * Gy, c_list )
			self.assertIn( D('max') < 1 * Gy, c_list )
			self.assertIn( D(80) < 1 * Gy, c_list )
			self.assertEqual( c_list.size, 3 )

		run_assertions(cl)

		# alternate initializations:
		cl2 = ConstraintList(cl)
		run_assertions(cl2)
		cl3 = ConstraintList(['Dmin > 0.1Gy', 'Dmax < 1 Gy', 'D80 < 1 Gy'])
		run_assertions(cl3)
		cl4 = ConstraintList('["Dmin > 0.1Gy", "Dmax < 1 Gy", "D80 < 1 Gy"]')
		run_assertions(cl4)

	def test_plotting_data(self):
		""" test ConstraintList object property plotting_data """
		cl = ConstraintList()
		cl += D('min') > 0.1 * Gy
		cl += D('max') < 1 * Gy
		cl += D(80) < 1 * Gy
		self.assertEqual( len(cl.plotting_data), 3 )
		self.assertTrue( all(listmap(
				lambda x: isinstance(x, tuple), cl.plotting_data)) )
		self.assertTrue( all(listmap(
				lambda x: len(x) == 2, cl.plotting_data)) )
		self.assertTrue( all(listmap(
				lambda x: isinstance(x[0], str), cl.plotting_data)) )
		self.assertTrue( all(listmap(
				lambda x: len(x[0]) == 7, cl.plotting_data)) )
		self.assertTrue( all(listmap(
				lambda x: isinstance(x[1], dict), cl.plotting_data)) )

class DVHTestCase(ConradTestCase):
	def test_init(self):
		""" test DVH object init """
		m = 500
		dvh = DVH(m)
		self.assertFalse( dvh.populated )
		self.assertEqual( sum(dvh.data), 0 )
		self.assertEqual( dvh.min_dose, 0 )
		self.assertEqual( dvh.max_dose, 0 )
		self.assertEqual( len(dvh._DVH__dose_buffer), m )
		self.assertLessEqual( len(dvh._DVH__doses), dvh.MAX_LENGTH + 1 )

		m = 2 * dvh.MAX_LENGTH
		dvh = DVH(m)
		self.assertEqual( len(dvh._DVH__dose_buffer), m )
		self.assertLessEqual( len(dvh._DVH__doses), dvh.MAX_LENGTH + 1 )

		m = 2 * dvh.MAX_LENGTH
		dvh = DVH(m, maxlength=m + 1)
		self.assertEqual( len(dvh._DVH__dose_buffer), m )
		self.assertGreater( len(dvh._DVH__doses), dvh.MAX_LENGTH + 1 )

		dvh = DVH(1000, maxlength=1000)
		self.assertEqual( dvh._DVH__stride, 1 )
		dvh = DVH(2000, maxlength=1000)
		self.assertEqual( dvh._DVH__stride, 2 )
		dvh = DVH(3000, maxlength=1000)
		self.assertEqual( dvh._DVH__stride, 3 )

	def test_data(self):
		""" test DVH object property data """
		m = 3000
		y = np.random.rand(m)
		dvh = DVH(m)
		dvh.data = y
		self.assertNotEqual( sum(dvh.data), 0 )

		# test sorting
		self.assertEqual( sum(np.diff(dvh.data.argsort()) != 1), 0 )

		# test min, max
		self.assertEqual( dvh.min_dose, y.min() )
		self.assertEqual( dvh.max_dose, y.max() )

	def test_interpolate_percentile(self):
		""" test DVH object static method interpolate_percentile """
		dvh = DVH(100)
		first = 1
		second = 5
		interp = first + (second - first) * np.random.rand()
		alpha = dvh._DVH__interpolate_percentile(first, second, interp)

		self.assert_scalar_equal(alpha * first + (1 - alpha) * second, interp,
								 1e-7, 1e-7)

		alpha_identity = dvh._DVH__interpolate_percentile(
				second, second, second)
		self.assertEqual( alpha_identity, 1 )

		with self.assertRaises(ValueError):
			alpha_fail = dvh._DVH__interpolate_percentile(
					second, second, interp)

	def test_dose_at_percentile(self):
		""" test DVH object method dose_at_percentile """

		# percentile resolution = 0.1%
		m = 3000
		y = np.random.rand(m)
		y_sort = np.zeros(m)
		y_sort += y
		y_sort.sort()
		dvh = DVH(m)
		dvh.data = y

		self.assertEqual( dvh.dose_at_percentile(100), y.min() )
		self.assertEqual( dvh.dose_at_percentile(0), y.max() )

		percentile = 60.2
		idx_lower = int(0.01 * (100 - percentile - 0.5) * m)
		idx_upper = int(0.01 * (100 - percentile + 0.5) * m)
		dose_lower = y_sort[idx_lower]
		dose_upper = y_sort[idx_upper]

		dose_retrieved = dvh.dose_at_percentile(percentile)
		self.assertLessEqual( dose_lower, dose_retrieved)
		self.assertLessEqual( dose_retrieved, dose_upper)

		# percentile resolution > 0.5%, interpolation required
		m = 60
		y = np.random.rand(m)
		y_sort = np.zeros(m)
		y_sort += y
		y_sort.sort()
		dvh = DVH(m)
		dvh.data = y

		self.assertEqual( dvh.dose_at_percentile(100), y.min() )
		self.assertEqual( dvh.dose_at_percentile(0), y.max() )

		percentile = 60.2
		idx_lower = int(0.01 * (100 - percentile - 0.5) * m)
		idx_upper = int(0.01 * (100 - percentile + 0.5) * m)
		if idx_upper == idx_lower:
			idx_upper += 1
		dose_lower = y_sort[idx_lower]
		dose_upper = y_sort[idx_upper]

		dose_retrieved = dvh.dose_at_percentile(percentile)
		self.assertLessEqual( dose_lower, dose_retrieved)
		self.assertLessEqual( dose_retrieved, dose_upper)

	def test_plotting_data(self):
		""" test DVH object property plotting_data """
		m = 2500
		y = np.random.rand(m)
		dvh = DVH(m)
		dvh.data = y

		pd = dvh.plotting_data
		self.assertIsInstance( pd, dict )
		self.assertIn( 'percentile', pd )
		self.assertIn( 'dose', pd )
		self.assertIsInstance( pd['percentile'], np.ndarray )
		self.assertIsInstance( pd['dose'], np.ndarray )
		self.assertEqual( len(pd['dose']), len(pd['percentile']) )

		# test sorting
		self.assertEqual( sum(np.diff(pd['dose'].argsort()) != 1), 0 )
		self.assertEqual( sum(np.diff(pd['percentile'].argsort()) != -1), 0 )

		# test values
		self.assertEqual( pd['dose'][0], 0 )
		self.assertEqual( pd['percentile'][0], 100 )
		self.assertEqual( pd['percentile'][-1], 0 )

		# test re-sampling
		maxlength = 20
		dvh_r = dvh.resample(maxlength)
		pd = dvh_r.plotting_data
		self.assertLessEqual( len(pd['dose']), maxlength + 2 )
