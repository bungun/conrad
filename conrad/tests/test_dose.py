import numpy as np
import scipy.sparse as sp

from conrad.compat import *
from conrad.defs import CONRAD_DEBUG_PRINT
from conrad.physics.units import DeliveredDose, Gy, cGy, percent, mm3, cm3
from conrad.medicine.dose import *
from conrad.tests.base import *

class ConstraintTestCase(ConradTestCase):
	def test_relops(self):
		""" test contents of __ConstraintRelops object """
		self.assertTrue( RELOPS.GEQ == '>=' )
		self.assertTrue( RELOPS.LEQ == '<=' )
		self.assertTrue( RELOPS.INDEFINITE == '<>' )

	def test_generic_constraint_init(self):
		""" test Constraint object intialization and properties """
		c = Constraint()
		self.assertFalse( c.resolved )
		self.assert_nan( c.dose )
		self.assertTrue( c.relop == RELOPS.INDEFINITE )
		self.assertTrue( c.threshold == None )

		# expect exception, since constraint direction indefinite
		try:
			result = c.upper
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		self.assertTrue( isinstance(c.rx_dose, DeliveredDose) )
		self.assert_nan( c.rx_dose.value )

		# expect exception, since dual_value is nan
		try:
			result = c.active
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		self.assert_nan( c.dual_value )
		self.assertTrue( c.slack == 0. )

		# expect exception, since constraint direction indefinite
		try:
			result = c.dose_achieved
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		self.assertTrue( c.priority == 1 )
		self.assertTrue( c.symbol == '<>' )

	def test_generic_constraint_setters(self):
		""" test Constraint object setters """
		c = Constraint()
		c.dose = 30 * Gy
		self.assertTrue( c.dose.value == 30.)
		c.dose = 30 * cGy
		self.assertTrue( c.dose.to_Gy.value == 0.3 )
		c.dose = 40 * percent
		self.assert_nan( c.dose.value )
		c.rx_dose = 50 * Gy
		self.assertTrue( c.rx_dose.value == 50 )
		self.assertTrue( c.dose.value == 0.4 * 50 )

		# trigger exceptions with unitless input
		try:
			c.dose = 12.
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			c.dose = 'string'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			c.rx_dose = 50
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			c.rx_dose = 'string'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		c.relop = '>'
		self.assertTrue( c.relop == RELOPS.GEQ )
		self.assertFalse( c.upper )
		c.relop = '>='
		self.assertTrue( c.relop == RELOPS.GEQ )
		self.assertFalse( c.upper )
		c.relop = '<'
		self.assertTrue( c.relop == RELOPS.LEQ )
		self.assertTrue( c.upper )
		c.relop = '<='
		self.assertTrue( c.relop == RELOPS.LEQ )
		self.assertTrue( c.upper )

		c.dual_value = 0.
		self.assertTrue( c.dual_value == 0 )
		self.assertFalse( c.active )
		c.dual_value = np.random.rand()
		self.assertTrue( c.dual_value > 0 )
		self.assertTrue( c.active )


		c.slack = 12.
		self.assertTrue( c.slack == 12 )
		c.relop = '<'
		self.assertTrue( c.dose_achieved.value == c.dose.value + 12 )
		c.relop = '>'
		self.assertTrue( c.dose_achieved.value == c.dose.value - 12 )

		# trigger exceptions with bad input
		try:
			c.slack = -23
			self.assertTrue( False )
		except:
			self.assertTrue( True )
		try:
			c.slack = 'random string'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		for i in xrange(4):
			c.priority = i
			self.assertTrue( c.priority == i )

		try:
			c.priority = -1
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			c.priority = 4
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		try:
			c.priority = 'random string'
			self.assertTrue( False )
		except:
			self.assertTrue( True )

	def test_generic_constraint_operators(self):
		""" test Constraint object overloaded operators """
		c = Constraint()

		c > 30 * Gy
		self.assertTrue( c.relop == RELOPS.GEQ )
		self.assertTrue( c.dose.value == 30. )
		self.assertTrue( isinstance(c.dose, type(Gy)) )
		c < 20 * Gy
		self.assertTrue( c.relop == RELOPS.LEQ )
		self.assertTrue( c.dose.value == 20. )
		self.assertTrue( isinstance(c.dose, type(Gy)) )
		c >= 32 * Gy
		self.assertTrue( c.relop == RELOPS.GEQ )
		self.assertTrue( c.dose.value == 32. )
		self.assertTrue( isinstance(c.dose, type(Gy)) )
		c <= 25 * Gy
		self.assertTrue( c.relop == RELOPS.LEQ )
		self.assertTrue( c.dose.value == 25. )
		self.assertTrue( isinstance(c.dose, type(Gy)) )

		# defer test of Constraint.__eq__() method to DTestCase below

	def test_generic_constraint_print(self):
		""" test Constraint object __str__() function """
		c = Constraint()
		self.assertTrue( str(c) == 'DNone <> nan' )
		c > 30 * Gy
		self.assertTrue( str(c) == 'DNone >= 30.0 Gy' )
		c < 25 * Gy
		self.assertTrue( str(c) == 'DNone <= 25.0 Gy' )

class PercentileConstraintTestCase(ConradTestCase):
	def test_percentile_constraint(self):
		pc = PercentileConstraint()
		self.assertTrue( pc.relop == RELOPS.INDEFINITE )
		self.assert_nan( pc.dose )
		self.assertTrue( pc.percentile is None )
		pc.percentile = 12
		self.assertTrue( isinstance(pc.percentile, type(percent)) )
		self.assertTrue( pc.percentile.value == 12 )
		pc.percentile = 43 * percent
		self.assertTrue( pc.percentile.value == 43 )
		pc > 22. * Gy
		self.assertTrue( pc.relop == RELOPS.GEQ )
		self.assertTrue( pc.dose.value == 22. )

		pc = PercentileConstraint(percentile=32., relop='<', dose=12 * Gy)
		self.assertTrue( pc.percentile.value == 32 )
		self.assertTrue( pc.relop == RELOPS.LEQ )
		self.assertTrue( pc.dose.value == 12 )

		d = pc.plotting_data
		self.assertTrue( d['type'] == 'percentile' )
		self.assertTrue( d['percentile'][0] == pc.percentile.value )
		self.assertTrue( d['percentile'][1] == pc.percentile.value )
		self.assertTrue( d['dose'][0] == pc.dose.value )
		self.assertTrue( d['dose'][1] == pc.dose_achieved.value )
		self.assertTrue( d['symbol'] == '<' )

	def test_maxmargin_fulfillers(self):
		pc = PercentileConstraint()
		pc.percentile = 12
		pc <= 5 * Gy
		pc.slack = 3

		self.assertTrue( pc.dose.value == 5 )
		self.assertTrue( pc.dose_achieved.value == 8 )

		dose_vec = 10 * np.random.rand(200)

		# without slack, upper limit
		n_fulfilled = int(np.ceil((1 - pc.percentile.fraction) * len(dose_vec)))
		maxmargin_fulfillers = pc.get_maxmargin_fulfillers(dose_vec)
		self.assertTrue( len(maxmargin_fulfillers) == n_fulfilled )
		dose_sub = dose_vec[maxmargin_fulfillers]

		# confirm sorted
		self.assertTrue( sum(np.diff(dose_sub.argsort()) != 1) == 0 )

		# without slack, lower limit
		pc >= 5 * Gy
		n_fulfilled = int(np.ceil(pc.percentile.fraction * len(dose_vec)))
		maxmargin_fulfillers = pc.get_maxmargin_fulfillers(dose_vec)
		self.assertTrue( len(maxmargin_fulfillers) == n_fulfilled )
		dose_sub = dose_vec[maxmargin_fulfillers]

		# confirm sorted
		self.assertTrue( sum(np.diff(dose_sub.argsort()) != 1) == 0 )

		# with slack, upper limit
		maxmargin_fulfillers_slack = pc.get_maxmargin_fulfillers(
				dose_vec, had_slack=True)
		self.assertTrue( len(maxmargin_fulfillers_slack) == n_fulfilled )
		dose_sub_slack = dose_vec[maxmargin_fulfillers_slack]

		# confirm sorted
		self.assertTrue( sum(np.diff(dose_sub_slack.argsort()) != 1) == 0 )

# class AbsoluteVolumeConstraintTestCase(ConradTestCase):
# 	def test_absolutevolumeconstraint(self):
# 		avc = AbsoluteVolumeConstraint()
# 		self.assertTrue( avc.volume.value is nan )
# 		self.assertTrue( avc.total_volume.value is nan )
# 		self.assertTrue( avc.relop == RELOPS.INDEFINITE )
# 		self.assertTrue( avc.dose is nan )

# 		avc.volume = 500 * mm3
# 		self.assertTrue( avc.volume.value == 500 )
# 		avc <= 1 * Gy

# 		# conversion to percentile constraint should fail when total volume
# 		# not set
# 		try:
# 			pc_failed = avc.to_percentile_constraint
# 			self.assertTrue( False )
# 		except:
# 			self.assertTrue( True )

# 		avc.total_volume = 40 * cm3

# 		# constraint resolution status is exception
# 		try:
# 			status = avc.resolved
# 			self.assertTrue( False )
# 		except:
# 			self.assertTrue( True )

# 		pc = avc.to_percentile_constraint
# 		self.assertTrue( pc.resolved )

# 		# exception: constrained volume / total volume > 1
# 		avc.total_volume = 100 * mm3
# 		try:
# 			pc_failed = avc.to_percentile_constraint
# 			self.assertTrue( False )
# 		except:
# 			self.assertTrue( True )

# 		# exception: constrained volume / total volume == 0
# 		avc.volume = 0 * mm3
# 		try:
# 			pc_failed = avc.to_percentile_constraint
# 			self.assertTrue( False )
# 		except:
# 			self.assertTrue( True )

class MeanConstraintTestCase(ConradTestCase):
	def test_mean_constraint(self):
		mc = MeanConstraint()
		mc <= 5 * Gy
		self.assertTrue( mc.relop == RELOPS.LEQ )
		self.assertTrue( mc.dose.value == 5 )

		d = mc.plotting_data
		self.assertTrue( d['type'] == 'mean' )
		self.assertTrue( d['dose'][0] == mc.dose.value )
		self.assertTrue( d['dose'][1] == mc.dose_achieved.value )
		self.assertTrue( d['symbol'] == '<' )

class MaxConstraintTestCase(ConradTestCase):
	def test_max_constraint(self):
		maxc = MaxConstraint()
		maxc <= 5 * Gy
		self.assertTrue( maxc.relop == RELOPS.LEQ )
		self.assertTrue( maxc.dose.value == 5 )

		d = maxc.plotting_data
		self.assertTrue( d['type'] == 'max' )
		self.assertTrue( d['dose'][0] == maxc.dose.value )
		self.assertTrue( d['dose'][1] == maxc.dose_achieved.value )
		self.assertTrue( d['symbol'] == '<' )

		# test exception handling (Dmax > X not allowed)
		try:
			maxc >= 3 * Gy
			self.assertTrue( False )
		except:
			self.assertTrue( True )

class MinConstraintTestCase(ConradTestCase):
	def test_min_constraint(self):
		minc = MinConstraint()
		minc >= 2 * Gy
		self.assertTrue( minc.relop == RELOPS.GEQ )
		self.assertTrue( minc.dose.value == 2 )

		d = minc.plotting_data
		self.assertTrue( d['type'] == 'min' )
		self.assertTrue( d['dose'][0] == minc.dose.value )
		self.assertTrue( d['dose'][1] == minc.dose_achieved.value )
		self.assertTrue( d['symbol'] == '>' )

		# test exception handling (Dmax > X not allowed)
		try:
			minc <= 1 * Gy
			self.assertTrue( False )
		except:
			self.assertTrue( True )

class DTestCase(ConradTestCase):
	def test_D(self):
		""" test function D() """

		# percentile
		# - valid dose
		c = D(80) < 12 * Gy
		self.assertTrue( isinstance(c, PercentileConstraint) )
		self.assertTrue( c.dose.value == 12 )
		self.assertTrue( isinstance(c.threshold, Percent) )
		self.assertTrue( c.threshold.value == 80 )
		self.assertTrue( c.relop == RELOPS.LEQ )

		c = D(80) > 12 * Gy
		self.assertTrue( isinstance(c, PercentileConstraint) )
		self.assertTrue( c.dose.value == 12 )
		self.assertTrue( isinstance(c.threshold, Percent) )
		self.assertTrue( c.threshold.value == 80 )
		self.assertTrue( c.relop == RELOPS.GEQ )

		c = D(80) < 95 * percent
		self.assertTrue( isinstance(c, PercentileConstraint) )
		self.assert_nan( c.dose.value )
		self.assertTrue( isinstance(c.threshold, Percent) )
		self.assertTrue( c.threshold.value == 80 )
		self.assertTrue( c.relop == RELOPS.LEQ )
		c.rx_dose = 10 * Gy
		self.assertTrue( c.dose.value == 0.95 * 10 )

		c = D(80) > 95 * percent
		self.assertTrue( isinstance(c, PercentileConstraint) )
		self.assert_nan( c.dose.value )
		self.assertTrue( isinstance(c.threshold, Percent) )
		self.assertTrue( c.threshold.value == 80 )
		self.assertTrue( c.relop == RELOPS.GEQ )
		c.rx_dose = 10 * Gy
		self.assertTrue( c.dose.value == 0.95 * 10 )

		c = D(80.2) < 12 * Gy
		self.assertTrue( isinstance(c, PercentileConstraint) )
		self.assertTrue( c.dose.value == 12 )
		self.assertTrue( isinstance(c.threshold, Percent) )
		self.assertTrue( c.threshold.value == 80.2 )
		self.assertTrue( c.relop == RELOPS.LEQ )

		# - invalid dose
		try:
			c = D(80) < 12
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		# mean
		# - valid dose
		c = D('mean') < 12 * Gy
		self.assertTrue( isinstance(c, MeanConstraint) )
		self.assertTrue( c.dose.value == 12 )
		self.assertTrue( c.threshold == 'mean' )
		self.assertTrue( c.relop == RELOPS.LEQ )

		c = D('mean') > 12 * Gy
		self.assertTrue( isinstance(c, MeanConstraint) )
		self.assertTrue( c.dose.value == 12 )
		self.assertTrue( c.threshold == 'mean' )
		self.assertTrue( c.relop == RELOPS.GEQ )

		c = D('mean') < 95 * percent
		self.assertTrue( isinstance(c, MeanConstraint) )
		self.assert_nan( c.dose.value )
		self.assertTrue( c.threshold == 'mean' )
		self.assertTrue( c.relop == RELOPS.LEQ )
		c.rx_dose = 10 * Gy
		self.assertTrue( c.dose.value == 0.95 * 10 )


		c = D('mean') > 95 * percent
		self.assertTrue( isinstance(c, MeanConstraint) )
		self.assert_nan( c.dose.value )
		self.assertTrue( c.threshold == 'mean' )
		self.assertTrue( c.relop == RELOPS.GEQ )
		c.rx_dose = 10 * Gy
		self.assertTrue( c.dose.value == 0.95 * 10 )

		# - invalid dose
		try:
			c = D('mean') > 12
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		# min
		# - valid dose, relop
		c = D('min') > 10 * Gy
		self.assertTrue( isinstance(c, MinConstraint) )
		self.assertTrue( c.dose.value == 10 )
		self.assertTrue( c.threshold == 'min' )
		self.assertTrue( c.relop == RELOPS.GEQ )

		c = D('min') > 90 * percent
		self.assertTrue( isinstance(c, MinConstraint) )
		self.assert_nan( c.dose.value )
		self.assertTrue( c.threshold == 'min' )
		self.assertTrue( c.relop == RELOPS.GEQ )
		c.rx_dose = 10 * Gy
		self.assertTrue( c.dose.value == 0.9 * 10 )

		# - invalid dose
		try:
			c = D('min') > 90
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		# - invalid relop
		try:
			c = D('min') < 90 * percent
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		# max
		# - valid dose, relop
		c = D('max') < 32 * Gy
		self.assertTrue( isinstance(c, MaxConstraint) )
		self.assertTrue( c.dose.value == 32 )
		self.assertTrue( c.threshold == 'max' )
		self.assertTrue( c.relop == RELOPS.LEQ )

		c = D('max') < 110 * percent
		self.assertTrue( isinstance(c, MaxConstraint) )
		self.assert_nan( c.dose.value )
		self.assertTrue( c.threshold == 'max' )
		self.assertTrue( c.relop == RELOPS.LEQ )
		c.rx_dose = 10 * Gy
		self.assertTrue( c.dose.value == 1.1 * 10 )

		# - invalid dose
		try:
			c = D('max') < 32
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		# - invalid relop
		try:
			c = D('max') > 32 * Gy
			self.assertTrue( False )
		except:
			self.assertTrue( True )

		# test Constraint.__eq__() method here, deferred from
		# ConstraintTestCase above
		self.assertEqual( D(80) >= 30 * Gy, D(80.0) > 30. * Gy )
		self.assertEqual( D('mean') >= 30 * Gy, D('mean') > 30. * Gy )
		self.assertEqual( D('min') >= 30 * Gy, D('min') > 30. * Gy )
		self.assertEqual( D('max') <= 30 * Gy, D('max') < 30. * Gy )

# class VTestCase(ConradTestCase):
# 	pass

class ConstraintListTestCase(ConradTestCase):
	def test_constraint_list(self):
		""" test ConstraintList object initialization, methods and
			properties
		"""
		cl = ConstraintList()
		self.assertTrue( isinstance(cl.items, dict) )
		self.assertTrue( cl.last_key is None )
		self.assertTrue( cl.size == 0 )
		self.assertTrue( cl.mean_only )

		# test constraint addition
		cl += D('mean') > 0.2 * Gy
		self.assertTrue( cl.size > 0 )
		last_key = cl.last_key
		self.assertTrue( last_key in cl )
		self.assertTrue( isinstance(cl[cl.last_key], Constraint) )
		self.assertTrue( isinstance(cl[cl.last_key], MeanConstraint) )

		# test constraint removal by key
		cl -= last_key
		self.assertTrue( cl.size == 0 )

		# test constraint removal by info (in general won't work since
		#  constraint properties are mutable)
		cl += D('mean') > 0.2 * Gy
		self.assertTrue( cl.size > 0 )
		last_key_2 = cl.last_key
		cl -= D('mean') > 0.2 * Gy
		self.assertTrue( cl.size == 0 )

		# test that constraints are mean only
		self.assertTrue( cl.mean_only )
		cl += D('min') > 0.1 * Gy
		self.assertTrue( isinstance(cl[cl.last_key], Constraint) )
		self.assertTrue( isinstance(cl[cl.last_key], MinConstraint) )
		self.assertFalse( cl.mean_only )
		cl -= cl.last_key
		self.assertTrue( cl.mean_only )
		cl += D('max') < 1 * Gy
		self.assertTrue( isinstance(cl[cl.last_key], Constraint) )
		self.assertTrue( isinstance(cl[cl.last_key], MaxConstraint) )
		self.assertFalse( cl.mean_only )
		cl -= cl.last_key
		self.assertTrue( cl.mean_only )
		cl += D(80) < 1 * Gy
		self.assertTrue( isinstance(cl[cl.last_key], Constraint) )
		self.assertTrue( isinstance(cl[cl.last_key], PercentileConstraint) )
		self.assertFalse( cl.mean_only )

		cl += D('min') > 0.1 * Gy
		cl += D('max') < 1 * Gy
		cl += D(80) < 1 * Gy
		self.assertTrue( cl.size == 4 )
		cl.clear()
		self.assertTrue( cl.size == 0 )

	def test_plotting_data(self):
		""" test ConstraintList object property plotting_data """
		cl = ConstraintList()
		cl += D('min') > 0.1 * Gy
		cl += D('max') < 1 * Gy
		cl += D(80) < 1 * Gy
		self.assertTrue( len(cl.plotting_data) == 3 )
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
		self.assertTrue( sum(dvh.data) == 0 )
		self.assertTrue( dvh.min_dose == 0 )
		self.assertTrue( dvh.max_dose == 0 )
		self.assertTrue( len(dvh._DVH__dose_buffer) == m )
		self.assertTrue( len(dvh._DVH__doses) <= dvh.MAX_LENGTH + 1 )

		m = 2 * dvh.MAX_LENGTH
		dvh = DVH(m)
		self.assertTrue( len(dvh._DVH__dose_buffer) == m )
		self.assertTrue( len(dvh._DVH__doses) <= dvh.MAX_LENGTH + 1 )

		m = 2 * dvh.MAX_LENGTH
		dvh = DVH(m, maxlength=m + 1)
		self.assertTrue( len(dvh._DVH__dose_buffer) == m )
		self.assertTrue( len(dvh._DVH__doses) > dvh.MAX_LENGTH + 1 )

		dvh = DVH(1000, maxlength=1000)
	 	self.assertTrue( dvh._DVH__stride == 1 )
		dvh = DVH(2000, maxlength=1000)
		self.assertTrue( dvh._DVH__stride == 2 )
		dvh = DVH(3000, maxlength=1000)
		self.assertTrue( dvh._DVH__stride == 3 )

	def test_data(self):
		""" test DVH object property data """
		m = 3000
		y = np.random.rand(m)
		dvh = DVH(m)
		dvh.data = y
		self.assertTrue( sum(dvh.data) != 0 )

		# test sorting
		self.assertTrue( sum(np.diff(dvh.data.argsort()) != 1) == 0 )

		# test min, max
		self.assertTrue( dvh.min_dose == y.min() )
		self.assertTrue( dvh.max_dose == y.max() )

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
		self.assertTrue( alpha_identity == 1 )

		try:
			alpha_fail = dvh._DVH__interpolate_percentile(
					second, second, interp)
			self.assertTrue( False )
		except:
			self.assertTrue( True )

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

		self.assertTrue( dvh.dose_at_percentile(100) == y.min() )
		self.assertTrue( dvh.dose_at_percentile(0) == y.max() )

		percentile = 60.2
		idx_lower = int(0.01 * (100 - percentile - 0.5) * m)
		idx_upper = int(0.01 * (100 - percentile + 0.5) * m)
		dose_lower = y_sort[idx_lower]
		dose_upper = y_sort[idx_upper]

		dose_retrieved = dvh.dose_at_percentile(percentile)
		self.assertTrue( dose_lower <= dose_retrieved)
		self.assertTrue( dose_retrieved <= dose_upper)

		# percentile resolution > 0.5%, interpolation required
		m = 60
		y = np.random.rand(m)
		y_sort = np.zeros(m)
		y_sort += y
		y_sort.sort()
		dvh = DVH(m)
		dvh.data = y

		self.assertTrue( dvh.dose_at_percentile(100) == y.min() )
		self.assertTrue( dvh.dose_at_percentile(0) == y.max() )

		percentile = 60.2
		idx_lower = int(0.01 * (100 - percentile - 0.5) * m)
		idx_upper = int(0.01 * (100 - percentile + 0.5) * m)
		if idx_upper == idx_lower:
			idx_upper += 1
		dose_lower = y_sort[idx_lower]
		dose_upper = y_sort[idx_upper]

		dose_retrieved = dvh.dose_at_percentile(percentile)
		self.assertTrue( dose_lower <= dose_retrieved)
		self.assertTrue( dose_retrieved <= dose_upper)

	def test_plotting_data(self):
		""" test DVH object property plotting_data """
		m = 2500
		y = np.random.rand(m)
		dvh = DVH(m)
		dvh.data = y

		pd = dvh.plotting_data
		self.assertTrue( isinstance(pd, dict) )
		self.assertTrue( 'percentile' in pd )
		self.assertTrue( 'dose' in pd )
		self.assertTrue( isinstance(pd['percentile'], np.ndarray) )
		self.assertTrue( isinstance(pd['dose'], np.ndarray) )
		self.assertTrue( len(pd['dose']) == len(pd['percentile']) )

		# test sorting
		self.assertTrue( sum(np.diff(pd['dose'].argsort()) != 1) == 0)
		self.assertTrue( sum(np.diff(pd['percentile'].argsort()) != -1) == 0)

		# test values
		self.assertTrue( pd['dose'][0] == 0 )
		self.assertTrue( pd['percentile'][0] == 100 )
		self.assertTrue( pd['percentile'][-1] == 0 )
