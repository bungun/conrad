from conrad.compat import *
from conrad.physics.string import *
from conrad.tests.base import *

class TestPhysics(ConradTestCase):
	def test_percent_parse(self):
		input_ = '7 %'
		output = strip_percent_units(input_)
		f_output = float_value_from_percent_string(input_)
		value = percent_from_string(input_)
		self.assertTrue( output == '7 ' )
		self.assertTrue( f_output == 7. )
		self.assertTrue( isinstance(value, Percent) )
		self.assertTrue( value.value == 7 )

		input_ = '7 percent'
		output = strip_percent_units(input_)
		f_output = float_value_from_percent_string(input_)
		value = percent_from_string(input_)
		self.assertTrue( output == '7 ' )
		self.assertTrue( f_output == 7. )
		self.assertTrue( isinstance(value, Percent) )
		self.assertTrue( value.value == 7 )

		input_ = '7 Percent'
		output = strip_percent_units(input_)
		f_output = float_value_from_percent_string(input_)
		value = percent_from_string(input_)
		self.assertTrue( output == '7 ' )
		self.assertTrue( f_output == 7. )
		self.assertTrue( isinstance(value, Percent) )
		self.assertTrue( value.value == 7 )

		input_ = '7 PERCENT'
		output = strip_percent_units(input_)
		f_output = float_value_from_percent_string(input_)
		value = percent_from_string(input_)
		self.assertTrue( output == '7 ' )
		self.assertTrue( f_output == 7. )
		self.assertTrue( isinstance(value, Percent) )
		self.assertTrue( value.value == 7 )

		input_ = '7 pct'
		output = strip_percent_units(input_)
		f_output = float_value_from_percent_string(input_)
		value = percent_from_string(input_)
		self.assertTrue( output == '7 ' )
		self.assertTrue( f_output == 7. )
		self.assertTrue( isinstance(value, Percent) )
		self.assertTrue( value.value == 7 )

		input_ = '7 PCT'
		output = strip_percent_units(input_)
		f_output = float_value_from_percent_string(input_)
		value = percent_from_string(input_)
		self.assertTrue( output == '7 ' )
		self.assertTrue( f_output == 7. )
		self.assertTrue( isinstance(value, Percent) )
		self.assertTrue( value.value == 7 )

	def test_volume_parse(self):
		input_ = '3 cc'
		unit = volume_unit_from_string(input_)
		self.assertTrue( isinstance(unit, CM3) )
		output = strip_volume_units(input_)
		self.assertTrue( output == '3 ' )
		f_output = float_value_from_volume_string(input_)
		self.assertTrue( f_output == 3 )
		v_output = volume_from_string(input_)
		self.assertTrue( isinstance(v_output, CM3) )
		self.assertTrue( v_output.value == 3. )

		input_ = '3 cm3'
		unit = volume_unit_from_string(input_)
		self.assertTrue( isinstance(unit, CM3) )
		output = strip_volume_units(input_)
		self.assertTrue( output == '3 ' )
		f_output = float_value_from_volume_string(input_)
		self.assertTrue( f_output == 3 )
		v_output = volume_from_string(input_)
		self.assertTrue( isinstance(v_output, CM3) )
		self.assertTrue( v_output.value == 3. )

		input_ = '3 cm^3'
		unit = volume_unit_from_string(input_)
		self.assertTrue( isinstance(unit, CM3) )
		output = strip_volume_units(input_)
		self.assertTrue( output == '3 ' )
		f_output = float_value_from_volume_string(input_)
		self.assertTrue( f_output == 3 )
		v_output = volume_from_string(input_)
		self.assertTrue( isinstance(v_output, CM3) )
		self.assertTrue( v_output.value == 3. )

		input_ = '3 CC'
		unit = volume_unit_from_string(input_)
		self.assertTrue( isinstance(unit, CM3) )
		output = strip_volume_units(input_)
		self.assertTrue( output == '3 ' )
		f_output = float_value_from_volume_string(input_)
		self.assertTrue( f_output == 3 )
		v_output = volume_from_string(input_)
		self.assertTrue( isinstance(v_output, CM3) )
		self.assertTrue( v_output.value == 3. )

		input_ = '3 CM3'
		unit = volume_unit_from_string(input_)
		self.assertTrue( isinstance(unit, CM3) )
		output = strip_volume_units(input_)
		self.assertTrue( output == '3 ' )
		f_output = float_value_from_volume_string(input_)
		self.assertTrue( f_output == 3 )
		v_output = volume_from_string(input_)
		self.assertTrue( isinstance(v_output, CM3) )
		self.assertTrue( v_output.value == 3. )

		input_ = '3 CM^3'
		unit = volume_unit_from_string(input_)
		self.assertTrue( isinstance(unit, CM3) )
		output = strip_volume_units(input_)
		self.assertTrue( output == '3 ' )
		f_output = float_value_from_volume_string(input_)
		self.assertTrue( f_output == 3 )
		v_output = volume_from_string(input_)
		self.assertTrue( isinstance(v_output, CM3) )
		self.assertTrue( v_output.value == 3. )

		input_ = '3 mm3'
		unit = volume_unit_from_string(input_)
		self.assertTrue( isinstance(unit, MM3) )
		output = strip_volume_units(input_)
		self.assertTrue( output == '3 ' )
		f_output = float_value_from_volume_string(input_)
		self.assertTrue( f_output == 3 )
		v_output = volume_from_string(input_)
		self.assertTrue( isinstance(v_output, MM3) )
		self.assertTrue( v_output.value == 3. )

		input_ = '3 mm^3'
		unit = volume_unit_from_string(input_)
		self.assertTrue( isinstance(unit, MM3) )
		output = strip_volume_units(input_)
		self.assertTrue( output == '3 ' )
		f_output = float_value_from_volume_string(input_)
		self.assertTrue( f_output == 3 )
		v_output = volume_from_string(input_)
		self.assertTrue( isinstance(v_output, MM3) )
		self.assertTrue( v_output.value == 3. )

		input_ = '3 MM3'
		unit = volume_unit_from_string(input_)
		self.assertTrue( isinstance(unit, MM3) )
		output = strip_volume_units(input_)
		self.assertTrue( output == '3 ' )
		f_output = float_value_from_volume_string(input_)
		self.assertTrue( f_output == 3 )
		v_output = volume_from_string(input_)
		self.assertTrue( isinstance(v_output, MM3) )
		self.assertTrue( v_output.value == 3. )

		input_ = '3 MM^3'
		unit = volume_unit_from_string(input_)
		self.assertTrue( isinstance(unit, MM3) )
		output = strip_volume_units(input_)
		self.assertTrue( output == '3 ' )
		f_output = float_value_from_volume_string(input_)
		self.assertTrue( f_output == 3 )
		v_output = volume_from_string(input_)
		self.assertTrue( isinstance(v_output, MM3) )
		self.assertTrue( v_output.value == 3. )

	def test_dose_parse(self):
		input_ = '5 gy'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, Gray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, Gray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 Gy'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, Gray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, Gray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 GY'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, Gray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, Gray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 gray'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, Gray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, Gray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 Gray'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, Gray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, Gray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 GRAY'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, Gray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, Gray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 cgy'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, centiGray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, centiGray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 cGy'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, centiGray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, centiGray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 cGY'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, centiGray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, centiGray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 CGY'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, centiGray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, centiGray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 centigray'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, centiGray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, centiGray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 centiGray'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, centiGray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, centiGray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 centiGRAY'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, centiGray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, centiGray) )
		self.assertTrue( d_output.value == 5 )

		input_ = '5 CENTIGRAY'
		unit = dose_unit_from_string(input_)
		self.assertTrue( isinstance(unit, centiGray) )
		output = strip_dose_units(input_)
		self.assertTrue( output == '5 ' )
		f_output = float_value_from_dose_string(input_)
		self.assertTrue( f_output == 5. )
		d_output = dose_from_string(input_)
		self.assertTrue( isinstance(d_output, centiGray) )
		self.assertTrue( d_output.value == 5 )

	def test_dose_or_percent_parse(self):
		input_ = '5 Gy'
		output = percent_or_dose_from_string(input_)
		self.assertTrue( isinstance(output, Gray) )
		self.assertTrue( output.value == 5 )

		input_ = '50 %'
		output = percent_or_dose_from_string(input_)
		self.assertTrue( isinstance(output, Percent) )
		self.assertTrue( output.value == 50 )

		input_ = '5'
		output = percent_or_dose_from_string(input_)
		self.assertTrue( output is None )