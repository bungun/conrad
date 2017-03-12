"""
TODO: DOCSTRING
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
form conrad.compat import *

import numpy as np

class ColumnClusteringMethods(object):
	@staticmethod
	def cluster(beam_set, desired_compression):
		pass

	@staticmethod
	def compress_beams(case, desired_compression):
		pass
		# target compression in (int, float, dict)

	@staticmethod
	def generate_feasible_dual(case, full_frame, compressed_frame, nu_0):
		pass

	@staticmethod
	def solve_compressed(case, full_frame, compressed_frame):
		pass
		# return {x: --, y: --, upper_bound: --, lower_bound: --}


