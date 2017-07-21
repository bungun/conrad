"""
Unit tests for :mod:`conrad.optimization.problem`.
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

from conrad.physics.units import Gy
from conrad.medicine import Structure, Anatomy
from conrad.medicine.dose import D
from conrad.optimization.clustering import *
from conrad.tests.base import *

class ClusteredProblemTestCase(ConradTestCase):
	pass

class UnconstrainedVoxClusProblemTestCase(ConradTestCase):
	pass

class UnconstrainedBeamClusProblemTestCase(ConradTestCase):
	pass

