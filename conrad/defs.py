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
from os import getenv
from pip import get_installed_distributions
from numpy import nan, squeeze, array, ndarray
from scipy.sparse import csr_matrix, csc_matrix
from conrad.compat import *

def println(*args):
	print(args)

CONRAD_DEBUG = getenv('CONRAD_DEBUG', False)
CONRAD_DEBUG_PRINT = println if CONRAD_DEBUG else lambda x : None

SOLVER_OPTIONS = ['ECOS', 'SCS']
MAX_VERBOSITY = 1

CONRAD_MATRIX_TYPES = (ndarray, csr_matrix, csc_matrix)

def vec(vectorlike):
	return squeeze(array(vectorlike))

def is_vector(vectorlike):
	if isinstance(vectorlike, ndarray):
		return len(vectorlike) == vectorlike.shape[0]
	return False

def sparse_or_dense(matrixlike):
	if matrixlike is None:
		return False

	dense = isinstance(matrixlike, ndarray)
	if dense:
		dense &= len(matrixlike.shape) == 2
	sparse = isinstance(matrixlike, (csr_matrix, csc_matrix))
	return sparse or dense

def positive_real_valued(val):
	if val is not None and val is not nan and isinstance(val, (int, float)):
		if val > 0:
			return True
	else:
		return False

def module_installed(name, version_string=None):
	modules = get_installed_distributions()
	index = -1

	installed = False
	for idx, module in enumerate(modules):
		if name in str(module):
			index = idx
			installed = True

	if version_string:
		installed &= str(version_string) in str(modules[index])

	return installed