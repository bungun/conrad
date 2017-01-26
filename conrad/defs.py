"""
Frequently used constants and utilities.

Attributes:
	CONRAD_DEBUG (:obj:`bool`): Toggle for debugging-specific code
		branches.
	CONRAD_DEBUG_PRINT (:obj:`bool`): Toggle for debugging-specifc print
		statements.
	SOLVER_OPTIONS (:obj:`list` of :obj:`str`): Enumeration of solver
		names usable with :mod:`cvpxy`.
	CONRAD_MATRIX_TYPES (:obj:`tuple`): Enumeration of :mod:`numpy` and
		:mod:`scipy` matrix types used in :mod:`conrad`.
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
import pip
import numpy as np
import scipy.sparse as sp

def println(*args):
	print(args)

CONRAD_DEBUG = os.getenv('CONRAD_DEBUG', False)
CONRAD_DEBUG_PRINT = println if CONRAD_DEBUG else lambda x : None

SOLVER_OPTIONS = ['ECOS', 'SCS']

CONRAD_MATRIX_TYPES = (np.ndarray, sp.csr_matrix, sp.csc_matrix)

def vec(vectorlike):
	""" Convert input to one-dimensional :class:`~numpy.ndarray`. """
	return np.squeeze(np.array(vectorlike))

def is_vector(vectorlike):
	""" ``True`` if input is one-dimensional :class:`~numpy.ndarray`. """
	if isinstance(vectorlike, np.ndarray):
		return vectorlike.size == vectorlike.shape[0]
	return False

def sparse_or_dense(matrixlike):
	"""
	``True`` if input is a :mod:`conrad`-recognized matrix type.

	Accepted types include: two-dimensional :class:`ndarray`,
	:class:`sp.csr_matrx` and :class:`sp.csc_matrix`.
	"""
	if matrixlike is None:
		return False

	dense = isinstance(matrixlike, np.ndarray)
	if dense:
		dense &= len(matrixlike.shape) == 2
	sparse = isinstance(matrixlike, (sp.csr_matrix, sp.csc_matrix))
	return sparse or dense

def vec_or_mat_formatted(arraylike):
	return is_vector(arraylike) or sparse_or_dense(arraylike)

def positive_real_valued(val):
	""" ``True`` if input is a positive scalar. """
	if val is not None and val is not np.nan and isinstance(val, (int, float)):
		if val > 0:
			return True
	else:
		return False

def module_installed(name, version_string=None):
	"""
	Test whether queried module is installed.

	Arguments:
		name (:obj:`str`): Name of module to query.
		version_string (:obj:`str`, optional): Specific module version
			to query.

	Returns:
		:obj:`bool`: ``True`` if queried module has a matching string in
		dictionary values returned by
		:func:`pip.get_installed_distributions`.
	"""
	modules = pip.get_installed_distributions()
	index = -1

	installed = False
	for idx, module in enumerate(modules):
		if name in str(module):
			index = idx
			installed = True

	if version_string:
		installed &= str(version_string) in str(modules[index])

	return installed