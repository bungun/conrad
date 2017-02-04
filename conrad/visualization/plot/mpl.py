"""
Dose volume histogram plotting imports.

Attributes:
	PLOTTING_INSTALLED (:obj:`bool`): ``True`` if :mod:`matplotlib` is
		not available. If so, :class:`DVHPlot` and :class:`CasePlotter`
		types are replaced with lambdas that match the initialization
		argument signature and each yield ``None`` instead.



	DISPLAY_AVAILABLE (:obj:`bool`): ``True`` if :mod:`matplotlib` is
		available and connected to a displayable backend.
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

from conrad.defs import module_installed

# allow for CONRAD use without plotting by making visualization types
# optional
PLOTTING_INSTALLED = False
DISPLAY_AVAILABLE = False
if module_installed('matplotlib'):
	PLOTTING_INSTALLED = True

	import matplotlib as mpl
	import matplotlib.lines
	import matplotlib.axes
	import matplotlib.figure
	import matplotlib.colors
	if os.getenv('DISPLAY') is None:
		mpl.use('Agg')
	else:
		DISPLAY_AVAILABLE = True
	import matplotlib.pyplot as plt