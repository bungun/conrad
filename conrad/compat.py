from __future__ import print_function
"""
Python 2/3 comptability module for CONRAD.

Attributes:
	xrange: Alias python3 :func:`range` in namespace to match python2
		:func:`xrange`.
	listmap: Wrap python3 :func:`map` to match python2 implementation.
	listfilter: Wrap python3 :func: `filter` to match python2
		implementation.
	reduce: Import python3 :func:`functools.reduce` into namespace.
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
import six
from six import add_metaclass

def listmap(f, *args):
	return list(six.moves.map(f, *args))
def listfilter(f, *args):
	return list(six.moves.filter(f, *args))

if six.PY3:
	from six.moves import xrange
	from six.moves import reduce

