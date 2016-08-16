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
from __future__ import print_function
from sys import version_info

if version_info.major > 2:
	xrange = range
	def listmap(f, *args):
		return list(map(f, *args))
	def listfilter(f, *args):
		return list(filter(f, *args))
	from functools import reduce
	CONRAD_PY_VERSION = 3
else:
	listmap = map
	listfilter = filter
	CONRAD_PY_VERSION = 2

