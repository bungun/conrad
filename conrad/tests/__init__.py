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


"""
Test/dependency hierarchy:

(module tests by test file name: test_{}.py, where {} = module name)

0:
	units
	abstract_vector
	abstract_matrix
	abstract_mapping
	solver
	plot_elements
	io_schema

1:
	grid
	physics_string
	physics_containers
	io_database
	io_filesystem

2:
	beams
	voxels
	dose
	objectives

3:
	physics
	structure
	solver

4:
	anatomy
	preprocessing

5:
	prescription
	solver_cvxpy
	solver_optkit
	plot_collections

6:
	history [test written to use Prescription, otherwise lvl=0]
	problem
	plot_canvases

7:
	case

8:
	io_accessors
	plot_plotter

9:
	io

"""