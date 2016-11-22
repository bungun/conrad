"""
Define :class:`CaseIO` for loading and saving treatment planning cases.
"""
"""
# Copyright 2016 Baris Ungun, Anqi Fu

# This file is part of CONRAD.

# CONRAD is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# CONRAD is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CONRAD.  If not, see <http://www.gnu.org/licenses/>.
"""
from conrad.compat import *

import yaml
from os import path
from abc import ABCMeta
import numpy as np
import scipy.sparse as sp

from conrad.case.case import Case
from conrad.medicine import Anatomy, Structure, Prescription
from conrad.physics.physics import Physics

class ConradIO(ABCMeta):
	@abstractmethod
	def read(self, file, key):
		pass

	@abstractmethod
	def write(self, file, data, overwrite=False):
		pass

class LocalNumpyIO(ConradIO):
	def read(self, file, key):
		if '.npy' in file:
			return np.load(file)
		elif '.npz' in file:
			return np.load(filename)[key]
		elif '.txt' in file:
			return np.loadtxt(filename)
		else:
			raise ValueError('file extension must be one of {}'.format(
							('.npz', '.npy', '.txt')))

	@staticmethod
	def write(self, file, data, overwrite=False):
		extension = '.npz' if isinstance(data, dict) else '.npy'
		if not extension in file:
			file += extension

		if path.exists(file) and not overwrite:
			raise OSError('file "{}" exists; please specify keyword '
						  'argument `overwrite=True` to proceed with '
						  'save operation'.format(file))

		if isinstance(data, dict):
			np.savez(file, **data)
		else:
			np.save(file, **data)


class ConradDB(ABCMeta):
	@abstractmethod
	def get(self, key):
		pass

	@abstractmethod
	def set(self, key, value):
		pass

	@abstractmethod
	def ingest_yaml(self, filename):
		pass

	@abstractmethod
	def dump_to_yaml(self, filename):
		pass

class LocalPythonDB(ConradDB):
	def __init__(self, yaml_file=None):
		pass

	def get(self, key):
		pass

	def set(self, key, value):
		pass

	def ingest_yaml(self, filename):
		pass

	def dump_to_yaml(self, filename):
		pass

class DoseFrameMeta(object):
	def __init__(self):
		self.__name = None
		self.__frame = None
		self.__saved = False

	def write(self)

class CaseMeta(object):
	def __init__(self):
		self.__name = None
		self.__voxel_grid = None
		self.__beams = None
		self.__frames = {}
		self.__presciption = None

	def add_frame()


	@property
	def frames(self):
		return self.__frames.keys()


class CaseIO(object):
	def __init__(self):
		self.__DB = LocalPythonDB()
		self.__IO = LocalNumpyIO()
		self.__meta = CaseMeta()
		self.__case = None

	# STUB
	def set_filesystem(self, filesystem=None):
		if filesystem is None:
			self.__IO = LocalNumpyIO()
		else:
			return NotImplemented

	def set_database(self, database=None):
		if database is None:
			self.__IO = LocalYamlDB()
		else:
			return NotImplemented

	def read_file(self, file, key):
		return self.__IO.read(file, key)

	def write_file(self, file, data, overwrite=False):
		self.__IO.write(file, data, overwrite=overwrite)

	@staticmethod
	def fragment_empty(self, yaml_fragment):
		status = yaml_fragment is not None
		if 'data' in yaml_fragment:
			status &= yaml_fragment['data'] is not None
			yaml_fragment = yaml_fragment['data']
		if 'values' in yaml_fragment:
			status &= yaml_fragment['values'] is not None
			yaml_fragment = yaml_fragment['values']
		if 'file' in yaml_fragment:
			status &= yaml_fragment['file'] is not None

	def load_vec(self, yaml_fragment, base_path=''):
		if self.fragment_empty(yaml_fragment):
			return None

		if 'data' in yaml_fragment:
			assert yaml_fragment['type'] == 'vector'
			yaml_fragment = yaml_fragment['data']

		filename = path.join(base_path, yaml_fragment['file'])
		return self.read_file(filename, yaml_fragment['key'])

	def load_dense_mat(self, yaml_fragment, base_path=''):
		if self.fragment_empty(yaml_fragment):
			return None

		assert yaml_fragment['type'] == 'dense matrix'

		filename = path.join(base_path, yaml_fragment['data']['file'])
		return np.array(
				self.read_file(filename, yaml_fragment['key']),
				order='C' if rowmajor else 'F')

	def load_sparse_mat(self, yaml_fragment, base_path=''):
		if self.fragment_empty(yaml_fragment):
			return None

		assert yaml_fragment['type'] == 'sparse matrix'

		ptr = self.load_vec(yaml_fragment['pointers'], base_path=base_path)
		ind = self.load_vec(yaml_fragment['indices'], base_path=base_path)
		val = self.load_vec(yaml_fragment['values'], base_path=base_path)

		if yaml_fragment['layout']['fortran_indexing']:
			ptr -= 1
			ind -= 1
		constructor = sp.csr_matrix if yaml_fragment['layour']['CSR'] else \
					  sp.csc_matrix

		return constructor((val, ind, ptr), shape=yaml_fragment['shape'])

	def load_mat(self, yaml_fragment, base_path=''):
		if self.fragment_empty(yaml_fragment):
			return None
		elif yaml_fragment['type'] == 'sparse matrix':
			return self.load_sparse_mat(yaml_fragment)
		else:
			return self.load_dense_mat(yaml_fragment)

	def add_frame(self, frame_name):


	def load_frame(self, yaml_spec, case=None, frame_name='default'):
		if frame_name == 'default':
			frame_name = 'full'

		for frame in yaml_spec['frames']:
			if frame['name'] == frame_name:
				base_dir = frame['dir']

				if case is None:
					case = Case()

				if frame_name not in case.physics.available_frames:
					case.physics.add_dose_frame(frame_name)
				case.physics.change_dose_frame(frame_name)

				case.physics.dose_matrix = self.load_mat(
						frame['dose_matrix'], base_dir)
				case.physics.voxel_labels = self.load_mat(
						frame['voxel_labels'], base_dir)

				voxel_weights = self.load_mat(frame['voxel_weights'], base_dir)
				if voxel_weights is not None:
					case.physics.frame.voxel_weights = voxel_weights

				beam_labels = self.load_mat(frame['beam_labels'], base_dir)
				if beam_labels is not None:
					case.physics.frame.beam_labels = beam_labels

				beam_weights = self.load_mat(frame['beam_weights'], base_dir)
				if beam_weights is not None:
					case.physics.frame.beam_weights = voxel_weights

				return case

		raise ValueError('YAML does not contain data for dose frame "{}"'
						 ''.format(frame_name))


	def load_pogs_solver_cache(self, yaml_fragment):
		base_dir = yaml_fragment['dir']
		d = self.load_vec(yaml_fragment['left_preconditioner'], base_dir)
		A = self.load_mat(yaml_fragment['matrix'], base_dir)
		e = self.load_vec(yaml_fragment['right_preconditioner'],
								base_dir)
		L = self.load_mat(yaml_fragment['projector']['matrix'], base_dir)
		return d, A, e, L

	def load_solver_cache(self, yaml_spec, frame_name='default'):
		if frame_name == 'default':
			frame_name = 'full'

		if not 'caches' in yaml_spec:
			raise ValueError('YAML spec has no cache data')

		for cache in yaml_spec['caches']:
			if cache['frame'] == frame_name:
				if cache['solver'] == 'POGS':
					return self.load_pogs_solver_cache(cache)
				else:
					raise ValueError('solver caching only supported for POGS')
		raise ValueError('solver cache for frame "{}" not found'.format(
						 frame_name))

	def load_solution(self, yaml_spec, solution_name, frame_name='default'):
		if frame_name == 'default':
			frame_name = 'full'

		if not 'solutions' in yaml_spec:
			raise ValueError('YAML spec has no solution data')

		for solution in yaml_spec['solution']:
			if solution['frame'] == frame_name:
				x = self.load_vec(solution['x'])
				y = self.load_vec(solution['y'])
				mu = self.load_vec(solution['mu'])
				nu = self.load_vec(solution['nu'])
				return x, y, mu, nu

		raise ValueError('solution with name "{}" associated with frame "{}" '
						 'not found'.format(solution_name, frame_name))

	def load prescription(self, yaml_spec):
		return Prescription(yaml_spec['prescription'])

	def load_case_yaml(self, filepath):
		return yaml.safe_load(filepath)


	def write_case_yaml(self, case, filepath):
		f = open(filepath, 'w')

		# write case
		data = {'case': }
		# write prescription


	def construct(self, yaml_spec, case=None, **kwargs):


