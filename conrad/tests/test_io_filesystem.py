"""
Unit tests for :mod:`conrad.io.filesystem`.
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
import numpy as np
import scipy.sparse as sp

from conrad.io.filesystem import *
from conrad.tests.base import *

class FilesystemTestBase(ConradFilesystemBase):
	def check_dir(self, directory):
		return NotImplemented
	def join_mkdir(self, directory, *subdir):
		return NotImplemented
	def read(self, file, key):
		return NotImplemented
	def read_all(self, file):
		return NotImplemented
	def write(self, file, data, overwrite=False):
		return NotImplemented

class FilesystemTestNaming(ConradFilesystemBase):
	def check_dir(self, directory):
		return True
	def join_mkdir(self, directory, *subdir):
		d = str(directory)
		for s in subdir:
			d += '/' + str(subdir)
		return d
	def read(self, file, key):
		return 'reading at file `{}` with key `{}`'.format(file, key)
	def read_all(self, file):
		return NotImplemented
	def write(self, file, data, overwrite=False):
		return {'file': 'writing at file `{}`'.format(file), 'key': None}

class FilesystemTestCaching(ConradFilesystemBase):
	def __init__(self):
		ConradFilesystemBase.__init__(self)
		self.__files = {}

	@property
	def files(self):
		return {k:v for k, v in self.__files.items()}

	def check_dir(self, directory):
		return True
	def join_mkdir(self, directory, *subdir):
		d = str(directory)
		for s in subdir:
			if not d.endswith('/'):
				d += '/'
			if not d.endswith(str(s)):
				d += str(subdir)
			if not d.endswith('/'):
				d += '/'
		return d

	def read(self, file, key):
		if str(file) in self.__files:
			return self.__files[file]
		else:
			return None

	def read_all(self, file):
		return NotImplemented

	def write(self, file, data, overwrite=False):
		file = str(file)
		if not file.endswith('.npy'):
			file += '.npy'
		if overwrite or file not in self.__files:
			self.__files[file] = data
		return {'file': file, 'key': None}

class FilesystemBaseTestCase(ConradTestCase):
	def test_fsbase_init(self):
		fs = FilesystemTestBase()
		self.assertTrue( isinstance(fs._ConradFilesystemBase__DIGEST, dict) )
		for type_ in (
				int, float, str, dict, type(None), DataDictionaryEntry,
				VectorEntry, DenseMatrixEntry, SparseMatrixEntry):
			self.assertTrue( type_ in fs._ConradFilesystemBase__DIGEST)

		self.assertTrue( isinstance(fs._ConradFilesystemBase__DUMP, dict) )
		for type_ in (
				int, float, str, dict, type(None), np.ndarray, sp.csr_matrix,
				sp.csc_matrix):
			self.assertTrue( type_ in fs._ConradFilesystemBase__DUMP)

	def test_fsbase_abstract(self):
		fs = FilesystemTestBase()
		self.assertTrue( fs.check_dir('directory') is NotImplemented )
		self.assertTrue(
				fs.join_mkdir('directory', 'subdir') is NotImplemented )
		self.assertTrue( fs.read('file', 'key') is NotImplemented )
		self.assertTrue( fs.write('file', 'data') is NotImplemented )

	def test_fsbase_write_vector(self):
		fs = FilesystemTestNaming()
		ve = fs.write_vector('dir', 'name', np.random.rand(30))
		self.assertTrue( isinstance(ve, VectorEntry) )
		self.assertTrue( ve.data_file == 'writing at file `dir/name`' )

		with self.assertRaises(TypeError):
			fs.write_vector(np.random.rand(30, 20))
		with self.assertRaises(TypeError):
			fs.write_vector(
					'dir', 'name', sp.rand(30, 20, 0.2, format='csr'))

	def test_fsbase_write_dense_matrix(self):
		fs = FilesystemTestNaming()
		dme = fs.write_dense_matrix('dir', 'name', np.random.rand(30, 20))
		self.assertTrue( isinstance(dme, DenseMatrixEntry) )
		self.assertTrue( dme.layout_rowmajor )
		self.assertTrue( dme.data_file == 'writing at file `dir/name`' )

		with self.assertRaises(TypeError):
			fs.write_dense_matrix(np.random.rand(30))
		with self.assertRaises(TypeError):
			fs.write_dense_matrix(
					'dir', 'name', sp.rand(30, 20, 0.2, format='csr'))

	def test_fsbase_write_sparse_matrix(self):
		fs = FilesystemTestNaming()
		sme = fs.write_sparse_matrix(
				'dir', 'name', sp.rand(30, 20, 0.2, format='csr'))
		self.assertTrue( isinstance(sme, SparseMatrixEntry) )
		self.assertTrue( sme.layout_CSR )
		self.assertTrue( sme.layout_fortran_indexing is not None )
		self.assertFalse( sme.layout_fortran_indexing )
		self.assertTrue( sme.shape == (30, 20) )
		self.assertTrue(
				sme.data_pointers_file ==
				'writing at file `dir/name_pointers`' )
		self.assertTrue(
				sme.data_indices_file ==
				'writing at file `dir/name_indices`' )
		self.assertTrue(
				sme.data_values_file ==
				'writing at file `dir/name_values`' )

		sme2 = fs.write_sparse_matrix(
				'dir', 'name', sp.rand(30, 20, 0.2, format='csc'))
		self.assertTrue( isinstance(sme2, SparseMatrixEntry) )
		self.assertTrue( sme.layout_CSR is not None )
		self.assertFalse( sme2.layout_CSR )
		self.assertTrue( sme.layout_fortran_indexing is not None )
		self.assertFalse( sme2.layout_fortran_indexing )
		self.assertTrue( sme2.shape == (30, 20) )
		self.assertTrue(
				sme2.data_pointers_file ==
				'writing at file `dir/name_pointers`' )
		self.assertTrue(
				sme2.data_indices_file ==
				'writing at file `dir/name_indices`' )
		self.assertTrue(
				sme2.data_values_file ==
				'writing at file `dir/name_values`' )

		with self.assertRaises(TypeError):
			fs.write_sparse_matrix(np.random.rand(30))
		with self.assertRaises(TypeError):
			fs.write_sparse_matrix(np.random.rand(30, 20))
		with self.assertRaises(TypeError):
			fs.write_sparse_matrix(
					'dir', 'name', sp.rand(30, 20, 0.2, format='coo'))

	def test_fsbase_write_ndarray(self):
		fs = FilesystemTestNaming()
		ve = fs.write_ndarray('dir', 'name', np.random.rand(30))
		self.assertTrue( isinstance(ve, VectorEntry) )
		dme = fs.write_ndarray('dir', 'name', np.random.rand(30, 20))
		self.assertTrue( isinstance(dme, DenseMatrixEntry) )

		with self.assertRaises(TypeError):
			fs.write_ndarray(sp.rand(30, 20, 0.2, format='csr'))

	def test_fsbase_write_matrix(self):
		fs = FilesystemTestNaming()

		dme = fs.write_matrix('dir', 'name', np.random.rand(30, 20))
		self.assertTrue( isinstance(dme, DenseMatrixEntry) )
		sme = fs.write_matrix(
				'dir', 'name', sp.rand(30, 20, 0.2, format='csr'))
		self.assertTrue( isinstance(sme, SparseMatrixEntry) )
		sme2 = fs.write_matrix(
				'dir', 'name', sp.rand(30, 20, 0.2, format='csc'))
		self.assertTrue( isinstance(sme2, SparseMatrixEntry) )

		with self.assertRaises(TypeError):
			fs.write_ndarray(np.random.rand(30))

	def test_fsbase_write_data_dictionary(self):
		fs = FilesystemTestNaming()

		dd = {'1': 'string', '2': 'numeric'}
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertTrue( dd == dd_out )

		dd['3'] = np.random.rand(30)
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertTrue( isinstance(dd_out, DataDictionaryEntry) )
		self.assertTrue( isinstance(dd_out.entries['3'], VectorEntry) )
		self.assertTrue(
				dd_out.entries['3'].data_file ==
				'writing at file `dir/name_3`' )

		dd['4'] = np.random.rand(30, 20)
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertTrue( isinstance(dd_out, DataDictionaryEntry) )
		self.assertTrue( isinstance(dd_out.entries['4'], DenseMatrixEntry) )
		self.assertTrue(
				dd_out.entries['4'].data_file ==
				'writing at file `dir/name_4`' )

		dd['5'] = sp.rand(30, 20, 0.2, format='csr')
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertTrue( isinstance(dd_out, DataDictionaryEntry) )
		self.assertTrue( isinstance(dd_out.entries['5'], SparseMatrixEntry) )
		self.assertTrue(
				dd_out.entries['5'].data_pointers_file ==
				'writing at file `dir/name_5_pointers`' )
		self.assertTrue(
				dd_out.entries['5'].data_indices_file ==
				'writing at file `dir/name_5_indices`' )
		self.assertTrue(
				dd_out.entries['5'].data_values_file ==
				'writing at file `dir/name_5_values`' )

		dd['6'] = {'key': 'value'}
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertTrue( isinstance(dd_out, DataDictionaryEntry) )
		self.assertTrue( dd_out.entries['6'] == dd['6'] )

		dd['7'] = {'key': np.random.rand(30)}
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertTrue( isinstance(dd_out, DataDictionaryEntry) )
		self.assertTrue( isinstance(dd_out.entries['7'], DataDictionaryEntry) )
		self.assertTrue( isinstance(
				dd_out.entries['7'].entries['key'], VectorEntry) )
		self.assertTrue(
				dd_out.entries['7'].entries['key'].data_file ==
				'writing at file `dir/name_7_key`' )

	def test_fsbase_write_generic(self):
		fs = FilesystemTestNaming()
		int_val = fs.write_data('dir', 'name', 2)
		self.assertTrue( isinstance(int_val, int) )
		flt_val = fs.write_data('dir', 'name', 2.)
		self.assertTrue( isinstance(flt_val, float) )
		str_val = fs.write_data('dir', 'name', '2')
		self.assertTrue( isinstance(str_val, str) )
		dct_val = fs.write_data('dir', 'name', {'key': 'value'})
		self.assertTrue( isinstance(dct_val, dict) )

		with self.assertRaises(TypeError):
			fs.write_data('dir', 'name', [])
		with self.assertRaises(TypeError):
			fs.write_data('dir', 'name', set())

		ve = fs.write_data('dir', 'name', np.random.rand(30))
		self.assertTrue( isinstance(ve, VectorEntry) )
		self.assertTrue( '`dir/name`' in ve.data_file )

		dme = fs.write_data('dir', 'name', np.random.rand(30, 20))
		self.assertTrue( isinstance(dme, DenseMatrixEntry) )
		self.assertTrue( '`dir/name`' in dme.data_file )

		sme = fs.write_data('dir', 'name', sp.rand(30, 20, 0.2, format='csr'))
		self.assertTrue( isinstance(sme, SparseMatrixEntry) )
		self.assertTrue( '`dir/name_pointers`' in sme.data_pointers_file )
		self.assertTrue( '`dir/name_indices`' in sme.data_indices_file )
		self.assertTrue( '`dir/name_values`' in sme.data_values_file )

		dde = fs.write_data('dir', 'name', {
				'first_key': np.random.rand(30),
				'second_key': np.random.rand(30),
				'third_key': np.random.rand(30, 20),
				'fourth_key': sp.rand(30, 20, 0.2, format='csr'),
				'fifth_key': {'key': 'value'},
				'sixth_key': 2,
				'seventh_key': 3.,
				'eighth_key': 'string',
				'ninth_key': {'key': np.random.rand(30)},
		})
		self.assertTrue( isinstance(dde, DataDictionaryEntry) )
		self.assertTrue(
				isinstance(dde.entries['first_key'], VectorEntry))
		self.assertTrue(
				'`dir/name_first_key`' in
				dde.entries['first_key'].data_file)
		self.assertTrue(
				isinstance(dde.entries['second_key'], VectorEntry))
		self.assertTrue(
				'`dir/name_second_key`' in
				dde.entries['second_key'].data_file)
		self.assertTrue(
				isinstance(dde.entries['third_key'], DenseMatrixEntry))
		self.assertTrue(
				'`dir/name_third_key`' in
				dde.entries['third_key'].data_file)
		self.assertTrue(
				isinstance(dde.entries['fourth_key'], SparseMatrixEntry))
		self.assertTrue(
				'`dir/name_fourth_key_pointers`' in
				dde.entries['fourth_key'].data_pointers_file)
		self.assertTrue(
				'`dir/name_fourth_key_indices`' in
				dde.entries['fourth_key'].data_indices_file)
		self.assertTrue(
				'`dir/name_fourth_key_values`' in
				dde.entries['fourth_key'].data_values_file)
		self.assertTrue( isinstance(dde.entries['fifth_key'], dict) )
		self.assertTrue( isinstance(dde.entries['sixth_key'], int) )
		self.assertTrue( isinstance(dde.entries['seventh_key'], float) )
		self.assertTrue( isinstance(dde.entries['eighth_key'], str) )
		self.assertTrue(
				isinstance(dde.entries['ninth_key'], DataDictionaryEntry) )
		self.assertTrue( isinstance(
			dde.entries['ninth_key'].entries['key'], VectorEntry) )
		self.assertTrue(
				'`dir/name_ninth_key_key`' in
				dde.entries['ninth_key'].entries['key'].data_file )

	def test_fsbase_read_vector(self):
		fs = FilesystemTestCaching()
		vec = np.random.rand(30)
		vec_entry = fs.write_vector('dir', 'vec1', vec)
		vec_back = fs.to_vector(vec_entry)
		self.assert_vector_equal(vec, vec_back)

	def test_fsbase_read_dense_matrix(self):
		fs = FilesystemTestCaching()
		mat = np.random.rand(30, 20)
		mat_entry = fs.write_dense_matrix('dir', 'mat1', mat)
		mat_back = fs.to_dense_matrix(mat_entry)
		self.assert_vector_equal(mat, mat_back)

	def test_fsbase_read_sparse_matrix(self):
		fs = FilesystemTestCaching()
		mat = sp.rand(30, 20, 0.2, format='csr')
		mat_entry = fs.write_sparse_matrix('dir', 'mat1', mat)
		mat_back = fs.to_sparse_matrix(mat_entry)
		self.assertTrue( isinstance(mat_back, sp.csr_matrix) )
		self.assert_vector_equal(mat.indptr, mat_back.indptr)
		self.assert_vector_equal(mat.indices, mat_back.indices)
		self.assert_vector_equal(mat.data, mat_back.data)

		mat = sp.rand(30, 20, 0.2, format='csc')
		mat_entry = fs.write_sparse_matrix('dir', 'mat2', mat)
		mat_back = fs.to_sparse_matrix(mat_entry)
		self.assertTrue( isinstance(mat_back, sp.csc_matrix) )
		self.assert_vector_equal(mat.indptr, mat_back.indptr)
		self.assert_vector_equal(mat.indices, mat_back.indices)
		self.assert_vector_equal(mat.data, mat_back.data)

	def test_fsbase_read_data_dictionary(self):
		fs = FilesystemTestCaching()
		data_dict = {
				1: np.random.rand(30), 2: np.random.rand(30),
				3: np.random.rand(30)
		}
		dict_entry = fs.write_data_dictionary('dir', 'dict1', data_dict)
		dict_back = fs.to_data_dictionary(dict_entry)
		self.assertTrue( isinstance(dict_back, dict) )
		for k in data_dict:
			self.assert_vector_equal(data_dict[k], dict_back[k])

	def test_fsbase_read_generic(self):
		fs = FilesystemTestCaching()
		input_ = {
				1: np.random.rand(30),
				2: np.random.rand(30, 20),
				3: sp.rand(30, 20, 0.2, format='csr'),
				4: sp.rand(30, 20, 0.2, format='csc'),
				5: 2,
				6: 2.,
				7: '2',
				8: {2: 3, 4: 5},
				9: {'part1': np.random.rand(30), 'part2': np.random.rand(30)}
		}
		output_ = fs.read_data(fs.write_data('dir', 'file', input_))
		self.assertTrue( isinstance(output_, dict) )
		for k in input_:
			self.assertTrue( k in output_ )
			self.assertTrue( type(input_[k]) == type(output_[k]) )
			if k <= 2:
				self.assert_vector_equal( input_[k], output_[k] )
			elif k <= 4:
				self.assert_vector_equal( input_[k].indptr, output_[k].indptr )
				self.assert_vector_equal(
						input_[k].indices, output_[k].indices )
				self.assert_vector_equal( input_[k].data, output_[k].data )
			elif k <= 8:
				self.assertTrue( input_[k] == output_[k] )
			else:
				for subk in input_[k]:
					self.assertTrue( subk in output_[k] )
					self.assert_vector_equal(
							input_[k][subk], output_[k][subk] )

class LocaFilesystemTestCase(ConradTestCase):
	def test_lfs_init(self):
		lfs = LocalFilesystem()
		self.assertTrue( isinstance(lfs, ConradFilesystemBase) )

	@classmethod
	def setUpClass(self):
		self.file_tag = 'CONRAD_IO_TEST'
		self.directories = []

	def tearDown(self):
		for f in os.listdir(os.getcwd()):
			if self.file_tag in f:
				os.remove(f)
		while len(self.directories) > 0:
			d = self.directories.pop()
			for f in os.listdir(d):
				os.remove(f)
			if os.path.exists(d):
				os.rmdir(d)

	def test_lfs_checkdir(self):
		lfs = LocalFilesystem()

		d = os.getcwd()
		lfs.check_dir(d)

		with self.assertRaises(OSError):
			lfs.check_dir(os.path.join(d, 'asldksldskjsajasj'))

	def test_lfs_join_mkdir(self):
		lfs = LocalFilesystem()

		d = os.getcwd()

		d1_ = os.path.join(d, 'test')
		self.directories.append(d1_)
		self.assertFalse( os.path.exists(d1_) )
		d1 = lfs.join_mkdir(d, 'test')
		self.assertTrue( d1 == d1_ )
		self.assertTrue( os.path.exists(d1_) )

		d2_ = os.path.join(d, 'test1', 'test2')
		self.directories.append(os.path.join(d, 'test1'))
		self.directories.append(d2_)
		self.assertFalse( os.path.exists(d2_) )
		d2 = lfs.join_mkdir(d, 'test1', 'test2')
		self.assertTrue( d2 == d2_ )
		self.assertTrue( os.path.exists(d2_) )

		d3_ = os.path.join(d2_, 'test3')
		self.directories.append(d3_)
		self.assertFalse( os.path.exists(d3_) )
		d3 = lfs.join_mkdir(d, 'test1', 'test2', 'test2', 'test3')
		self.assertTrue( d3 == d3_ )
		self.assertTrue( os.path.exists(d3_) )

	def test_lfs_read(self):
		lfs = LocalFilesystem()

		f_bad = os.path.join(os.getcwd(), 'sadsadas.npy')
		with self.assertRaises(OSError):
			lfs.read(f_bad, None)

		for suffix in ['', '.jpg', '.yml', '.npx']:
			with self.assertRaises(ValueError):
				f_bad = os.path.join(
						os.getcwd(), 'sadsadas' + self.file_tag + suffix)
				fd = open(f_bad, 'a')
				fd.close()
				lfs.read(f_bad, None)

		with self.assertRaises(ValueError):
			f_bad = os.path.join(
					os.getcwd(), 'asdsdsa' + self.file_tag + '.npz')
			fd = open(f_bad, 'a')
			fd.close()
			lfs.read(f_bad, None)

		v = np.random.rand(30)
		f_ = os.path.join(os.getcwd(), self.file_tag + 'vec')
		v_written = lfs.write(f_, v)
		v_read = lfs.read(v_written['file'], v_written['key'])
		self.assert_vector_equal( v, v_read )

	def test_lfs_write(self):
		lfs = LocalFilesystem()

		f_ = os.path.join(os.getcwd(), self.file_tag + 'vec1')
		f = lfs.write(f_, np.random.rand(30))
		self.assertTrue( f_ in f['file'] )
		self.assertTrue( f['file'].endswith('.npy') )

		with self.assertRaises(OSError):
			f2 = lfs.write(f_, np.random.rand(30))

		f3 = lfs.write(f_, np.random.rand(30), overwrite=True)

		f4 = lfs.write(f_, {'1': np.random.rand(30)})
		self.assertTrue( f_ in f4['1']['file'] )
		self.assertTrue( f4['1']['file'].endswith('.npz') )

	def test_lfs_functionality(self):
		lfs = LocalFilesystem()
		input_ = {
				1: np.random.rand(30),
				2: np.random.rand(30, 20),
				3: sp.rand(30, 20, 0.2, format='csr'),
				4: sp.rand(30, 20, 0.2, format='csc'),
				5: 2,
				6: 2.,
				7: '2',
				8: {2: 3, 4: 5},
				9: {'part1': np.random.rand(30), 'part2': np.random.rand(30)}
		}
		output_ = lfs.read_data(lfs.write_data(
				os.getcwd(), self.file_tag, input_))
		self.assertTrue( isinstance(output_, dict) )
		for k in input_:
			self.assertTrue( k in output_ )
			self.assertTrue( type(input_[k]) == type(output_[k]) )
			if k <= 2:
				self.assert_vector_equal( input_[k], output_[k] )
			elif k <= 4:
				self.assert_vector_equal( input_[k].indptr, output_[k].indptr )
				self.assert_vector_equal(
						input_[k].indices, output_[k].indices )
				self.assert_vector_equal( input_[k].data, output_[k].data )
			elif k <= 8:
				self.assertTrue( input_[k] == output_[k] )
			else:
				for subk in input_[k]:
					self.assertTrue( subk in output_[k] )
					self.assert_vector_equal(
							input_[k][subk], output_[k][subk] )