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
		raise NotImplementedError
	def join_mkdir(self, directory, *subdir):
		raise NotImplementedError
	def read(self, file, key):
		raise NotImplementedError
	def read_all(self, file):
		raise NotImplementedError
	def write(self, file, data, overwrite=False):
		raise NotImplementedError

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
		raise NotImplementedError
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
		raise NotImplementedError

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
		self.assertIsInstance(fs._ConradFilesystemBase__DIGEST, dict )
		for type_ in (
				int, float, str, dict, type(None), DataDictionaryEntry,
				VectorEntry, DenseMatrixEntry, SparseMatrixEntry):
			self.assertIn( type_, fs._ConradFilesystemBase__DIGEST)

		self.assertIsInstance(fs._ConradFilesystemBase__DUMP, dict )
		for type_ in (
				int, float, str, dict, type(None), np.ndarray, sp.csr_matrix,
				sp.csc_matrix):
			self.assertIn( type_, fs._ConradFilesystemBase__DUMP)

	def test_fsbase_abstract(self):
		fs = FilesystemTestBase()
		with self.assertRaises(NotImplementedError):
			fs.check_dir('directory')
		with self.assertRaises(NotImplementedError):
			fs.join_mkdir('directory', 'subdir')
		with self.assertRaises(NotImplementedError):
			fs.read('file', 'key')
		with self.assertRaises(NotImplementedError):
			fs.write('file', 'data')

	def test_fsbase_write_vector(self):
		fs = FilesystemTestNaming()
		ve = fs.write_vector('dir', 'name', np.random.rand(30))
		self.assertIsInstance(ve, VectorEntry )
		self.assertEqual( ve.data_file, 'writing at file `dir/name`' )

		with self.assertRaises(TypeError):
			fs.write_vector(np.random.rand(30, 20))
		with self.assertRaises(TypeError):
			fs.write_vector(
					'dir', 'name', sp.rand(30, 20, 0.2, format='csr'))

	def test_fsbase_write_dense_matrix(self):
		fs = FilesystemTestNaming()
		dme = fs.write_dense_matrix('dir', 'name', np.random.rand(30, 20))
		self.assertIsInstance(dme, DenseMatrixEntry )
		self.assertTrue( dme.layout_rowmajor )
		self.assertEqual( dme.data_file, 'writing at file `dir/name`' )

		with self.assertRaises(TypeError):
			fs.write_dense_matrix(np.random.rand(30))
		with self.assertRaises(TypeError):
			fs.write_dense_matrix(
					'dir', 'name', sp.rand(30, 20, 0.2, format='csr'))

	def test_fsbase_write_sparse_matrix(self):
		fs = FilesystemTestNaming()
		sme = fs.write_sparse_matrix(
				'dir', 'name', sp.rand(30, 20, 0.2, format='csr'))
		self.assertIsInstance(sme, SparseMatrixEntry )
		self.assertTrue( sme.layout_CSR )
		self.assertIsNotNone( sme.layout_fortran_indexing )
		self.assertFalse( sme.layout_fortran_indexing )
		self.assertEqual( sme.shape, (30, 20) )
		self.assertEqual(
				sme.data_pointers_file,
				'writing at file `dir/name_pointers`' )
		self.assertEqual(
				sme.data_indices_file,
				'writing at file `dir/name_indices`' )
		self.assertEqual(
				sme.data_values_file,
				'writing at file `dir/name_values`' )

		sme2 = fs.write_sparse_matrix(
				'dir', 'name', sp.rand(30, 20, 0.2, format='csc'))
		self.assertIsInstance(sme2, SparseMatrixEntry )
		self.assertIsNotNone( sme.layout_CSR )
		self.assertFalse( sme2.layout_CSR )
		self.assertIsNotNone( sme.layout_fortran_indexing )
		self.assertFalse( sme2.layout_fortran_indexing )
		self.assertEqual( sme2.shape, (30, 20) )
		self.assertEqual(
				sme2.data_pointers_file,
				'writing at file `dir/name_pointers`' )
		self.assertEqual(
				sme2.data_indices_file,
				'writing at file `dir/name_indices`' )
		self.assertEqual(
				sme2.data_values_file,
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
		self.assertIsInstance(ve, VectorEntry )
		dme = fs.write_ndarray('dir', 'name', np.random.rand(30, 20))
		self.assertIsInstance(dme, DenseMatrixEntry )

		with self.assertRaises(TypeError):
			fs.write_ndarray(sp.rand(30, 20, 0.2, format='csr'))

	def test_fsbase_write_matrix(self):
		fs = FilesystemTestNaming()

		dme = fs.write_matrix('dir', 'name', np.random.rand(30, 20))
		self.assertIsInstance(dme, DenseMatrixEntry )
		sme = fs.write_matrix(
				'dir', 'name', sp.rand(30, 20, 0.2, format='csr'))
		self.assertIsInstance(sme, SparseMatrixEntry )
		sme2 = fs.write_matrix(
				'dir', 'name', sp.rand(30, 20, 0.2, format='csc'))
		self.assertIsInstance(sme2, SparseMatrixEntry )

		with self.assertRaises(TypeError):
			fs.write_ndarray(np.random.rand(30))

	def test_fsbase_write_data_dictionary(self):
		fs = FilesystemTestNaming()

		dd = {'1': 'string', '2': 'numeric'}
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertEqual( dd, dd_out )

		dd['3'] = np.random.rand(30)
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertIsInstance(dd_out, DataDictionaryEntry )
		self.assertIsInstance(dd_out.entries['3'], VectorEntry )
		self.assertEqual(
				dd_out.entries['3'].data_file,
				'writing at file `dir/name_3`' )

		dd['4'] = np.random.rand(30, 20)
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertIsInstance(dd_out, DataDictionaryEntry )
		self.assertIsInstance(dd_out.entries['4'], DenseMatrixEntry )
		self.assertEqual(
				dd_out.entries['4'].data_file,
				'writing at file `dir/name_4`' )

		dd['5'] = sp.rand(30, 20, 0.2, format='csr')
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertIsInstance(dd_out, DataDictionaryEntry )
		self.assertIsInstance(dd_out.entries['5'], SparseMatrixEntry )
		self.assertEqual(
				dd_out.entries['5'].data_pointers_file,
				'writing at file `dir/name_5_pointers`' )
		self.assertEqual(
				dd_out.entries['5'].data_indices_file,
				'writing at file `dir/name_5_indices`' )
		self.assertEqual(
				dd_out.entries['5'].data_values_file,
				'writing at file `dir/name_5_values`' )

		dd['6'] = {'key': 'value'}
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertIsInstance(dd_out, DataDictionaryEntry )
		self.assertEqual( dd_out.entries['6'], dd['6'] )

		dd['7'] = {'key': np.random.rand(30)}
		dd_out = fs.write_data_dictionary('dir', 'name', dd)
		self.assertIsInstance(dd_out, DataDictionaryEntry )
		self.assertIsInstance(dd_out.entries['7'], DataDictionaryEntry )
		self.assertIsInstance(
				dd_out.entries['7'].entries['key'], VectorEntry )
		self.assertEqual(
				dd_out.entries['7'].entries['key'].data_file,
				'writing at file `dir/name_7_key`' )

	def test_fsbase_write_generic(self):
		fs = FilesystemTestNaming()
		int_val = fs.write_data('dir', 'name', 2)
		self.assertIsInstance(int_val, int )
		flt_val = fs.write_data('dir', 'name', 2.)
		self.assertIsInstance(flt_val, float )
		str_val = fs.write_data('dir', 'name', '2')
		self.assertIsInstance(str_val, str )
		dct_val = fs.write_data('dir', 'name', {'key': 'value'})
		self.assertIsInstance(dct_val, dict )

		with self.assertRaises(TypeError):
			fs.write_data('dir', 'name', [])
		with self.assertRaises(TypeError):
			fs.write_data('dir', 'name', set())

		ve = fs.write_data('dir', 'name', np.random.rand(30))
		self.assertIsInstance(ve, VectorEntry )
		self.assertIn( '`dir/name`', ve.data_file )

		dme = fs.write_data('dir', 'name', np.random.rand(30, 20))
		self.assertIsInstance(dme, DenseMatrixEntry )
		self.assertIn( '`dir/name`', dme.data_file )

		sme = fs.write_data('dir', 'name', sp.rand(30, 20, 0.2, format='csr'))
		self.assertIsInstance(sme, SparseMatrixEntry )
		self.assertIn( '`dir/name_pointers`', sme.data_pointers_file )
		self.assertIn( '`dir/name_indices`', sme.data_indices_file )
		self.assertIn( '`dir/name_values`', sme.data_values_file )

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
		self.assertIsInstance(dde, DataDictionaryEntry )
		self.assertIsInstance( dde.entries['first_key'], VectorEntry )
		self.assertIn(
				'`dir/name_first_key`',
				dde.entries['first_key'].data_file)
		self.assertIsInstance( dde.entries['second_key'], VectorEntry )
		self.assertIn(
				'`dir/name_second_key`',
				dde.entries['second_key'].data_file)
		self.assertIsInstance( dde.entries['third_key'], DenseMatrixEntry )
		self.assertIn(
				'`dir/name_third_key`',
				dde.entries['third_key'].data_file)
		self.assertIsInstance( dde.entries['fourth_key'], SparseMatrixEntry )
		self.assertIn(
				'`dir/name_fourth_key_pointers`',
				dde.entries['fourth_key'].data_pointers_file)
		self.assertIn(
				'`dir/name_fourth_key_indices`',
				dde.entries['fourth_key'].data_indices_file)
		self.assertIn(
				'`dir/name_fourth_key_values`',
				dde.entries['fourth_key'].data_values_file)
		self.assertIsInstance(dde.entries['fifth_key'], dict )
		self.assertIsInstance(dde.entries['sixth_key'], int )
		self.assertIsInstance(dde.entries['seventh_key'], float )
		self.assertIsInstance(dde.entries['eighth_key'], str )
		self.assertIsInstance( dde.entries['ninth_key'], DataDictionaryEntry )
		self.assertIsInstance(
			dde.entries['ninth_key'].entries['key'], VectorEntry )
		self.assertIn(
				'`dir/name_ninth_key_key`',
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
		self.assertIsInstance(mat_back, sp.csr_matrix )
		self.assert_vector_equal(mat.indptr, mat_back.indptr)
		self.assert_vector_equal(mat.indices, mat_back.indices)
		self.assert_vector_equal(mat.data, mat_back.data)

		mat = sp.rand(30, 20, 0.2, format='csc')
		mat_entry = fs.write_sparse_matrix('dir', 'mat2', mat)
		mat_back = fs.to_sparse_matrix(mat_entry)
		self.assertIsInstance(mat_back, sp.csc_matrix )
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
		self.assertIsInstance(dict_back, dict )
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
		self.assertIsInstance(output_, dict )
		for k in input_:
			self.assertIn( k, output_ )
			self.assertEqual( type(input_[k]), type(output_[k]) )
			if k <= 2:
				self.assert_vector_equal( input_[k], output_[k] )
			elif k <= 4:
				self.assert_vector_equal( input_[k].indptr, output_[k].indptr )
				self.assert_vector_equal(
						input_[k].indices, output_[k].indices )
				self.assert_vector_equal( input_[k].data, output_[k].data )
			elif k <= 8:
				self.assertEqual( input_[k], output_[k] )
			else:
				for subk in input_[k]:
					self.assertIn( subk, output_[k] )
					self.assert_vector_equal(
							input_[k][subk], output_[k][subk] )

class LocaFilesystemTestCase(ConradTestCase):
	def test_lfs_init(self):
		lfs = LocalFilesystem()
		self.assertIsInstance(lfs, ConradFilesystemBase )

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
				print("REMOVING {}".format(f))
				if os.path.exists(f):
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
		self.assertEqual( d1, d1_ )
		self.assertTrue( os.path.exists(d1_) )

		d2_ = os.path.join(d, 'test1', 'test2')
		self.directories.append(os.path.join(d, 'test1'))
		self.directories.append(d2_)
		self.assertFalse( os.path.exists(d2_) )
		d2 = lfs.join_mkdir(d, 'test1', 'test2')
		self.assertEqual( d2, d2_ )
		self.assertTrue( os.path.exists(d2_) )

		d3_ = os.path.join(d2_, 'test3')
		self.directories.append(d3_)
		self.assertFalse( os.path.exists(d3_) )
		d3 = lfs.join_mkdir(d, 'test1', 'test2', 'test2', 'test3')
		self.assertEqual( d3, d3_ )
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
		self.assertIn( f_, f['file'] )
		self.assertTrue( f['file'].endswith('.npy') )

		# This assertion is obsolete since OSError was changed to
		# warnings.warn()
		# with self.assertRaises(OSError):
			# f2 = lfs.write(f_, np.random.rand(30))

		f3 = lfs.write(f_, np.random.rand(30), overwrite=True)

		f4 = lfs.write(f_, {'1': np.random.rand(30)})
		self.assertIn( f_, f4['1']['file'] )
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
		self.assertIsInstance(output_, dict )
		for k in input_:
			self.assertIn( k, output_ )
			self.assertEqual( type(input_[k]), type(output_[k]) )
			if k <= 2:
				self.assert_vector_equal( input_[k], output_[k] )
			elif k <= 4:
				self.assert_vector_equal( input_[k].indptr, output_[k].indptr )
				self.assert_vector_equal(
						input_[k].indices, output_[k].indices )
				self.assert_vector_equal( input_[k].data, output_[k].data )
			elif k <= 8:
				self.assertEqual( input_[k], output_[k] )
			else:
				for subk in input_[k]:
					self.assertIn( subk, output_[k] )
					self.assert_vector_equal(
							input_[k][subk], output_[k][subk] )