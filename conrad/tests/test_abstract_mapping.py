"""
Unit tests for :mod:`conrad.abstract.mapping`.
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

from conrad.abstract.mapping import *
from conrad.tests.base import *

class DiscreteMappingTestCase(ConradTestCase):
	def test_discrete_mapping_init(self):
		dmap = DiscreteMapping(range(10))

		self.assertIsInstance( dmap.vec, np.ndarray )
		self.assert_vector_equal( dmap.vec, [i for i in xrange(10)] )
		self.assertEqual( dmap.n_frame0, 10 )
		self.assertEqual( dmap.n_frame1, 10 )

		dmap = DiscreteMapping([0, 3, 5])
		self.assertEqual( dmap.n_frame0, 3 )
		self.assertEqual( dmap.n_frame1, 6 )

		for i in xrange(dmap.n_frame0):
			self.assertEqual( dmap[i], dmap.vec[i] )

	def test_discrete_mapping_01(self):
		dmap = DiscreteMapping([1, 3, 5])
		reverse = {1: 0, 3: 1, 5: 2}

		vec_0 = np.random.rand(dmap.n_frame0)
		vec_1 = np.zeros(dmap.n_frame1)

		for clear, t in [(False, 1), (False, 2), (True, 1)]:
			dmap.frame0_to_1_inplace(vec_0, vec_1, clear_output=clear)
			vec_1a = dmap.frame0_to_1(vec_0)

			self.assertIsInstance( vec_1a, np.ndarray )
			for index in range(dmap.n_frame1):
				if index in reverse:
					self.assertEqual( vec_1[index], t * vec_0[reverse[index]] )
					self.assertEqual( vec_1a[index], vec_0[reverse[index]] )
				else:
					self.assertEqual( vec_1[index], 0 )
					self.assertEqual( vec_1a[index], 0 )
			vec_1a = None

		n = 5
		mat_0 = np.random.rand(dmap.n_frame0, n)
		mat_1 = np.zeros((dmap.n_frame1, n))

		for clear, t in [(False, 1), (False, 2), (True, 1)]:
			dmap.frame0_to_1_inplace(mat_0, mat_1, clear_output=clear)
			mat_1a = dmap.frame0_to_1(mat_0)
			self.assertIsInstance( mat_1a, np.ndarray )
			for index in range(dmap.n_frame1):
				if index in reverse:
					self.assert_vector_equal(
							mat_1[index, :], t * mat_0[reverse[index], :])
					self.assert_vector_equal(
							mat_1a[index, :], mat_0[reverse[index], :])

				else:
					self.assertEqual( np.sum(mat_1[index, :] != 0), 0 )
					self.assertEqual( np.sum(mat_1a[index, :] != 0), 0 )
			mat_1a = None

		dmap = DiscreteMapping([0, 1, 1])
		i = np.random.rand(3)
		o = dmap.frame0_to_1(i)
		self.assertEqual( o[0], i[0] )
		self.assertEqual( o[1], i[1] + i[2] )

		i = np.random.rand(3, 4)
		o = dmap.frame0_to_1(i)
		self.assert_vector_equal( o[0, :], i[0, :] )
		self.assert_vector_equal( o[1, :], i[1, :] + i[2, :] )

		with self.assertRaises(TypeError):
			dmap.frame0_to_1_inplace(vec_0, mat_1)
		with self.assertRaises(TypeError):
			dmap.frame0_to_1_inplace(mat_1, vec_0)
		with self.assertRaises(ValueError):
			dmap.frame0_to_1_inplace(vec_1, np.zeros(2 * dmap.n_frame1))

	def test_discrete_mapping_10(self):
		dmap = DiscreteMapping([1, 3, 5])
		reverse = {1: 0, 3: 1, 5: 2}

		vec_0 = np.zeros(dmap.n_frame0)
		vec_1 = np.random.rand(dmap.n_frame1)

		for clear, t in [(False, 1), (False, 2), (True, 1)]:
			dmap.frame1_to_0_inplace(vec_1, vec_0, clear_output=clear)
			vec_0a = dmap.frame1_to_0(vec_1)

			self.assertIsInstance( vec_0a, np.ndarray )
			for index in range(dmap.n_frame1):
				if index in reverse:
					self.assertEqual( vec_0[reverse[index]], t * vec_1[index] )
					self.assertEqual( vec_0a[reverse[index]], vec_1[index] )
			vec_0a = None

		n = 5
		mat_0 = np.zeros((dmap.n_frame0, n))
		mat_1 = np.random.rand(dmap.n_frame1, n)

		for clear, t in [(False, 1), (False, 2), (True, 1)]:
			dmap.frame1_to_0_inplace(mat_1, mat_0, clear_output=clear)
			mat_0a = dmap.frame1_to_0(mat_1)
			self.assertIsInstance( mat_0a, np.ndarray )
			for index in range(dmap.n_frame1):
				if index in reverse:
					self.assert_vector_equal(
							mat_0[reverse[index], :], t * mat_1[index, :])
					self.assert_vector_equal(
							mat_0a[reverse[index], :], mat_1[index, :])

			mat_1a = None

		dmap = DiscreteMapping([0, 1, 1])
		i = np.random.rand(2)
		o = dmap.frame1_to_0(i)
		self.assertEqual( o[0], i[0] )
		self.assertEqual( o[1], i[1] )
		self.assertEqual( o[2], i[1] )

		i = np.random.rand(2, 4)
		o = dmap.frame1_to_0(i)
		self.assert_vector_equal( o[0, :], i[0, :] )
		self.assert_vector_equal( o[1, :], i[1, :] )
		self.assert_vector_equal( o[2, :], i[1, :] )

		with self.assertRaises(TypeError):
			dmap.frame1_to_0_inplace(vec_1, mat_0)
		with self.assertRaises(TypeError):
			dmap.frame1_to_0_inplace(mat_1, vec_0)
		with self.assertRaises(ValueError):
			dmap.frame1_to_0_inplace(vec_1, np.zeros(2 * dmap.n_frame0))

class ClusterMappingTestCase(ConradTestCase):
	def test_cluster_mapping_init(self):
		cmap = ClusterMapping([1, 2, 2, 3, 3, 3])
		self.assertEqual( cmap.n_points, 6 )
		self.assertEqual( cmap.n_clusters, 4 )
		self.assert_vector_equal( cmap.cluster_weights, np.array([range(4)]) )

	def test_cluster_mapping_rescale_points(self):
		cmap = ClusterMapping([1, 2, 2, 3, 3, 3])

		vec = np.random.rand(cmap.n_points)
		expect = 1 * vec
		for idx, weight in enumerate(map(float, [1, 2, 2, 3, 3, 3])):
			expect[idx] *= 1. / weight
		cmap._ClusterMapping__rescale_len_points(vec)
		self.assert_vector_equal( vec, expect )

		mat = np.random.rand(cmap.n_points, 5)
		expect = 1 * mat
		for idx, weight in enumerate(map(float, [1, 2, 2, 3, 3, 3])):
			expect[idx, :] *= 1. / weight
		cmap._ClusterMapping__rescale_len_points(mat)
		self.assert_vector_equal( mat, expect )

	def test_cluster_mapping_rescale_clusters(self):
		cmap = ClusterMapping([1, 2, 2, 3, 3, 3])

		vec = np.random.rand(cmap.n_clusters)
		vec[0] = 0
		expect = 1 * vec
		for idx, weight in enumerate(map(float, [0, 1, 2, 3])):
			if weight > 0:
				expect[idx] *= 1. / weight
		cmap._ClusterMapping__rescale_len_clusters(vec)
		self.assert_vector_equal( vec, expect )

		mat = np.random.rand(cmap.n_clusters, 5)
		mat[0, :] = 0

		expect = 1 * mat
		for idx, weight in enumerate(map(float, [0, 1, 2, 3])):
			if weight > 0:
				expect[idx, :] *= 1. / weight
		cmap._ClusterMapping__rescale_len_clusters(mat)
		self.assert_vector_equal( mat, expect )

	def test_cluster_mapping_downsample(self):
		cmap = ClusterMapping([0, 1, 1, 2, 2, 2])

		pts = np.ones(cmap.n_points)
		clus = np.zeros(cmap.n_clusters)

		for rescale, expect in [(True, [1, 1, 1]), (False, [1, 2, 3])]:
			# in-place
			cmap.downsample_inplace(
					pts, clus, rescale_output=rescale, clear_output=True)
			self.assert_vector_equal( clus, expect )

			# allocating
			cgen = cmap.downsample(pts, rescale_output=rescale)
			self.assert_vector_equal( cgen, expect )

		n = 5
		pts = np.ones((cmap.n_points, n))
		clus = np.zeros((cmap.n_clusters, n))

		for rescale, expect in [(True, [1, 1, 1]), (False, [1, 2, 3])]:
			# in-place
			cmap.downsample_inplace(
					pts, clus, rescale_output=rescale, clear_output=True)
			for i, e in enumerate(expect):
				self.assert_vector_equal( clus[i, :], e * np.ones(n) )

			# allocating
			cgen = cmap.downsample(pts, rescale_output=rescale)
			for i, e in enumerate(expect):
				self.assert_vector_equal( cgen[i, :], e * np.ones(n) )

	def test_cluster_mapping_upsample(self):
		cmap = ClusterMapping([0, 1, 1, 2, 2, 2])

		pts = np.zeros(cmap.n_points)
		clus = np.ones(cmap.n_clusters)

		expect_a = 1. / (cmap.vec + 1)
		expect_b = np.ones(cmap.n_points)

		for rescale, expect in [(True, expect_a), (False, expect_b)]:
			# in-place
			cmap.upsample_inplace(
					clus, pts, rescale_output=rescale, clear_output=True)
			self.assert_vector_equal( pts, expect )

			# allocating
			pgen = cmap.upsample(clus, rescale_output=rescale)
			self.assert_vector_equal( pgen, expect )

		n = 5
		pts = np.zeros((cmap.n_points, n))
		clus = np.ones((cmap.n_clusters, n))

		for rescale, expect in [(True, expect_a), (False, expect_b)]:
			# in-place
			cmap.upsample_inplace(
					clus, pts, rescale_output=rescale, clear_output=True)

			for i, e in enumerate(expect):
				self.assert_vector_equal( pts[i, :], e * np.ones(n) )

			# allocating
			pgen = cmap.upsample(clus, rescale_output=rescale)
			for i, e in enumerate(expect):
				self.assert_vector_equal( pgen[i, :], e * np.ones(n) )

		ri = np.random.rand(cmap.n_clusters)
		ro = cmap.downsample(cmap.upsample(ri))
		self.assert_vector_equal( ri, ro )

	def test_cluster_mapping_to_contiguous(self):
		cmap = ClusterMapping([0, 1, 1, 2])
		self.assertIs( cmap.contiguous, cmap )

		cmap = ClusterMapping([1, 3, 3, 5, 3])
		cmap_c = cmap.contiguous
		self.assertIsNot( cmap_c, cmap )

		self.assert_vector_equal( cmap_c.vec, [0, 1, 1, 2, 1] )

class PermutationMappingTestCase(ConradTestCase):
	def test_permutation_mapping_init(self):
		PermutationMapping(range(10))
		PermutationMapping(np.random.rand(10).argsort())
		with self.assertRaises(ValueError):
			PermutationMapping([1, 3, 4, 5])
		with self.assertRaises(ValueError):
			PermutationMapping([1, 1, 2, 3, 4, 5])

class IdentityMappingTestCase(ConradTestCase):
	def test_identiy_mapping_init(self):
		im = IdentityMapping(10)
		self.assertEqual( im.n_frame0, 10 )
		self.assertEqual( im.n_frame1, 10 )
		self.assert_vector_equal( im.vec, xrange(10) )
		x = np.random.rand(10)
		y = np.zeros(10)
		for f in (im.frame0_to_1, im.frame1_to_0):
			self.assert_vector_equal( x, f(x) )
		for f in (im.frame0_to_1_inplace, im.frame1_to_0_inplace):
			self.assert_vector_equal( x, f(x, y, clear_output=True) )

class DictionaryMappingTestCase(ConradTestCase):
	def test_dictionary_mapping_init(self):
		m0 = IdentityMapping(5)
		m1 = DiscreteMapping([0, 4])
		m2 = ClusterMapping([0, 0, 1, 2, 3, 3, 4])
		m3 = PermutationMapping([0, 3, 2, 1, 4])
		dm = DictionaryMapping({0: m0, 1: m1, 2: m2, 3: m3})

		maps = [m0, m1, m2, m3]
		sizes = [m.n_frame0 for m in maps]
		out_sizes = [m.n_frame1 for m in maps]
		N_maps = len(maps)

		# key order
		self.assert_vector_equal( dm.key_order, [0, 1, 2, 3] )

		# contains
		for i in xrange(4):
			self.assertIn( i, dm )

		# getitem
		for i in xrange(4):
			self.assertIs( dm[i], maps[i] )

		vec = [0, 1, 2, 3, 4, 5, 9, 10, 10, 11, 12, 13, 13, 14, 15, 18, 17, 16, 19]
		self.assertEqual( dm.n_frame0, sum(sizes) )
		self.assertEqual( dm.n_frame1, sum(out_sizes) )
		self.assertIsInstance( dm.concatenated_map, DiscreteMapping )
		self.assert_vector_equal( dm.concatenated_map.vec, vec )

		x0 = {i: np.random.rand(sizes[i]) for i in xrange(N_maps)}
		y1_expect = {i: maps[i].frame0_to_1(x0[i]) for i in xrange(N_maps)}
		y1 = dm.frame0_to_1(x0)
		for key in x0:
			self.assert_vector_equal( y1_expect[key], y1[key] )

		y1_expect[2] = m2.downsample(x0[2], rescale_output=True)
		dm.frame0_to_1_inplace(x0, y1, rescale_output=True, clear_output=True)
		for key in x0:
			self.assert_vector_equal( y1_expect[key], y1[key] )

		x1 = {i: np.random.rand(5) for i in xrange(N_maps)}
		y0_expect = {i: maps[i].frame1_to_0(x1[i]) for i in xrange(N_maps)}
		y0 = dm.frame1_to_0(x1)
		for key in x1:
			self.assert_vector_equal( y0_expect[key], y0[key] )

		y0_expect[2] = m2.upsample(x1[2], rescale_output=True)
		dm.frame1_to_0_inplace(x1, y0, rescale_output=True, clear_output=True)
		for key in x1:
			self.assert_vector_equal( y0_expect[key], y0[key] )

class DictionaryClusterMappingTestCase(ConradTestCase):
	def test_dictionary_cluster_mapping_init(self):
		m0 = IdentityMapping(5)
		m1 = DiscreteMapping([0, 1, 4])
		m2 = ClusterMapping([0, 0, 1, 2, 3, 3, 4])
		m3 = PermutationMapping([0, 3, 2, 1, 4])
		with self.assertRaises(TypeError):
			dm = DictionaryClusterMapping({0: m0, 1: m1, 2: m2, 3: m3})
		dm = DictionaryClusterMapping({0: m0, 2: m2, 3: m3})

		maps = [m0, m2, m3]
		keys = [0, 2, 3]
		sizes = [m.n_frame0 for m in maps]
		out_sizes = [m.n_frame1 for m in maps]
		N_maps = len(maps)

		vec = [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 13, 12, 11, 14]
		self.assertEqual( dm.n_frame0, 17 )
		self.assertEqual( dm.n_frame1, 15 )
		self.assertIsInstance( dm.concatenated_cluster_map, ClusterMapping )
		self.assert_vector_equal( dm.concatenated_map.vec, vec )

		x0 = {keys[i]: np.random.rand(sizes[i]) for i in xrange(N_maps)}
		y1_expect = {
				keys[i]: maps[i].frame0_to_1(x0[keys[i]])
				for i in xrange(N_maps)}
		y1 = dm.downsample(x0)
		for key in x0:
			self.assert_vector_equal( y1_expect[key], y1[key] )

		y1_expect[2] = m2.downsample(x0[2], rescale_output=True)
		dm.downsample_inplace(x0, y1, rescale_output=True, clear_output=True)
		for key in x0:
			self.assert_vector_equal( y1_expect[key], y1[key] )

		x1 = {keys[i]: np.random.rand(5) for i in xrange(N_maps)}
		y0_expect = {
				keys[i]: maps[i].frame1_to_0(x1[keys[i]])
				for i in xrange(N_maps)}
		y0 = dm.upsample(x1)
		for key in x1:
			self.assert_vector_equal( y0_expect[key], y0[key] )

		y0_expect[2] = m2.upsample(x1[2], rescale_output=True)
		dm.upsample_inplace(x1, y0, rescale_output=True, clear_output=True)
		for key in x1:
			self.assert_vector_equal( y0_expect[key], y0[key] )

class MappingMethodsTestCase(ConradTestCase):
	def test_map_type_to_string(self):
		self.assertEqual(
				map_type_to_string(PermutationMapping(range(10))),
				'permutation' )
		self.assertEqual(
				map_type_to_string(ClusterMapping(range(10))),
				'cluster' )
		self.assertEqual(
				map_type_to_string(DiscreteMapping(range(10))), 'discrete' )
		self.assertEqual(
				map_type_to_string(IdentityMapping(10)), 'identity' )
		dmap = DictionaryMapping({0: DiscreteMapping(range(10))})
		self.assertEqual( map_type_to_string(dmap), 'dictionary' )
		dcmap = DictionaryClusterMapping({0: ClusterMapping(range(10))})
		self.assertEqual( map_type_to_string(dcmap), 'dictionary' )
		with self.assertRaises(TypeError):
			map_type_to_string(range(10))

	def test_string_to_map_constructor(self):
		self.assertEqual(
				string_to_map_constructor('permutation'), PermutationMapping )
		self.assertEqual(
				string_to_map_constructor('cluster'), ClusterMapping )
		self.assertEqual(
				string_to_map_constructor('identity'), IdentityMapping )
		self.assertEqual(
				string_to_map_constructor('dictionary'), DictionaryMapping )
		self.assertEqual(
				string_to_map_constructor('cluster dictionary'),
				DictionaryClusterMapping )
		self.assertEqual(
				string_to_map_constructor(''), DiscreteMapping )