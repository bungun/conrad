from conrad.physics.units import cm, mm
from conrad.physics.beams import *
from conrad.tests.base import *

class BeamsTestCase(ConradTestCase):
	def test_bixel_grid(self):
		nx = 12
		ny = 9
		n = nx * ny

		dx = 1 * cm
		dy = 0.5 * cm

		g = BixelGrid(nx, ny)
		self.assertEqual(g.bixels, n)

		idx = int(n * rand())
		g.set_order('xy')
		self.assertEqual(g.position2index(*g.index2position(idx)), idx)
		g.set_order('yx')
		self.assertEqual(g.position2index(*g.index2position(idx)), idx)

		indices = (n * rand(10)).astype(int)
		idx_recovered = listmap(
				lambda pos : g.position2index(*pos),
				listmap(g.index2position, indices))
		self.assertTrue(
				all(listmap(lambda a, b : a == b, indices, idx_recovered)))

	def test_beam_set(self):
		pass