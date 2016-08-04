from conrad.compat import *
from conrad.tests.base import *

class CaseTestCase(ConradTestCase):
	@classmethod
	def setUpClass(self):
		self.anatomy = Anatomy()
		# self.
	# def setUp(self):
	# 	# Construct dose matrix
	# 	A_targ = 1.2 * np.random.rand(self.m_targ, self.n)
	# 	A_oar = 0.3 * np.random.rand(self.m_oar, self.n)
	# 	self.A = np.vstack((A_targ, A_oar))

	# # Runs once before all unit tests
	# def setUpClass(self):
	# 	mx = 10
	# 	my = 10
	# 	mz = 5
	# 	self.m = mx * my * mz
	# 	self.m_targ = 100
	# 	self.m_oar = self.m - self.m_targ
	# 	self.n = 200

	# 	self.voxels = VoxelGrid(mx, my, mz)
	# 	self.beams = BeamSet(n_beams=self.n)
	# 	self.physics = Physics(beams=self.beams, dose_grid=self.voxels)

	# 	# Structure labels
	# 	self.lab_tum = 0
	# 	self.lab_oar = 1

	# 	# Prescription for each structure
	# 	self.rx = [
	# 			{
	# 				'label': self.lab_tum,
	# 				'name': 'tumor',
	# 				'is_target': True,
	# 				'dose': 1.,
	# 				'constraints': None
	# 			},{
	# 				'label': self.lab_oar,
	# 				'name': 'oar',
	# 				'is_target': False,
	# 				'dose': 0.,
	# 				'constraints': None
	# 			}]

	# 	# Voxel labels on beam matrix
	# 	self.label_order = [self.lab_tum, self.lab_oar]
	# 	self.voxel_labels = [self.lab_tum] * self.m_targ + \
	# 						[self.lab_oar] * self.m_oar

	# 	self.anatomy = Anatomy()
	# 	for s in self.rx:
	# 		self.anatomy += Structure(s['label'], s['name'], s['is_target'],
	# 								  dose=s['dose'])

	# 	self.physics.voxel_labels = self.voxel_labels

	# # Runs once after all unit tests
	# def tearDownClass(self):
	# 	files_to_delete = ['test_plotting.pdf']
	# 	for fname in files_to_delete:
	# 		fpath = path.join(path.abspath(path.dirname(__file__)), fname)
	# 		if path.isfile(fpath): os_remove(fpath)

	# setUpClass = classmethod(setUpClass)
	# tearDownClass = classmethod(tearDownClass)


	def tearDown(self):
		pass
		# self.

	""" Unit tests using example problems. """
	def test_case_init(self):
		pass

	def test_rx_to_anatomy(self):
		pass

	def test_constraint_manipulation(self):
		pass

	def test_objective_manipulation(self):
		pass

	def test_physics_to_anatomy(self):
		pass

	def test_calculate_doses(self):
		pass

	def test_plan(self):
		pass
	# def test_basic(self):
	# 	# Construct unconstrained case
	# 	case = Case(physics=self.physics, anatomy=self.anatomy,
	# 				prescription=self.rx)

	# 	# Add DVH constraints and solve
	# 	case.anatomy['tumor'].constraints += D(20) <= 1.15
	# 	case.anatomy['tumor'].constraints += D(80) >= 0.95
	# 	case.anatomy['oar'].constraints += D(50) < 0.30
	# 	case.anatomy['oar'].constraints += D(10) < 0.55
	# 	run = case.plan()
	# 	print 'solution found in {} seconds\n'.format(run.solvetime)
	# 	print 'dose summary:\n', case.anatomy.dose_summary_string
	# 	print run.x
	# 	print self.A.dot(run.x)

	# def test_2pass_no_constr(self):
	# 	# Construct unconstrained case
	# 	rx = Prescription(self.rx)
	# 	anatomy = Anatomy(prescription=rx)
	# 	case = Case(self.physics, self.anatomy, self.rx)

	# 	# Solve with slack in single pass
	# 	case.plan(solver='ECOS')
	# 	res_x = case.x
	# 	res_obj = case.solver_info['objective']

	# 	# Check results from 2-pass identical if no DVH constraints
	# 	case.plan(solver='ECOS', dvh_exact=True)
	# 	res_x_2pass = case.x
	# 	res_obj_2pass = case.solver_info['objective']
	# 	self.assertItemsEqual(res_x, res_x_2pass)
	# 	self.assertEqual(res_obj, res_obj_2pass)

	# def test_2pass_noslack(self):
	# 	# Construct unconstrained case
	# 	case = Case(physics=self.physics, anatomy=self.anatomy,
	# 				prescription=self.rx)

	# 	# Add DVH constraints and solve
	# 	case.anatomy['tumor'].constraints += D(20) <= 1.15
	# 	case.anatomy['tumor'].constraints += D(80) >= 0.95
	# 	case.anatomy['oar'].constraints += D(50) < 0.30
	# 	run1 = case.plan(solver='ECOS', dvh_slack=False)
	# 	res_obj = case.problem.solver.objective.value

	# 	# Check objective from 2nd pass <= 1st pass
	# 	# (since 1st constraints more restrictive)
	# 	run2 = case.plan(solver='ECOS', dvh_slack=False, dvh_exact=True)
	# 	res_obj_2pass = case.problem.solver.objective.value
	# 	self.assertTrue(res_obj_2pass <= res_obj)


	def test_plotting_data(self):
		pass

