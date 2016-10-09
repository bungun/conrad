"""
Define :class:`Case`, the top level interface for treatment planning.
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

from operator import add
from warnings import warn
from numpy import ndarray, array, squeeze, zeros, nan

from conrad.physics import Physics
from conrad.medicine import Anatomy, Prescription
from conrad.optimization.problem import PlanningProblem
from conrad.optimization.history import RunRecord, PlanningHistory

class Case(object):
	"""
	Top level interface for treatment planning.

	A :class:`Case` has four major components.

	:attr:`Case.physics` is of type :class:`Physics`, and contains
	physical information for the case, including the number of
	voxels, beams, beam layout, voxel labels and dose influence
	matrix.

	:attr:`Case.anatomy` is of type :class:`Antomy`, and manages the
	structures in the patient anatomy, including optimization
	objectives and dose constraints applied to each structure.

	:attr:`Case.prescription` is of type :class:`Prescription`, and
	specifies a clinical prescription for the case, including prescribed
	doses for target structures and prescribed dose constraints (e.g.,
	RTOG recommendations).

	:attr:`Case.problem` is of type :class:`PlanningProblem`, and is a
	tool that forms and manages the mathematical representation of
	treatment planning problem specified by case anatomy, physics and
	prescription; it serves as the interface to convex solvers that run
	the treatment plan optimization.
	"""

	def __init__(self, anatomy=None, physics=None, prescription=None,
				 suppress_rx_constraints=False):
		"""
		Initialize case with anatomy, physics and prescription data.

		If ``prescription`` provided and ``anatomy`` not provided,
		:attr:`Case.anatomy` populated with structures from
		``prescription``.

		Arguments:
			anatomy (optional): Must be compatible with :class:`Anatomy`
				initializer.
			physics (optional): Must be compatible with :class:`Physics`
				initializer.
			prescription (optional): Must be compatible with
				:class:`Presription` initializer; i.e., can be
				:class:`Prescription`, a suitably formatted :obj:`list`
				with prescription data, or the path to a valid JSON or
				YAML file with suitably formatted prescription data.
			suppress_rx_constraints (:obj:`bool`, optional): Suppress
				constraints in ``prescription`` from being attached to
				structures in :attr:`Case.anatomy`.
		"""
		self.__physics = None
		self.__anatomy = None
		self.__prescription = None
		self.__problem = None

		self.physics = physics
		self.anatomy = anatomy
		self.prescription = prescription
		self.__problem = PlanningProblem()

		# append prescription constraints unless suppressed:
		if not suppress_rx_constraints:
			self.transfer_rx_constraints_to_anatomy()

	@property
	def physics(self):
		""" Patient anatomy, contains all dose physics information. """
		return self.__physics

	@physics.setter
	def physics(self, physics):
		self.__physics = Physics(physics)

	@property
	def anatomy(self):
		""" Container for all planning structures. """
		return self.__anatomy

	@anatomy.setter
	def anatomy(self, anatomy):
		self.__anatomy = Anatomy(anatomy)

	@property
	def prescription(self):
		"""
		Container for clinical goals and limits.

		Structure list from prescription used to populate
		:attr:`Case.anatomy` if anatomy is empty when
		:attr:`Case.prescription` setter is invoked.
		"""
		return self.__prescription

	@prescription.setter
	def prescription(self, prescription):
		self.__prescription = Prescription(prescription)
		if self.anatomy.is_empty:
			self.anatomy.structures = self.prescription.structure_dict

	@property
	def problem(self):
		""" Object managing numerical optimization setup and results. """
		return self.__problem

	@property
	def structures(self):
		""" Dictionary of structures contained in :attr:`Case.anatomy`. """
		return self.anatomy.structures

	@property
	def A(self):
		"""
		Dose matrix from current planning frame of :attr:`Case.physics`.
		"""
		if self.physics is None:
			return None
		return self.physics.dose_matrix

	@property
	def n_structures(self):
		""" Number of structures in :attr:`Case.anatomy`. """
		return self.anatomy.n_structures

	@property
	def n_voxels(self):
		"""
		Number of voxels in current planning frame of :attr:`Case.physics`.
		"""
		if self.physics.voxels is nan:
			return None
		return self.physics.voxels

	@property
	def n_beams(self):
		"""
		Number of beams in current planning frame of :attr:`Case.physics`.
		"""
		if self.physics.beams is nan:
			return None
		return self.physics.beams

	def transfer_rx_constraints_to_anatomy(self):
		"""
		Push constraints in prescription onto structures in anatomy.

		Assume each structure label represented in
		:attr:`Case.prescription` is represented in
		:attr:`Case.anatomy`. Any existing constraints on structures in
		:attr:`Case.anatomy` are preserved.

		Arguments:
			None

		Returns:
			None
		"""
		constraint_dict = self.prescription.constraints_by_label
		for structure_label, constr_list in constraint_dict.items():
			self.anatomy[structure_label].constraints += constr_list

	def add_constraint(self, structure_label, constraint):
		"""
		Add ``constraint`` to structure specified by ``structure_label``.

		Arguments:
			structure_label: Must correspond to label or name of a
				:class:`~conrad.medicine.Structure` in
				:attr:`Case.anatomy`.
			constraint (:class:`conrad.medicine.Constraint`): Dose
				constraint to add to constraint list of specified
				structure.

		Returns:
			None
		"""
		self.anatomy[structure_label].constraints += constraint
		return self.anatomy[structure_label].constraints.last_key

	def drop_constraint(self, constr_id):
		"""
		Remove constraint from case.

		If ``constr_id`` is a valid key to a constraint in the
		:class:`~conrad.medicine.dose.ConstraintList` attached to one of
		the structures in :attr:`Case.anatomy`, that constraint will be
		removed from the structure's constraint list. Call is no-op if
		key does not exist.

		Arguments:
			constr_id: Key to a constraint on one of the structures in
				:attr:`Case.anatomy`.

		Returns:
			None
		"""
		for s in self.anatomy:
			if constr_id in s.constraints:
				s.constraints -= constr_id
				break

	def clear_constraints(self):
		"""
		Remove all constraints from all structures in :class:`Case`.

		Arguments:
			None

		Returns:
			None
		"""
		self.anatomy.clear_constraints()

	def change_constraint(self, constr_id, threshold=None, direction=None,
						  dose=None):
		"""
		Modify constraint in :class:`Case`.

		If ``constr_id`` is a valid key to a constraint in the
		:class:`~conrad.medicine.dose.ConstraintList` attached to one of
		the structures in :attr:`Case.anatomy`, that constraint will be
		modified according to the remaining arguments. Call is no-op if
		key does not exist.

		Arguments:
			constr_id: Key to a constraint on one of the structures in
				:attr:`Case.anatomy`.
			threshold (optional): If constraint in question is a
				:class:`~conrad.medicine.dose.PercentileConstraint`,
				percentile threshold set to this value. No effect
				otherwise.
			direction (:obj:`str`, optional): Constraint direction set
				to this value. Should be one of: '<' or '>'.
			dose (:class:`~conrad.physics.units.DeliveredDose`, optional):
				Constraint dose level set to this value.

		Returns:
			None

		"""
		for s in self.anatomy:
			if constr_id in s.constraints:
				s.set_constraint(constr_id, threshold, direction, dose)
				break

	def change_objective(self, label, dose=None, w_under=None, w_over=None):
		"""
		Modify objective for structure in :class:`Case`.

		Arguments:
			label: Label or name of a :class:`~conrad.medicine.Structure`
				in :attr:`Case.anatomy`.
			dose (:class:`~conrad.physics.units.DeliveredDose`, optional):
				Set target dose for structure.
			w_under (:obj:`float`, optional): Set underdose weight for
				structure.
			w_over (:obj:`float`, optional): Set overdose weight for
				structure.

		Returns:
			None
		"""
		if self.anatomy[label].is_target:
			if dose is not None:
				self.anatomy[label].dose = dose
			if w_under is not None:
				self.anatomy[label].w_under = w_under
		if w_over is not None:
			self.anatomy[label].w_over = w_over

	def load_physics_to_anatomy(self, overwrite=False):
		"""
		Transfer data from physics to each structure.

		The label associated with each structure in :attr:`Case.anatomy`
		is used to retrieve the dose matrix data and voxel weights from
		:attr:`Case.physics` for the voxels bearing that label.

		The method marks the :attr:`Case.physics.dose_matrix` as seen,
		in order to prevent redundant data transfers.

		Arguments:
			overwrite(:obj:`bool`, optional): If ``True``, dose matrix
				data from :attr:`Case.physics` will overwrite dose
				matrices assigned to each structure in
				:attr:`Case.anatomy`.

		Returns:
			None

		Raises:
			ValueError: If :attr:`Case.anatomy` has assigned dose
				matrices, :attr:`Case.physics` not marked as having
				updated dose matrix data, and flag ``overwrite`` set to
				``False``.

		"""
		if not overwrite:
			if self.anatomy.plannable and self.physics.data_loaded:
				raise ValueError('case.anatomy already has valid dose '
								 'matrix data loaded to each structure.\n'
								 'call this method with option '
								 '"overwrite=True" to load new dose matrix '
								 'data')
			if self.physics.data_loaded:
				return

		for structure in self.anatomy:
			label = structure.label
			structure.A_full = self.physics.dose_matrix_by_label(label)
			structure.voxel_weights = self.physics.voxel_weights_by_label(
					label)
		self.physics.mark_data_as_loaded()

	def calculate_doses(self, x):
		"""
		Calculate voxel doses for each structure in :attr:`Case.anatomy`.

		Arguments:
			x: Vector-like array of beam intensities.

		Returns:
			None
		"""
		self.anatomy.calculate_doses(x)

	@property
	def plannable(self):
		"""
		``True`` if case meets minimum requirements for :meth:`Case.plan` call.

		Arguments:
			None

		Returns:
			:obj:`bool`: ``True`` if anatomy has one or more target
			structures and dose matrices from the case physics.
		"""
		plannable = True
		if not self.physics.plannable:
			plannable &= False
		elif not self.anatomy.plannable:
			self.load_physics_to_anatomy()
			plannable &= self.anatomy.plannable
		return plannable

	def plan(self, use_slack=True, use_2pass=False, **options):
		"""
		Invoke numerical solver to optimize plan, given state of :class:`Case`.

		At call time, the objectives, dose constraints, dose matrix,
		and other relevant data associated with each structure in
		:attr:`Case.anatomy` is passed to :attr:`Case.problem` to build
		and solve a convex optimization problem.

		Arguments:
			use_slack (:obj:`bool`, optional): Allow slacks on each dose
				constraint.
			use_2pass (:obj:`bool`, optional): Execute two-pass planing
				method to enforce exact versions, rather than convex
				restrictions of any percentile-type dose constraints
				included in the plan.
			**options: Arbitrary keyword arguments.

		Returns:
			:obj:`tuple`: Tuple with :obj:`bool` indicator of planning
			problem feasibility and a
			:class:`~conrad.optimization.history.RunRecord` with data
			from the setup, execution and output of the planning run.

		Raises:
			ValueError: If case not plannable due to missing information.
		"""
		if not self.plannable:
			raise ValueError('case not plannable in current state.\n'
							 'minimum requirements:\n'
							 '---------------------\n'
							 '-"case.physics.dose_matrix" is set\n'
							 '-"case.physics.voxel_labels" is set\n'
							 '-"case.anatomy" contains at least one '
							 'structure marked as target')


		# two pass planning for DVH constraints: OFF by default
		use_2pass = options.pop('dvh_exact', use_2pass)
		# dose constraint slack: ON by default
		use_slack = options.pop('dvh_slack', use_slack)

		# objective weight for slack minimization
		gamma = options['gamma'] = options.pop('slack_penalty', None)

		run = RunRecord(
				self.anatomy.list,
				use_2pass=use_2pass,
				use_slack=use_slack,
				gamma=gamma)

		# solve problem
		feas = self.problem.solve(self.anatomy.list, run.output,
								 slack=use_slack, exact_constraints=use_2pass,
								 **options)


		# update doses
		if run.feasible:
			run.plotting_data[0] = self.plotting_data(x=run.x)
			if use_2pass:
				run.plotting_data['exact'] = self.plotting_data(x=run.x_exact)
		else:
			warn('Problem infeasible as formulated')

		status = (feas == int(1 + int(use_2pass)))
		return status, run

	def plotting_data(self, x=None, constraints_only=False, maxlength=None):
		"""
		Dictionary of :mod:`matplotlib`-compatible plotting data.

		Includes data for dose volume histograms, prescribed doses, and
		dose volume (percentile) constraints for each structure in
		:attr:`Case.anatomy`.

		Arguments:
			x (optional): Vector of beam intensities from which to
				calculate structure doses prior to emitting plotting
				data.
			constraints_only (:obj:`bool`, optional): If ``True``, only
				include each structure's constraint data in returned
				dictionary.
			maxlength (:obj:`int`, optional): If specified, re-sample
				each structure's DVH plotting data to have a maximum
				series length of ``maxlength``.

		Returns:
			:obj:`dict`: Plotting data for each structure, keyed by
			structure label.
		"""
		if constraints_only:
			return self.anatomy.plotting_data(constraints_only=True)
		else:
			if x is not None:
				self.calculate_doses(x)
			return self.anatomy.plotting_data(maxlength=maxlength)