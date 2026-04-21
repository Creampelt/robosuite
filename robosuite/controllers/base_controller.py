from __future__ import annotations

import abc
from collections.abc import Iterable

import mujoco
import numpy as np
import torch

import robosuite.macros as macros
from robosuite.utils.binding_utils import MjSimWarp


class Controller(object, metaclass=abc.ABCMeta):
    """
    General controller interface.

    Requires reference to mujoco sim object, eef_name of specific robot, relevant joint_indexes to that robot, and
    whether an initial_joint is used for nullspace torques or not

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range
    """

    def __init__(
        self,
        sim,
        eef_name,
        joint_indexes,
        actuator_range,
    ):

        # Actuator range
        self.actuator_min = actuator_range[0]
        self.actuator_max = actuator_range[1]

        # Attributes for scaling / clipping inputs to outputs
        self.action_scale = None
        self.action_input_transform = None
        self.action_output_transform = None

        # Private property attributes
        self.control_dim = None
        self.output_min = None
        self.output_max = None
        self.input_min = None
        self.input_max = None

        # mujoco simulator state
        self.sim = sim
        self.model_timestep = macros.SIMULATION_TIMESTEP
        self.eef_name = eef_name
        self.joint_index = joint_indexes["joints"]
        self.qpos_index = joint_indexes["qpos"]
        self.qvel_index = joint_indexes["qvel"]

        # robot states
        self.ee_pos = None
        self.ee_ori_mat = None
        self.ee_pos_vel = None
        self.ee_ori_vel = None
        self.joint_pos = None
        self.joint_vel = None

        # dynamics and kinematics
        self.J_pos = None
        self.J_ori = None
        self.J_full = None
        self.mass_matrix = None

        # Joint dimension
        self.joint_dim = len(joint_indexes["joints"])

        # Torques being outputted by the controller
        self.torques = None

        # Update flag to prevent redundant update calls
        self.new_update = True

        # Move forward one timestep to propagate updates before taking first update
        self.sim.forward()

        # Initialize controller by updating internal state and setting the initial joint, pos, and ori
        self.update()
        self.initial_joint, self.initial_ee_pos, self.initial_ee_ori_mat = self._extract_initial_state()

    @abc.abstractmethod
    def run_controller(self):
        """
        Abstract method that should be implemented in all subclass controllers, and should convert a given action
        into torques (pre gravity compensation) to be executed on the robot.
        Additionally, resets the self.new_update flag so that the next self.update call will occur
        """
        self.new_update = True

    def scale_action(self, action):
        """
        Clips @action to be within self.input_min and self.input_max, and then re-scale the values to be within
        the range self.output_min and self.output_max

        Args:
            action (Iterable): Actions to scale

        Returns:
            np.array: Re-scaled action
        """

        if self.action_scale is None:
            self.action_scale = abs(self.output_max - self.output_min) / abs(self.input_max - self.input_min)
            self.action_output_transform = (self.output_max + self.output_min) / 2.0
            self.action_input_transform = (self.input_max + self.input_min) / 2.0
            if isinstance(self.sim, MjSimWarp):
                dev = torch.device("cuda")
                self.input_min = torch.as_tensor(self.input_min, device=dev, dtype=torch.float32)
                self.input_max = torch.as_tensor(self.input_max, device=dev, dtype=torch.float32)
                self.action_scale = torch.as_tensor(self.action_scale, device=dev, dtype=torch.float32)
                self.action_input_transform = torch.as_tensor(
                    self.action_input_transform, device=dev, dtype=torch.float32
                )
                self.action_output_transform = torch.as_tensor(
                    self.action_output_transform, device=dev, dtype=torch.float32
                )

        if isinstance(action, torch.Tensor):
            action = torch.clamp(action, self.input_min, self.input_max)
        else:
            action = np.clip(action, self.input_min, self.input_max)
        return (action - self.action_input_transform) * self.action_scale + self.action_output_transform

    def update(self, force=False):
        """
        Updates the state of the robot arm, including end effector pose / orientation / velocity, joint pos/vel,
        jacobian, and mass matrix. By default, since this is a non-negligible computation, multiple redundant calls
        will be ignored via the self.new_update attribute flag. However, if the @force flag is set, the update will
        occur regardless of that state of self.new_update. This base class method of @run_controller resets the
        self.new_update flag

        Args:
            force (bool): Whether to force an update to occur or not
        """

        # Only run update if self.new_update or force flag is set
        if self.new_update or force:
            # For warp, forward() is already called in the main step loop before _pre_action.
            # Calling it again here would be a redundant GPU kernel launch.
            if not isinstance(self.sim, MjSimWarp):
                self.sim.forward()

            if isinstance(self.sim, MjSimWarp):
                import torch

                # All state variables are CUDA torch.Tensors.
                # Shapes mirror the batched (num_envs, ...) convention.
                self.ee_pos = self.sim.data.get_site_xpos(self.eef_name)      # (num_envs, 3)
                self.ee_ori_mat = self.sim.data.get_site_xmat(self.eef_name)  # (num_envs, 3, 3)

                # qpos/qvel subsetted to this controller's DOFs, as CUDA torch.Tensors.
                self.joint_pos = self.sim.data.qpos[self.qpos_index]  # (num_envs, ndof)
                self.joint_vel = self.sim.data.qvel[self.qvel_index]  # (num_envs, ndof)

                # Cache full-nv Jacs so partial_update_warp skips the jac kernel relaunch.
                J_pos_full, J_ori_full = self.sim.data.get_site_jacs(self.eef_name)  # (B, 3, nv)
                self._J_pos_full = J_pos_full  # (num_envs, 3, nv) -- cached for velocity use
                self._J_ori_full = J_ori_full  # (num_envs, 3, nv)
                # Stacked (6, nv) Jacobian for single-bmm velocity computation in partial_update_warp().
                self._J_vel = torch.cat([J_pos_full, J_ori_full], dim=1)  # (num_envs, 6, nv)
                self.J_pos = J_pos_full[:, :, self.qvel_index]    # (num_envs, 3, ndof)
                self.J_ori = J_ori_full[:, :, self.qvel_index]    # (num_envs, 3, ndof)
                self.J_full = torch.cat([self.J_pos, self.J_ori], dim=1)  # (num_envs, 6, ndof)

                # Velocities from Jacobians x qvel -- one bmm using the stacked Jacobian.
                qvel_full = self.sim.data.qvel.tensor              # (num_envs, nv)
                vel_both = torch.bmm(self._J_vel, qvel_full.unsqueeze(-1)).squeeze(-1)  # (B, 6)
                self.ee_pos_vel = vel_both[:, :3]                  # (B, 3)
                self.ee_ori_vel = vel_both[:, 3:]                  # (B, 3)

                # qM top-left nvxnv is the dense symmetric mass matrix; sub-index per env,
                # not env-0 only -- mass matrix is config-dependent and broadcasting destabilises.
                nv = self.sim.model.nv
                qM_full = self.sim.data.qM  # (num_envs, padded_nv, padded_nv)
                mm_sub = qM_full[:, :nv, :nv][:, self.qvel_index, :][:, :, self.qvel_index]
                self.mass_matrix = mm_sub.float()  # (num_envs, ndof, ndof)
            else:
                self.ee_pos = np.array(self.sim.data.site_xpos[self.sim.model.site_name2id(self.eef_name)])
                self.ee_ori_mat = np.array(
                    self.sim.data.site_xmat[self.sim.model.site_name2id(self.eef_name)].reshape([3, 3])
                )
                self.ee_pos_vel = np.array(self.sim.data.get_site_xvelp(self.eef_name))
                self.ee_ori_vel = np.array(self.sim.data.get_site_xvelr(self.eef_name))

                self.joint_pos = np.array(self.sim.data.qpos[self.qpos_index])
                self.joint_vel = np.array(self.sim.data.qvel[self.qvel_index])

                self.J_pos = np.array(
                    self.sim.data.get_site_jacp(self.eef_name).reshape((3, -1))[:, self.qvel_index]
                )
                self.J_ori = np.array(
                    self.sim.data.get_site_jacr(self.eef_name).reshape((3, -1))[:, self.qvel_index]
                )
                self.J_full = np.array(np.vstack([self.J_pos, self.J_ori]))

                mass_matrix = np.ndarray(shape=(self.sim.model.nv, self.sim.model.nv), dtype=np.float64, order="C")
                mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)
                mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
                self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]

            # Clear self.new_update
            self.new_update = False

    def partial_update_warp(self):
        """
        Cheap per-substep state refresh for warp: updates positions, orientations,
        and velocities without re-launching the Jacobian kernel.

        Velocities are computed as ``J_cached @ qvel_fresh`` using the full Jacobians
        cached by the last ``update()`` call.  Jacobians and mass matrix are kept
        from the beginning of the policy step; they change slowly enough at 500 Hz
        that this approximation is accurate.

        Only called from :meth:`OSCController._run_controller_warp` on substeps 2-25.
        """
        import torch

        data = self.sim.data
        self.ee_pos = data.get_site_xpos(self.eef_name)         # (B, 3)
        self.ee_ori_mat = data.get_site_xmat(self.eef_name)     # (B, 3, 3)
        self.joint_pos = data.qpos[self.qpos_index]             # (B, ndof)
        qvel_full = data.qvel.tensor                            # (B, nv)
        self.joint_vel = qvel_full[:, self.qvel_index]          # (B, ndof)
        # Velocities: single bmm with the stacked (6, nv) Jacobian cached by update().
        vel_both = torch.bmm(self._J_vel, qvel_full.unsqueeze(-1)).squeeze(-1)  # (B, 6)
        self.ee_pos_vel = vel_both[:, :3]                       # (B, 3)
        self.ee_ori_vel = vel_both[:, 3:]                       # (B, 3)

    def update_base_pose(self, base_pos, base_ori):
        """
        Optional function to implement in subclass controllers that will take in @base_pos and @base_ori and update
        internal configuration to account for changes in the respective states. Useful for controllers e.g. IK, which
        is based on pybullet and requires knowledge of simulator state deviations between pybullet and mujoco

        Args:
            base_pos (3-tuple): x,y,z position of robot base in mujoco world coordinates
            base_ori (4-tuple): x,y,z,w orientation or robot base in mujoco world coordinates
        """
        pass

    def update_initial_joints(self, initial_joints):
        """
        Updates the internal attribute self.initial_joints. This is useful for updating changes in controller-specific
        behavior, such as with OSC where self.initial_joints is used for determine nullspace actions

        This function can also be extended by subclassed controllers for additional controller-specific updates

        Args:
            initial_joints (Iterable): Array of joint position values to update the initial joints
        """
        self.initial_joint = np.array(initial_joints)
        self.update(force=True)
        _, self.initial_ee_pos, self.initial_ee_ori_mat = self._extract_initial_state()

    def _extract_initial_state(self):
        """
        Extract ``(initial_joint, initial_ee_pos, initial_ee_ori_mat)`` as plain
        numpy arrays suitable for use as controller goals.

        For standard sims these are just the current state attributes.  For
        :class:`MjSimWarp` the attributes are batched warp arrays, so we pull
        env 0 to CPU.
        """
        if isinstance(self.sim, MjSimWarp):
            # joint_pos already sliced to qpos_index; double-index is OOB for tasks
            # whose arm joints don't start at global qpos[0] (e.g. HammerCleanup drawer).
            joint = self.joint_pos[0].cpu().numpy()
            ee_pos = self.ee_pos[0].cpu().numpy()
            ee_ori_mat = self.ee_ori_mat[0].cpu().numpy()
        else:
            joint = self.joint_pos
            ee_pos = self.ee_pos
            ee_ori_mat = self.ee_ori_mat
        return joint, ee_pos, ee_ori_mat

    def clip_torques(self, torques):
        """
        Clips the torques to be within the actuator limits

        Args:
            torques (Iterable): Torques to clip

        Returns:
            np.array: Clipped torques
        """
        return np.clip(torques, self.actuator_min, self.actuator_max)

    def reset_goal(self):
        """
        Resets the goal -- usually by setting to the goal to all zeros, but in some cases may be different (e.g.: OSC)
        """
        raise NotImplementedError

    @staticmethod
    def nums2array(nums, dim):
        """
        Convert input @nums into numpy array of length @dim. If @nums is a single number, broadcasts it to the
        corresponding dimension size @dim before converting into a numpy array

        Args:
            nums (numeric or Iterable): Either single value or array of numbers
            dim (int): Size of array to broadcast input to env.sim.data.actuator_force

        Returns:
            np.array: Array filled with values specified in @nums
        """
        # First run sanity check to make sure no strings are being inputted
        if isinstance(nums, str):
            raise TypeError("Error: Only numeric inputs are supported for this function, nums2array!")

        # Check if input is an Iterable, if so, we simply convert the input to np.array and return
        # Else, input is a single value, so we map to a numpy array of correct size and return
        return np.array(nums) if isinstance(nums, Iterable) else np.ones(dim) * nums

    @property
    def torque_compensation(self):
        """
        Gravity compensation for this robot arm

        Returns:
            np.array or torch.Tensor: torques, shape ``(num_envs, ndof)`` for warp
        """
        return self.sim.data.qfrc_bias[self.qvel_index]

    @property
    def actuator_limits(self):
        """
        Torque limits for this controller

        Returns:
            2-tuple:

                - (np.array) minimum actuator torques
                - (np.array) maximum actuator torques
        """
        return self.actuator_min, self.actuator_max

    @property
    def control_limits(self):
        """
        Limits over this controller's action space, which defaults to input min/max

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        return self.input_min, self.input_max

    @property
    def name(self):
        """
        Name of this controller

        Returns:
            str: controller name
        """
        raise NotImplementedError
