"""
Gripper with two fingers for Rethink Robots.
"""
from __future__ import annotations

import numpy as np
import torch

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class RethinkGripperBase(GripperModel):
    """
    Gripper with long two-fingered parallel jaw.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/rethink_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.020833, -0.020833])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["l_finger_g0", "l_finger_g1", "l_fingertip_g0", "l_fingerpad_g0"],
            "right_finger": ["r_finger_g0", "r_finger_g1", "r_fingertip_g0", "r_fingerpad_g0"],
            "left_fingerpad": ["l_fingerpad_g0"],
            "right_fingerpad": ["r_fingerpad_g0"],
        }


class RethinkGripper(RethinkGripperBase):
    """
    Modifies two finger base to only take one action.
    """

    def format_action(self, action: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action: gripper-specific action, shape (dof,), (num_envs, dof) numpy or torch tensor

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert action.shape[-1] == self.dof
        if isinstance(action, torch.Tensor):
            dev, dtype = action.device, action.dtype
            if not isinstance(self.current_action, torch.Tensor):
                self.current_action = torch.as_tensor(self.current_action, device=dev, dtype=dtype)
            if action.ndim == 2 and self.current_action.ndim == 1:
                self.current_action = self.current_action.unsqueeze(0).expand(action.shape[0], -1).clone()
            scale = torch.tensor([1.0, -1.0], device=dev, dtype=dtype)
            self.current_action = torch.clamp(
                self.current_action + scale * self.speed * torch.sign(action), -1.0, 1.0
            )
        else:
            if action.ndim == 2 and self.current_action.ndim == 1:
                self.current_action = np.tile(self.current_action, (action.shape[0], 1))
            self.current_action = np.clip(
                self.current_action + np.array([1.0, -1.0]) * self.speed * np.sign(action), -1.0, 1.0
            )
        return self.current_action

    @property
    def speed(self):
        return 0.01

    @property
    def dof(self):
        return 1
