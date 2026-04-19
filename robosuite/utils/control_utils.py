from __future__ import annotations

import numpy as np
import torch

import robosuite.utils.transform_utils as trans
from robosuite.utils.numba import jit_decorator


@jit_decorator
def nullspace_torques(
    mass_matrix: np.ndarray,
    nullspace_matrix: np.ndarray,
    initial_joint: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_kp: float = 10,
) -> np.ndarray:
    """
    For a robot with redundant DOF(s), a nullspace exists which is orthogonal to the remainder of the controllable
    subspace of the robot's joints. Therefore, an additional secondary objective that does not impact the original
    controller objective may attempt to be maintained using these nullspace torques.

    This utility function specifically calculates nullspace torques that attempt to maintain a given robot joint
    positions @initial_joint with zero velocity using proportinal gain @joint_kp

    :Note: @mass_matrix, @nullspace_matrix, @joint_pos, and @joint_vel should reflect the robot's state at the current
    timestep

    Args:
        mass_matrix (np.array): 2d array representing the mass matrix of the robot
        nullspace_matrix (np.array): 2d array representing the nullspace matrix of the robot
        initial_joint (np.array): Joint configuration to be used for calculating nullspace torques
        joint_pos (np.array): Current joint positions
        joint_vel (np.array): Current joint velocities
        joint_kp (float): Proportional control gain when calculating nullspace torques

    Returns:
          np.array: nullspace torques
    """

    # kv calculated below corresponds to critical damping
    joint_kv = np.sqrt(joint_kp) * 2

    # calculate desired torques based on gains and error
    pose_torques = np.dot(mass_matrix, (joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel))

    # map desired torques to null subspace within joint torque actuator space
    nullspace_torques = np.dot(nullspace_matrix.transpose(), pose_torques)
    return nullspace_torques


@jit_decorator
def opspace_matrices(
    mass_matrix: np.ndarray,
    J_full: np.ndarray,
    J_pos: np.ndarray,
    J_ori: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the relevant matrices used in the operational space control algorithm

    Args:
        mass_matrix (np.array): 2d array representing the mass matrix of the robot
        J_full (np.array): 2d array representing the full Jacobian matrix of the robot
        J_pos (np.array): 2d array representing the position components of the Jacobian matrix of the robot
        J_ori (np.array): 2d array representing the orientation components of the Jacobian matrix of the robot

    Returns:
        4-tuple:

            - (np.array): full lambda matrix (as 2d array)
            - (np.array): position components of lambda matrix (as 2d array)
            - (np.array): orientation components of lambda matrix (as 2d array)
            - (np.array): nullspace matrix (as 2d array)
    """
    mass_matrix_inv = np.linalg.inv(mass_matrix)

    # J M^-1 J^T
    lambda_full_inv = np.dot(np.dot(J_full, mass_matrix_inv), J_full.transpose())

    # Jx M^-1 Jx^T
    lambda_pos_inv = np.dot(np.dot(J_pos, mass_matrix_inv), J_pos.transpose())

    # Jr M^-1 Jr^T
    lambda_ori_inv = np.dot(np.dot(J_ori, mass_matrix_inv), J_ori.transpose())

    # take the inverses, but zero out small singular values for stability
    lambda_full = np.linalg.pinv(lambda_full_inv)
    lambda_pos = np.linalg.pinv(lambda_pos_inv)
    lambda_ori = np.linalg.pinv(lambda_ori_inv)

    # nullspace
    Jbar = np.dot(mass_matrix_inv, J_full.transpose()).dot(lambda_full)
    nullspace_matrix = np.eye(J_full.shape[-1], J_full.shape[-1]) - np.dot(Jbar, J_full)

    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix


@jit_decorator
def orientation_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """
    This function calculates a 3-dimensional orientation error vector for use in the
    impedance controller. It does this by computing the delta rotation between the
    inputs and converting that rotation to exponential coordinates (axis-angle
    representation, where the 3d vector is axis * angle).
    See https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation for more information.
    Optimized function to determine orientation error from matrices

    Args:
        desired (np.array): 2d array representing target orientation matrix
        current (np.array): 2d array representing current orientation matrix

    Returns:
        np.array: 2d array representing orientation error as a matrix
    """
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]

    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))

    return error


def set_goal_position(
    delta: np.ndarray,
    current_position: np.ndarray,
    position_limit: np.ndarray | None = None,
    set_pos: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculates and returns the desired goal position, clipping the result accordingly to @position_limits.
    @delta and @current_position must be specified if a relative goal is requested, else @set_pos must be
    specified to define a global goal position

    Args:
        delta (np.array): Desired relative change in position
        current_position (np.array): Current position
        position_limit (None or np.array): 2d array defining the (min, max) limits of permissible position goal commands
        set_pos (None or np.array): If set, will ignore @delta and set the goal position to this value

    Returns:
        np.array: calculated goal position in absolute coordinates

    Raises:
        ValueError: [Invalid position_limit shape]
    """
    n = len(current_position)
    if set_pos is not None:
        goal_position = set_pos
    else:
        goal_position = current_position + delta

    if position_limit is not None:
        if position_limit.shape != (2, n):
            raise ValueError(
                "Position limit should be shaped (2,{}) " "but is instead: {}".format(n, position_limit.shape)
            )

        # Clip goal position
        goal_position = np.clip(goal_position, position_limit[0], position_limit[1])

    return goal_position


def set_goal_orientation(
    delta: np.ndarray,
    current_orientation: np.ndarray,
    orientation_limit: np.ndarray | None = None,
    set_ori: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculates and returns the desired goal orientation, clipping the result accordingly to @orientation_limits.
    @delta and @current_orientation must be specified if a relative goal is requested, else @set_ori must be
    an orientation matrix specified to define a global orientation

    Args:
        delta (np.array): Desired relative change in orientation, in axis-angle form [ax, ay, az]
        current_orientation (np.array): Current orientation, in rotation matrix form
        orientation_limit (None or np.array): 2d array defining the (min, max) limits of permissible orientation goal commands
        set_ori (None or np.array): If set, will ignore @delta and set the goal orientation to this value

    Returns:
        np.array: calculated goal orientation in absolute coordinates

    Raises:
        ValueError: [Invalid orientation_limit shape]
    """
    # directly set orientation
    if set_ori is not None:
        goal_orientation = set_ori

    # otherwise use delta to set goal orientation
    else:
        # convert axis-angle value to rotation matrix
        quat_error = trans.axisangle2quat(delta)
        rotation_mat_error = trans.quat2mat(quat_error)
        goal_orientation = np.dot(rotation_mat_error, current_orientation)

    # check for orientation limits
    if np.array(orientation_limit).any():
        if orientation_limit.shape != (2, 3):
            raise ValueError(
                "Orientation limit should be shaped (2,3) " "but is instead: {}".format(orientation_limit.shape)
            )

        # Convert to euler angles for clipping
        euler = trans.mat2euler(goal_orientation)

        # Clip euler angles according to specified limits
        limited = False
        for idx in range(3):
            if orientation_limit[0][idx] < orientation_limit[1][idx]:  # Normal angle sector meaning
                if orientation_limit[0][idx] < euler[idx] < orientation_limit[1][idx]:
                    continue
                else:
                    limited = True
                    dist_to_lower = euler[idx] - orientation_limit[0][idx]
                    if dist_to_lower > np.pi:
                        dist_to_lower -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_lower += 2 * np.pi

                    dist_to_higher = euler[idx] - orientation_limit[1][idx]
                    if dist_to_lower > np.pi:
                        dist_to_higher -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_higher += 2 * np.pi

                    if dist_to_lower < dist_to_higher:
                        euler[idx] = orientation_limit[0][idx]
                    else:
                        euler[idx] = orientation_limit[1][idx]
            else:  # Inverted angle sector meaning
                if orientation_limit[0][idx] < euler[idx] or euler[idx] < orientation_limit[1][idx]:
                    continue
                else:
                    limited = True
                    dist_to_lower = euler[idx] - orientation_limit[0][idx]
                    if dist_to_lower > np.pi:
                        dist_to_lower -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_lower += 2 * np.pi

                    dist_to_higher = euler[idx] - orientation_limit[1][idx]
                    if dist_to_lower > np.pi:
                        dist_to_higher -= 2 * np.pi
                    elif dist_to_lower < -np.pi:
                        dist_to_higher += 2 * np.pi

                    if dist_to_lower < dist_to_higher:
                        euler[idx] = orientation_limit[0][idx]
                    else:
                        euler[idx] = orientation_limit[1][idx]
        if limited:
            goal_orientation = trans.euler2mat(np.array([euler[0], euler[1], euler[2]]))
    return goal_orientation


@torch.jit.script
def opspace_matrices_torch(
    mass_matrix: torch.Tensor,
    J_full: torch.Tensor,
    J_pos: torch.Tensor,
    J_ori: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched torch version of opspace_matrices.

    Args:
        mass_matrix: (B, ndof, ndof) — per-env mass matrix
        J_full: (B, 6, ndof)
        J_pos: (B, 3, ndof)
        J_ori: (B, 3, ndof)

    Returns:
        lambda_full (B, 6, 6), lambda_pos (B, 3, 3), lambda_ori (B, 3, 3), nullspace_matrix (B, ndof, ndof)
    """
    # mass_matrix is (B, ndof, ndof) — per-env, so each env gets its own inertia tensor.
    # pinv (not inv) so a single corrupted env — e.g. NaN/near-singular qM from a post-
    # divergence env that hasn't been scrubbed yet — doesn't raise and take down the
    # whole batched controller launch. Post-step obs-NaN detection resets the bad env.
    M_inv = torch.linalg.pinv(mass_matrix, rcond=1e-6)  # (B, ndof, ndof)

    # rcond=1e-4 mirrors numpy's pinv rcond threshold and prevents NaN/inf when the
    # Jacobian is near-singular (e.g. after a contact event with rapid joint movement).
    lambda_full = torch.linalg.pinv(J_full @ M_inv @ J_full.mT, rcond=1e-4)   # (B, 6, 6)
    lambda_pos  = torch.linalg.pinv(J_pos  @ M_inv @ J_pos.mT,  rcond=1e-4)   # (B, 3, 3)
    lambda_ori  = torch.linalg.pinv(J_ori  @ M_inv @ J_ori.mT,  rcond=1e-4)   # (B, 3, 3)

    Jbar = M_inv @ J_full.mT @ lambda_full                         # (B, ndof, 6)
    n = J_full.shape[-1]
    eye = torch.eye(n, device=J_full.device, dtype=J_full.dtype)
    nullspace_matrix = eye - Jbar @ J_full                         # (B, ndof, ndof)

    return lambda_full, lambda_pos, lambda_ori, nullspace_matrix


@torch.jit.script
def nullspace_torques_torch(
    mass_matrix: torch.Tensor,
    nullspace_matrix: torch.Tensor,
    initial_joint: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_kp: float = 10.0,
) -> torch.Tensor:
    """Batched torch version of nullspace_torques.

    Args:
        mass_matrix: (ndof, ndof)
        nullspace_matrix: (B, ndof, ndof)
        initial_joint: (ndof,)
        joint_pos: (B, ndof)
        joint_vel: (B, ndof)
        joint_kp: proportional gain

    Returns:
        (B, ndof)
    """
    joint_kv = (joint_kp ** 0.5) * 2
    error = joint_kp * (initial_joint - joint_pos) - joint_kv * joint_vel  # (B, ndof)
    pose_torques = (mass_matrix @ error.unsqueeze(-1)).squeeze(-1)          # (B, ndof)
    return (nullspace_matrix.mT @ pose_torques.unsqueeze(-1)).squeeze(-1)   # (B, ndof)


@torch.jit.script
def orientation_error_torch(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Batched torch version of orientation_error.

    Args:
        desired: (B, 3, 3)
        current: (B, 3, 3)

    Returns:
        (B, 3)
    """
    rc1, rc2, rc3 = current[:, :, 0], current[:, :, 1], current[:, :, 2]
    rd1, rd2, rd3 = desired[:, :, 0], desired[:, :, 1], desired[:, :, 2]
    return 0.5 * (
        torch.linalg.cross(rc1, rd1) +
        torch.linalg.cross(rc2, rd2) +
        torch.linalg.cross(rc3, rd3)
    )
