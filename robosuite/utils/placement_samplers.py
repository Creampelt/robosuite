from __future__ import annotations

import collections
from copy import copy

import numpy as np

from robosuite.models.objects import MujocoObject
from robosuite.utils import RandomizationError
from robosuite.utils.transform_utils import quat_multiply


class ObjectPositionSampler:
    """
    Base class of object placement sampler.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        ensure_object_boundary_in_range (bool): If True, will ensure that the object is enclosed within a given boundary
            (should be implemented by subclass)

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
    ):
        # Setup attributes
        self.name = name
        if mujoco_objects is None:
            self.mujoco_objects = []
        else:
            # Shallow copy the list so we don't modify the inputted list but still keep the object references
            self.mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else copy(mujoco_objects)
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.ensure_valid_placement = ensure_valid_placement
        self.reference_pos = reference_pos
        self.z_offset = z_offset

    def add_objects(self, mujoco_objects):
        """
        Add additional objects to this sampler. Checks to make sure there's no identical objects already stored.

        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): single model or list of MJCF object models
        """
        mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else mujoco_objects
        for obj in mujoco_objects:
            assert obj not in self.mujoco_objects, "Object '{}' already in sampler!".format(obj.name)
            self.mujoco_objects.append(obj)

    def reset(self):
        """
        Resets this sampler. Removes all mujoco objects from this sampler.
        """
        self.mujoco_objects = []

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample on a surface (not necessarily table surface).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object.

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        """
        raise NotImplementedError


class UniformRandomSampler(ObjectPositionSampler):
    """
    Places all objects within the table uniformly random.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects

        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects

        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation

        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_range=(0, 0),
        y_range=(0, 0),
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.rotation = rotation
        self.rotation_axis = rotation_axis

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_x(self, object_horizontal_radius):
        """
        Samples the x location for a given object

        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for

        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_range
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        """
        Samples the y location for a given object

        Args:
            object_horizontal_radius (float): Radius of the object currently being sampled for

        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_range
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object

        Returns:
            np.array: sampled object quaternion in (w,x,y,z) form

        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = np.random.uniform(high=max(self.rotation), low=min(self.rotation))
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(self.rotation_axis)
            )

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(base_offset)

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert obj.name not in placed_objects, "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            for i in range(5000):  # 5000 retries
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                # objects cannot overlap
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]):
                            location_valid = False
                            break

                if location_valid:
                    # random rotation
                    quat = self._sample_quat()

                    # multiply this quat by the object's initial rotation if it has the attribute specified
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # location is valid, put the object down
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects

    def sample_batch(
        self,
        n: int,
        fixtures: dict | None = None,
        reference=None,
        on_top: bool = True,
    ) -> dict:
        """Batched variant of :meth:`sample` for ``n`` independent envs.

        Returns a dict with the same structure as ``sample()`` except pos is a
        ``(n, 3)`` array and quat is a ``(n, 4)`` array (wxyz). Fixture entries
        passed in as scalar ``(pos3, quat4, obj)`` are broadcast to batched
        shape for use in the per-row valid-placement check.

        Semantics match ``sample()`` row-for-row. Rejection sampling is
        vectorised: at each iteration only the still-invalid rows resample,
        so this runs in 1-2 iterations for well-separated ranges.

        Raises:
            RandomizationError: if any row still has an invalid placement
                after 5000 iterations.
        """
        placed_objects: dict = {} if fixtures is None else self._broadcast_fixtures(fixtures, n)
        if reference is None:
            base_offset = np.asarray(self.reference_pos, dtype=np.float64)
            base_offset = np.broadcast_to(base_offset, (n, 3)).copy()
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, _, ref_obj = placed_objects[reference]
            base_offset = np.asarray(ref_pos, dtype=np.float64).copy()
            if base_offset.shape == (3,):
                base_offset = np.broadcast_to(base_offset, (n, 3)).copy()
            if on_top:
                base_offset[:, 2] += ref_obj.top_offset[-1]
        else:
            base_offset = np.asarray(reference, dtype=np.float64)
            assert (
                base_offset.shape[-1] == 3
            ), "Invalid reference received. Should be (x,y,z), but got shape: {}".format(base_offset.shape)
            if base_offset.shape == (3,):
                base_offset = np.broadcast_to(base_offset, (n, 3)).copy()

        out: dict = dict(placed_objects)
        for obj in self.mujoco_objects:
            assert obj.name not in out, "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset

            x_min, x_max = self.x_range
            y_min, y_max = self.y_range
            if self.ensure_object_boundary_in_range:
                x_min += horizontal_radius
                x_max -= horizontal_radius
                y_min += horizontal_radius
                y_max -= horizontal_radius

            pos = np.empty((n, 3), dtype=np.float64)
            pos[:, 0] = np.random.uniform(low=x_min, high=x_max, size=n) + base_offset[:, 0]
            pos[:, 1] = np.random.uniform(low=y_min, high=y_max, size=n) + base_offset[:, 1]
            pos[:, 2] = self.z_offset + base_offset[:, 2]
            if on_top:
                pos[:, 2] -= bottom_offset[-1]

            invalid = np.zeros(n, dtype=bool)
            if self.ensure_valid_placement and out:
                invalid = self._compute_overlap_mask(pos, out, horizontal_radius, bottom_offset)

            # Row-level rejection; typically 0-1 iterations on non-overlapping ranges.
            for _ in range(5000):
                if not invalid.any():
                    break
                m = invalid
                k = int(m.sum())
                pos[m, 0] = np.random.uniform(low=x_min, high=x_max, size=k) + base_offset[m, 0]
                pos[m, 1] = np.random.uniform(low=y_min, high=y_max, size=k) + base_offset[m, 1]
                invalid = self._compute_overlap_mask(pos, out, horizontal_radius, bottom_offset)
            else:
                if invalid.any():
                    raise RandomizationError(
                        f"Cannot place '{obj.name}' for {int(invalid.sum())}/{n} envs"
                    )

            quat = self._sample_quat_batch(n)
            if hasattr(obj, "init_quat"):
                quat = _quat_multiply_batch(quat, np.asarray(obj.init_quat, dtype=np.float64))

            out[obj.name] = (pos, quat, obj)

        return out

    @staticmethod
    def _broadcast_fixtures(fixtures: dict, n: int) -> dict:
        """Broadcast any scalar (pos3, quat4) fixture entries up to (n, ...)."""
        bc: dict = {}
        for k, (pos, quat, obj) in fixtures.items():
            pos = np.asarray(pos, dtype=np.float64)
            quat = np.asarray(quat, dtype=np.float64)
            if pos.ndim == 1:
                pos = np.broadcast_to(pos, (n, 3)).copy()
            if quat.ndim == 1:
                quat = np.broadcast_to(quat, (n, 4)).copy()
            bc[k] = (pos, quat, obj)
        return bc

    @staticmethod
    def _compute_overlap_mask(
        pos: np.ndarray,
        placed: dict,
        horizontal_radius: float,
        bottom_offset: np.ndarray,
    ) -> np.ndarray:
        """Return a bool mask of rows whose candidate ``pos`` overlaps any
        already-placed (batched) object, matching the scalar valid-placement
        test in :meth:`sample`."""
        invalid = np.zeros(pos.shape[0], dtype=bool)
        for other_pos, _, other_obj in placed.values():
            if other_pos.ndim == 1:
                other_pos = np.broadcast_to(other_pos, pos.shape)
            dxy = pos[:, :2] - other_pos[:, :2]
            hdist = np.linalg.norm(dxy, axis=-1)
            dz = pos[:, 2] - other_pos[:, 2]
            overlap = (hdist <= other_obj.horizontal_radius + horizontal_radius) & (
                dz <= other_obj.top_offset[-1] - bottom_offset[-1]
            )
            invalid |= overlap
        return invalid

    def _sample_quat_batch(self, n: int) -> np.ndarray:
        """Batched wxyz quaternion sampler matching :meth:`_sample_quat`."""
        if self.rotation is None:
            rot_angle = np.random.uniform(low=0.0, high=2 * np.pi, size=n)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = np.random.uniform(
                low=min(self.rotation), high=max(self.rotation), size=n
            )
        else:
            rot_angle = np.full(n, float(self.rotation))

        c = np.cos(rot_angle / 2.0)
        s = np.sin(rot_angle / 2.0)
        quat = np.zeros((n, 4), dtype=np.float64)
        quat[:, 0] = c  # w
        if self.rotation_axis == "x":
            quat[:, 1] = s
        elif self.rotation_axis == "y":
            quat[:, 2] = s
        elif self.rotation_axis == "z":
            quat[:, 3] = s
        else:
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )
        return quat


def _quat_multiply_batch(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """Batched version of ``robosuite.utils.transform_utils.quat_multiply``.

    Mirrors the scalar function's element ordering bit-for-bit (it labels the
    input indices as ``(x, y, z, w)`` and outputs the corresponding composition).
    Accepts any broadcast-compatible combination of ``(n, 4)`` and ``(4,)``
    inputs, returns ``(n, 4)``.
    """
    q1a = np.asarray(q1, dtype=np.float64)
    q0a = np.asarray(q0, dtype=np.float64)
    if q1a.ndim == 1:
        q1a = q1a[None, :]
    if q0a.ndim == 1:
        q0a = q0a[None, :]
    n = max(q1a.shape[0], q0a.shape[0])
    q1a = np.broadcast_to(q1a, (n, 4))
    q0a = np.broadcast_to(q0a, (n, 4))
    x1, y1, z1, w1 = q1a[:, 0], q1a[:, 1], q1a[:, 2], q1a[:, 3]
    x0, y0, z0, w0 = q0a[:, 0], q0a[:, 1], q0a[:, 2], q0a[:, 3]
    out = np.empty((n, 4), dtype=np.float64)
    out[:, 0] = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
    out[:, 1] = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
    out[:, 2] = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
    out[:, 3] = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    return out


class SequentialCompositeSampler(ObjectPositionSampler):
    """
    Samples position for each object sequentially. Allows chaining
    multiple placement initializers together - so that object locations can
    be sampled on top of other objects or relative to other object placements.

    Args:
        name (str): Name of this sampler.
    """

    def __init__(self, name):
        # Samplers / args will be filled in later
        self.samplers = collections.OrderedDict()
        self.sample_args = collections.OrderedDict()

        super().__init__(name=name)

    def append_sampler(self, sampler, sample_args=None):
        """
        Adds a new placement initializer with corresponding @sampler and arguments

        Args:
            sampler (ObjectPositionSampler): sampler to add
            sample_args (None or dict): If specified, should be additional arguments to pass to @sampler's sample()
                call. Should map corresponding sampler's arguments to values (excluding @fixtures argument)

        Raises:
            AssertionError: [Object name in samplers]
        """
        # Verify that all added mujoco objects haven't already been added, and add to this sampler's objects dict
        for obj in sampler.mujoco_objects:
            assert obj not in self.mujoco_objects, f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        self.samplers[sampler.name] = sampler
        self.sample_args[sampler.name] = sample_args

    def hide(self, mujoco_objects):
        """
        Helper method to remove an object from the workspace.

        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to hide
        """
        sampler = UniformRandomSampler(
            name="HideSampler",
            mujoco_objects=mujoco_objects,
            x_range=[-10, -20],
            y_range=[-10, -20],
            rotation=[0, 0],
            rotation_axis="z",
            z_offset=10,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
        )
        self.append_sampler(sampler=sampler)

    def add_objects(self, mujoco_objects):
        """
        Override super method to make sure user doesn't call this (all objects should implicitly belong to sub-samplers)
        """
        raise AttributeError("add_objects() should not be called for SequentialCompsiteSamplers!")

    def add_objects_to_sampler(self, sampler_name, mujoco_objects):
        """
        Adds specified @mujoco_objects to sub-sampler with specified @sampler_name.

        Args:
            sampler_name (str): Existing sub-sampler name
            mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to add
        """
        # First verify that all mujoco objects haven't already been added, and add to this sampler's objects dict
        mujoco_objects = [mujoco_objects] if isinstance(mujoco_objects, MujocoObject) else mujoco_objects
        for obj in mujoco_objects:
            assert obj not in self.mujoco_objects, f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        # Make sure sampler_name exists
        assert (
            sampler_name in self.samplers.keys()
        ), "Invalid sub-sampler specified, valid options are: {}, " "requested: {}".format(
            self.samplers.keys(), sampler_name
        )
        # Add the mujoco objects to the requested sub-sampler
        self.samplers[sampler_name].add_objects(mujoco_objects)

    def reset(self):
        """
        Resets this sampler. In addition to base method, iterates over all sub-samplers and resets them
        """
        super().reset()
        for sampler in self.samplers.values():
            sampler.reset()

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Sample from each placement initializer sequentially, in the order
        that they were appended.

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. This will override each
                sampler's @reference argument if not already specified. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object. This will override each
                sampler's @on_top argument if not already specified. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            RandomizationError: [Cannot place all objects]
        """
        # Standardize inputs
        placed_objects = {} if fixtures is None else copy(fixtures)

        # Iterate through all samplers to sample
        for sampler, s_args in zip(self.samplers.values(), self.sample_args.values()):
            # Pre-process sampler args
            if s_args is None:
                s_args = {}
            for arg_name, arg in zip(("reference", "on_top"), (reference, on_top)):
                if arg_name not in s_args:
                    s_args[arg_name] = arg
            # Run sampler
            new_placements = sampler.sample(fixtures=placed_objects, **s_args)
            # Update placements
            placed_objects.update(new_placements)

        return placed_objects

    def sample_batch(
        self,
        n: int,
        fixtures: dict | None = None,
        reference=None,
        on_top: bool = True,
    ) -> dict:
        """Batched composite sample for ``n`` envs. See
        :meth:`UniformRandomSampler.sample_batch`.
        """
        placed_objects: dict = {} if fixtures is None else UniformRandomSampler._broadcast_fixtures(fixtures, n)
        for sampler, s_args in zip(self.samplers.values(), self.sample_args.values()):
            if s_args is None:
                s_args = {}
            for arg_name, arg in zip(("reference", "on_top"), (reference, on_top)):
                if arg_name not in s_args:
                    s_args[arg_name] = arg
            if not hasattr(sampler, "sample_batch"):
                raise NotImplementedError(
                    f"sample_batch not implemented on sub-sampler {type(sampler).__name__}"
                )
            new_placements = sampler.sample_batch(n, fixtures=placed_objects, **s_args)
            placed_objects.update(new_placements)
        return placed_objects
