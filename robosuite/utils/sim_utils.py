"""
Collection of useful simulation utilities
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from robosuite.models.base import MujocoModel

if TYPE_CHECKING:
    import warp as wp


def np_to_warp(np_arr: np.ndarray, template: "wp.array") -> "wp.array":
    """
    Convert a numpy array into a warp array whose dtype and device match *template*.

    Args:
        np_arr: Source numpy array (will be cast to float32 before conversion).
        template: A ``wp.array`` whose ``.dtype`` and ``.device`` are used as the
                  conversion target.

    Returns:
        wp.array: A new warp array on the same device as *template* with the same
                  element dtype.

    Raises:
        TypeError: If *template*'s element dtype is not one of the supported warp
                   structured types.
    """
    import warp as wp

    dtype = template.dtype
    device = template.device
    shape = template.shape
    src = np.asarray(np_arr, dtype=np.float32)
    dtype_name: str = dtype.__name__ if hasattr(dtype, "__name__") else str(dtype)
    if dtype_name == "float32":
        return wp.from_numpy(src, device=device)
    elif dtype_name in ("vec2f", "vec2"):
        return wp.from_numpy(src.reshape(*shape, 2), dtype=dtype, device=device)
    elif dtype_name in ("vec3f", "vec3"):
        return wp.from_numpy(src.reshape(*shape, 3), dtype=dtype, device=device)
    elif dtype_name in ("vec4f", "vec4"):
        return wp.from_numpy(src.reshape(*shape, 4), dtype=dtype, device=device)
    elif dtype_name in ("quatf", "quat"):
        return wp.from_numpy(src.reshape(*shape, 4), dtype=dtype, device=device)
    elif dtype_name in ("mat33f", "mat33"):
        return wp.from_numpy(src.reshape(*shape, 3, 3), dtype=dtype, device=device)
    elif dtype_name in ("spatial_vectorf", "spatial_vector"):
        return wp.from_numpy(src.reshape(*shape, 6), dtype=dtype, device=device)
    else:
        raise TypeError(f"Unsupported warp element dtype for conversion: {dtype_name!r}")


def check_contact(sim, geoms_1, geoms_2=None):
    """
    Finds contact between two geom groups.
    Args:
        sim (MjSim): Current simulation object
        geoms_1 (str or list of str or MujocoModel): an individual geom name or list of geom names or a model. If
            a MujocoModel is specified, the geoms checked will be its contact_geoms
        geoms_2 (str or list of str or MujocoModel or None): another individual geom name or list of geom names.
            If a MujocoModel is specified, the geoms checked will be its contact_geoms. If None, will check
            any collision with @geoms_1 to any other geom in the environment
    Returns:
        bool: True if any geom in @geoms_1 is in contact with any geom in @geoms_2.
    """
    # Check if either geoms_1 or geoms_2 is a string, convert to list if so
    if type(geoms_1) is str:
        geoms_1 = [geoms_1]
    elif isinstance(geoms_1, MujocoModel):
        geoms_1 = geoms_1.contact_geoms
    if type(geoms_2) is str:
        geoms_2 = [geoms_2]
    elif isinstance(geoms_2, MujocoModel):
        geoms_2 = geoms_2.contact_geoms
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        # check contact geom in geoms
        c1_in_g1 = sim.model.geom_id2name(contact.geom1) in geoms_1
        c2_in_g2 = sim.model.geom_id2name(contact.geom2) in geoms_2 if geoms_2 is not None else True
        # check contact geom in geoms (flipped)
        c2_in_g1 = sim.model.geom_id2name(contact.geom2) in geoms_1
        c1_in_g2 = sim.model.geom_id2name(contact.geom1) in geoms_2 if geoms_2 is not None else True
        if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
            return True
    return False


def get_contacts(sim, model):
    """
    Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
    geom names currently in contact with that model (excluding the geoms that are part of the model itself).
    Args:
        sim (MjSim): Current simulation model
        model (MujocoModel): Model to check contacts for.
    Returns:
        set: Unique geoms that are actively in contact with this model.
    Raises:
        AssertionError: [Invalid input type]
    """
    # Make sure model is MujocoModel type
    assert isinstance(model, MujocoModel), "Inputted model must be of type MujocoModel; got type {} instead!".format(
        type(model)
    )
    contact_set = set()
    for contact in sim.data.contact[: sim.data.ncon]:
        # check contact geom in geoms; add to contact set if match is found
        g1, g2 = sim.model.geom_id2name(contact.geom1), sim.model.geom_id2name(contact.geom2)
        if g1 in model.contact_geoms and g2 not in model.contact_geoms:
            contact_set.add(g2)
        elif g2 in model.contact_geoms and g1 not in model.contact_geoms:
            contact_set.add(g1)
    return contact_set
