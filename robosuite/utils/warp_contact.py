"""
Per-env contact-group queries against the flat mujoco-warp contact SoA.

mujoco-warp stores all active contacts for every world in a single flat buffer
of size ``naconmax`` (total across worlds), with each slot carrying
``(worldid, geom[2], dist, ...)``. There is no per-world ``ncon[w]``, so
asking "does world w have a contact between group A and group B?" needs a
kernel that partitions by ``worldid``.

This module provides that primitive. Import is cheap (just imports ``warp``);
the ``@wp.kernel`` decorator registers a kernel that JIT-compiles on first
launch and is cached thereafter by warp.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import warp as wp


@wp.kernel
def _contact_group_kernel(
    nacon: wp.array(dtype=wp.int32),
    worldid: wp.array(dtype=wp.int32),
    geompair: wp.array(dtype=wp.vec2i),
    mask_a: wp.array(dtype=wp.int32),
    mask_b: wp.array(dtype=wp.int32),
    use_b: wp.int32,
    out: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    if tid >= nacon[0]:
        return
    g = geompair[tid]
    g1 = g[0]
    g2 = g[1]
    hit = wp.int32(0)
    if use_b == 0:
        # "any contact touching group A"
        if mask_a[g1] != 0 or mask_a[g2] != 0:
            hit = 1
    else:
        # symmetric: (g1 in A and g2 in B) or (g2 in A and g1 in B)
        if (mask_a[g1] != 0 and mask_b[g2] != 0) or (mask_a[g2] != 0 and mask_b[g1] != 0):
            hit = 1
    if hit != 0:
        wp.atomic_max(out, worldid[tid], 1)


def build_geom_mask(ngeom: int, geom_ids: Iterable[int], device) -> wp.array:
    """Build a length-``ngeom`` int32 membership mask on *device*."""
    mask = np.zeros(ngeom, dtype=np.int32)
    for gid in geom_ids:
        mask[int(gid)] = 1
    return wp.from_numpy(mask, dtype=wp.int32, device=device)


def launch_contact_group_kernel(
    warp_data,
    num_envs: int,
    mask_a: wp.array,
    mask_b: Optional[wp.array],
    out: wp.array,
) -> None:
    """
    Zero ``out`` and launch the contact-group kernel.

    ``out`` must be a length-``num_envs`` int32 warp array; after the launch,
    ``out[w] == 1`` iff world ``w`` has at least one contact matching the
    (A, [B]) group query.
    """
    out.zero_()
    use_b = 0 if mask_b is None else 1
    mask_b_arg = mask_b if mask_b is not None else mask_a
    naconmax = warp_data.contact.worldid.shape[0]
    wp.launch(
        kernel=_contact_group_kernel,
        dim=naconmax,
        inputs=[
            warp_data.nacon,
            warp_data.contact.worldid,
            warp_data.contact.geom,
            mask_a,
            mask_b_arg,
            int(use_b),
            out,
        ],
        device=warp_data.qpos.device,
    )
