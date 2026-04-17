"""
Useful classes for supporting DeepMind MuJoCo binding.
"""

from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import mujoco_warp
    import torch
    import warp as wp
import ctypes
import ctypes.util
import platform
from tempfile import TemporaryDirectory

# DIRTY HACK copied from mujoco-py - a global lock on rendering
from threading import Lock

import mujoco
import numpy as np

import robosuite.macros as macros
from robosuite.utils.sim_utils import np_to_warp  # noqa: F401  (re-exported for backwards compat)

_MjSim_render_lock = Lock()

_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    ctypes.WinDLL(os.path.join(os.path.dirname(__file__), "mujoco.dll"))

CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if CUDA_VISIBLE_DEVICES != "":
    MUJOCO_EGL_DEVICE_ID = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    if MUJOCO_EGL_DEVICE_ID is not None:
        assert MUJOCO_EGL_DEVICE_ID.isdigit() and (MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES), (
            "MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES"
        )

if macros.MUJOCO_GPU_RENDERING and os.environ.get("MUJOCO_GL", None) not in ["osmesa", "glx"]:
    # If gpu rendering is specified in macros, then we enforce gpu
    # option for rendering
    if _SYSTEM == "Darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    else:
        os.environ["MUJOCO_GL"] = "egl"
_MUJOCO_GL = os.environ.get("MUJOCO_GL", "").lower().strip()
if _MUJOCO_GL not in ("disable", "disabled", "off", "false", "0"):
    _VALID_MUJOCO_GL = ("enable", "enabled", "on", "true", "1", "glfw", "")
    if _SYSTEM == "Linux":
        _VALID_MUJOCO_GL += ("glx", "egl", "osmesa")
    elif _SYSTEM == "Windows":
        _VALID_MUJOCO_GL += ("wgl",)
    elif _SYSTEM == "Darwin":
        _VALID_MUJOCO_GL += ("cgl",)
    if _MUJOCO_GL not in _VALID_MUJOCO_GL:
        raise RuntimeError(f"invalid value for environment variable MUJOCO_GL: {_MUJOCO_GL}")
    if _SYSTEM == "Linux" and _MUJOCO_GL == "osmesa":
        from robosuite.renderers.context.osmesa_context import OSMesaGLContext as GLContext
    elif _SYSTEM == "Linux" and _MUJOCO_GL == "egl":
        from robosuite.renderers.context.egl_context import EGLGLContext as GLContext
    else:
        from robosuite.renderers.context.glfw_context import GLFWGLContext as GLContext


class MjRenderContext:
    """
    Class that encapsulates rendering functionality for a
    MuJoCo simulation.

    See https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjrendercontext.pyx
    """

    def __init__(self, sim, offscreen=True, device_id=-1, max_width=640, max_height=480):
        assert offscreen, "only offscreen supported for now"
        self.sim = sim
        self.offscreen = offscreen
        self.device_id = device_id

        # setup GL context with defaults for now
        self.gl_ctx = GLContext(max_width=max_width, max_height=max_height, device_id=self.device_id)
        self.gl_ctx.make_current()

        # Ensure the model data has been updated so that there
        # is something to render
        sim.forward()
        # make sure sim has this context
        sim.add_render_context(self)

        self.model = sim.model
        self.data = sim.data

        # create default scene
        self.scn = mujoco.MjvScene(sim.model._model, maxgeom=1000)

        # camera
        self.cam = mujoco.MjvCamera()
        self.cam.fixedcamid = 0
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # options for visual / collision mesh can be set externally, e.g. vopt.geomgroup[0], vopt.geomgroup[1]
        self.vopt = mujoco.MjvOption()

        self.pert = mujoco.MjvPerturb()
        self.pert.active = 0
        self.pert.select = 0
        self.pert.skinselect = -1

        # self._markers = []
        # self._overlay = {}

        self._set_mujoco_context_and_buffers()

    def _set_mujoco_context_and_buffers(self):
        self.con = mujoco.MjrContext(self.model._model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

    def update_offscreen_size(self, width, height):
        if (width != self.con.offWidth) or (height != self.con.offHeight):
            self.model.vis.global_.offwidth = width
            self.model.vis.global_.offheight = height
            self.con.free()
            del self.con
            self._set_mujoco_context_and_buffers()

    def upload_texture(self, tex_id):
        """Uploads given texture to the GPU"""
        self.gl_ctx.make_current()
        mujoco.mjr_uploadTexture(self.model, self.con, tex_id)

    def render(self, width, height, camera_id=None, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)

        # if self.sim.render_callback is not None:
        #     self.sim.render_callback(self.sim, self)

        # update width and height of rendering context if necessary
        if width > self.con.offWidth or height > self.con.offHeight:
            new_width = max(width, self.model.vis.global_.offwidth)
            new_height = max(height, self.model.vis.global_.offheight)
            self.update_offscreen_size(new_width, new_height)

        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        mujoco.mjv_updateScene(
            self.model._model, self.data._data, self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        # for marker_params in self._markers:
        #     self._add_marker_to_scene(marker_params)

        mujoco.mjr_render(viewport=viewport, scn=self.scn, con=self.con)
        # for gridpos, (text1, text2) in self._overlay.items():
        #     mjr_overlay(const.FONTSCALE_150, gridpos, rect, text1.encode(), text2.encode(), &self._con)

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 0

    def read_pixels(self, width, height, depth=False, segmentation=False):
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb_img = np.empty((height, width, 3), dtype=np.uint8)
        depth_img = np.empty((height, width), dtype=np.float32) if depth else None

        mujoco.mjr_readPixels(rgb=rgb_img, depth=depth_img, viewport=viewport, con=self.con)

        ret_img = rgb_img
        if segmentation:
            seg_img = rgb_img[:, :, 0] + rgb_img[:, :, 1] * (2**8) + rgb_img[:, :, 2] * (2**16)
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
            seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)

            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]

        if depth:
            return (ret_img, depth_img)
        else:
            return ret_img

    def upload_texture(self, tex_id):
        """Uploads given texture to the GPU."""
        self.gl_ctx.make_current()
        mujoco.mjr_uploadTexture(self.model, self.con, tex_id)

    def __del__(self):
        # free mujoco rendering context and GL rendering context.
        # Guarded because interpreter shutdown may run atexit hooks (notably
        # eglTerminate) before this finalizer, which would otherwise raise
        # EGL_NOT_INITIALIZED during render-context teardown.
        try:
            self.con.free()
        except Exception:
            pass
        try:
            self.gl_ctx.free()
        except Exception:
            pass
        del self.con
        del self.gl_ctx
        del self.scn
        del self.cam
        del self.vopt
        del self.pert


class MjRenderContextOffscreen(MjRenderContext):
    def __init__(self, sim, device_id, max_width=640, max_height=480):
        super().__init__(sim, offscreen=True, device_id=device_id, max_width=max_width, max_height=max_height)


class MjSimState:
    """
    A mujoco simulation state.
    """

    def __init__(self, time, qpos, qvel):
        self.time = time
        self.qpos = qpos
        self.qvel = qvel

    @classmethod
    def from_flattened(cls, array, sim):
        """
        Takes flat mjstate array and MjSim instance and
        returns MjSimState.
        """
        idx_time = 0
        idx_qpos = idx_time + 1
        idx_qvel = idx_qpos + sim.model.nq

        time = array[idx_time]
        qpos = array[idx_qpos : idx_qpos + sim.model.nq]
        qvel = array[idx_qvel : idx_qvel + sim.model.nv]
        assert sim.model.na == 0

        return cls(time=time, qpos=qpos, qvel=qvel)

    def flatten(self):
        return np.concatenate([[self.time], self.qpos, self.qvel], axis=0)


class _MjModelMeta(type):
    """
    Metaclass which allows MjModel below to delegate to mujoco.MjModel.

    Taken from dm_control: https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/wrapper/core.py#L244
    """

    def __new__(cls, name, bases, dct):
        for attr in dir(mujoco.MjModel):
            if not attr.startswith("_"):
                if attr not in dct:
                    # pylint: disable=protected-access
                    fget = lambda self, attr=attr: getattr(self._model, attr)
                    fset = lambda self, value, attr=attr: setattr(self._model, attr, value)
                    # pylint: enable=protected-access
                    dct[attr] = property(fget, fset)
        return super().__new__(cls, name, bases, dct)


class MjModel(metaclass=_MjModelMeta):
    """Wrapper class for a MuJoCo 'mjModel' instance.
    MjModel encapsulates features of the model that are expected to remain
    constant. It also contains simulation and visualization options which may be
    changed occasionally, although this is done explicitly by the user.
    """

    _HAS_DYNAMIC_ATTRIBUTES = True

    def __init__(self, model_ptr):
        """Creates a new MjModel instance from a mujoco.MjModel."""
        self._model = model_ptr

        # make useful mappings such as _body_name2id and _body_id2name
        self.make_mappings()

    @classmethod
    def from_xml_path(cls, xml_path):
        """Creates an MjModel instance from a path to a model XML file."""
        model_ptr = _get_model_ptr_from_xml(xml_path=xml_path)
        return cls(model_ptr)

    def __del__(self):
        # free mujoco model
        del self._model

    """
    Some methods supported by sim.model in mujoco-py.
    Copied from https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2611
    """

    def _extract_mj_names(self, name_adr, num_obj, obj_type):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1127
        """

        ### TODO: fix this to use @name_adr like mujoco-py - more robust than assuming IDs are continuous ###

        # objects don't need to be named in the XML, so name might be None
        id2name = {i: None for i in range(num_obj)}
        name2id = {}
        for i in range(num_obj):
            name = mujoco.mj_id2name(self._model, obj_type, i)
            name2id[name] = i
            id2name[i] = name

        # # objects don't need to be named in the XML, so name might be None
        # id2name = { i: None for i in range(num_obj) }
        # name2id = {}
        # for i in range(num_obj):
        #     name = self.model.names[name_adr[i]]
        #     decoded_name = name.decode()
        #     if decoded_name:
        #         obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        #         assert (0 <= obj_id < num_obj) and (id2name[obj_id] is None)
        #         name2id[decoded_name] = obj_id
        #         id2name[obj_id] = decoded_name

        # sort names by increasing id to keep order deterministic
        return tuple(id2name[nid] for nid in sorted(name2id.values())), name2id, id2name

    def make_mappings(self):
        """
        Make some useful internal mappings that mujoco-py supported.
        """
        p = self
        self.body_names, self._body_name2id, self._body_id2name = self._extract_mj_names(
            p.name_bodyadr, p.nbody, mujoco.mjtObj.mjOBJ_BODY
        )
        self.joint_names, self._joint_name2id, self._joint_id2name = self._extract_mj_names(
            p.name_jntadr, p.njnt, mujoco.mjtObj.mjOBJ_JOINT
        )
        self.geom_names, self._geom_name2id, self._geom_id2name = self._extract_mj_names(
            p.name_geomadr, p.ngeom, mujoco.mjtObj.mjOBJ_GEOM
        )
        self.site_names, self._site_name2id, self._site_id2name = self._extract_mj_names(
            p.name_siteadr, p.nsite, mujoco.mjtObj.mjOBJ_SITE
        )
        self.light_names, self._light_name2id, self._light_id2name = self._extract_mj_names(
            p.name_lightadr, p.nlight, mujoco.mjtObj.mjOBJ_LIGHT
        )
        self.camera_names, self._camera_name2id, self._camera_id2name = self._extract_mj_names(
            p.name_camadr, p.ncam, mujoco.mjtObj.mjOBJ_CAMERA
        )
        self.actuator_names, self._actuator_name2id, self._actuator_id2name = self._extract_mj_names(
            p.name_actuatoradr, p.nu, mujoco.mjtObj.mjOBJ_ACTUATOR
        )
        self.sensor_names, self._sensor_name2id, self._sensor_id2name = self._extract_mj_names(
            p.name_sensoradr, p.nsensor, mujoco.mjtObj.mjOBJ_SENSOR
        )
        self.tendon_names, self._tendon_name2id, self._tendon_id2name = self._extract_mj_names(
            p.name_tendonadr, p.ntendon, mujoco.mjtObj.mjOBJ_TENDON
        )
        self.mesh_names, self._mesh_name2id, self._mesh_id2name = self._extract_mj_names(
            p.name_meshadr, p.nmesh, mujoco.mjtObj.mjOBJ_MESH
        )

    def body_id2name(self, id):
        """Get body name from mujoco body id."""
        if id not in self._body_id2name:
            raise ValueError("No body with id %d exists." % id)
        return self._body_id2name[id]

    def body_name2id(self, name):
        """Get body id from mujoco body name."""
        if name not in self._body_name2id:
            raise ValueError('No "body" with name %s exists. Available "body" names = %s.' % (name, self.body_names))
        return self._body_name2id[name]

    def joint_id2name(self, id):
        """Get joint name from mujoco joint id."""
        if id not in self._joint_id2name:
            raise ValueError("No joint with id %d exists." % id)
        return self._joint_id2name[id]

    def joint_name2id(self, name):
        """Get joint id from joint name."""
        if name not in self._joint_name2id:
            raise ValueError('No "joint" with name %s exists. Available "joint" names = %s.' % (name, self.joint_names))
        return self._joint_name2id[name]

    def geom_id2name(self, id):
        """Get geom name from  geom id."""
        if id not in self._geom_id2name:
            raise ValueError("No geom with id %d exists." % id)
        return self._geom_id2name[id]

    def geom_name2id(self, name):
        """Get geom id from  geom name."""
        if name not in self._geom_name2id:
            raise ValueError('No "geom" with name %s exists. Available "geom" names = %s.' % (name, self.geom_names))
        return self._geom_name2id[name]

    def site_id2name(self, id):
        """Get site name from site id."""
        if id not in self._site_id2name:
            raise ValueError("No site with id %d exists." % id)
        return self._site_id2name[id]

    def site_name2id(self, name):
        """Get site id from site name."""
        if name not in self._site_name2id:
            raise ValueError('No "site" with name %s exists. Available "site" names = %s.' % (name, self.site_names))
        return self._site_name2id[name]

    def light_id2name(self, id):
        """Get light name from light id."""
        if id not in self._light_id2name:
            raise ValueError("No light with id %d exists." % id)
        return self._light_id2name[id]

    def light_name2id(self, name):
        """Get light id from light name."""
        if name not in self._light_name2id:
            raise ValueError('No "light" with name %s exists. Available "light" names = %s.' % (name, self.light_names))
        return self._light_name2id[name]

    def camera_id2name(self, id):
        """Get camera name from camera id."""
        if id not in self._camera_id2name:
            raise ValueError("No camera with id %d exists." % id)
        return self._camera_id2name[id]

    def camera_name2id(self, name):
        """Get camera id from  camera name."""
        if name not in self._camera_name2id:
            raise ValueError(
                'No "camera" with name %s exists. Available "camera" names = %s.' % (name, self.camera_names)
            )
        return self._camera_name2id[name]

    def actuator_id2name(self, id):
        """Get actuator name from actuator id."""
        if id not in self._actuator_id2name:
            raise ValueError("No actuator with id %d exists." % id)
        return self._actuator_id2name[id]

    def actuator_name2id(self, name):
        """Get actuator id from actuator name."""
        if name not in self._actuator_name2id:
            raise ValueError(
                'No "actuator" with name %s exists. Available "actuator" names = %s.' % (name, self.actuator_names)
            )
        return self._actuator_name2id[name]

    def sensor_id2name(self, id):
        """Get sensor name from sensor id."""
        if id not in self._sensor_id2name:
            raise ValueError("No sensor with id %d exists." % id)
        return self._sensor_id2name[id]

    def sensor_name2id(self, name):
        """Get sensor id from sensor name."""
        if name not in self._sensor_name2id:
            raise ValueError(
                'No "sensor" with name %s exists. Available "sensor" names = %s.' % (name, self.sensor_names)
            )
        return self._sensor_name2id[name]

    def tendon_id2name(self, id):
        """Get tendon name from tendon id."""
        if id not in self._tendon_id2name:
            raise ValueError("No tendon with id %d exists." % id)
        return self._tendon_id2name[id]

    def tendon_name2id(self, name):
        """Get tendon id from tendon name."""
        if name not in self._tendon_name2id:
            raise ValueError(
                'No "tendon" with name %s exists. Available "tendon" names = %s.' % (name, self.tendon_names)
            )
        return self._tendon_name2id[name]

    def mesh_id2name(self, id):
        """Get mesh name from  mesh id."""
        if id not in self._mesh_id2name:
            raise ValueError("No mesh with id %d exists." % id)
        return self._mesh_id2name[id]

    def mesh_name2id(self, name):
        """Get mesh id from mesh name."""
        if name not in self._mesh_name2id:
            raise ValueError('No "mesh" with name %s exists. Available "mesh" names = %s.' % (name, self.mesh_names))
        return self._mesh_name2id[name]

    # def userdata_id2name(self, id):
    #     if id not in self._userdata_id2name:
    #         raise ValueError("No userdata with id %d exists." % id)
    #     return self._userdata_id2name[id]

    # def userdata_name2id(self, name):
    #     if name not in self._userdata_name2id:
    #         raise ValueError("No \"userdata\" with name %s exists. Available \"userdata\" names = %s." % (name, self.userdata_names))
    #     return self._userdata_name2id[name]

    def get_xml(self):
        with TemporaryDirectory() as td:
            filename = os.path.join(td, "model.xml")
            ret = mujoco.mj_saveLastXML(filename.encode(), self._model)
            return open(filename).read()

    def get_joint_qpos_addr(self, name):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1178

        Returns the qpos address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for pos[start:end] access.
        """
        joint_id = self.joint_name2id(name)
        joint_type = self.jnt_type[joint_id]
        joint_addr = self.jnt_qposadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    def get_joint_qvel_addr(self, name):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1202

        Returns the qvel address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for vel[start:end] access.
        """
        joint_id = self.joint_name2id(name)
        joint_type = self.jnt_type[joint_id]
        joint_addr = self.jnt_dofadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 3
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)


class _MjDataMeta(type):
    """
    Metaclass which allows MjData below to delegate to mujoco.MjData.

    Taken from dm_control.
    """

    def __new__(cls, name, bases, dct):
        for attr in dir(mujoco.MjData):
            if not attr.startswith("_"):
                if attr not in dct:
                    # pylint: disable=protected-access
                    fget = lambda self, attr=attr: getattr(self._data, attr)
                    fset = lambda self, value, attr=attr: setattr(self._data, attr, value)
                    # pylint: enable=protected-access
                    dct[attr] = property(fget, fset)
        return super().__new__(cls, name, bases, dct)


class MjData(metaclass=_MjDataMeta):
    """Wrapper class for a MuJoCo 'mjData' instance.
    MjData contains all of the dynamic variables and intermediate results produced
    by the simulation. These are expected to change on each simulation timestep.
    The properties without docstrings are defined in mujoco source code from https://github.com/deepmind/mujoco/blob/062cb53a4a14b2a7a900453613a7ce498728f9d8/include/mujoco/mjdata.h#L126.
    """

    def __init__(self, model):
        """Construct a new MjData instance.
        Args:
          model: An MjModel instance.
        """
        self._model = model
        self._data = mujoco.MjData(model._model)

    @property
    def model(self):
        """The parent MjModel for this MjData instance."""
        return self._model

    def __del__(self):
        # free mujoco data
        del self._data

    """
    Some methods supported by sim.data in mujoco-py.
    Copied from https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2611
    """

    @property
    def body_xpos(self):
        """
        Note: mujoco-py used to support sim.data.body_xpos but DM mujoco bindings requires sim.data.xpos,
              so we explicitly expose this as a property
        """
        return self._data.xpos

    @property
    def body_xquat(self):
        """
        Note: mujoco-py used to support sim.data.body_xquat but DM mujoco bindings requires sim.data.xquat,
              so we explicitly expose this as a property
        """
        return self._data.xquat

    @property
    def body_xmat(self):
        """
        Note: mujoco-py used to support sim.data.body_xmat but DM mujoco bindings requires sim.data.xmax,
              so we explicitly expose this as a property
        """
        return self._data.xmat

    def get_body_xpos(self, name):
        """
        Query cartesian position of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xpos (np.ndarray): The xpos value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        return self.xpos[bid]

    def get_body_xquat(self, name):
        """
        Query the rotation of a mujoco body in quaternion (in wxyz convention) using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xquat (np.ndarray): The xquat value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        return self.xquat[bid]

    def get_body_xmat(self, name):
        """
        Query the rotation of a mujoco body in a rotation matrix using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xmat (np.ndarray): The xmat value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        return self.xmat[bid].reshape((3, 3))

    def get_body_jacp(self, name):
        """
        Query the position jacobian of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            jacp (np.ndarray): The jacp value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model._model, self._data, jacp, None, bid)
        return jacp

    def get_body_jacr(self, name):
        """
        Query the rotation jacobian of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            jacr (np.ndarray): The jacr value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model._model, self._data, None, jacr, bid)
        return jacr

    def get_body_xvelp(self, name):
        """
        Query the translational velocity of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xvelp (np.ndarray): The translational velocity of the mujoco body.
        """
        jacp = self.get_body_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_body_xvelr(self, name):
        """
        Query the rotational velocity of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xvelr (np.ndarray): The rotational velocity of the mujoco body.
        """
        jacr = self.get_body_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_geom_xpos(self, name):
        """
        Query the cartesian position of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            geom_xpos (np.ndarray): The cartesian position of the mujoco body.
        """
        gid = self.model.geom_name2id(name)
        return self.geom_xpos[gid]

    def get_geom_xmat(self, name):
        """
        Query the rotation of a mujoco geom in a rotation matrix using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            geom_xmat (np.ndarray): The 3x3 rotation matrix of the mujoco geom.
        """
        gid = self.model.geom_name2id(name)
        return self.geom_xmat[gid].reshape((3, 3))

    def get_geom_jacp(self, name):
        """
        Query the position jacobian of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            jacp (np.ndarray): The jacp value of the mujoco geom
        """
        gid = self.model.geom_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model._model, self._data, jacp, None, gid)
        return jacp

    def get_geom_jacr(self, name):
        """
        Query the rotation jacobian of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            jacr (np.ndarray): The jacr value of the mujoco geom
        """
        gid = self.model.geom_name2id(name)
        jacv = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model._model, self._data, None, jacv, gid)
        return jacr

    def get_geom_xvelp(self, name):
        """
        Query the translational velocity of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            xvelp (np.ndarray): The translational velocity of the mujoco geom
        """
        jacp = self.get_geom_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_geom_xvelr(self, name):
        """
        Query the rotational velocity of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            xvelr (np.ndarray): The rotational velocity of the mujoco geom
        """
        jacr = self.get_geom_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_site_xpos(self, name):
        """
        Query the cartesian position of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            site_xpos (np.ndarray): The carteisan position of the mujoco site
        """
        sid = self.model.site_name2id(name)
        return self.site_xpos[sid]

    def get_site_xmat(self, name):
        """
        Query the rotation of a mujoco site in a rotation matrix using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            site_xmat (np.ndarray): The 3x3 rotation matrix of the mujoco site.
        """
        sid = self.model.site_name2id(name)
        return self.site_xmat[sid].reshape((3, 3))

    def get_site_jacp(self, name):
        """
        Query the position jacobian of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            jacp (np.ndarray): The jacp value of the mujoco site
        """
        sid = self.model.site_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model._model, self._data, jacp, None, sid)
        return jacp

    def get_site_jacr(self, name):
        """
        Query the rotation jacobian of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            jacr (np.ndarray): The jacr value of the mujoco site
        """
        sid = self.model.site_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model._model, self._data, None, jacr, sid)
        return jacr

    def get_site_xvelp(self, name):
        """
        Query the translational velocity of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            xvelp (np.ndarray): The translational velocity of the mujoco site
        """
        jacp = self.get_site_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_site_xvelr(self, name):
        """
        Query the rotational velocity of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            xvelr (np.ndarray): The rotational velocity of the mujoco site
        """
        jacr = self.get_site_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_camera_xpos(self, name):
        """
        Get the cartesian position of a camera using name

        Args:
            name (str): The name of a camera
        Returns:
            cam_xpos (np.ndarray): The cartesian position of a camera
        """
        cid = self.model.camera_name2id(name)
        return self.cam_xpos[cid]

    def get_camera_xmat(self, name):
        """
        Get the rotation of a camera in a rotation matrix using name

        Args:
            name (str): The name of a camera
        Returns:
            cam_xmat (np.ndarray): The 3x3 rotation matrix of a camera
        """
        cid = self.model.camera_name2id(name)
        return self.cam_xmat[cid].reshape((3, 3))

    def get_light_xpos(self, name):
        """
        Get cartesian position of a light source

        Args:
            name (str): The name of a lighting source
        Returns:
            light_xpos (np.ndarray): The cartesian position of the light source
        """
        lid = self.model.light_name2id(name)
        return self.light_xpos[lid]

    def get_light_xdir(self, name):
        """
        Get the direction of a light source using name

        Args:
            name (str): The name of a light
        Returns:
            light_xdir (np.ndarray): The direction vector of the lightsource
        """
        lid = self.model.light_name2id(name)
        return self.light_xdir[lid]

    def get_sensor(self, name):
        """
        Get the data of a sensor using name

        Args:
            name (str): The name of a sensor
        Returns:
            sensordata (np.ndarray): The sensor data vector
        """
        sid = self.model.sensor_name2id(name)
        return self.sensordata[sid]

    def get_mocap_pos(self, name):
        """
        Get the position of a mocap body using name.

        Args:
            name (str): The name of a joint
        Returns:
            mocap_pos (np.ndarray): The current position of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        return self.mocap_pos[mocap_id]

    def set_mocap_pos(self, name, value):
        """
        Set the quaternion of a mocap body using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint position of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.mocap_pos[mocap_id] = value

    def get_mocap_quat(self, name):
        """
        Get the quaternion of a mocap body using name.

        Args:
            name (str): The name of a joint
        Returns:
            mocap_quat (np.ndarray): The current quaternion of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        return self.mocap_quat[mocap_id]

    def set_mocap_quat(self, name, value):
        """
        Set the quaternion of a mocap body using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint quaternion of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.mocap_quat[mocap_id] = value

    def get_joint_qpos(self, name):
        """
        Get the position of a joint using name.

        Args:
            name (str): The name of a joint

        Returns:
            qpos (np.ndarray): The current position of a joint.
        """
        addr = self.model.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.qpos[addr]
        else:
            start_i, end_i = addr
            return self.qpos[start_i:end_i]

    def set_joint_qpos(self, name, value):
        """
        Set the velocities of a joint using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint velocity of a joint.
        """
        addr = self.model.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.qpos[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (end_i - start_i,), "Value has incorrect shape %s: %s" % (name, value)
            self.qpos[start_i:end_i] = value

    def get_joint_qvel(self, name):
        """
        Get the velocity of a joint using name.

        Args:
            name (str): The name of a joint

        Returns:
            qvel (np.ndarray): The current velocity of a joint.
        """
        addr = self.model.get_joint_qvel_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.qvel[addr]
        else:
            start_i, end_i = addr
            return self.qvel[start_i:end_i]

    def set_joint_qvel(self, name, value):
        """
        Set the velocities of a mjo using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint velocity of a joint.
        """
        addr = self.model.get_joint_qvel_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.qvel[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (end_i - start_i,), "Value has incorrect shape %s: %s" % (name, value)
            self.qvel[start_i:end_i] = value


class MjSim:
    """
    Meant to somewhat replicate functionality in mujoco-py's MjSim object
    (see https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjsim.pyx).
    """

    def __init__(self, model):
        """
        Args:
            model: should be an MjModel instance created via a factory function
                such as mujoco.MjModel.from_xml_string(xml)
        """
        self.model = MjModel(model)
        self.data = MjData(self.model)

        # offscreen render context object
        self._render_context_offscreen = None

    @classmethod
    def from_xml_string(cls, xml):
        model = mujoco.MjModel.from_xml_string(xml)
        return cls(model)

    @classmethod
    def from_xml_file(cls, xml_file):
        f = open(xml_file, "r")
        xml = f.read()
        f.close()
        return cls.from_xml_string(xml)

    def reset(self):
        """Reset simulation."""
        mujoco.mj_resetData(self.model._model, self.data._data)

    def forward(self):
        """Forward call to synchronize derived quantities."""
        mujoco.mj_forward(self.model._model, self.data._data)

    def step(self, with_udd=True):
        """Step simulation."""
        mujoco.mj_step(self.model._model, self.data._data)

    def render(
        self,
        width=None,
        height=None,
        *,
        camera_name=None,
        depth=False,
        mode="offscreen",
        device_id=-1,
        segmentation=False,
    ):
        """
        Renders view from a camera and returns image as an `numpy.ndarray`.
        Args:
        - width (int): desired image width.
        - height (int): desired image height.
        - camera_name (str): name of camera in model. If None, the free
            camera will be used.
        - depth (bool): if True, also return depth buffer
        - device (int): device to use for rendering (only for GPU-backed
            rendering).
        Returns:
        - rgb (uint8 array): image buffer from camera
        - depth (float array): depth buffer from camera (only returned
            if depth=True)
        """
        if camera_name is None:
            camera_id = None
        else:
            camera_id = self.model.camera_name2id(camera_name)

        assert mode == "offscreen", "only offscreen supported for now"
        assert self._render_context_offscreen is not None
        with _MjSim_render_lock:
            self._render_context_offscreen.render(
                width=width, height=height, camera_id=camera_id, segmentation=segmentation
            )
            return self._render_context_offscreen.read_pixels(width, height, depth=depth, segmentation=segmentation)

    def add_render_context(self, render_context):
        assert render_context.offscreen
        if self._render_context_offscreen is not None:
            # free context
            del self._render_context_offscreen
        self._render_context_offscreen = render_context

    def get_state(self):
        """Return MjSimState instance for current state."""
        return MjSimState(
            time=self.data.time,
            qpos=np.copy(self.data.qpos),
            qvel=np.copy(self.data.qvel),
        )

    def set_state(self, value):
        """
        Set internal state from MjSimState instance. Should
        call @forward afterwards to synchronize derived quantities.
        """
        self.data.time = value.time
        self.data.qpos[:] = np.copy(value.qpos)
        self.data.qvel[:] = np.copy(value.qvel)

    def set_state_from_flattened(self, value):
        """
        Set internal mujoco state using flat mjstate array. Should
        call @forward afterwards to synchronize derived quantities.

        See https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjsimstate.pyx#L54
        """
        state = MjSimState.from_flattened(value, self)

        # do this instead of @set_state to avoid extra copy of qpos and qvel
        self.data.time = state.time
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel

    def free(self):
        # clean up here to prevent memory leaks
        del self._render_context_offscreen
        del self.data
        del self.model
        del self


# ---------------------------------------------------------------------------
# MuJoCo Warp — GPU-parallelised simulation
# ---------------------------------------------------------------------------


class MjDataWarp:
    """
    Accessor for ``mujoco_warp.Data`` that mirrors the :class:`MjData`
    interface while exposing batched GPU arrays.

    Unlike :class:`MjData` (single environment), every array property returns
    a ``warp.array`` whose first dimension is ``num_envs``.  Writes must also
    supply ``warp.array`` values; they are copied directly onto the GPU without
    an intermediate CPU round-trip.

    Named getters (``get_body_xpos``, ``get_joint_qpos``, …) return new
    ``warp.array`` objects extracted from the relevant slice of the full batch.

    Named setters (``set_joint_qpos``, ``set_mocap_pos``, …) accept
    ``warp.array`` values and write them back into the batch.

    .. note::
        Jacobian methods use ``mujoco_warp.jac`` and run entirely on the GPU,
        returning ``(num_envs, 3, nv)`` numpy arrays.  Velocity methods
        (``get_*_xvelp/r``) are derived from the Jacobians and return
        ``(num_envs, 3)`` numpy arrays.
    """

    def __init__(
        self, model: MjModel, warp_data: "mujoco_warp.Data", warp_model: "mujoco_warp.Model", num_envs: int
    ) -> None:
        """
        Args:
            model: shared :class:`MjModel` instance.
            warp_data: ``mujoco_warp.Data`` holding GPU arrays for all envs.
            warp_model: ``mujoco_warp.Model`` on the GPU.
            num_envs: number of parallel worlds.
        """
        self._model = model
        self._data = warp_data
        self._warp_model = warp_model
        self.num_envs = num_envs
        # Cache for body_wp arrays (constant per bodyid, avoids repeated CPU→GPU copies)
        self._body_wp_cache: dict = {}

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    @property
    def model(self) -> MjModel:
        return self._model

    def __getattr__(self, name: str) -> "wp.array":
        """Delegate to warp data, returning the raw ``wp.array``."""
        try:
            return getattr(self._data, name)
        except AttributeError:
            raise AttributeError(f"MjDataWarp has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        """Copy a value into the corresponding GPU storage array.

        Accepts both ``wp.array`` and ``torch.Tensor`` as *value*; tensors are
        converted to numpy before the warp copy.
        """
        if name.startswith("_") or name == "num_envs":
            object.__setattr__(self, name, value)
            return
        try:
            target = getattr(self._data, name)
        except AttributeError:
            object.__setattr__(self, name, value)
            return
        if hasattr(target, "numpy"):  # it is a warp array
            import torch
            import warp as wp

            if isinstance(value, torch.Tensor):
                value = wp.from_numpy(value.cpu().numpy().astype(np.float32), device=target.device)
            wp.copy(target, value)
        else:
            object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Zero-copy array proxies — _BatchedArray wraps wp.to_torch so that
    # arr[i] returns (num_envs, ...) without knowing the batch dimension.
    # Scalar arrays (qpos, qvel, qfrc_bias) are returned as raw tensors
    # since callers already use explicit [:, idx] slicing on them.
    # ------------------------------------------------------------------

    @property
    def ctrl(self) -> "_BatchedArray":
        """Zero-copy ``(num_envs, nu)`` writable proxy."""
        return _BatchedArray(self._data.ctrl)

    @property
    def body_xpos(self) -> "_BatchedArray":
        """Zero-copy ``(num_envs, nbody, 3)`` proxy."""
        return _BatchedArray(self._data.xpos)

    @property
    def body_xquat(self) -> "_BatchedArray":
        """Zero-copy ``(num_envs, nbody, 4)`` proxy (wxyz convention)."""
        return _BatchedArray(self._data.xquat)

    @property
    def body_xmat(self) -> "_BatchedArray":
        """Zero-copy ``(num_envs, nbody, 3, 3)`` proxy."""
        return _BatchedArray(self._data.xmat)

    @property
    def geom_xpos(self) -> "_BatchedArray":
        """Zero-copy ``(num_envs, ngeom, 3)`` proxy."""
        return _BatchedArray(self._data.geom_xpos)

    @property
    def site_xpos(self) -> "_BatchedArray":
        """Zero-copy ``(num_envs, nsite, 3)`` proxy."""
        return _BatchedArray(self._data.site_xpos)

    @property
    def site_xmat(self) -> "_BatchedArray":
        """Zero-copy ``(num_envs, nsite, 3, 3)`` proxy."""
        return _BatchedArray(self._data.site_xmat)

    @property
    def qpos(self) -> "_BatchedArray":
        """Zero-copy ``(num_envs, nq)`` proxy. Use ``.tensor`` for the raw tensor."""
        return _BatchedArray(self._data.qpos)

    @property
    def qvel(self) -> "_BatchedArray":
        """Zero-copy ``(num_envs, nv)`` proxy. Use ``.tensor`` for the raw tensor."""
        return _BatchedArray(self._data.qvel)

    @property
    def qfrc_bias(self) -> "_BatchedArray":
        """Zero-copy ``(num_envs, nv)`` proxy (gravity/Coriolis forces)."""
        return _BatchedArray(self._data.qfrc_bias)

    @property
    def qM(self) -> "torch.Tensor":
        """Dense mass matrix: ``(num_envs, padded_nv, padded_nv)`` CUDA float32.

        MuJoCo Warp stores the full symmetric mass matrix in a dense padded
        block (padded to the next warp SIMD width).  The top-left ``nv × nv``
        sub-block contains the actual values — no ``mj_fullM`` call required.
        """
        import warp as wp
        return wp.to_torch(self._data.qM)

    # ------------------------------------------------------------------
    # Named getters — delegate to array proxies (zero-copy)
    # ------------------------------------------------------------------


    def get_body_xpos(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` CUDA tensor."""
        return self.body_xpos[self._model.body_name2id(name)]

    def get_body_xquat(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 4)`` CUDA tensor (wxyz convention)."""
        return self.body_xquat[self._model.body_name2id(name)]

    def get_body_xmat(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3, 3)`` CUDA tensor."""
        return self.body_xmat[self._model.body_name2id(name)]

    def get_geom_xpos(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` CUDA tensor."""
        return self.geom_xpos[self._model.geom_name2id(name)]

    def get_geom_xmat(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3, 3)`` CUDA tensor."""
        import warp as wp
        gid = self._model.geom_name2id(name)
        return wp.to_torch(self._data.geom_xmat)[: , gid]

    def get_site_xpos(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` CUDA tensor."""
        return self.site_xpos[self._model.site_name2id(name)]

    def get_site_xmat(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3, 3)`` CUDA tensor."""
        return self.site_xmat[self._model.site_name2id(name)]

    def get_camera_xpos(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` CUDA tensor."""
        import warp as wp
        cid = self._model.camera_name2id(name)
        return wp.to_torch(self._data.cam_xpos)[:, cid]

    def get_camera_xmat(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3, 3)`` CUDA tensor."""
        import warp as wp
        cid = self._model.camera_name2id(name)
        return wp.to_torch(self._data.cam_xmat)[:, cid]

    def get_sensor(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs,)`` CUDA tensor."""
        import warp as wp
        sid = self._model.sensor_name2id(name)
        return wp.to_torch(self._data.sensordata)[:, sid]

    # ------------------------------------------------------------------
    # Joint position / velocity getters and setters
    # ------------------------------------------------------------------

    def get_joint_qpos(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs,)`` or ``(num_envs, ndim)`` CUDA tensor."""
        import warp as wp
        addr = self._model.get_joint_qpos_addr(name)
        t = wp.to_torch(self._data.qpos)
        if isinstance(addr, (int, np.int32, np.int64)):
            return t[:, addr]
        return t[:, addr[0]:addr[1]]

    def set_joint_qpos(self, name: str, value) -> None:
        """Set joint position for all envs. Accepts ndarray, wp.array, or torch.Tensor."""
        import torch
        import warp as wp
        addr = self._model.get_joint_qpos_addr(name)
        val = value if isinstance(value, torch.Tensor) else torch.as_tensor(
            value if isinstance(value, np.ndarray) else value.numpy(),
            device="cuda", dtype=torch.float32,
        )
        t = wp.to_torch(self._data.qpos)
        if isinstance(addr, (int, np.int32, np.int64)):
            t[:, addr] = val
        else:
            t[:, addr[0]:addr[1]] = val

    def get_joint_qvel(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs,)`` or ``(num_envs, ndim)`` CUDA tensor."""
        import warp as wp
        addr = self._model.get_joint_qvel_addr(name)
        t = wp.to_torch(self._data.qvel)
        if isinstance(addr, (int, np.int32, np.int64)):
            return t[:, addr]
        return t[:, addr[0]:addr[1]]

    def set_joint_qvel(self, name: str, value) -> None:
        """Set joint velocity for all envs. Accepts ndarray, wp.array, or torch.Tensor."""
        import torch
        import warp as wp
        addr = self._model.get_joint_qvel_addr(name)
        val = value if isinstance(value, torch.Tensor) else torch.as_tensor(
            value if isinstance(value, np.ndarray) else value.numpy(),
            device="cuda", dtype=torch.float32,
        )
        t = wp.to_torch(self._data.qvel)
        if isinstance(addr, (int, np.int32, np.int64)):
            t[:, addr] = val
        else:
            t[:, addr[0]:addr[1]] = val

    def set_qpos_indexed(self, indexes, values) -> None:
        """Set ``qpos[:, indexes] = values`` for all envs. Accepts ndarray or torch.Tensor."""
        import torch
        import warp as wp
        val = values if isinstance(values, torch.Tensor) else torch.as_tensor(
            values, device="cuda", dtype=torch.float32
        )
        wp.to_torch(self._data.qpos)[:, indexes] = val

    # ------------------------------------------------------------------
    # Mocap getters and setters
    # ------------------------------------------------------------------

    def get_mocap_pos(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` CUDA tensor."""
        import warp as wp
        body_id = self._model.body_name2id(name)
        mocap_id = self._model.body_mocapid[body_id]
        return wp.to_torch(self._data.mocap_pos)[:, mocap_id]

    def set_mocap_pos(self, name: str, value) -> None:
        """Set mocap body position for all envs. Accepts ndarray, wp.array, or torch.Tensor."""
        import torch
        import warp as wp
        body_id = self._model.body_name2id(name)
        mocap_id = self._model.body_mocapid[body_id]
        val = value if isinstance(value, torch.Tensor) else torch.as_tensor(
            value if isinstance(value, np.ndarray) else value.numpy(),
            device="cuda", dtype=torch.float32,
        )
        wp.to_torch(self._data.mocap_pos)[:, mocap_id] = val

    def get_mocap_quat(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 4)`` CUDA tensor (wxyz convention)."""
        import warp as wp
        body_id = self._model.body_name2id(name)
        mocap_id = self._model.body_mocapid[body_id]
        return wp.to_torch(self._data.mocap_quat)[:, mocap_id]

    def set_mocap_quat(self, name: str, value) -> None:
        """Set mocap body orientation for all envs. Accepts ndarray, wp.array, or torch.Tensor."""
        import torch
        import warp as wp
        body_id = self._model.body_name2id(name)
        mocap_id = self._model.body_mocapid[body_id]
        val = value if isinstance(value, torch.Tensor) else torch.as_tensor(
            value if isinstance(value, np.ndarray) else value.numpy(),
            device="cuda", dtype=torch.float32,
        )
        wp.to_torch(self._data.mocap_quat)[:, mocap_id] = val

    # ------------------------------------------------------------------
    # Jacobians — computed on GPU via mujoco_warp.jac
    # ------------------------------------------------------------------

    def _compute_jacs(self, point: "torch.Tensor", bodyid: int) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Compute translational and rotational Jacobians for a point on a body,
        across all envs simultaneously on the GPU.

        Args:
            point: ``(num_envs, 3)`` CUDA float32 tensor — point in world coords per env.
            bodyid: integer body ID (same for all envs).

        Returns:
            Tuple ``(jacp, jacr)`` each as a ``(num_envs, 3, nv)`` CUDA tensor.
        """
        import mujoco_warp as mjwarp
        import warp as wp

        nv = self._model.nv
        device = self._data.qpos.device
        jacp_wp = wp.zeros((self.num_envs, 3, nv), dtype=float, device=device)
        jacr_wp = wp.zeros((self.num_envs, 3, nv), dtype=float, device=device)
        # Cache the body_wp array — bodyid is constant for a given site/body, so
        # building [bodyid]*num_envs and uploading to GPU only happens once.
        if bodyid not in self._body_wp_cache:
            self._body_wp_cache[bodyid] = wp.array(
                [bodyid] * self.num_envs, dtype=wp.int32, device=device
            )
        body_wp = self._body_wp_cache[bodyid]
        # Reinterpret (num_envs, 3) float32 tensor as (num_envs,) vec3f — zero-copy.
        point_wp = wp.from_torch(point.contiguous(), dtype=wp.vec3f)
        mjwarp.jac(self._warp_model, self._data, jacp_wp, jacr_wp, point_wp, body_wp)
        return wp.to_torch(jacp_wp), wp.to_torch(jacr_wp)

    def get_site_jacs(self, name: str) -> "tuple[torch.Tensor, torch.Tensor]":
        """Returns ``(jacp, jacr)`` both as ``(num_envs, 3, nv)`` CUDA tensors in one kernel call."""
        sid = self._model.site_name2id(name)
        bid = int(self._model._model.site_bodyid[sid])
        return self._compute_jacs(self.site_xpos[sid], bid)

    def get_body_jacp(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3, nv)`` position Jacobian as a zero-copy CUDA tensor."""
        bid = self._model.body_name2id(name)
        jacp, _ = self._compute_jacs(self.body_xpos[bid], bid)
        return jacp

    def get_body_jacr(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3, nv)`` rotation Jacobian as a zero-copy CUDA tensor."""
        bid = self._model.body_name2id(name)
        _, jacr = self._compute_jacs(self.body_xpos[bid], bid)
        return jacr

    def get_site_jacp(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3, nv)`` position Jacobian as a zero-copy CUDA tensor."""
        sid = self._model.site_name2id(name)
        bid = int(self._model._model.site_bodyid[sid])
        jacp, _ = self._compute_jacs(self.site_xpos[sid], bid)
        return jacp

    def get_site_jacr(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3, nv)`` rotation Jacobian as a zero-copy CUDA tensor."""
        sid = self._model.site_name2id(name)
        bid = int(self._model._model.site_bodyid[sid])
        _, jacr = self._compute_jacs(self.site_xpos[sid], bid)
        return jacr

    def get_geom_jacp(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3, nv)`` position Jacobian as a zero-copy CUDA tensor."""
        gid = self._model.geom_name2id(name)
        bid = int(self._model._model.geom_bodyid[gid])
        jacp, _ = self._compute_jacs(self.geom_xpos[gid], bid)
        return jacp

    def get_geom_jacr(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3, nv)`` rotation Jacobian as a zero-copy CUDA tensor."""
        gid = self._model.geom_name2id(name)
        bid = int(self._model._model.geom_bodyid[gid])
        _, jacr = self._compute_jacs(self.geom_xpos[gid], bid)
        return jacr

    # ------------------------------------------------------------------
    # Velocity getters — derived from Jacobians × qvel, shape (num_envs, 3)
    # ------------------------------------------------------------------

    def _jac_times_qvel(self, jac: "torch.Tensor") -> "torch.Tensor":
        """Batched matrix-vector multiply: ``jac @ qvel`` per env.

        Args:
            jac: ``(num_envs, 3, nv)`` CUDA torch.Tensor.

        Returns:
            ``(num_envs, 3)`` CUDA torch.Tensor.
        """
        import torch

        return torch.einsum("eij,ej->ei", jac, self.qvel.tensor)

    def get_body_xvelp(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` translational velocity for a body as a CUDA torch.Tensor."""
        return self._jac_times_qvel(self.get_body_jacp(name))

    def get_body_xvelr(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` rotational velocity for a body as a CUDA torch.Tensor."""
        return self._jac_times_qvel(self.get_body_jacr(name))

    def get_site_xvelp(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` translational velocity for a site as a CUDA torch.Tensor."""
        return self._jac_times_qvel(self.get_site_jacp(name))

    def get_site_xvelr(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` rotational velocity for a site as a CUDA torch.Tensor."""
        return self._jac_times_qvel(self.get_site_jacr(name))

    def get_geom_xvelp(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` translational velocity for a geom as a CUDA torch.Tensor."""
        return self._jac_times_qvel(self.get_geom_jacp(name))

    def get_geom_xvelr(self, name: str) -> "torch.Tensor":
        """Returns ``(num_envs, 3)`` rotational velocity for a geom as a CUDA torch.Tensor."""
        return self._jac_times_qvel(self.get_geom_jacr(name))


class MjModelWarp(MjModel):
    """MjModel subclass for warp sims — returns CUDA tensors for array properties."""

    @property
    def actuator_ctrlrange(self) -> "torch.Tensor":
        """``(nu, 2)`` CUDA tensor of actuator control ranges."""
        import torch
        return torch.as_tensor(self._model.actuator_ctrlrange, dtype=torch.float32, device="cuda")


class _BatchedArray:
    """Zero-copy proxy for a ``(num_envs, n, ...)`` warp array.

    Wraps via ``wp.to_torch`` (shared GPU memory — no copies).  Plain indexing
    is treated as column indexing across the env dimension so callers do not
    need to know about the batch dimension:

        arr[i]      →  tensor[:, i]       # (num_envs, ...)
        arr[i] = v  →  tensor[:, i] = v   # in-place, propagates to warp
        arr.tensor  →  full (num_envs, n, ...) tensor
    """

    __slots__ = ("_tensor",)

    def __init__(self, warp_arr: "wp.array") -> None:
        import warp as wp
        self._tensor: "torch.Tensor" = wp.to_torch(warp_arr)

    @property
    def tensor(self) -> "torch.Tensor":
        """Full ``(num_envs, n, ...)`` CUDA tensor (zero-copy view)."""
        return self._tensor

    def __getitem__(self, idx) -> "torch.Tensor":
        return self._tensor[:, idx]

    def __setitem__(self, idx, value) -> None:
        import torch
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, device=self._tensor.device, dtype=self._tensor.dtype)
        self._tensor[:, idx] = value


class MjSimWarp(MjSim):
    """
    GPU-parallelised simulation using MuJoCo Warp.

    Manages ``num_envs`` independent physics worlds that share the same model
    XML and are stepped simultaneously on the GPU.  The public interface
    mirrors :class:`MjSim` where possible, with the key difference that
    ``sim.data`` is an :class:`MjDataWarp` whose arrays carry a leading
    ``num_envs`` batch dimension and live on the GPU.

    Typical usage::

        sim = MjSimWarp.from_xml_string(xml, num_envs=64)
        sim.reset()

        # Set controls for all envs at once (warp array on GPU)
        ctrl = wp.zeros((64, sim.model.nu), dtype=wp.float32)
        sim.data.ctrl = ctrl

        sim.step()                              # advance all 64 envs

        qpos: wp.array = sim.data.qpos          # (64, nq) on GPU

        # Pull one env to CPU for rendering or Jacobian queries
        mjdata_0: mujoco.MjData = sim.get_env_data(0)
        img = sim.render(env_idx=0, width=256, height=256)
    """

    # Solver settings tuned for parallel warp rollouts. Per mujoco-warp's
    # put_data contract: njmax and nconmax are per-world caps; naconmax is the
    # total contact-buffer size across all worlds. ccd_iterations must be
    # sufficient for the most complex geometry in the scene.
    #
    # Sizing these involves a tradeoff with num_envs:
    #   efc.J memory  = nworld * njmax * nv_pad  (dense Jacobian, f32)
    #   EPA scratch   = naccdmax * (440 + 164*ccd_iterations) bytes
    # where naccdmax defaults to naconmax = _NACONMAX_PER_ENV * nworld. At very
    # large nworld you may need to shrink these; at smaller nworld (< ~1000)
    # the defaults here leave ample headroom.
    #
    # njmax is per-world: if you see "nefc overflow - please increase njmax to
    # N", bump _NJMAX_PER_ENV above N. Mimicgen tasks (Coffee, Threading, etc.)
    # commonly need ~3000 per world due to mesh-based collision geoms.
    _NJMAX_PER_ENV: int = 3500
    _NCONMAX_PER_ENV: int = 128
    _NACONMAX_PER_ENV: int = 60
    # ccd_iterations is the EPA iteration cap per contact pair. Mujoco-warp
    # warns "opt.ccd_iterations needs to be increased" when it hits this cap
    # without converging; bump if you see the warning recurring.
    _CCD_ITERATIONS: int = 200

    # Currently-active per-task overrides, set by ``robosuite.make()`` when the
    # task's env_kwargs include ``njmax_per_env`` / ``naconmax_per_env``. Read
    # here with precedence: per-instance kwarg > active override > class default.
    # Stored at class scope so that hard_reset rebuilds (which call
    # ``from_xml_string`` directly rather than going through ``robosuite.make``)
    # still pick up the task's override.
    _ACTIVE_NJMAX_PER_ENV: Optional[int] = None
    _ACTIVE_NACONMAX_PER_ENV: Optional[int] = None

    def __init__(
        self,
        model: mujoco.MjModel,
        num_envs: int = 1,
        njmax_per_env: Optional[int] = None,
        naconmax_per_env: Optional[int] = None,
    ) -> None:
        """
        Args:
            model: a ``mujoco.MjModel`` instance (not yet wrapped).
            num_envs: number of parallel worlds to simulate.
            njmax_per_env: per-world cap for active efc constraints. When
                ``None``, falls back to ``_ACTIVE_NJMAX_PER_ENV`` (set by
                ``robosuite.make`` from task env_kwargs), then to
                ``_NJMAX_PER_ENV``.
            naconmax_per_env: per-env contribution to the total contact-buffer
                pool (total ``naconmax`` = this x ``num_envs``). Same fallback
                order as ``njmax_per_env``.
        """
        import mujoco_warp as mjwarp
        import warp as wp

        self._mjwarp = mjwarp
        self._wp = wp
        self.num_envs: int = num_envs

        # Optional env-var knobs (for benchmarking speed-vs-accuracy tradeoffs).
        # Defaults preserve existing behaviour.
        #   ROBOSUITE_WARP_TOLERANCE_CLAMP=1  -> accept mujoco-warp's 1e-6 clamp
        #   ROBOSUITE_WARP_SOLVER_ITERS=<int> -> override opt.iterations
        #   ROBOSUITE_WARP_LS_ITERS=<int>     -> override opt.ls_iterations
        #   ROBOSUITE_WARP_CONE=pyramidal|elliptic -> override opt.cone
        _accept_tol_clamp = os.environ.get("ROBOSUITE_WARP_TOLERANCE_CLAMP", "0") == "1"
        _solver_iters = os.environ.get("ROBOSUITE_WARP_SOLVER_ITERS")
        _ls_iters = os.environ.get("ROBOSUITE_WARP_LS_ITERS")
        _cone_env = os.environ.get("ROBOSUITE_WARP_CONE")

        if _cone_env is not None:
            cone_map = {"pyramidal": 0, "elliptic": 1}
            if _cone_env not in cone_map:
                raise ValueError(f"ROBOSUITE_WARP_CONE={_cone_env!r}; expected pyramidal|elliptic")
            model.opt.cone = cone_map[_cone_env]
        if _solver_iters is not None:
            model.opt.iterations = int(_solver_iters)
        if _ls_iters is not None:
            model.opt.ls_iterations = int(_ls_iters)

        # Shared model wrapper — identical for every env
        self.model = MjModelWarp(model)

        # Warp model: increase CCD iterations to avoid solver warnings under
        # parallel load (the default of 35 is too low for multi-env rollouts).
        self._warp_model = mjwarp.put_model(model)
        self._warp_model.opt.ccd_iterations = self._CCD_ITERATIONS
        # put_model copies from MjModel for most fields, but mirror iterations
        # explicitly in case the warp layout diverges.
        if _solver_iters is not None:
            self._warp_model.opt.iterations = int(_solver_iters)
        if _ls_iters is not None:
            self._warp_model.opt.ls_iterations = int(_ls_iters)

        # Restore the XML-specified solver tolerance. mujoco-warp's put_model
        # unconditionally clamps to max(tolerance, 1e-6) "because f32 GPU", but
        # for contact-heavy manipulation tasks this costs a lot of fidelity
        # relative to mujoco-python's f64 behaviour.
        if not _accept_tol_clamp:
            self._warp_model.opt.tolerance.fill_(float(model.opt.tolerance))

        # Resolve effective buffer sizes: per-instance kwarg > class-level
        # active override (set by robosuite.make from task env_kwargs) > class
        # default. Stored on self for auditability.
        effective_njmax = (
            njmax_per_env
            if njmax_per_env is not None
            else (self._ACTIVE_NJMAX_PER_ENV if self._ACTIVE_NJMAX_PER_ENV is not None else self._NJMAX_PER_ENV)
        )
        effective_naconmax_per_env = (
            naconmax_per_env
            if naconmax_per_env is not None
            else (self._ACTIVE_NACONMAX_PER_ENV if self._ACTIVE_NACONMAX_PER_ENV is not None else self._NACONMAX_PER_ENV)
        )
        self._effective_njmax_per_env: int = int(effective_njmax)
        self._effective_naconmax_per_env: int = int(effective_naconmax_per_env)

        # Warp data: njmax and nconmax are per-world; naconmax is the total
        # contact-buffer size across all worlds (see mujoco_warp.put_data docs).
        _ref_data = mujoco.MjData(model)
        self._warp_data = mjwarp.put_data(
            model, _ref_data, nworld=num_envs,
            njmax=self._effective_njmax_per_env,
            nconmax=self._NCONMAX_PER_ENV,
            naconmax=self._effective_naconmax_per_env * num_envs,
        )

        self.data = MjDataWarp(self.model, self._warp_data, self._warp_model, num_envs)
        self._render_context_offscreen = None

        # CUDA graph capture state (opt-in via ROBOSUITE_WARP_GRAPH=1).
        # First N steps run eagerly so all kernels JIT-compile; next step is
        # captured into a graph; subsequent steps replay via capture_launch.
        # Writing to d.ctrl between launches is fine — graph captures kernel
        # sequence, not input values. Reset and kinematics_forward are outside
        # the captured region.
        self._graph_enabled: bool = os.environ.get("ROBOSUITE_WARP_GRAPH", "0") == "1"
        self._graph_warmup_steps: int = int(os.environ.get("ROBOSUITE_WARP_GRAPH_WARMUP", "3"))
        self._graph_steps_done: int = 0
        self._step_graph = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_xml_string(
        cls,
        xml: str,
        num_envs: int = 1,
        njmax_per_env: Optional[int] = None,
        naconmax_per_env: Optional[int] = None,
    ) -> "MjSimWarp":
        model = mujoco.MjModel.from_xml_string(xml)
        return cls(
            model,
            num_envs=num_envs,
            njmax_per_env=njmax_per_env,
            naconmax_per_env=naconmax_per_env,
        )

    @classmethod
    def from_xml_file(
        cls,
        xml_file: str,
        num_envs: int = 1,
        njmax_per_env: Optional[int] = None,
        naconmax_per_env: Optional[int] = None,
    ) -> "MjSimWarp":
        with open(xml_file, "r") as f:
            xml = f.read()
        return cls.from_xml_string(
            xml,
            num_envs=num_envs,
            njmax_per_env=njmax_per_env,
            naconmax_per_env=naconmax_per_env,
        )

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def reset(self, env_indices: Optional[List[int]] = None) -> None:
        """
        Reset simulation state for the specified envs.

        Args:
            env_indices: list of env indices to reset, or ``None`` to reset
                         all envs.
        """
        if env_indices is None:
            self._mjwarp.reset_data(self._warp_model, self._warp_data)
        else:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[list(env_indices)] = True
            device = self._warp_data.qpos.device
            warp_mask = self._wp.array(mask, dtype=self._wp.bool, device=device)
            self._mjwarp.reset_data(self._warp_model, self._warp_data, reset=warp_mask)

    def forward(self) -> None:
        """Run forward dynamics for all envs simultaneously."""
        self._mjwarp.forward(self._warp_model, self._warp_data)

    def kinematics_forward(self) -> None:
        """Run only kinematics + passive forces for all envs (cheap pre-controller pass).

        This is cheaper than ``forward()`` because it skips collision detection,
        constraint making, factorisation, and the constraint solver.  It updates
        ``site_xpos``, ``site_xmat``, body positions/orientations, and ``qfrc_bias``
        (gravity + Coriolis) — everything the OSC controller reads.  The full
        ``forward()`` is still called internally by ``step()``, so constraint
        accuracy is not affected.
        """
        import mujoco_warp._src.smooth as _smooth
        m, d = self._warp_model, self._warp_data
        _smooth.kinematics(m, d)   # site_xpos, site_xmat, body frames
        _smooth.com_pos(m, d)      # needed by rne (subtree CoM for gravity)
        _smooth.com_vel(m, d)      # subtree velocity (needed by rne Coriolis)
        _smooth.rne(m, d)          # qfrc_bias = gravity + Coriolis

    def step(self, with_udd: bool = True) -> None:
        """Advance all envs by one timestep on the GPU."""
        if not self._graph_enabled:
            self._mjwarp.step(self._warp_model, self._warp_data)
            return

        if self._step_graph is not None:
            self._wp.capture_launch(self._step_graph)
            return

        if self._graph_steps_done < self._graph_warmup_steps:
            self._mjwarp.step(self._warp_model, self._warp_data)
            self._graph_steps_done += 1
            return

        # Capture: the scope runs the step once as it records the kernel graph.
        self._wp.synchronize()
        with self._wp.ScopedCapture() as cap:
            self._mjwarp.step(self._warp_model, self._warp_data)
        self._step_graph = cap.graph

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def get_state(self, env_idx: Optional[int] = None) -> "MjSimState | List[MjSimState]":
        """
        Return the current simulation state.

        Args:
            env_idx: if given, returns a single :class:`MjSimState` for that
                     env; if ``None``, returns a list for all envs.
        """
        qpos = self._warp_data.qpos.numpy()  # (num_envs, nq)
        qvel = self._warp_data.qvel.numpy()  # (num_envs, nv)
        time = self._warp_data.time.numpy()  # (num_envs,)

        if env_idx is not None:
            return MjSimState(
                time=float(time[env_idx]),
                qpos=qpos[env_idx].copy(),
                qvel=qvel[env_idx].copy(),
            )
        return [MjSimState(time=float(time[i]), qpos=qpos[i].copy(), qvel=qvel[i].copy()) for i in range(self.num_envs)]

    def set_state(
        self,
        value: "MjSimState | List[MjSimState]",
        env_idx: Optional[int] = None,
    ) -> None:
        """
        Set simulation state from one or more :class:`MjSimState` objects.

        Args:
            value: a single :class:`MjSimState` applied to *env_idx* (or
                   broadcast to all envs when *env_idx* is ``None``), or a
                   list of :class:`MjSimState` of length ``num_envs``.
            env_idx: target env index; ignored when *value* is a list.
        """
        device = self._warp_data.qpos.device

        if isinstance(value, MjSimState):
            if env_idx is None:
                qpos = np.tile(value.qpos, (self.num_envs, 1)).astype(np.float32)
                qvel = np.tile(value.qvel, (self.num_envs, 1)).astype(np.float32)
                time = np.full(self.num_envs, value.time, dtype=np.float32)
            else:
                qpos = self._warp_data.qpos.numpy().copy()
                qvel = self._warp_data.qvel.numpy().copy()
                time = self._warp_data.time.numpy().copy()
                qpos[env_idx] = value.qpos
                qvel[env_idx] = value.qvel
                time[env_idx] = value.time
        else:
            states = list(value)
            assert len(states) == self.num_envs, f"Expected {self.num_envs} states, got {len(states)}"
            qpos = np.stack([s.qpos for s in states]).astype(np.float32)
            qvel = np.stack([s.qvel for s in states]).astype(np.float32)
            time = np.array([s.time for s in states], dtype=np.float32)

        self._wp.copy(self._warp_data.qpos, self._wp.from_numpy(qpos, device=device))
        self._wp.copy(self._warp_data.qvel, self._wp.from_numpy(qvel, device=device))
        self._wp.copy(self._warp_data.time, self._wp.from_numpy(time, device=device))

    def set_state_from_flattened(
        self,
        value: np.ndarray,
        env_idx: Optional[int] = None,
    ) -> None:
        """
        Set state from a flat ``(1 + nq + nv,)`` array, or a batch
        ``(num_envs, 1 + nq + nv)`` array.

        Args:
            value: 1-D array for a single env, or 2-D for all envs.
            env_idx: target env index when *value* is 1-D; ignored for 2-D.
        """
        value = np.asarray(value)
        if value.ndim == 1:
            self.set_state(MjSimState.from_flattened(value, self), env_idx=env_idx)
        else:
            self.set_state([MjSimState.from_flattened(value[i], self) for i in range(len(value))])

    # ------------------------------------------------------------------
    # Single-env data extraction
    # ------------------------------------------------------------------

    def get_env_data(self, env_idx: int) -> mujoco.MjData:
        """
        Copy GPU state for one env into a ``mujoco.MjData`` on CPU.

        Useful for rendering, Jacobian queries, or any single-env operation
        that requires CPU-side mujoco.

        Args:
            env_idx: 0-based world index.

        Returns:
            ``mujoco.MjData`` populated with the current state of env
            *env_idx*.
        """
        result = mujoco.MjData(self.model._model)
        self._mjwarp.get_data_into(result, self.model._model, self._warp_data, world_id=env_idx)
        return result

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        *,
        camera_name: Optional[str] = None,
        depth: bool = False,
        mode: str = "offscreen",
        device_id: int = -1,
        segmentation: bool = False,
        env_idx: int = 0,
    ) -> np.ndarray:
        """
        Render a single env and return an image array.

        Pulls GPU state for *env_idx* to CPU, then delegates to the
        standard MuJoCo offscreen renderer attached to this sim.

        Args:
            env_idx: which parallel world to render (default 0).

        Returns:
            ``uint8`` RGB image array, or ``(rgb, depth)`` tuple when
            ``depth=True``.
        """
        assert mode == "offscreen", "only offscreen rendering is supported"
        assert self._render_context_offscreen is not None

        camera_id = None if camera_name is None else self.model.camera_name2id(camera_name)

        # Bring the selected env's state to CPU and render via the shared context
        env_mj_data = self.get_env_data(env_idx)

        # Temporarily redirect the render context's data pointer
        saved_data_ptr = self._render_context_offscreen.data._data
        self._render_context_offscreen.data._data = env_mj_data
        try:
            with _MjSim_render_lock:
                self._render_context_offscreen.render(
                    width=width,
                    height=height,
                    camera_id=camera_id,
                    segmentation=segmentation,
                )
                return self._render_context_offscreen.read_pixels(width, height, depth=depth, segmentation=segmentation)
        finally:
            self._render_context_offscreen.data._data = saved_data_ptr

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def free(self) -> None:
        del self._render_context_offscreen
        del self._warp_data
        del self._warp_model
        del self.data
        del self.model
        gc.collect()
