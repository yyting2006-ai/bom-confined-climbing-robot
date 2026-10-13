"""Frozen, new-scene geometry pilots for the physical four-motor robot.

Requires MuJoCo 2.3.3 for the original global-coordinate MJCF.
All lengths are metres; angles in generated MJCF are degrees. These are new
scenes, not reproductions of the archived seven-motor/low-gravity experiments.
The only output side effect of build_scene is creating the requested XML.
Infeasible proposed shapes are reported, never rescaled to fit the corridor.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from typing import Sequence
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


SOURCE_PROJECT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = SOURCE_PROJECT / "models/parallel_original_8joint_4act_forceclean.xml"
SOURCE_GENERATOR = SOURCE_PROJECT / "src/parallel_eight_rod_morphology.py"
SOURCE_CONTRACT = SOURCE_PROJECT / "ORIGINAL_4ACTUATOR_CONTRACT.md"
SCENE_NAMES = ("stepped", "curved")
LENGTH_SCALE_BOUNDS = (0.90, 1.10)
BLOCKS = ((0, 1), (2, 3), (4, 5), (6, 7))
INITIAL_GAP_M = 1.79
CORRIDOR_CENTER_X_M = 0.82
INITIAL_ROBOT_Z_TRANSLATION_M = 0.080
TERRAIN_START_Z_M = 2.10
WALL_TOP_Z_M = 10.0
WALL_HALF_THICKNESS_M = 0.125
WALL_HALF_WIDTH_Y_M = 2.0
MAX_INWARD_OFFSET_M = 0.15
CURVE_END_Z_M = 3.90
CURVE_SEGMENTS = 36
INITIAL_PENETRATION_LIMIT_M = 0.001
ACTIVE_ACTUATORS = ("1_joint", "111_joint", "11111_joint", "1111111_joint")
ALL_ACTUATORS = (
    "1_joint", "11_joint", "111_joint", "1111_joint",
    "11111_joint", "111111_joint", "1111111_joint", "0_joint",
)


def _source_morphology_module():
    """Load the actual source generator without changing sys.path."""
    name = "_new_scene_original_eight_rod_morphology"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SOURCE_GENERATOR)
        if spec is None or spec.loader is None:
            raise ImportError(str(SOURCE_GENERATOR))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _fmt(values) -> str:
    return " ".join(f"{float(value):.15g}" for value in values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _translate_robot_global(root: ET.Element, dz: float) -> None:
    """Rigidly translate every global robot coordinate; leave refs unchanged."""
    robot = root.find("./worldbody/body[@name='0']")
    if robot is None:
        raise ValueError("source robot body missing")
    delta = np.array([0.0, 0.0, dz])
    for node in robot.iter():
        if "pos" in node.attrib:
            node.set("pos", _fmt(np.fromstring(node.attrib["pos"], sep=" ") + delta))
        if "fromto" in node.attrib:
            endpoints = np.fromstring(node.attrib["fromto"], sep=" ").reshape(2, 3)
            node.set("fromto", _fmt((endpoints + delta).ravel()))
    connect = root.find("./equality/connect[@name='kinematic_link']")
    if connect is None:
        raise ValueError("source closure missing")
    connect.set("anchor", _fmt(np.fromstring(connect.attrib["anchor"], sep=" ") + delta))


def _wall_sections(scene: str) -> list[dict]:
    """Return exact inner face segments, including flat approach/upper walls."""
    right_face_x = CORRIDOR_CENTER_X_M + INITIAL_GAP_M / 2.0
    if scene == "stepped":
        bands = ((0.0, 2.10, 0.0), (2.10, 2.70, 0.05),
                 (2.70, 3.30, 0.10), (3.30, 10.0, 0.15))
        return [
            {"name": "wall_right" if i == 0 else f"wall_right_step_{i:02d}",
             "start_xz_m": [right_face_x - offset, low],
             "end_xz_m": [right_face_x - offset, high]}
            for i, (low, high, offset) in enumerate(bands)
        ]
    sections = [{"name": "wall_right", "start_xz_m": [right_face_x, 0.0],
                 "end_xz_m": [right_face_x, TERRAIN_START_Z_M]}]
    zs = np.linspace(TERRAIN_START_Z_M, CURVE_END_Z_M, CURVE_SEGMENTS + 1)
    # A local smooth bulge: displacement is 0 -> .15 -> 0, with zero end slope.
    phase = (zs - TERRAIN_START_Z_M) / (CURVE_END_Z_M - TERRAIN_START_Z_M)
    offsets = 0.5 * MAX_INWARD_OFFSET_M * (1.0 - np.cos(2.0 * np.pi * phase))
    for i in range(CURVE_SEGMENTS):
        sections.append({"name": f"wall_right_curve_{i:03d}",
                         "start_xz_m": [float(right_face_x - offsets[i]), float(zs[i])],
                         "end_xz_m": [float(right_face_x - offsets[i + 1]), float(zs[i + 1])]})
    sections.append({"name": "wall_right_upper", "start_xz_m": [right_face_x, CURVE_END_Z_M],
                     "end_xz_m": [right_face_x, WALL_TOP_Z_M]})
    return sections


def _make_wall_box(worldbody: ET.Element, template: dict, section: dict) -> None:
    """Put the box's local -x face exactly on the declared segment."""
    start, end = np.asarray(section["start_xz_m"]), np.asarray(section["end_xz_m"])
    delta = end - start
    length = float(np.linalg.norm(delta))
    normal = np.array([delta[1], -delta[0]]) / length  # outward into right wall
    center = 0.5 * (start + end) + WALL_HALF_THICKNESS_M * normal
    angle_deg = math.degrees(math.atan2(float(delta[0]), float(delta[1])))
    attrs = dict(template)
    attrs.update(name=section["name"], type="box",
                 pos=_fmt([center[0], 0.0, center[1]]),
                 size=_fmt([WALL_HALF_THICKNESS_M, WALL_HALF_WIDTH_Y_M, 0.5 * length]),
                 euler=_fmt([0.0, angle_deg, 0.0]))
    ET.SubElement(worldbody, "geom", attrs)
    section["box_center_xyz_m"] = [float(center[0]), 0.0, float(center[1])]
    section["box_halfsize_xyz_m"] = [WALL_HALF_THICKNESS_M, WALL_HALF_WIDTH_Y_M, 0.5 * length]
    section["box_euler_xyz_deg"] = [0.0, angle_deg, 0.0]


def _audit(model, source_model, data, sections: list[dict]) -> dict:
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                      for i in range(model.nu)]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                   for i in range(model.njnt)]
    source_joints = [mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                     for i in range(source_model.njnt)]
    active_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                  for name in ACTIVE_ACTUATORS]
    contract_ok = bool(
        (model.nq, model.nv, model.nu) == (10, 10, 8)
        and actuator_names == list(ALL_ACTUATORS) and active_ids == [0, 2, 4, 6]
        and joint_names == source_joints
        and np.array_equal(model.jnt_type, source_model.jnt_type)
        and np.array_equal(model.jnt_range, source_model.jnt_range)
        and np.array_equal(model.actuator_trnid, source_model.actuator_trnid)
        and np.array_equal(model.actuator_gear, source_model.actuator_gear)
        and np.array_equal(model.actuator_ctrlrange, source_model.actuator_ctrlrange)
        and np.array_equal(model.qpos0, source_model.qpos0)
        and np.array_equal(model.opt.gravity, [0.0, 0.0, -9.81])
    )
    contacts = []
    for contact in data.contact:
        ids = [int(contact.geom1), int(contact.geom2)]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in ids]
        if all(model.geom_bodyid[i] == 0 for i in ids):
            continue
        category = ("wall" if any(n.startswith("wall_") for n in names) else
                    "floor" if "floor" in names else "self")
        contacts.append({"geom_names": names, "category": category,
                         "distance_m": float(contact.dist),
                         "penetration_m": max(0.0, -float(contact.dist))})
    penetration = max((c["penetration_m"] for c in contacts), default=0.0)
    wall_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, s["name"]) for s in sections]
    world_walls = bool(all(i >= 0 and int(model.geom_bodyid[i]) == 0 for i in wall_ids))
    return {
        "four_motor_audit": {"passed": contract_ok, "action_dim": 4,
                             "nq": model.nq, "nv": model.nv, "actuator_slots": model.nu,
                             "actuator_names": actuator_names, "joint_names": joint_names,
                             "active_ids": active_ids, "inactive_ids": [1, 3, 5, 7],
                             "active_names": list(ACTIVE_ACTUATORS),
                             "mapping": "[a0,a1,a2,a3] -> [a0,0,a1,0,a2,0,a3,0]",
                             "right_wall_geoms_are_worldbody": world_walls},
        "feasibility": {"checked_state": "qpos0, qvel=0, ctrl=0; mj_forward; no integration",
                        "initial_qpos": data.qpos.tolist(),
                        "max_initial_penetration_m": penetration,
                        "penetration_limit_m": INITIAL_PENETRATION_LIMIT_M,
                        "contacts": contacts,
                        "feasible": bool(contract_ok and world_walls
                                         and penetration <= INITIAL_PENETRATION_LIMIT_M + 1e-12)},
        "mass_audit": {"rule": "source inertiafromgeom=true; original geom density/mass attributes retained",
                       "body_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                                      for i in range(1, model.nbody)],
                       "source_body_mass_kg": source_model.body_mass[1:].tolist(),
                       "candidate_body_mass_kg": model.body_mass[1:].tolist(),
                       "source_total_mass_kg": float(source_model.body_mass.sum()),
                       "candidate_total_mass_kg": float(model.body_mass.sum())},
    }


def build_scene(output_xml: Path, scene: str, length_scales: Sequence[float]) -> dict:
    """Create a scene and return JSON-serializable provenance/audits.

    ``length_scales`` is eight independent ratios in [0.9, 1.1], in rod chain
    order. ``scene`` is case-insensitive Stepped/Curved. Existing destinations
    are refused. A valid but initially penetrating proposal is still written
    and returns ``feasible=False``; callers must not train it. Reset noise is
    not included in this static check and needs a separate per-reset guard.
    """
    if mujoco.__version__ != "2.3.3":
        raise RuntimeError("Use the project Python 3.11 environment with mujoco==2.3.3; do not convert coordinates")
    output_xml = Path(output_xml).resolve()
    if output_xml.exists():
        raise FileExistsError(output_xml)
    scene = str(scene).lower()
    if scene not in SCENE_NAMES:
        raise ValueError(f"scene must be one of {SCENE_NAMES}")
    scales = np.asarray(length_scales, dtype=np.float64)
    if scales.shape != (8,) or not np.isfinite(scales).all():
        raise ValueError("length_scales must contain eight finite numbers")
    if np.any(scales < LENGTH_SCALE_BOUNDS[0]) or np.any(scales > LENGTH_SCALE_BOUNDS[1]):
        raise ValueError("length scales must lie in [0.90, 1.10]")
    morphology_module = _source_morphology_module()
    lengths = morphology_module.NOMINAL_LENGTHS_M * scales
    morphology = morphology_module.EightRodMorphology(*map(float, lengths))
    vertices = morphology.vertices()
    root = ET.parse(SOURCE_MODEL).getroot()
    if root.find("compiler").get("coordinate") != "global":
        raise ValueError("source coordinate contract changed")
    root.set("model", f"new_scene_pilot_four_motor_{scene}")
    # The same updates as the source generator, made in memory to avoid an
    # intermediate model. All untouched physical attributes remain original.
    for i, (body_name, geom_name) in enumerate(zip(morphology_module.BODY_NAMES,
                                                  morphology_module.GEOM_NAMES)):
        root.find(f".//geom[@name='{geom_name}']").set(
            "fromto", _fmt(np.concatenate((vertices[i], vertices[(i + 1) % 8]))))
        if i > 0:
            root.find(f".//joint[@name='{body_name}_joint']").set("pos", _fmt(vertices[i]))
    root.find(".//site[@name='closure_endpoint']").set("pos", _fmt(vertices[0]))
    root.find("./equality/connect[@name='kinematic_link']").set("anchor", _fmt(vertices[0]))
    _translate_robot_global(root, INITIAL_ROBOT_Z_TRANSLATION_M)
    worldbody = root.find("worldbody")
    left = worldbody.find("geom[@name='wall_left']")
    left_x, _ = morphology_module.wall_positions_for_gap(INITIAL_GAP_M)
    left.set("pos", _fmt([left_x, 0.0, WALL_TOP_Z_M / 2.0]))
    right = worldbody.find("geom[@name='wall_right']")
    right_template = dict(right.attrib)
    worldbody.remove(right)
    sections = _wall_sections(scene)
    for section in sections:
        _make_wall_box(worldbody, right_template, section)
    # XML text is audited before publication, and source files are read-only.
    xml_text = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_text)
    source_model = mujoco.MjModel.from_xml_path(str(SOURCE_MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    audits = _audit(model, source_model, data, sections)
    if not audits["four_motor_audit"]["passed"]:
        raise AssertionError("four-motor physical contract drift")
    initial_top = float(vertices[:, 2].max() + INITIAL_ROBOT_Z_TRANSLATION_M + 0.05)
    if initial_top >= TERRAIN_START_Z_M:
        raise AssertionError("terrain must begin above the complete initial robot")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    with output_xml.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n' + xml_text + "\n")
    curve_height = CURVE_END_Z_M - TERRAIN_START_Z_M
    dz = curve_height / CURVE_SEGMENTS
    metadata = {
        "schema": "new_four_motor_scene_pilot_v1", "scene": scene,
        "experiment_scope": "new-scene pilot; not an archived seven-motor result reproduction",
        "output_xml": str(output_xml), "output_sha256": _sha256(output_xml),
        "mujoco_version": mujoco.__version__, "units": {"length": "m", "mass": "kg", "gravity": "m/s^2"},
        "source_files": {str(p): _sha256(p) for p in (SOURCE_MODEL, SOURCE_GENERATOR, SOURCE_CONTRACT)},
        "length_scales": scales.tolist(), "rod_lengths_m": lengths.tolist(),
        "scale_bounds": list(LENGTH_SCALE_BOUNDS), "blocks": [list(b) for b in BLOCKS],
        "gravity_xyz_m_s2": model.opt.gravity.tolist(),
        "geometry": {"initial_gap_m": INITIAL_GAP_M, "corridor_center_x_m": CORRIDOR_CENTER_X_M,
                     "initial_robot_translation_xyz_m": [0.0, 0.0, INITIAL_ROBOT_Z_TRANSLATION_M],
                     "initial_robot_collision_top_z_m": initial_top,
                     "terrain_start_z_m": TERRAIN_START_Z_M, "wall_top_z_m": WALL_TOP_Z_M,
                     "wall_half_thickness_m": WALL_HALF_THICKNESS_M,
                     "wall_half_width_y_m": WALL_HALF_WIDTH_Y_M,
                     "max_inward_offset_m": MAX_INWARD_OFFSET_M,
                     "minimum_gap_m": INITIAL_GAP_M - MAX_INWARD_OFFSET_M,
                     "right_wall_segments": sections,
                     "curve_definition": ("d(z)=0.075*(1-cos(2*pi*(z-2.10)/1.80)) for 2.10<=z<=3.90; 0 outside"
                                          if scene == "curved" else None),
                     "curve_num_sloped_boxes": CURVE_SEGMENTS if scene == "curved" else 0,
                     "curve_vertical_resolution_m": dz if scene == "curved" else None,
                     "curve_linear_interpolation_max_error_bound_m": (
                         MAX_INWARD_OFFSET_M * 2.0 * math.pi**2 / curve_height**2 * dz**2 / 8.0
                         if scene == "curved" else None),
                     "contact_classification": "ForceRuntime uses wall_left*/wall_right* names; all walls belong to worldbody",
                     "curve_contact_surface": "piecewise planar joined inner faces, no mesh or heightfield" if scene == "curved" else None},
        **audits,
    }
    metadata["feasible"] = metadata["feasibility"]["feasible"]
    return metadata
