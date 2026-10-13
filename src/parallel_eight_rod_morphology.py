"""Strict eight-rod length morphology generator for the physical robot model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


BODY_NAMES = ("0", "1", "11", "111", "1111", "11111", "111111", "1111111")
GEOM_NAMES = tuple(f"collision_body_{name}" for name in BODY_NAMES)
PARAMETER_NAMES = tuple(f"rod{index}_m" for index in range(8))
NOMINAL_GAP_M = 1.71
NOMINAL_VERTICES_XZ = np.asarray(
    [
        [0.0, 0.0],
        [0.8, 0.3],
        [1.6, 0.0],
        [1.6, 0.8],
        [1.6, 1.6],
        [0.8, 1.3],
        [0.0, 1.6],
        [0.0, 0.8],
    ],
    dtype=np.float64,
)
NOMINAL_EDGES_XZ = np.roll(NOMINAL_VERTICES_XZ, -1, axis=0) - NOMINAL_VERTICES_XZ
NOMINAL_LENGTHS_M = np.linalg.norm(NOMINAL_EDGES_XZ, axis=1)
NOMINAL_ANGLES_RAD = np.arctan2(NOMINAL_EDGES_XZ[:, 1], NOMINAL_EDGES_XZ[:, 0])
LENGTH_SCALE_BOUNDS = (0.75, 1.25)
LENGTH_BOUNDS_M = tuple(
    (float(length * LENGTH_SCALE_BOUNDS[0]), float(length * LENGTH_SCALE_BOUNDS[1]))
    for length in NOMINAL_LENGTHS_M
)


@dataclass(frozen=True)
class EightRodMorphology:
    """Eight independently proposed physical rod lengths, in chain order."""

    rod0_m: float
    rod1_m: float
    rod2_m: float
    rod3_m: float
    rod4_m: float
    rod5_m: float
    rod6_m: float
    rod7_m: float

    @classmethod
    def nominal(cls) -> "EightRodMorphology":
        return cls(*map(float, NOMINAL_LENGTHS_M))

    def array(self) -> np.ndarray:
        return np.asarray([getattr(self, name) for name in PARAMETER_NAMES], dtype=np.float64)

    def values(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    def validate(self) -> None:
        lengths = self.array()
        for index, (value, bounds) in enumerate(zip(lengths, LENGTH_BOUNDS_M)):
            low, high = bounds
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{PARAMETER_NAMES[index]}={value} outside [{low}, {high}]")
        if 2.0 * float(np.max(lengths)) >= float(np.sum(lengths)):
            raise ValueError("rod lengths cannot form a closed polygon")

    def vertices(self) -> np.ndarray:
        self.validate()
        lengths = self.array()
        if np.array_equal(lengths, NOMINAL_LENGTHS_M):
            xz = NOMINAL_VERTICES_XZ.copy()
        else:
            xz = _solve_closed_vertices(lengths)
        return np.column_stack((xz[:, 0], np.zeros(8, dtype=np.float64), xz[:, 1]))

    def actual_lengths(self) -> np.ndarray:
        vertices = self.vertices()
        return np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)


def _closure(angles: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.dot(lengths, np.cos(angles)), np.dot(lengths, np.sin(angles))],
        dtype=np.float64,
    )


def _closure_jacobian(angles: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return np.vstack((-lengths * np.sin(angles), lengths * np.cos(angles)))


def _signed_area(vertices_xz: np.ndarray) -> float:
    x = vertices_xz[:, 0]
    z = vertices_xz[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - z * np.roll(x, -1)))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    first = _orientation(a, b, c) * _orientation(a, b, d)
    second = _orientation(c, d, a) * _orientation(c, d, b)
    return first < -1e-12 and second < -1e-12


def _has_self_intersection(vertices_xz: np.ndarray) -> bool:
    count = len(vertices_xz)
    for first in range(count):
        a = vertices_xz[first]
        b = vertices_xz[(first + 1) % count]
        for second in range(first + 1, count):
            if second == first or second == (first + 1) % count:
                continue
            if first == 0 and second == count - 1:
                continue
            c = vertices_xz[second]
            d = vertices_xz[(second + 1) % count]
            if _segments_cross(a, b, c, d):
                return True
    return False


def _solve_closed_vertices(lengths: np.ndarray) -> np.ndarray:
    """Project nominal link angles to exact closure using only NumPy."""

    angles = NOMINAL_ANGLES_RAD.copy()
    closure_error = float("inf")
    for _ in range(100):
        residual = _closure(angles, lengths)
        closure_error = float(np.linalg.norm(residual))
        if closure_error <= 1e-11:
            break
        jacobian = _closure_jacobian(angles, lengths)
        gram = jacobian @ jacobian.T + 1e-14 * np.eye(2)
        step = jacobian.T @ np.linalg.solve(gram, residual)
        scale = 1.0
        while scale >= 1e-6:
            candidate = angles - scale * step
            if float(np.linalg.norm(_closure(candidate, lengths))) < closure_error:
                angles = candidate
                break
            scale *= 0.5
        else:
            break
    closure_error = float(np.linalg.norm(_closure(angles, lengths)))
    if closure_error > 1e-8:
        raise ValueError(f"eight-rod linkage solve failed; closure={closure_error}")
    if float(np.max(np.abs(angles - NOMINAL_ANGLES_RAD))) > 0.5 * np.pi:
        raise ValueError("linked morphology rotated beyond the physical configuration branch")

    edges = np.column_stack((lengths * np.cos(angles), lengths * np.sin(angles)))
    vertices = np.vstack((np.zeros(2, dtype=np.float64), np.cumsum(edges, axis=0)[:-1]))
    vertices[:, 0] += 0.8 - 0.5 * float(vertices[:, 0].min() + vertices[:, 0].max())
    vertices[:, 1] -= float(vertices[:, 1].min())

    actual = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    if not np.allclose(actual, lengths, rtol=0.0, atol=1e-8):
        raise ValueError(f"linked rod lengths drifted by {float(np.max(np.abs(actual - lengths)))} m")
    if _signed_area(vertices) <= 0.1 or _has_self_intersection(vertices):
        raise ValueError("linked morphology is folded or self-intersecting")
    return vertices


def _format(values: np.ndarray | list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def wall_positions_for_gap(
    gap_m: float,
    center_x: float = 0.82,
    half_thickness_m: float = 0.125,
) -> tuple[float, float]:
    gap = float(gap_m)
    if not np.isfinite(gap) or not 1.50 <= gap <= 1.90:
        raise ValueError("gap_m must lie in [1.50, 1.90]")
    return center_x - 0.5 * gap - half_thickness_m, center_x + 0.5 * gap + half_thickness_m


def generate_model(
    source: Path,
    target: Path,
    morphology: EightRodMorphology,
    *,
    gap_m: float = NOMINAL_GAP_M,
) -> None:
    """Generate one model while changing only the eight physical rod lengths."""

    morphology.validate()
    source = Path(source).resolve()
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(target)
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "parallel_original_8joint_4motor_eight_rod_morphology")
    vertices = morphology.vertices()

    for index, (body_name, geom_name) in enumerate(zip(BODY_NAMES, GEOM_NAMES)):
        geom = root.find(f".//geom[@name='{geom_name}']")
        if geom is None:
            raise ValueError(f"missing geom {geom_name}")
        first = vertices[index]
        second = vertices[(index + 1) % len(vertices)]
        geom.set("fromto", _format(np.concatenate((first, second))))
        if index > 0:
            joint = root.find(f".//joint[@name='{body_name}_joint']")
            if joint is None:
                raise ValueError(f"missing joint {body_name}_joint")
            joint.set("pos", _format(first))

    closure_site = root.find(".//site[@name='closure_endpoint']")
    equality = root.find(".//equality/connect[@name='kinematic_link']")
    if closure_site is None or equality is None:
        raise ValueError("missing closure declarations")
    closure_site.set("pos", _format(vertices[0]))
    equality.set("anchor", _format(vertices[0]))

    left_x, right_x = wall_positions_for_gap(gap_m)
    left = root.find(".//geom[@name='wall_left']")
    right = root.find(".//geom[@name='wall_right']")
    if left is None or right is None:
        raise ValueError("missing walls")
    left_pos = np.fromstring(left.get("pos", ""), sep=" ")
    right_pos = np.fromstring(right.get("pos", ""), sep=" ")
    left_pos[0] = left_x
    right_pos[0] = right_x
    left.set("pos", _format(left_pos))
    right.set("pos", _format(right_pos))

    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="utf-8", xml_declaration=True)
