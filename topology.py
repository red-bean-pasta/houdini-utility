import math
from typing import Sequence, Callable

import hou
import numpy as np

from common import fill_face


def fill_pentagon(
    geo: hou.Geometry,
    points: Sequence[hou.Point],
    midedge: tuple[hou.Point, hou.Point],
) -> tuple[hou.Point, hou.Point]:
    """Subdivide a pentagon into 3 quads by placing a midpoint on one edge and an internal floating point.

    :param geo: The Houdini geometry.
    :param points: 5 cyclic points of the pentagon.
    :param midedge: Tuple of 2 adjacent points defining the edge to split.
    :return: Tuple of (midpoint on specified midedge, interior float point).
    """
    assert len(set(points)) == 5, "Expected 5 distinct points for fill_pentagon"

    p_a, p_b = midedge
    assert p_a in points and p_b in points and p_a != p_b, f"mid_edge {midedge} must be in points"
    idx_a = points.index(p_a)
    idx_b = points.index(p_b)
    diff = (idx_b - idx_a) % 5
    assert diff in (1, 4), f"mid_edge points must be adjacent in points sequence, got diff {diff}"

    ordered = (
        [points[(idx_a + k) % 5] for k in range(5)]
        if diff == 1 else
        [points[(idx_a - k) % 5] for k in range(5)]
    )
    p0, p1, p2, p3, p4 = ordered

    midpos = (p0.position() + p1.position()) / 2.0
    midpoint = geo.createPoint()
    midpoint.setPosition(midpos)

    v_edge = p1.position() - p0.position()
    edge_len = v_edge.length()
    assert edge_len > 1e-6, "Expected nonzero mid_edge length"

    def dist_to_line(pt: hou.Point) -> float:
        v = pt.position() - p0.position()
        return v.cross(v_edge).length() / edge_len

    dist_p2 = dist_to_line(p2)
    dist_p4 = dist_to_line(p4)
    if dist_p2 <= dist_p4:
        m_base = (p0.position() + p4.position()) / 2.0
        f_pos = (p2.position() + m_base) / 2.0
    else:
        m_base = (p1.position() + p2.position()) / 2.0
        f_pos = (p4.position() + m_base) / 2.0

    floatpoint = geo.createPoint()
    floatpoint.setPosition(f_pos)

    fill_face(geo, [p1, midpoint, floatpoint, p2])
    fill_face(geo, [midpoint, p0, p4, floatpoint])
    fill_face(geo, [floatpoint, p4, p3, p2])

    return midpoint, floatpoint


def classify_after_inset(
        geo: hou.Geometry,
        prim_count_before: int,
        horizontal_pack_size: int = 1,
        vertical_pack_size: int = 1,
) -> tuple[
        list[tuple[hou.Prim, ...]],
        list[tuple[hou.Prim, ...]],
]:
    """Classify primitives output by PolyInset SOP into central panes and border sills.

    Assumes a grid of connected quad faces of size (horizontal_pack_size x vertical_pack_size)
    was inset. In Houdini's PolyInset SOP:
    - Central inset primitives retain their original indices preceding the newly generated geometry.
    - Border/sill primitives are appended contiguously in deterministic order.

    :param geo: Geometry after PolyInset.
    :param prim_count_before: Number of primitives before the inset operation.
    :param horizontal_pack_size: Number of columns in the quad grid.
    :param vertical_pack_size: Number of rows in the quad grid.
    :return: Tuple of (panes_groups, sills_groups).
    """
    assert horizontal_pack_size > 0 and vertical_pack_size > 0
    assert prim_count_before >= horizontal_pack_size * vertical_pack_size
    prims_after: tuple[hou.Prim, ...] = geo.prims()
    assert len(prims_after) > prim_count_before

    prims_added = prims_after[prim_count_before:]

    grid_size = horizontal_pack_size * vertical_pack_size
    sill_size = 4 * grid_size - (horizontal_pack_size - 1) * 2 - (vertical_pack_size - 1) * 2
    assert len(prims_added) % sill_size == 0, f"primitives added {len(prims_added)} should be a multiple of sill size {sill_size}"
    group_count = len(prims_added) // sill_size

    assert prim_count_before >= group_count * grid_size
    pane_start = prim_count_before - group_count * grid_size

    panes: list[tuple[hou.Prim, ...]] = []
    sills: list[tuple[hou.Prim, ...]] = []
    for i in range(group_count):
        panes.append(prims_after[pane_start + i * grid_size : pane_start + (i + 1) * grid_size])
        sills.append(prims_added[i * sill_size : (i + 1) * sill_size])

    return panes, sills


def get_point_on_ellipse_2d(
        origin: hou.Vector3,
        vertical_end: hou.Vector3,
        side_end: hou.Vector3,
        rad_from_y: float = math.pi / 4,
) -> hou.Vector3:
    """Evaluate a point on a planar ellipse arc parameterized by its semi-axes vectors."""
    v_upper = vertical_end - origin
    v_left = side_end - origin

    assert v_upper.length() > 1e-6, "vertical_end must not coincide with origin"
    assert v_left.length() > 1e-6, "side_end must not coincide with origin"
    assert math.isclose(v_upper.dot(v_left), 0.0, abs_tol=1e-5), f"upper-origin ({v_upper}) and side-origin ({v_left}) axes must be perpendicular"

    return origin + v_upper * math.cos(rad_from_y) + v_left * math.sin(rad_from_y)


def interpolate_conic(
        p0: hou.Vector3 | hou.Point,
        p1: hou.Vector3 | hou.Point,
        p2: hou.Vector3 | hou.Point,
        slope0: hou.Vector3,
        slope1: hou.Vector3,
) -> Callable[[float], tuple[hou.Vector3, ...]]:
    """Construct a planar conic passing through p0, p1, p2 with tangents slope0 at p0 and slope1 at p1.

    The method projects the 3D problem into a 2D local orthonormal coordinate plane:
      - along_axis: direction from p0 to p1
      - across_axis: in-plane normal orthogonal to along_axis
    It sets up an algebraic conic equation:
      A * x² + B * x * y + C * y² + D * x + E * y = 0
    where (x, y) = (along, across) with p0 at (0, 0).
    The remaining 4 constraints (passage through p1, p2, and tangent directions at p0, p1)
    form a 4x5 linear system solved via SVD for the 1D null space.

    :param p0: First point on conic (origin of the local 2D coordinate system).
    :param p1: Second point on conic.
    :param p2: Intermediate third point on conic.
    :param slope0: Tangent vector at p0.
    :param slope1: Tangent vector at p1.
    :return: A function that accepts a signed distance along the p0->p1 axis,
             and returns the 0, 1, or 2 3D points on the conic.
    """
    vector0 = p0.position() if isinstance(p0, hou.Point) else hou.Vector3(p0)
    vector1 = p1.position() if isinstance(p1, hou.Point) else hou.Vector3(p1)
    vector2 = p2.position() if isinstance(p2, hou.Point) else hou.Vector3(p2)

    delta01 = vector1 - vector0
    delta02 = vector2 - vector0
    assert delta01.length() != 0.0, "p0 and p1 must be distinct"

    along_axis = delta01.normalized()
    normal = along_axis.cross(delta02)
    assert normal.length() != 0.0, "p0, p1, and p2 must not be collinear"
    normal = normal.normalized()
    across_axis = normal.cross(along_axis).normalized()

    along1 = delta01.length()
    along2 = delta02.dot(along_axis)
    across2 = delta02.dot(across_axis)

    def project_slope(slope: hou.Vector3) -> tuple[float, float]:
        tangent = slope - slope.dot(normal) * normal
        assert tangent.length() != 0.0, "Slope must have a non-zero component in the conic plane"
        tangent = tangent.normalized()
        return tangent.dot(along_axis), tangent.dot(across_axis)

    along_slope0, across_slope0 = project_slope(slope0)
    along_slope1, across_slope1 = project_slope(slope1)

    # A*x² + B*x*y + C*y² + D*x + E*y = 0
    matrix = np.array([
        [along1**2, 0.0, 0.0, along1, 0.0],
        [along2**2, along2 * across2, across2**2, along2, across2],
        [0.0, 0.0, 0.0, along_slope0, across_slope0],
        [2.0 * along1 * along_slope1, along1 * across_slope1, 0.0, along_slope1, across_slope1],
    ], dtype=float)
    _, singular_values, vh = np.linalg.svd(matrix)
    A, B, C, D, E = map(float, vh[-1])

    tolerance = np.finfo(float).eps * max(matrix.shape) * singular_values[0]
    assert np.sum(singular_values > tolerance) == 4, "The supplied points and slopes do not determine a unique conic"

    coefficient_scale = max(abs(A), abs(B), abs(C), abs(D), abs(E))
    assert coefficient_scale != 0.0, "Failed to construct a valid conic"
    A /= coefficient_scale
    B /= coefficient_scale
    C /= coefficient_scale
    D /= coefficient_scale
    E /= coefficient_scale

    def to_3d(along: float, across: float) -> hou.Vector3:
        return vector0 + along * along_axis + across * across_axis

    def evaluate(along: float) -> tuple[hou.Vector3, ...]:
        quadratic = C
        linear = B * along + E
        constant = A * along**2 + D * along
        eps = 1e-12

        if abs(quadratic) <= eps:
            if abs(linear) <= eps:
                return ()
            across = -constant / linear
            return (to_3d(along, across),)

        discriminant = linear**2 - 4.0 * quadratic * constant
        if discriminant < -eps:
            return ()

        if abs(discriminant) <= eps:
            across = -linear / (2.0 * quadratic)
            return (to_3d(along, across),)

        sqrt_discriminant = math.sqrt(discriminant)
        across0 = (-linear + sqrt_discriminant) / (2.0 * quadratic)
        across1 = (-linear - sqrt_discriminant) / (2.0 * quadratic)
        return to_3d(along, across0), to_3d(along, across1)

    return evaluate
