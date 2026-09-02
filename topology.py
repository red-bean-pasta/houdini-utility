import math
from collections import defaultdict
from typing import Sequence, Callable, Any

import hou

from .common import fill_face, get_prim_normal
from .internal import elliptical_interpolator
from .internal.toplogy_helper import (
    Edge,
    partition_connected_prims,
    get_edge_prim_count,
    collect_prim_attrs,
    copy_point_data,
    fill_faces,
)


def inset(
        prims: list[hou.Prim],
        scalar: float,
        use_ratio: bool = True,
) -> list[hou.Prim]:
    """Perform an inset operation on a collection of polygon primitives.

    - Groups primitives into connected components and insets each component independently.
    - Offsets boundary vertices inward by distance or ratio and generates border quad faces (trapezoids).
    - Preserves primitive attributes and primitive group memberships across divided primitives.
    - Preserves point attributes and group memberships on newly generated inset points.

    :param prims: List of polygon primitives to inset.
    :param scalar: Offset ratio (in [0, 1]) or world distance.
    :param use_ratio: If True, scalar is interpreted as a ratio; otherwise as absolute distance.
    :return: List of inner inset primitives.
    """
    if not prims:
        return []

    geo = prims[0].geometry()
    components = partition_connected_prims(prims)

    inner_prims: list[hou.Prim] = []
    for comp in components:
        inner_prims.extend(_inset_connected(geo, comp, scalar, use_ratio))

    return inner_prims

def _inset_connected(
        geo: hou.Geometry,
        island: list[hou.Prim],
        scalar: float,
        is_scalar_ratio: bool,
) -> list[hou.Prim]:
    edge_counts = get_edge_prim_count(island)

    incoming_boundary_edge: dict[hou.Point, tuple[hou.Point, hou.Prim, float]] = {}
    outgoing_boundary_edge: dict[hou.Point, tuple[hou.Point, hou.Prim, float]] = {}
    boundary_edges_by_prim: defaultdict[hou.Prim, list[tuple[hou.Point, hou.Point]]] = defaultdict(list)
    for p in island:
        pts = [v.point() for v in p.vertices()]
        n = len(pts)
        for i in range(n):
            u, v = pts[i], pts[(i + 1) % n]
            edge = Edge.from_points(u, v, reorder=True)
            if edge_counts[edge] == 1:
                boundary_edges_by_prim[p].append((u, v))
                edge_len = (v.position() - u.position()).length()
                outgoing_boundary_edge[u] = (v, p, edge_len)
                incoming_boundary_edge[v] = (u, p, edge_len)

    inset_point_map: dict[hou.Point, hou.Point] = {}
    for pt in incoming_boundary_edge:
        u, p_in, l_in = incoming_boundary_edge[pt]
        w, p_out, l_out = outgoing_boundary_edge[pt]

        pos = pt.position()
        t_in = (pos - u.position()).normalized()
        t_out = (w.position() - pos).normalized()

        norm_in = get_prim_normal(p_in)
        norm_out = get_prim_normal(p_out)

        n1 = t_in.cross(norm_in).normalized()
        n2 = t_out.cross(norm_out).normalized()

        if is_scalar_ratio:
            d1 = l_in * scalar
            d2 = l_out * scalar
        else:
            d1 = scalar
            d2 = scalar

        cos_theta = n1.dot(n2)
        denom = 1.0 - cos_theta * cos_theta
        if denom > 1e-6:
            alpha = (d1 - d2 * cos_theta) / denom
            beta = (d2 - d1 * cos_theta) / denom
            delta_v = n1 * alpha + n2 * beta
        else:
            delta_v = (n1 + n2).normalized() * ((d1 + d2) * 0.5)

        new_pt = geo.createPoint()
        new_pt.setPosition(pos + delta_v)
        copy_point_data(pt, new_pt)
        inset_point_map[pt] = new_pt

    attr_data = collect_prim_attrs(island)

    prim_attr_map = dict(zip(island, attr_data))
    inner_face_points: list[list[hou.Point]] = []
    inner_attribs: list[tuple[dict[str, Any], list[str]]] = []
    border_face_points: list[list[hou.Point]] = []
    border_attribs: list[tuple[dict[str, Any], list[str]]] = []
    for p in island:
        pts = [v.point() for v in p.vertices()]
        inner_pts = [inset_point_map.get(pt, pt) for pt in pts]
        inner_face_points.append(inner_pts)
        inner_attribs.append(prim_attr_map[p])
        for u, v in boundary_edges_by_prim[p]:
            u_prime = inset_point_map[u]
            v_prime = inset_point_map[v]
            border_face_points.append([u, v, v_prime, u_prime])
            border_attribs.append(prim_attr_map[p])

    geo.deletePrims(island, keep_points=True)
    inner_prims = fill_faces(inner_face_points, inner_attribs)
    _ = fill_faces(border_face_points, border_attribs)
    return inner_prims


def loop_cut(
        prim: hou.Prim,
        start_point: hou.Point,
        end_point: hou.Point,
        scalar: float,
        use_ratio: bool = True,
) -> tuple[list[hou.Point], list[hou.Prim]]:
    """Perform a loop cut across adjacent quads starting from a specified edge of a quad primitive.

    - Places points interpolated between the start side and end side of each quad's cut edge.
    - Propagates across quad topology until reaching an open boundary, non-quad geometry, or looping back.
    - Deletes affected primitives and refills faces (end-side faces first, followed by start-side faces).
    - Preserves all primitive attributes and primitive group memberships across divided primitives.

    :param prim: The initial quad primitive.
    :param start_point: Starting point of the initial edge to cut.
    :param end_point: Ending point of the initial edge to cut.
    :param scalar: Ratio (in [0, 1]) or distance from start_point along the edge.
    :param use_ratio: If True, scalar is interpreted as a ratio; otherwise as a distance.
    :return: Tuple of (added_points, start_side_prims).
    """
    assert isinstance(prim, (hou.Prim, hou.Face, hou.Polygon)), f"Expected a primitive, got {type(prim)}"
    assert len(prim.vertices()) == 4, f"Expected quad prim with 4 vertices, got {len(prim.vertices())}"

    geo = prim.geometry()

    pts = list(prim.points())
    assert start_point in pts and end_point in pts, "start_point and end_point must be on prim"
    assert start_point != end_point, "start_point and end_point must be distinct"
    idx_s = pts.index(start_point)
    idx_e = pts.index(end_point)
    diff = (idx_e - idx_s) % 4
    assert diff in (1, 3), f"start_point and end_point must form an edge on prim, got indices {idx_s} and {idx_e}"

    visited_prims: list[hou.Prim] = []
    added_points: list[hou.Point] = []
    end_side_face_points: list[list[hou.Point]] = []
    start_side_face_points: list[list[hou.Point]] = []

    curr_prim: hou.Prim | None = prim
    s_pt: hou.Point = start_point
    e_pt: hou.Point = end_point

    m_start = geo.createPoint()
    m_start.setPosition(_calc_loop_cut_position(s_pt, e_pt, scalar, use_ratio))
    added_points.append(m_start)
    m_curr = m_start

    while curr_prim is not None:
        visited_prims.append(curr_prim)
        pts = list(curr_prim.points())
        idx_s = pts.index(s_pt)
        idx_e = pts.index(e_pt)
        diff = (idx_e - idx_s) % 4

        if diff == 1:
            e_next = pts[(idx_s + 2) % 4]
            s_next = pts[(idx_s + 3) % 4]
        else:
            s_next = pts[(idx_e + 2) % 4]
            e_next = pts[(idx_e + 3) % 4]

        # Check if the opposite edge closes back to the start edge or start primitive
        if {s_next, e_next} == {start_point, end_point}:
            m_next = m_start
            next_prim = None
        else:
            edge = geo.findEdge(s_next, e_next)
            adj_prims = [p for p in edge.prims() if p != curr_prim] if edge is not None else []
            if prim in adj_prims:
                m_next = m_start
                next_prim = None
            else:
                candidate_prims = [p for p in adj_prims if p not in visited_prims and len(p.vertices()) == 4]
                m_next = geo.createPoint()
                m_next.setPosition(_calc_loop_cut_position(s_next, e_next, scalar, use_ratio))
                added_points.append(m_next)
                if len(candidate_prims) == 1:
                    next_prim = candidate_prims[0]
                else:
                    next_prim = None

        if diff == 1:
            end_side_face_points.append([m_curr, e_pt, e_next, m_next])
            start_side_face_points.append([s_pt, m_curr, m_next, s_next])
        else:
            end_side_face_points.append([e_pt, m_curr, m_next, e_next])
            start_side_face_points.append([m_curr, s_pt, s_next, m_next])

        if next_prim is not None:
            curr_prim = next_prim
            s_pt = s_next
            e_pt = e_next
            m_curr = m_next
        else:
            curr_prim = None

    attr_data = collect_prim_attrs(visited_prims)
    geo.deletePrims(visited_prims, keep_points=True)
    _ = fill_faces(end_side_face_points, attr_data)
    start_side_prims = fill_faces(start_side_face_points, attr_data)
    return added_points, start_side_prims

def _calc_loop_cut_position(p_start: hou.Point, p_end: hou.Point, scalar: float, use_ratio: bool) -> hou.Vector3:
    pos_s = p_start.position()
    pos_e = p_end.position()
    if use_ratio:
        return pos_s * (1.0 - scalar) + pos_e * scalar
    else:
        v = pos_e - pos_s
        length = v.length()
        if length < 1e-8:
            return pos_s
        return pos_s + (v / length) * scalar


def fill_pentagon(
    geo: hou.Geometry,
    points: Sequence[hou.Point],
    midpoint_edge: tuple[hou.Point, hou.Point],
    reverse: bool = False,
) -> tuple[hou.Point, hou.Point]:
    """Subdivide a pentagon into 3 quads by placing a midpoint on one edge and an internal floating point.

    :param geo: The Houdini geometry.
    :param points: 5 cyclic points of the pentagon.
    :param midpoint_edge: Tuple of 2 adjacent points defining the edge to split.
    :return: Tuple of (midpoint on specified midedge, interior float point).
    """
    assert len(set(points)) == 5, "Expected 5 distinct points for fill_pentagon"

    p_a, p_b = midpoint_edge
    assert p_a in points and p_b in points and p_a != p_b, f"mid_edge {midpoint_edge} must be in points"
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

    fill_face(geo, [p1, midpoint, floatpoint, p2], reverse)
    fill_face(geo, [midpoint, p0, p4, floatpoint], reverse)
    fill_face(geo, [floatpoint, p4, p3, p2], reverse)

    return midpoint, floatpoint


def fill_pentagon_with_buffer(
    geo: hou.Geometry,
    points: Sequence[hou.Point],
    buffer_edge: tuple[hou.Point, hou.Point],
    buffer_ratio: float,
    midpoint_edge: tuple[hou.Point, hou.Point],
    reverse: bool = False,
) -> tuple[hou.Point, hou.Point, hou.Point, hou.Point]:
    """Subdivide a pentagon with a buffer quad adjacent to buffer_edge, then subdivide the remainder into 3 quads.

    :param geo: The Houdini geometry.
    :param points: 5 cyclic points of the pentagon.
    :param buffer_edge: Tuple of 2 adjacent points defining the edge to buffer.
    :param buffer_ratio: Ratio along the connected edges from buffer_edge (in [0, 1)).
    :param midpoint_edge: Tuple of 2 adjacent points defining the edge to split in the pentagon.
    :param reverse: Whether to reverse primitive vertex ordering.
    :return: Tuple of (midpoint, interior float point, buffer_point_a, buffer_point_b).
    """
    assert len(set(points)) == 5, "Expected 5 distinct points for fill_pentagon_with_buffer"
    assert 0.0 <= buffer_ratio < 1.0, f"buffer_ratio must be in [0.0, 1.0), got {buffer_ratio}"
    assert set(midpoint_edge) != set(buffer_edge), f"midpoint_edge {midpoint_edge} cannot be buffer_edge {buffer_edge}"

    pb_a, pb_b = buffer_edge
    assert pb_a in points and pb_b in points and pb_a != pb_b, f"buffer_edge {buffer_edge} must be in points"
    idx_a = points.index(pb_a)
    idx_b = points.index(pb_b)
    diff = (idx_b - idx_a) % 5
    assert diff in (1, 4), f"buffer_edge points must be adjacent in points sequence, got diff {diff}"

    if buffer_ratio == 0.0:
        midpoint, floatpoint = fill_pentagon(geo, points, midpoint_edge, reverse)
        return midpoint, floatpoint, pb_a, pb_b

    if diff == 1:
        p_start, p_end = pb_a, pb_b
        p_prev = points[(idx_a - 1) % 5]
        p_next = points[(idx_b + 1) % 5]
    else:
        p_start, p_end = pb_b, pb_a
        p_prev = points[(idx_b - 1) % 5]
        p_next = points[(idx_a + 1) % 5]

    b_start_pos = p_start.position() * (1.0 - buffer_ratio) + p_prev.position() * buffer_ratio
    b_start = geo.createPoint()
    b_start.setPosition(b_start_pos)

    b_end_pos = p_end.position() * (1.0 - buffer_ratio) + p_next.position() * buffer_ratio
    b_end = geo.createPoint()
    b_end.setPosition(b_end_pos)

    fill_face(geo, [p_start, p_end, b_end, b_start], reverse)

    new_points = [
        b_start if p == p_start else (b_end if p == p_end else p)
        for p in points
    ]
    new_midpoint_edge = (
        b_start if midpoint_edge[0] == p_start else (b_end if midpoint_edge[0] == p_end else midpoint_edge[0]),
        b_start if midpoint_edge[1] == p_start else (b_end if midpoint_edge[1] == p_end else midpoint_edge[1]),
    )

    midpoint, floatpoint = fill_pentagon(geo, new_points, new_midpoint_edge, reverse)
    b_a = b_start if pb_a == p_start else b_end
    b_b = b_end if pb_a == p_start else b_start
    return midpoint, floatpoint, b_a, b_b


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
    """
    See `internal.elliptical_interpolator.interpolate_conic`.
    """
    return elliptical_interpolator.get_point_on_ellipse_2d(origin, vertical_end, side_end, rad_from_y)

def interpolate_conic(
        p0: hou.Vector3 | hou.Point,
        p1: hou.Vector3 | hou.Point,
        p2: hou.Vector3 | hou.Point,
        normal0: hou.Vector3,
        normal1: hou.Vector3,
) -> Callable[[float], tuple[tuple[hou.Vector3, hou.Vector3], ...]]:
    """
    See `internal.elliptical_interpolator.interpolate_conic`.
    """
    return elliptical_interpolator.interpolate_conic(p0, p1, p2, normal0, normal1)


def interpolate_elliptical(
        p0: hou.Vector3 | hou.Point,
        p1: hou.Vector3 | hou.Point,
        p2: hou.Vector3 | hou.Point,
        normal0: hou.Vector3,
        normal2: hou.Vector3,
) -> Callable[[float], tuple[hou.Vector3, hou.Vector3]]:
    """
    See `internal.elliptical_interpolator.interpolate_elliptical`.
    """
    return elliptical_interpolator.interpolate_elliptical(p0, p1, p2, normal0, normal2)
