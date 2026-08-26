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
        normal0: hou.Vector3,
        normal1: hou.Vector3,
) -> Callable[[float], tuple[tuple[hou.Vector3, hou.Vector3], ...]]:
    """Construct a planar conic passing through p0, p1, p2 with normals normal0 at p0 and normal1 at p1.

    The method projects the 3D problem into a 2D local orthonormal coordinate plane:
      - along_axis: direction from p0 to p1
      - across_axis: in-plane normal orthogonal to along_axis
    It sets up an algebraic conic equation:
      A * x² + B * x * y + C * y² + D * x + E * y = 0
    where (x, y) = (along, across) with p0 at (0, 0).
    The remaining 4 constraints (passage through p1, p2, and normal directions at p0, p1)
    form a 4x5 linear system solved via SVD for the 1D null space.

    :param p0: First point on conic (origin of the local 2D coordinate system).
    :param p1: Second point on conic.
    :param p2: Intermediate third point on conic.
    :param normal0: Inward normal vector at p0.
    :param normal1: Inward normal vector at p1.
    :return: A function that accepts a signed distance along the p0->p1 axis,
             and returns 0, 1, or 2 point and normal pairs in 3D space on the conic.
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

    def project_normal(norm: hou.Vector3) -> tuple[float, float]:
        proj = norm - norm.dot(normal) * normal
        assert proj.length() != 0.0, "Normal must have a non-zero component in the conic plane"
        proj = proj.normalized()
        return proj.dot(along_axis), proj.dot(across_axis)

    along_normal0, across_normal0 = project_normal(normal0)
    along_normal1, across_normal1 = project_normal(normal1)

    # A*x² + B*x*y + C*y² + D*x + E*y = 0
    matrix = np.array([
        [along1**2, 0.0, 0.0, along1, 0.0],
        [along2**2, along2 * across2, across2**2, along2, across2],
        [0.0, 0.0, 0.0, across_normal0, -along_normal0],
        [2.0 * along1 * across_normal1, -along1 * along_normal1, 0.0, across_normal1, -along_normal1],
    ], dtype=float)
    _, singular_values, vh = np.linalg.svd(matrix)
    A, B, C, D, E = map(float, vh[-1])

    tolerance = np.finfo(float).eps * max(matrix.shape) * singular_values[0]
    assert np.sum(singular_values > tolerance) == 4, "The supplied points and normals do not determine a unique conic"

    coefficient_scale = max(abs(A), abs(B), abs(C), abs(D), abs(E))
    assert coefficient_scale != 0.0, "Failed to construct a valid conic"
    A /= coefficient_scale
    B /= coefficient_scale
    C /= coefficient_scale
    D /= coefficient_scale
    E /= coefficient_scale

    def to_3d(along: float, across: float) -> hou.Vector3:
        return vector0 + along * along_axis + across * across_axis

    def evaluate(along: float) -> tuple[tuple[hou.Vector3, hou.Vector3], ...]:
        def get_point_and_normal(p_across: float) -> tuple[hou.Vector3, hou.Vector3]:
            gradient_along = 2.0 * A * along + B * p_across + D
            gradient_across = B * along + 2.0 * C * p_across + E
            curvature = (
               ( 2.0 * A) * gradient_across ** 2
                - 2.0 * B * gradient_along * gradient_across
                + (2.0 * C) * gradient_along ** 2
            )
            point_normal = (
                along_axis * gradient_along
                + across_axis * gradient_across
            ).normalized()
            if curvature > 0.0:
                point_normal = -point_normal
            point = to_3d(along, p_across)
            return point, point_normal

        quadratic = C
        linear = B * along + E
        constant = A * along**2 + D * along
        eps = 1e-12

        if abs(quadratic) <= eps:
            if abs(linear) <= eps:
                return ()
            across = -constant / linear
            return get_point_and_normal(across),

        discriminant = linear**2 - 4.0 * quadratic * constant
        if discriminant < -eps:
            return ()

        if abs(discriminant) <= eps:
            across = -linear / (2.0 * quadratic)
            return get_point_and_normal(across),

        sqrt_discriminant = math.sqrt(discriminant)
        across0 = (-linear + sqrt_discriminant) / (2.0 * quadratic)
        across1 = (-linear - sqrt_discriminant) / (2.0 * quadratic)
        return get_point_and_normal(across0), get_point_and_normal(across1)

    return evaluate


def interpolate_elliptical(
        p0: hou.Vector3 | hou.Point,
        p1: hou.Vector3 | hou.Point,
        p2: hou.Vector3 | hou.Point,
        normal0: hou.Vector3,
        normal2: hou.Vector3,
) -> Callable[[float], tuple[hou.Vector3, hou.Vector3]]:
    """Construct a smooth elliptical 3D bulge passing through p0, p1, p2 with normals normal0 at p0 and normal2 at p2.

    The middle point p1 is explicitly treated as the apex/tip of the bulge relative to the baseline from p0 to p2.
    The curve is composed of two quintic polynomial segments (p0 -> p1 and p1 -> p2) joined at p1 with C² continuity.
    Tangent magnitudes and apex curvature are chosen by minimizing jerk energy.

    :param p0: Start point of the curve (t = 0.0).
    :param p1: Apex/peak point of the curve (bulge tip).
    :param p2: End point of the curve (t = 1.0).
    :param normal0: Inward normal vector at p0.
    :param normal2: Inward normal vector at p2.
    :return: A function that accepts a normalized parameter t in [0, 1]
            representing progress along the p0->p2 baseline chord axis (where 0.0 is p0 and 1.0 is p2),
            and returns the (position, in-plane inward normal) pair as hou.Vector3.
    """
    p0_arr, p1_arr, p2_arr = _to_np(p0), _to_np(p1), _to_np(p2)
    tangent_apex = _normalized(p2_arr - p0_arr)
    chord_length = np.linalg.norm(p2_arr - p0_arr)
    mid_along = float(np.dot(p1_arr - p0_arr, tangent_apex))

    height_vec = (p1_arr - p0_arr) - mid_along * tangent_apex
    height = np.linalg.norm(height_vec)
    assert height > 1e-6, "p1 must not be collinear with p0 and p2"
    outward = height_vec / height
    inward = -outward

    plane_normal = _normalized(np.cross(tangent_apex, outward))

    d0 = _normalized(np.cross(plane_normal, _project_on_plane(_to_np(normal0), plane_normal)))
    d2 = _normalized(np.cross(plane_normal, _project_on_plane(_to_np(normal2), plane_normal)))

    def build_spline(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        endpoint_speed, apex_speed, apex_accel = np.exp(np.clip(parameters, -20.0, 20.0))
        vel0, vel1, vel2 = endpoint_speed * d0, apex_speed * tangent_apex, endpoint_speed * d2
        acc0, acc1, acc2 = np.zeros(3), apex_accel * inward, np.zeros(3)
        return (
            _quintic_coefficients(p0_arr, vel0, acc0, p1_arr, vel1, acc1),
            _quintic_coefficients(p1_arr, vel1, acc1, p2_arr, vel2, acc2),
        )

    def objective(parameters: np.ndarray) -> float:
        left, right = build_spline(parameters)
        return _jerk_energy(left) + _jerk_energy(right)

    x0 = np.log([chord_length * 0.5, chord_length * 0.5, max(4.0 * height, 1e-6)])
    best_x = _bfgs_minimize(objective, x0)
    left_coeff, right_coeff = build_spline(best_x)

    def evaluate(t: float) -> tuple[hou.Vector3, hou.Vector3]:
        t = float(np.clip(t, 0.0, 1.0))
        target_along = t * chord_length

        if target_along <= mid_along:
            sub_t = _solve_t_for_along(left_coeff, p0_arr, tangent_apex, target_along)
            pos, vel = _eval_quintic(left_coeff, sub_t)
        else:
            sub_t = _solve_t_for_along(right_coeff, p0_arr, tangent_apex, target_along)
            pos, vel = _eval_quintic(right_coeff, sub_t)

        tangent = _normalized(vel) if np.linalg.norm(vel) > 1e-12 else tangent_apex
        in_plane_normal = np.cross(tangent, plane_normal)
        in_plane_normal = _normalized(in_plane_normal) if np.linalg.norm(in_plane_normal) > 1e-12 else inward

        return hou.Vector3(pos), hou.Vector3(in_plane_normal)

    return evaluate

def _to_np(v: hou.Vector3 | hou.Point | Sequence[float] | np.ndarray) -> np.ndarray:
    return np.array(v.position() if isinstance(v, hou.Point) else v, dtype=float)

def _normalized(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v)
    assert length > 1e-6, "Expected non-zero vector for normalization"
    return v / length

def _project_on_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return _normalized(v - np.dot(v, normal) * normal)

def _quintic_coefficients(
        p0: np.ndarray,
        v0: np.ndarray,
        a0: np.ndarray,
        p1: np.ndarray,
        v1: np.ndarray,
        a1: np.ndarray,
) -> np.ndarray:
    """Construct quintic polynomial C(t) = c0 + c1*t + ... + c5*t^5 for t in [0, 1]."""
    dp = p1 - p0
    c0 = p0
    c1 = v0
    c2 = 0.5 * a0
    c3 = 10.0 * dp - 6.0 * v0 - 4.0 * v1 - 1.5 * a0 + 0.5 * a1
    c4 = -15.0 * dp + 8.0 * v0 + 7.0 * v1 + 1.5 * a0 - a1
    c5 = 6.0 * dp - 3.0 * (v0 + v1) - 0.5 * (a0 - a1)
    return np.vstack([c0, c1, c2, c3, c4, c5])

def _jerk_energy(coeff: np.ndarray) -> float:
    """Compute exact integral of ||C'''(t)||^2 dt from 0 to 1 for a quintic polynomial."""
    A = 6.0 * coeff[3]
    B = 24.0 * coeff[4]
    C = 60.0 * coeff[5]

    return float(
        np.dot(A, A)
        + np.dot(A, B)
        + (2.0 / 3.0) * np.dot(A, C)
        + (1.0 / 3.0) * np.dot(B, B)
        + 0.5 * np.dot(B, C)
        + 0.2 * np.dot(C, C)
    )

def _eval_quintic(coeff: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate position and velocity of quintic polynomial at scalar t."""
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    pos = coeff[0] + coeff[1] * t + coeff[2] * t2 + coeff[3] * t3 + coeff[4] * t4 + coeff[5] * t5
    vel = coeff[1] + 2.0 * coeff[2] * t + 3.0 * coeff[3] * t2 + 4.0 * coeff[4] * t3 + 5.0 * coeff[5] * t4
    return pos, vel

def _solve_t_for_along(coeff: np.ndarray, p0: np.ndarray, along_axis: np.ndarray, target_along: float) -> float:
    """Solve for parameter t in [0, 1] such that (C(t) - p0) . along_axis = target_along."""
    k = np.dot(coeff, along_axis)
    k[0] -= (np.dot(p0, along_axis) + target_along)

    val0 = k[0]
    val1 = float(np.sum(k))
    if val0 * val1 >= 0.0:
        return 0.0 if abs(val0) < abs(val1) else 1.0

    low, high = 0.0, 1.0
    t = max(0.0, min(1.0, -val0 / (val1 - val0)))
    for _ in range(25):
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        val = k[0] + k[1] * t + k[2] * t2 + k[3] * t3 + k[4] * t4 + k[5] * t5
        if abs(val) < 1e-12:
            break
        if val < 0.0:
            low = t
        else:
            high = t
        dval = k[1] + 2.0 * k[2] * t + 3.0 * k[3] * t2 + 4.0 * k[4] * t3 + 5.0 * k[5] * t4
        if abs(dval) > 1e-12:
            t_new = t - val / dval
            if low < t_new < high:
                t = t_new
            else:
                t = 0.5 * (low + high)
        else:
            t = 0.5 * (low + high)

    return float(t)

def _bfgs_minimize(
        f: Callable[[np.ndarray], float],
        x0: np.ndarray | list[float],
        max_iter: int = 200,
        gtol: float = 1e-5,
        eps: float = 1e-6,
) -> np.ndarray:
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)

    def grad(x_curr: np.ndarray) -> np.ndarray:
        g = np.zeros(n)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            g[i] = (f(x_curr + dx) - f(x_curr - dx)) / (2.0 * eps)
        return g

    fx = f(x)
    g = grad(x)

    for _ in range(max_iter):
        if np.linalg.norm(g) < gtol:
            break
        p = -H @ g
        alpha = 1.0
        c1 = 1e-4
        while alpha > 1e-8:
            if f(x + alpha * p) <= fx + c1 * alpha * np.dot(g, p):
                break
            alpha *= 0.5

        x_next = x + alpha * p
        fx_next = f(x_next)
        g_next = grad(x_next)

        s = x_next - x
        y = g_next - g
        ys = np.dot(y, s)

        if ys > 1e-10:
            rho = 1.0 / ys
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        else:
            H = np.eye(n)

        x, fx, g = x_next, fx_next, g_next
        if np.linalg.norm(s) < 1e-8:
            break

    return x
