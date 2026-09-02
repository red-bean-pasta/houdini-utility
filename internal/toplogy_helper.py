from typing import Sequence, Any

import hou

from ..common import fill_face


class Edge:
    def __init__(self, point1: int, point2: int, reorder: bool = False) -> None:
        if reorder:
            self.start = min(point1, point2)
            self.end = max(point1, point2)
        else:
            self.start = point1
            self.end = point2

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Edge):
            return self.start == other.start and self.end == other.end
        return False

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __iter__(self):
        yield self.start
        yield self.end

    @staticmethod
    def from_points(point1: hou.Point, point2: hou.Point, reorder: bool = False) -> "Edge":
        return Edge(point1.number(), point2.number(), reorder)


def fill_faces(
        points: list[list[hou.Point]],
        attributes: Sequence[tuple[dict[str, Any], list[str]]] = ()
) -> list[hou.Prim]:
    prims: list[hou.Prim] = []
    for pts in points:
        geo = pts[0].geometry()
        prims.append(fill_face(geo, pts))
    if attributes:
        apply_prim_attrs(prims, attributes)
    return prims


def collect_prim_attrs(
        prims: Sequence[hou.Prim],
) -> list[tuple[dict[str, Any], list[str]]]:
    if not prims:
        return []
    assert is_same_geo(prims)
    geo = prims[0].geometry()
    prim_attribs = geo.primAttribs()
    prim_groups = geo.primGroups()
    return [
        (
            {attr.name(): p.attribValue(attr) for attr in prim_attribs},
            [g.name() for g in prim_groups if g.contains(p)],
        ) for p in prims
    ]

def apply_prim_attrs(
        prims: Sequence[hou.Prim],
        data: Sequence[tuple[dict[str, Any], list[str]]],
) -> None:
    for (attr_dict, group_list), prim in zip(data, prims):
        for attr_name, attr_val in attr_dict.items():
            prim.setAttribValue(attr_name, attr_val)
        for g_name in group_list:
            geo = prim.geometry()
            group = geo.findPrimGroup(g_name)
            if group is not None:
                group.add(prim)


def copy_point_data(
        src: hou.Point,
        dst: hou.Point
) -> None:
    assert is_same_geo([src, dst])
    geo = src.geometry()
    for attr in geo.pointAttribs():
        if attr.name() != "P":
            dst.setAttribValue(attr.name(), src.attribValue(attr))
    for grp in geo.pointGroups():
        if grp.contains(src):
            grp.add(dst)


def partition_connected_prims(
        prims: list[hou.Prim]
) -> list[list[hou.Prim]]:
    """

    :param prims:
    :return: islands of argument `prims`
    """
    edge_prim_map: dict[tuple[int, int], list[hou.Prim]] = {}
    for p in prims:
        pts = [v.point().number() for v in p.vertices()]
        n = len(pts)
        for i in range(n):
            edge = (pts[i], pts[(i + 1) % n]) if pts[i] < pts[(i + 1) % n] else (pts[(i + 1) % n], pts[i])
            edge_prim_map.setdefault(edge, []).append(p)

    adjacent_map: dict[hou.Prim, set[hou.Prim]] = {p: set() for p in prims}
    for shared_prims in edge_prim_map.values():
        for p1 in shared_prims:
            for p2 in shared_prims:
                if p1 != p2:
                    adjacent_map[p1].add(p2)

    components: list[list[hou.Prim]] = []
    visited: set[hou.Prim] = set()
    for p in prims:
        if p in visited:
            continue
        compo: list[hou.Prim] = []
        queue = [p]
        visited.add(p)
        while queue:
            current = queue.pop()
            compo.append(current)
            for neighbor in adjacent_map[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(compo)

    return components


def get_edge_prim_count(
        prims: list[hou.Prim]
) -> dict[Edge, int]:
    counts: dict[Edge, int] = {}
    for p in prims:
        pts = [v.point().number() for v in p.vertices()]
        n = len(pts)
        for i in range(n):
            edge = Edge(pts[i], pts[(i + 1) % n], reorder=True)
            counts[edge] = counts.get(edge, 0) + 1
    return counts


def is_same_geo(sequence: Sequence[Any]) -> bool:
    if len(sequence) <= 1:
        return True
    first_repr = repr(sequence[0].geometry())
    return all(
        repr(item.geometry()) == first_repr
        for item in sequence[1:]
    )
