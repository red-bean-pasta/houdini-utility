"""
Utility functions for attribute-based identification, naming, and indexing of geometry elements (points, prims).
"""
import re
from collections import defaultdict
from typing import Sequence, Callable, Literal

import hou

from topology import classify_after_inset
from common import (
    points_start_with,
    set_point_attr,
    fill_face,
    get_prim_centroid,
)


def unique_points_start_with(
    geo: hou.Geometry,
    attribute: str,
    prefixes: str | tuple[str, ...],
) -> dict[str, hou.Point]:
    result = {}
    for point, value in zip(
        geo.points(),
        geo.pointStringAttribValues(attribute),
    ):
        if not value:
            continue
        assert value not in result, f"Duplicate point {attribute}: {value}"
        if not value.startswith(prefixes):
            continue
        result[value] = point
    return result


def points_by_unique_attr(
        geo: hou.Geometry | hou.Prim | Sequence[hou.Prim],
        attribute: str,
) -> dict[str, hou.Point]:
    result = {}
    point: hou.Point
    if not isinstance(geo, Sequence):
        geo = (geo, )
    for g in geo:
        for point in g.points():
            value = point.stringAttribValue(attribute)
            if not value:
                continue
            assert value not in result, f"Duplicate point {attribute}: {value}"
            result[value] = point
    return result


def deduplicate_point_attributes(
        geo: hou.Geometry,
        attribute: str,
        prefix: str | tuple[str, ...] | None,
        add_affix: bool = False,
        keep_first: bool = True,
) -> None:
    """Deduplicate string attributes on points by either clearing duplicates or affixing sequential indices.

    :param geo: The Houdini geometry.
    :param attribute: The point attribute name.
    :param prefix: Optional prefix to filter points.
    :param add_affix: If true, affixes like "_1", "_2" will be added; otherwise duplicates after the first are cleared.
    :param keep_first: If true, first encountered point will keep the attribute, else the last;
    """
    points = points_start_with(geo, attribute, prefix) if prefix else geo.points()
    grouped: defaultdict[str, list[hou.Point]] = defaultdict(list)
    for p in points:
        v = p.stringAttribValue(attribute)
        grouped[v].append(p)
    for value, duplicates in grouped.items():
        if len(duplicates) <= 1:
            continue
        if add_affix:
            for i, point in enumerate(duplicates):
                set_point_attr(point, attribute, f"{value}_{i+1}")
        else:
            for point in (duplicates[1:] if keep_first else duplicates[:-1]):
                point.setAttribValue(attribute, "")


def rename_point_attr(
        geo: hou.Geometry,
        attribute: str,
        filtrate: Callable[[hou.Point], bool],
        rename: Callable[[str], str | Literal[False]],
) -> None:
    """Conditionally rename point attribute values matching a filter.

    :param geo: The Houdini geometry.
    :param attribute: The point attribute name.
    :param filtrate: Predicate to decide if point should be considered.
    :param rename: Transformation function returning new name or False to skip renaming.
    """
    for point in geo.points():
        if not filtrate(point):
            continue
        point_id = point.stringAttribValue(attribute)
        if not point_id:
            continue
        new_id = rename(point_id)
        if new_id is False:
            continue
        point.setAttribValue(attribute, new_id)


def indexed_attr_range(
        geo: hou.Geometry,
        attribute: str,
        prefix: str,
) -> tuple[int, int] | None:
    values = geo.pointStringAttribValues(attribute)
    numbers: list[int] = []
    for value in values:
        if prefix:
            if not value.startswith(prefix):
                continue
            remainder = value[len(prefix):]
            match = re.search(r"-?\d+", remainder)
        else:
            match = re.search(r"-?\d+", value)
        if match is not None:
            numbers.append(int(match.group()))

    if not numbers:
        return None
    return min(numbers), max(numbers)


def fill_face_by_attr(
    geo: hou.Geometry,
    attribute: str,
    values: list[str],
) -> hou.Polygon:
    all_points = points_by_unique_attr(geo, attribute)
    face_points: list[hou.Point] = []
    for v in values:
        p = all_points.get(v)
        assert p is not None, f"Expected point with id {v}"
        face_points.append(p)
    return fill_face(geo, face_points)


def attribute_after_inset(
        node: hou.SopNode,
        attribute: str,
        pane_prefix: str,
        sill_prefix: str,
        horizontal_pack_size: int = 1,
        vertical_pack_size: int = 1,
) -> tuple[
        list[tuple[hou.Prim, ...]],
        list[tuple[hou.Prim, ...]],
]:
    assert pane_prefix != sill_prefix, "Pane and sill prefixes must differ"
    geo = node.geometry()

    in0 = node.input(0)
    assert in0 is not None, f"Expected node {node.path()} to have an input connected"
    in0_in0 = in0.input(0)
    assert in0_in0 is not None, f"Expected upstream node {in0.path()} to have an input connected"

    prim_count_before = len(in0_in0.geometry().prims())
    panes, sills = classify_after_inset(geo, prim_count_before, horizontal_pack_size, vertical_pack_size)
    for prefix, compos in {pane_prefix: panes, sill_prefix: sills}.items():
        i1 = 1; i2 = -1
        for compo in compos:
            centroid = get_prim_centroid(compo)
            i = i2 if centroid.x() < 0 else i1
            if len(compo) == 1:
                compo[0].setAttribValue(attribute, prefix + str(i))
                continue
            for j, part in enumerate(compo, start=1):
                part.setAttribValue(attribute, f"{prefix}{i}_{j}")
            if centroid.x() < 0:
                i2 -= 1
            else:
                i1 += 1
    return panes, sills
