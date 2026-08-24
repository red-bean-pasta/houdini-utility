from collections import defaultdict
from typing import get_origin, TypeVar, get_args, Any, Sequence

import hou

T = TypeVar("T")


def affix_attribute_value(prefix: str, *affixes: int | str) -> str:
    """Concatenate a prefix directly with underscore-joined affixes.

    Example: affix_attribute_value("pane", 1, 2) -> "pane1_2"
             affix_attribute_value("pane_", 1, 2) -> "pane_1_2"
    """
    return prefix + "_".join(map(str, affixes))


def get_parent(node: hou.SopNode) -> hou.Node:
    parent = node.parent()
    assert isinstance(parent, (hou.SopNode, hou.ObjNode, hou.Node)), "Expected Python SOP to be inside a valid network"
    return parent


def get_float_parm(node: hou.SopNode, name: str) -> float:
    return get_parm(node, name, float)

def get_vector2_parm(node: hou.SopNode, name: str) -> hou.Vector2:
    return hou.Vector2(
        get_parm(node, name, tuple[float, float])
    )

def get_vector3_parm(node: hou.SopNode, name: str) -> hou.Vector3:
    return hou.Vector3(
        get_parm(node, name, tuple[float, float, float])
    )

def get_parm(node: hou.SopNode, name: str, cls: type[T]) -> T:
    runtime_type = get_origin(cls) or cls

    if runtime_type is tuple:
        value = node.evalParmTuple(name)
    else:
        value = node.evalParm(name)

    assert value is not None, f"Expected parameter {name!r} on {node.path()}"
    assert validate_type(value, cls), f"Expected parameter {name!r} on {node.path()}  to be {cls!r}, got {value!r}"
    return value

def validate_type(value, cls: type[T]) -> bool:
    origin = get_origin(cls)

    if origin is not tuple:
        return isinstance(value, cls)

    if not isinstance(value, tuple):
        return False

    args = get_args(cls)
    if not args:
        return True
    if len(args) == 2 and args[1] is Ellipsis:
        return all(isinstance(v, args[0]) for v in value)
    return (
        len(value) == len(args)
        and all(isinstance(v, t) for v, t in zip(value, args))
    )


def add_point_group(geo: hou.Geometry, name: str) -> hou.PointGroup:
    group = geo.findPointGroup(name)
    if group is None:
        group = geo.createPointGroup(name)
    return group

def add_edge_group(geo: hou.Geometry, name: str) -> hou.EdgeGroup:
    group = geo.findEdgeGroup(name)
    if group is None:
        group = geo.createEdgeGroup(name)
    return group

def add_prim_group(geo: hou.Geometry, name: str) -> hou.PrimGroup:
    group = geo.findPrimGroup(name)
    if group is None:
        group = geo.createPrimGroup(name)
    return group


def add_point_attr(geo: hou.Geometry, name: str, default: Any) -> hou.Attrib:
    return add_attr(geo, hou.attribType.Point, name, default)

def add_prim_attr(geo: hou.Geometry, name: str, default: Any) -> hou.Attrib:
    return add_attr(geo, hou.attribType.Prim, name, default)

def add_attr(
        geo: hou.Geometry,
        cls: hou.attribType,
        name: str,
        default: Any,
        skip_existing: bool = True,
) -> hou.Attrib:
    find_attrib = {
        hou.attribType.Point: geo.findPointAttrib,
        hou.attribType.Prim: geo.findPrimAttrib,
        hou.attribType.Vertex: geo.findVertexAttrib,
        hou.attribType.Global: geo.findGlobalAttrib,
    }[cls]
    found = find_attrib(name)
    if skip_existing and found:
        return found
    return geo.addAttrib(cls, name, default)


def points_by_attr(
        geo: hou.Geometry | hou.Prim | Sequence[hou.Prim],
        attribute: str,
        skip_blank: bool = False,
) -> dict[str, set[hou.Point]]:
    result: defaultdict[str, set[hou.Point]] = defaultdict(set)
    point: hou.Point
    if not isinstance(geo, Sequence):
        geo = (geo, )
    for g in geo:
        for point in g.points():
            value = point.stringAttribValue(attribute)
            assert value is not None
            if value or not skip_blank:
                result[value].add(point)
    return dict(result)

def points_start_with(
    geo: hou.Geometry,
    attribute: str,
    prefixes: str | tuple[str, ...],
) -> list[hou.Point]:
    return [
        point
        for point, value in zip(
            geo.points(),
            geo.pointStringAttribValues(attribute),
        )
        if value.startswith(prefixes)
    ]


def set_point_attr(
    point: hou.Point,
    attribute: str,
    value: str,
) -> None:
    point.setAttribValue(attribute, value)

def set_points_attr(
    points: Sequence[hou.Point],
    attribute: str,
    values: Sequence[str],
) -> None:
    assert len(points) == len(values), "Expected matching point and value array sizes"
    for p_point, p_id in zip(points, values):
        set_point_attr(p_point, attribute, p_id)


def remove_attrs(
        geo: hou.Geometry,
        point_attribs: str | tuple[str, ...] | None = None,
        prim_attribs: str | tuple[str, ...] | None = None,
        global_attribs: str | tuple[str, ...] | None = None,
) -> None:
    for attribs, finder in zip(
        (point_attribs, prim_attribs, global_attribs),
        (geo.findPointAttrib, geo.findPrimAttrib, geo.findGlobalAttrib),
    ):
        if attribs is None:
            continue
        if isinstance(attribs, str):
            attribs = (attribs,)
        for name in attribs:
            attrib = finder(name)
            if attrib is not None:
                attrib.destroy()

def remove_groups(
        geo: hou.Geometry,
        point_groups: str | tuple[str, ...] | None = None,
        edge_groups: str | tuple[str, ...] | None = None,
        prim_groups: str | tuple[str, ...] | None = None,
) -> None:
    for groups, finder in zip(
        (point_groups, edge_groups, prim_groups),
        (geo.findPointGroup, geo.findEdgeGroup, geo.findPrimGroup),
    ):
        if groups is None:
            continue
        if isinstance(groups, str):
            groups = (groups,)
        for name in groups:
            group = finder(name)
            if group is not None:
                group.destroy()


def fill_face(
    geo: hou.Geometry,
    points: Sequence[hou.Point],
) -> hou.Polygon:
    polygon = geo.createPolygon()
    for point in points:
        polygon.addVertex(point)
    return polygon


def get_prim_centroid(prims: hou.Prim | Sequence[hou.Prim]) -> hou.Vector3:
    if isinstance(prims, hou.Prim):
        prims = (prims,)
    assert len(prims) > 0, "Expected at least one primitive to compute centroid"
    center = hou.Vector3()
    for prim in prims:
        center += prim.boundingBox().center()
    return center / len(prims)
