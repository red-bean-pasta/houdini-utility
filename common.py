import re
from collections import defaultdict
from typing import get_origin, TypeVar, get_args, Any, Sequence, Generic, Iterator

import hou

T = TypeVar("T")


class MessagedResult(Generic[T]):
    def __init__(self, value: T, messages: Sequence[str] = ()):
        self.value = value
        self.messages = list(messages)

    @property
    def has_messages(self) -> bool:
        return bool(self.messages)

    def add_message(self, message: str) -> None:
        self.messages.append(message)

    def add_messages(self, messages: Sequence[str]) -> None:
        self.messages.extend(messages)

    def __iter__(self) -> Iterator[Any]:
        yield self.value
        yield self.messages

    def __getitem__(self, index: int) -> Any:
        return (self.value, self.messages)[index]

    def __len__(self) -> int:
        return 2

    @staticmethod
    def retain(value: T, *results: "MessagedResult") -> "MessagedResult[T]":
        messages: list[str] = []
        for r in results:
            if isinstance(r, MessagedResult):
                messages.extend(r.messages)
        return MessagedResult(value, messages)


def is_equal_approx(a: float, b: float, tol: float = 1e-5) -> True:
    return abs(a - b) <= tol


def snake_case(s):
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'[\s\-]+', '_', s)
    return s.lower()

def title_case(s):
    words = re.findall(r'[A-Za-z0-9]+', re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s))
    return ' '.join(word.capitalize() for word in words)

def pascal_case(s):
    return title_case(s).replace(' ', '')


class Parameters(dict[str, Any]):
    """A dictionary subclass supporting attribute-style parameter access (e.g., `param.width`)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Parameter '{name}' not found") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"Parameter '{name}' not found") from None


def get_params(
        node: hou.OpNode,
        exclude_internal: bool = True,
        use_tuple: bool = True,
) -> Parameters:
    """Read all evaluated parameters on a node into a dot-accessible Parameters dictionary."""
    params = Parameters()
    for pt in node.parmTuples():
        if exclude_internal and not pt.isSpare():
            continue

        tpl = pt.parmTemplate()
        if isinstance(tpl, (hou.ButtonParmTemplate, hou.LabelParmTemplate, hou.SeparatorParmTemplate)):
            continue

        parent = pt[0].parentMultiParm() if hasattr(pt[0], "parentMultiParm") else None
        if parent is not None and isinstance(parent.parmTemplate(), hou.RampParmTemplate):
            continue

        if isinstance(tpl, hou.ToggleParmTemplate):
            val = bool(pt[0].evalAsInt())
        elif isinstance(tpl, hou.RampParmTemplate):
            val = pt[0].evalAsRamp()
        elif len(pt) == 1:
            val = pt[0].eval()
        else:
            evaluated = pt.eval()
            if not use_tuple and isinstance(tpl, hou.FloatParmTemplate):
                is_vector = (
                    tpl.namingScheme() in (hou.parmNamingScheme.XYZW, hou.parmNamingScheme.UVW, hou.parmNamingScheme.XYWH)
                    or tpl.look() == hou.parmLook.Vector
                )
                if is_vector:
                    match len(pt):
                        case 2:
                            val = hou.Vector2(evaluated)
                        case 3:
                            val = hou.Vector3(evaluated)
                        case 4:
                            val = hou.Vector4(evaluated)
                        case _:
                            val = evaluated
                else:
                    val = evaluated
            else:
                val = evaluated

        params[pt.name()] = val

    return params


def points_to_positions(points: Sequence[hou.Point]) -> Iterator[hou.Vector3]:
    return (p.position() for p in points)


def add_folder(
        node: hou.OpNode,
        name: str,
        label: str = "",
        folder_type: hou.folderType = hou.folderType.Tabs,
        **kwargs,
) -> None:
    if not label:
        label = title_case(name)
    ptg = node.parmTemplateGroup()
    folder = ptg.findFolder(label)
    if not folder:
        folder = hou.FolderParmTemplate(name, label, folder_type=folder_type, **kwargs)
        ptg.append(folder)
    node.setParmTemplateGroup(ptg)

def add_heading(
        node: hou.OpNode,
        text: str,
        name: str = "",
        label: str = "",
        folder_label: str = "",
        **kwargs
) -> None:
    group = node.parmTemplateGroup()
    if not name:
        name = snake_case(text)
    heading = hou.LabelParmTemplate(
        name=name,
        label=label if label else text,
        **kwargs
    )
    heading.setLabelParmType(hou.labelParmType.Heading)

    _add_to_group(group, heading, folder_label)
    node.setParmTemplateGroup(group)
    node.parm(name).set(text)

def add_float_param(
        node: hou.OpNode,
        name: str,
        size: int = 1,
        default: float | tuple[float, ...] = (0.0,),
        min_max: tuple[float | None, float | None] = (None, None),
        naming_scheme: hou.parmNamingScheme = hou.parmNamingScheme.XYZW,
        label: str = "",
        folder_label: str = "",
        **kwargs
) -> None:
    group = node.parmTemplateGroup()

    if isinstance(default, (int, float)):
        default = (float(default),)
    else:
        default = tuple(float(x) for x in default)
    param = hou.FloatParmTemplate(
        name,
        label if label else title_case(name),
        num_components=size,
        default_value=default,
        naming_scheme=naming_scheme,
        **kwargs,
    )
    p_min, p_max = min_max
    if p_min is not None:
        param.setMinValue(p_min)
        param.setMinIsStrict(True)
    if p_max is not None:
        param.setMaxValue(p_max)
        param.setMaxIsStrict(True)

    _add_to_group(group, param, folder_label)
    node.setParmTemplateGroup(group)

def _add_to_group(
        group: hou.ParmTemplateGroup,
        param: hou.ParmTemplate,
        folder: str
) -> None:
    if not folder:
        group.append(param)
    else:
        group.appendToFolder(folder, param)


def affix_attribute_value(prefix: str, *affixes: int | str) -> str:
    """Concatenate a prefix directly with underscore-joined affixes.

    Example: affix_attribute_value("pane", 1, 2) -> "pane1_2"
             affix_attribute_value("pane_", 1, 2) -> "pane_1_2"
    """
    return prefix + "_".join(map(str, affixes))


def get_parent(node: hou.Node) -> hou.OpNode:
    parent = node.parent()
    assert isinstance(parent, hou.OpNode), "Expected Python SOP to be inside a valid network"
    return parent


def get_float_parm(node: hou.OpNode, name: str) -> float:
    return get_parm(node, name, float)

def get_vector2_parm(node: hou.OpNode, name: str) -> hou.Vector2:
    return hou.Vector2(
        get_parm(node, name, tuple[float, float])
    )

def get_vector3_parm(node: hou.OpNode, name: str) -> hou.Vector3:
    return hou.Vector3(
        get_parm(node, name, tuple[float, float, float])
    )

def get_parm(node: hou.OpNode, name: str, cls: type[T]) -> T:
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

def add_global_attr(geo: hou.Geometry, name: str, default: Any) -> hou.Attrib:
    return add_attr(geo, hou.attribType.Global, name, default)

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
    reverse: bool = False,
) -> hou.Polygon:
    clone = list(points)
    if reverse:
        clone.reverse()
    polygon = geo.createPolygon()
    for point in clone:
        polygon.addVertex(point)
    return polygon


def get_prim_normal(prim: hou.Prim) -> hou.Vector3:
    if isinstance(prim, hou.Face):
        return prim.normal()
    pts = [v.point().position() for v in prim.vertices()]
    n = hou.Vector3()
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        n += hou.Vector3(
            (p[1] - q[1]) * (p[2] + q[2]),
            (p[2] - q[2]) * (p[0] + q[0]),
            (p[0] - q[0]) * (p[1] + q[1])
        )
    return -n.normalized()

def get_prim_centroid(prims: hou.Prim | Sequence[hou.Prim]) -> hou.Vector3:
    if isinstance(prims, hou.Prim):
        prims = (prims,)
    assert len(prims) > 0, "Expected at least one primitive to compute centroid"
    center = hou.Vector3()
    for prim in prims:
        center += prim.boundingBox().center()
    return center / len(prims)


def rotation_to(a: hou.Vector3, b: hou.Vector3) -> hou.Quaternion:
    q = hou.Quaternion()
    q.setToVectors(a, b)
    return q

def rotate_positions(
        origin: hou.Vector3,
        positions: Sequence[hou.Vector3],
        axis: hou.Vector3,
        degrees: float,
) -> tuple[hou.Vector3, ...]:
    rot = hou.hmath.buildRotateAboutAxis(
        axis,
        degrees,
    )
    return tuple(
        (p - origin) * rot + origin
        for p in positions
    )
