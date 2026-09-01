import inspect
from typing import Callable

import hou

import developing
from common import add_attr, remove_attrs, title_case, add_heading


def sopify(
    parent: hou.SopNode,
    input_node: hou.SopNode | None,
    function: Callable[[], None] | Callable[[hou.SopNode], None]
) -> hou.SopNode:
    """Generate a Python SOP node invoking a given module-level python function."""
    assert "<locals>" not in function.__qualname__, "Python SOP functions must be module-level functions"

    module = function.__module__
    qualname = function.__qualname__
    has_arg = len(inspect.signature(function).parameters) >= 1

    node = parent.createNode("python", function.__name__.strip("_"))
    if input_node is not None:
        node.setInput(0, input_node)

    call_code = f"f(hou.pwd())" if has_arg else "f()"
    node.parm("python").set(
        f"import {module}\n"
        f"f = {module}.{qualname}\n"
        f"{call_code}\n"
    )
    return node


def add_reloadable_subnet(
        parent: hou.OpNode,
        name: str,
) -> hou.SopNode:
    subnet = parent.createNode("subnet", name)
    add_reload_button(subnet)
    return subnet

def add_reload_button(parent: hou.SopNode) -> hou.SopNode:
    node = sopify(parent, None, developing.reload_modules)

    templates = node.parmTemplateGroup()
    reload_button = hou.ButtonParmTemplate(
        "reload",
        "Reload",
        script_callback=inspect.cleandoc(r"""
            import hou
            import developing
            
            node = hou.pwd()
            
            developing.reload_modules()
            
            subnet = node.parent()
            if subnet:
                try:
                    for child in subnet.allSubChildren():
                        child.cook(force=True)
                    subnet.cook(force=True)
                except Exception as e:
                    node.addError(str(e))
                    raise
        """),
        script_callback_language=hou.scriptLanguage.Python,
    )
    templates.append(reload_button)

    node.setParmTemplateGroup(templates)
    return node


def propagate_parameters(
    parent: hou.SopNode,
    child: hou.SopNode,
    skip_params: str | tuple[str, ...] = (),
    heading: str = "",
) -> None:
    """Expose spare parameters from child node onto parent node and link them via expressions."""
    if not heading:
        heading = title_case(child.name())
    add_heading(parent, heading)

    if isinstance(skip_params, str):
        skip_params = (skip_params,)
    parameters = tuple(
        parameter for parameter in child.parmTuples()
        if parameter[0].isSpare() and parameter.name() not in skip_params
    )

    group = parent.parmTemplateGroup()
    prefix = child.name() + "_"
    existing_names = {t.name() for t in group.entries()}
    for source in parameters:
        target_name = f"{prefix}{source.name()}"
        if target_name in existing_names:
            continue
        param = source.parmTemplate().clone()
        param.setName(target_name)
        group.append(param)
    parent.setParmTemplateGroup(group)

    for source in parameters:
        target = parent.parmTuple(f"{prefix}{source.name()}")
        if target is None:
            continue
        template = source.parmTemplate()
        if isinstance(template, hou.LabelParmTemplate) and template.labelParmType() == hou.labelParmType.Heading:
            text = source[0].evalAsString()
            target[0].set(f"{heading} > {text}")
        else:
            target.set(source.eval())
            for source_parm, target_parm in zip(source, target):
                source_parm.set(target_parm)


def add_merge(
    parent: hou.SopNode,
    name: str,
    *inputs: hou.SopNode,
) -> hou.SopNode:
    merge = parent.createNode("merge", name)
    for index, node in enumerate(inputs):
        merge.setInput(index, node)
    return merge


def add_fuse(
    parent: hou.SopNode,
    name: str,
    p_input: hou.SopNode,
) -> hou.SopNode:
    fuse = parent.createNode("fuse", name)
    fuse.setInput(0, p_input)
    return fuse


def add_mirror(
    parent: hou.SopNode,
    name: str,
    p_input: hou.SopNode,
    axis: hou.Vector3 | tuple[float, float, float],
    keep_original: bool,
    consolidate_unshared: bool,
    *args,
) -> hou.SopNode:
    mirror = parent.createNode("mirror", name, *args)
    mirror.setInput(0, p_input)
    mirror.parm("keepOriginal").set(keep_original)
    mirror.parm("dirx").set(axis[0])
    mirror.parm("diry").set(axis[1])
    mirror.parm("dirz").set(axis[2])
    mirror.parm("consolidateunshared").set(consolidate_unshared)
    return mirror


def add_output(
    parent: hou.SopNode,
    name: str,
    p_input: hou.SopNode,
) -> hou.SopNode:
    output = parent.createNode("null", name)
    output.setInput(0, p_input)
    output.setDisplayFlag(True)
    output.setRenderFlag(True)
    return output


def add_outside_recalculation(
    parent: hou.SopNode,
    name: str,
    p_input: hou.SopNode,
    reverse: bool = False,
) -> hou.SopNode:
    subnet = parent.createNode("subnet", name)
    subnet.setInput(0, p_input)

    indirect_input = subnet.indirectInputs()[0]
    clean_orient = subnet.createNode("clean", "orient_polygons")
    clean_orient.setInput(0, indirect_input)
    clean_orient.parm("orientpoly").set(1)
    clean_orient.parm("reversewinding").set(0)
    clean_orient.parm("deldegengeo").set(0)
    clean_orient.parm("delunusedpts").set(0)
    clean_orient.parm("removeunusedgrp").set(0)
    clean_orient.parm("deleteoverlap").set(0)
    clean_orient.parm("delnans").set(0)
    clean_orient.parm("delete_small").set(0)
    clean_orient.parm("fixoverlap").set(0)
    clean_orient.parm("fusepts").set(0)

    calc = sopify(subnet, clean_orient, _check_majority_insides)

    clean_reverse = subnet.createNode("clean", "reverse_winding")
    clean_reverse.setInput(0, calc)
    clean_reverse.parm("orientpoly").set(0)
    expression = "1 - detail(0, \"tmp_reverse_winding\", 0)" if reverse else "detail(0, \"tmp_reverse_winding\", 0)"
    clean_reverse.parm("reversewinding").setExpression(expression)
    clean_reverse.parm("deldegengeo").set(0)
    clean_reverse.parm("delunusedpts").set(0)
    clean_reverse.parm("removeunusedgrp").set(0)
    clean_reverse.parm("deleteoverlap").set(0)
    clean_reverse.parm("delnans").set(0)
    clean_reverse.parm("delete_small").set(0)
    clean_reverse.parm("fixoverlap").set(0)
    clean_reverse.parm("fusepts").set(0)

    cleanup = sopify(subnet, clean_reverse, _cleanup_recalculate_outside)
    add_output(subnet, "OUT", cleanup)
    subnet.layoutChildren()
    return subnet


def _check_majority_insides(node: hou.SopNode) -> None:
    geo = node.geometry()
    prims = geo.prims()
    is_closed = _is_closed_manifold(geo)
    is_inside = False
    if is_closed:
        vol = sum(p.intrinsicValue("measuredvolume") for p in prims)
        is_inside = vol < 0
        node.addMessage(f"Geometry is closed (watertight manifold, measured volume: {vol:.2f}).")
    else:
        node.addMessage("Geometry is open (boundary or non-manifold edges detected).")
    add_attr(geo, hou.attribType.Global, "tmp_reverse_winding", 0)
    geo.setGlobalAttribValue("tmp_reverse_winding", 1 if is_inside else 0)


def _is_closed_manifold(geo: hou.Geometry) -> bool:
    prims = geo.prims()
    if not prims:
        return False
    edge_counts: dict[tuple[int, int], int] = {}
    for prim in prims:
        if not isinstance(prim, hou.Face):
            return False
        verts = prim.vertices()
        n = len(verts)
        if n < 3:
            return False
        for i in range(n):
            p1 = verts[i].point().number()
            p2 = verts[(i + 1) % n].point().number()
            edge = (p1, p2) if p1 < p2 else (p2, p1)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    return all(count == 2 for count in edge_counts.values())


def _cleanup_recalculate_outside(node: hou.SopNode) -> None:
    geo = node.geometry()
    remove_attrs(geo, global_attribs="tmp_reverse_winding")
