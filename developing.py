import importlib
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Callable

import hou


def save(
        path: Path,
        build: Callable[[], None],
        rebuild: bool = True
) -> None:
    """Execute a build callable and save to a Houdini .hip file, creating a backup in case the build fails."""
    assert path.is_absolute(), "Require absolute path"

    if path.exists() and not rebuild:
        print(f"{path}: File exists. Exiting...")
        return

    backup_path = path.with_suffix(path.suffix + ".bak") if path.exists() else None
    if backup_path is not None:
        shutil.copy2(path, backup_path)

    try:
        path.unlink(missing_ok=True)
        hou.hipFile.save(str(path))  # type: ignore
        hou.setSessionModuleSource(textwrap.dedent("""
            from pathlib import Path
            import sys
            import hou

            hip_dir = str(Path(hou.hipFile.path()).resolve().parent)
            if hip_dir not in sys.path:
                sys.path.insert(0, hip_dir)
        """))

        build()

        hou.hipFile.save()  # type: ignore
        if backup_path is not None and backup_path.exists():
            backup_path.unlink(missing_ok=True)
        print(f"Saved {path}")
    except Exception:
        if backup_path is not None and backup_path.exists():
            shutil.copy2(backup_path, path)
            backup_path.unlink(missing_ok=True)
        raise


def reload_modules() -> None:
    importlib.invalidate_caches()
    hip_dir = Path(hou.hipFile.path()).resolve().parent  # type: ignore

    modules = []
    seen = set()
    for name, module in list(sys.modules.items()):
        if module is None or id(module) in seen:
            continue

        is_hip_module = False
        module_file = getattr(module, "__file__", None)
        if module_file:
            path = Path(module_file)
            # Critical:
            # Embedded/virtual modules such as PySide/Shiboken use relative
            # __file__ values. Never resolve those against our working directory.
            if path.is_absolute():
                try:
                    resolved = path.resolve(strict=True)
                    if path.is_relative_to(hip_dir) or resolved.is_relative_to(hip_dir):
                        is_hip_module = True
                except (OSError, RuntimeError):
                    pass

        if not is_hip_module:
            # Handle packages (such as namespace packages) whose __file__ is None
            module_path = getattr(module, "__path__", None)
            if module_path:
                for p in module_path:
                    path = Path(p)
                    if path.is_absolute():
                        try:
                            resolved = path.resolve(strict=True)
                            if path.is_relative_to(hip_dir) or resolved.is_relative_to(hip_dir):
                                is_hip_module = True
                                break
                        except (OSError, RuntimeError):
                            pass

        if not is_hip_module:
            continue

        spec = getattr(module, "__spec__", None)
        if spec is None:
            continue
        # Only accept the canonical sys.modules entry for this module.
        # importlib.reload() uses __spec__.name.
        if spec.name != name:
            continue
        # reload() requires the parent of a dotted module name to be an actual package with __path__.
        parent_name = name.rpartition(".")[0]
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is None or not hasattr(parent, "__path__"):
                continue

        seen.add(id(module))
        modules.append((name, module))

    # Children first, then their packages.
    modules.sort(
        key=lambda item: item[0].count("."),
        reverse=True,
    )
    for name, module in modules:
        # Make sure nothing changed the module identity/spec while earlier modules were being reloaded.
        if sys.modules.get(name) is not module:
            continue
        spec = getattr(module, "__spec__", None)
        if spec is None or spec.name != name:
            continue
        importlib.reload(module)
