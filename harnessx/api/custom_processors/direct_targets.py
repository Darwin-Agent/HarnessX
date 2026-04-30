# Copyright 2026 Darwin-Agent
# SPDX-License-Identifier: MIT
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import re
import shutil
import sys
import tempfile
import types
import uuid
from pathlib import Path
from typing import Any

from harnessx.core.processor import MultiHookProcessor
from harnessx.home import agent_home

_BASE_DIR = agent_home() / "lab" / "custom_processors"
_IMPORTED_DIR = _BASE_DIR / "imported"
_LEGACY_PY_ROOT = _BASE_DIR / "py"
_LEGACY_PACKAGE = "hx_custom_processors"
_LEGACY_PACKAGE_ROOT = _LEGACY_PY_ROOT / _LEGACY_PACKAGE

_MAX_FILES = 200
_MAX_FILE_BYTES = 1024 * 1024


def _sanitize_slug(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (raw or "").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "custom"


def _ensure_dirs() -> None:
    for d in (_BASE_DIR, _IMPORTED_DIR):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Some runtimes (e.g. read-only sandboxes) cannot write AGENT_HOME.
            # Callers that require write access will fail later with a clear error.
            pass


def ensure_import_path() -> None:
    """Compatibility no-op.

    Kept for backward compatibility with older code paths that expected this
    helper when custom processors were installed into a managed package.
    """
    _ensure_dirs()
    # Legacy compat for historical managed package targets:
    #   hx_custom_processors.<slug>.processor.<ClassName>
    p = str(_LEGACY_PY_ROOT.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

    pkg_root = str(_LEGACY_PACKAGE_ROOT.resolve())
    pkg = sys.modules.get(_LEGACY_PACKAGE)
    if isinstance(pkg, types.ModuleType) and hasattr(pkg, "__path__"):
        pkg_paths = list(getattr(pkg, "__path__", []))
        if pkg_root not in pkg_paths:
            pkg.__path__.append(pkg_root)  # type: ignore[attr-defined]
    importlib.invalidate_caches()


def managed_processors_dir() -> Path:
    """Return AGENT_HOME managed custom-processor root."""
    _ensure_dirs()
    return _BASE_DIR


def make_file_target(file_path: str | Path, class_name: str) -> str:
    """Build canonical file-based _target_ string.

    Format: ``file://<absolute_path>::<ClassName>``.
    """
    p = Path(file_path).expanduser().resolve()
    return f"file://{p}::{class_name}"


def parse_file_target(target: str) -> tuple[Path, str]:
    """Parse ``file://...::ClassName`` target into ``(path, class_name)``."""
    if not isinstance(target, str) or not target.startswith("file://"):
        raise ValueError("target is not a file:// target")
    spec = target[len("file://") :]
    path_part, sep, class_name = spec.rpartition("::")
    if not sep or not path_part.strip() or not class_name.strip():
        raise ValueError("invalid file target; expected 'file:///abs/path.py::ClassName'")
    path = Path(path_part).expanduser().resolve()
    return path, class_name.strip()


def _doc_first_line(node: ast.ClassDef) -> str:
    doc = ast.get_docstring(node) or ""
    return doc.strip().splitlines()[0].strip() if doc.strip() else ""


def _collect_multihook_aliases(tree: ast.Module) -> set[str]:
    aliases = {"MultiHookProcessor"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.endswith("core.processor") or mod == "harnessx.core.processor":
                for name in node.names:
                    if name.name == "MultiHookProcessor":
                        aliases.add(name.asname or name.name)
    return aliases


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return ""


def _scan_python_text(text: str, file_path: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text, filename=file_path)
    except SyntaxError:
        return []

    aliases = _collect_multihook_aliases(tree)
    out: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = {_base_name(b) for b in node.bases}
        if not bases.intersection(aliases):
            continue
        out.append(
            {
                "class_name": node.name,
                "label": node.name,
                "file_path": file_path,
                "doc": _doc_first_line(node),
            }
        )
    return out


def _iter_python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix == ".py" else []

    files: list[Path] = []
    for p in path.rglob("*.py"):
        if any(part in {"__pycache__", ".git", ".venv", "node_modules"} for part in p.parts):
            continue
        files.append(p)
        if len(files) >= _MAX_FILES:
            break
    return sorted(files)


def scan_path_for_processors(path: str) -> list[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    candidates: list[dict[str, Any]] = []
    for f in _iter_python_files(p):
        try:
            if f.stat().st_size > _MAX_FILE_BYTES:
                continue
            text = f.read_text(encoding="utf-8")
        except Exception:
            continue
        candidates.extend(_scan_python_text(text, str(f)))
    return candidates


def scan_text_for_processors(filename: str, content: str) -> list[dict[str, Any]]:
    name = filename or "uploaded_processor.py"
    return _scan_python_text(content, name)


def _load_class_from_file(file_path: Path, class_name: str) -> type:
    module_name = f"_hx_custom_scan_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot build import spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Class '{class_name}' not found in {file_path}")
    if not isinstance(cls, type):
        raise TypeError(f"'{class_name}' is not a class in {file_path}")
    if not issubclass(cls, MultiHookProcessor):
        raise TypeError(f"'{class_name}' does not subclass MultiHookProcessor")
    return cls


def _ctor_required_params(cls: type) -> list[str]:
    sig = inspect.signature(cls.__init__)
    required: list[str] = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            required.append(p.name)
    return required


def test_processor_from_path(path: str, class_name: str) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    cls = _load_class_from_file(p, class_name)
    required = _ctor_required_params(cls)
    if required:
        return {
            "ok": True,
            "instantiable": False,
            "required_args": required,
            "message": f"Import ok; constructor requires args: {required}",
        }
    try:
        cls()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": True,
            "instantiable": False,
            "required_args": [],
            "message": f"Import ok; constructor check failed: {type(exc).__name__}: {exc}",
        }
    return {
        "ok": True,
        "instantiable": True,
        "required_args": [],
        "message": "Import and ctor check passed",
    }


def test_processor_from_content(filename: str, content: str, class_name: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hx_custom_proc_") as td:
        f = Path(td) / (filename or "uploaded_processor.py")
        f.write_text(content, encoding="utf-8")
        return test_processor_from_path(str(f), class_name)


def _entry_for(
    path: Path,
    class_name: str,
    label: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    target = make_file_target(path, class_name)
    return {
        "id": hashlib.sha1(target.encode("utf-8")).hexdigest()[:12],
        "label": (label or class_name).strip() or class_name,
        "class_name": class_name,
        "target": target,
        "source_path": source_path or str(path),
        "installed_path": str(path),
    }


def _content_id(content: bytes) -> str:
    """Return a 10-char content-based ID from the SHA-1 of *content*.

    Using file content instead of a random UUID makes imports idempotent:
    the same source file always produces the same managed filename, so
    existing ``file://`` targets remain valid after agent_home is rebuilt.
    """
    return hashlib.sha1(content).hexdigest()[:10]


def _copy_into_managed(file_path: Path, class_name: str) -> Path:
    _ensure_dirs()
    content = file_path.read_bytes()
    file_stem = _sanitize_slug(file_path.stem)
    class_stem = _sanitize_slug(class_name)
    content_id = _content_id(content)
    dst = _IMPORTED_DIR / f"{file_stem}_{class_stem}_{content_id}.py"
    if not dst.exists():
        shutil.copy2(file_path, dst)
    return dst


def import_processor_from_path(path: str, class_name: str, label: str | None = None) -> dict[str, Any]:
    src = Path(path).expanduser().resolve()
    _load_class_from_file(src, class_name)

    _ensure_dirs()
    base = _BASE_DIR.resolve()
    try:
        src.relative_to(base)
        dst = src
    except Exception:
        dst = _copy_into_managed(src, class_name)

    return _entry_for(dst, class_name, label=label, source_path=str(src))


def import_processor_from_content(
    filename: str,
    content: str,
    class_name: str,
    label: str | None = None,
) -> dict[str, Any]:
    _ensure_dirs()

    content_bytes = content.encode("utf-8")
    stem = _sanitize_slug(Path(filename or "uploaded_processor.py").stem)
    cls = _sanitize_slug(class_name)
    content_id = _content_id(content_bytes)
    dst = _IMPORTED_DIR / f"{stem}_{cls}_{content_id}.py"
    if not dst.exists():
        dst.write_bytes(content_bytes)

    # Validate the saved file contains the requested class.
    _load_class_from_file(dst, class_name)
    return _entry_for(dst, class_name, label=label, source_path=str(dst))


def list_custom_processors() -> list[dict[str, Any]]:
    _ensure_dirs()
    if not _BASE_DIR.exists():
        return []
    rows: list[dict[str, Any]] = []
    for c in scan_path_for_processors(str(_BASE_DIR)):
        p = Path(str(c.get("file_path", ""))).expanduser().resolve()
        class_name = str(c.get("class_name", "")).strip()
        if not class_name:
            continue
        rows.append(
            _entry_for(
                p,
                class_name,
                label=str(c.get("label", class_name)),
                source_path=str(p),
            )
        )
    return sorted(
        rows,
        key=lambda x: (str(x.get("label", "")).lower(), str(x.get("class_name", ""))),
    )


def remove_custom_processor(processor_id: str) -> bool:
    rows = list_custom_processors()
    row = next(
        (r for r in rows if str(r.get("id")) == str(processor_id) or str(r.get("target")) == str(processor_id)),
        None,
    )
    if row is None:
        return False

    p = Path(str(row.get("installed_path", ""))).expanduser().resolve()
    base = _BASE_DIR.resolve()
    try:
        p.relative_to(base)
    except Exception:
        return False

    if p.exists() and p.is_file():
        p.unlink(missing_ok=True)

    # Best-effort cache cleanup for same stem in __pycache__.
    pyc_dir = p.parent / "__pycache__"
    if pyc_dir.exists() and pyc_dir.is_dir():
        stem = p.stem
        for f in pyc_dir.glob(f"{stem}*.pyc"):
            f.unlink(missing_ok=True)
    return True


def build_custom_dimension() -> dict[str, Any] | None:
    entries = list_custom_processors()
    if not entries:
        return None

    options: list[dict[str, Any]] = []
    for e in entries:
        target = str(e.get("target", "")).strip()
        if not target:
            continue
        options.append(
            {
                "key": str(e.get("id", target)),
                "label": str(e.get("label", e.get("class_name", "Custom Processor"))),
                "description": f"{e.get('class_name', 'Processor')} · {e.get('installed_path', '')}",
                "processors": [{"_target_": target}],
            }
        )
    if not options:
        return None

    return {
        "key": "custom_processors",
        "label": "Custom Processors",
        "description": "Processors imported from AGENT_HOME/lab/custom_processors.",
        "icon": "flask",
        "multi_select": True,
        "options": options,
    }
