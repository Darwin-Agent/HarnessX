"""Lab custom processor helpers (API-local)."""

from .direct_targets import (
    build_custom_dimension,
    ensure_import_path,
    import_processor_from_content,
    import_processor_from_path,
    list_custom_processors,
    make_file_target,
    managed_processors_dir,
    parse_file_target,
    remove_custom_processor,
    scan_path_for_processors,
    scan_text_for_processors,
    test_processor_from_content,
    test_processor_from_path,
)

__all__ = [
    "build_custom_dimension",
    "ensure_import_path",
    "import_processor_from_content",
    "import_processor_from_path",
    "list_custom_processors",
    "make_file_target",
    "managed_processors_dir",
    "parse_file_target",
    "remove_custom_processor",
    "scan_path_for_processors",
    "scan_text_for_processors",
    "test_processor_from_content",
    "test_processor_from_path",
]
