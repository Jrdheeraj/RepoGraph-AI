from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

try:
    from parser.repo_scanner import scan_repository
    from parser.tree_sitter_parser import ClassInfo, FileMetadata, FunctionInfo, ImportInfo, parse_files
except ImportError:
    from repo_scanner import scan_repository
    from tree_sitter_parser import ClassInfo, FileMetadata, FunctionInfo, ImportInfo, parse_files

logger = logging.getLogger(__name__)

GraphJson = dict[str, list[dict[str, Any]]]


def _node_id(kind: str, identifier: str) -> str:
    return f"{kind}:{identifier}"


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def _add_file_node(graph: nx.DiGraph, metadata: FileMetadata) -> str:
    file_id = _node_id("file", str(metadata.path))
    graph.add_node(
        file_id,
        id=file_id,
        type="file",
        label=metadata.path.name,
        path=str(metadata.path),
        module_name=metadata.module_name,
        docstring=metadata.docstring,
        has_syntax_error=metadata.has_syntax_error,
    )
    return file_id


def _add_class_node(graph: nx.DiGraph, class_info: ClassInfo, file_id: str) -> str:
    class_id = _node_id("class", class_info.qualified_name)
    graph.add_node(
        class_id,
        id=class_id,
        type="class",
        label=class_info.name,
        qualified_name=class_info.qualified_name,
        path=str(class_info.file_path),
        line_start=class_info.line_start,
        line_end=class_info.line_end,
        docstring=class_info.docstring,
        bases=class_info.bases,
        decorators=class_info.decorators,
    )
    graph.add_edge(file_id, class_id, type="contains")
    return class_id


def _add_function_node(graph: nx.DiGraph, function_info: FunctionInfo, owner_id: str) -> str:
    function_id = _node_id("function", function_info.qualified_name)
    graph.add_node(
        function_id,
        id=function_id,
        type="function",
        label=function_info.name,
        qualified_name=function_info.qualified_name,
        path=str(function_info.file_path),
        line_start=function_info.line_start,
        line_end=function_info.line_end,
        docstring=function_info.docstring,
        parameters=function_info.parameters,
        decorators=function_info.decorators,
        parent_class=function_info.parent_class,
        is_async=function_info.is_async,
    )
    graph.add_edge(owner_id, function_id, type="contains")
    return function_id


def _import_target(import_info: ImportInfo) -> str:
    if import_info.statement_type == "import":
        return import_info.module or import_info.name or ""

    prefix = "." * import_info.level
    if import_info.module and import_info.name:
        return f"{prefix}{import_info.module}.{import_info.name}"
    if import_info.module:
        return f"{prefix}{import_info.module}"
    return f"{prefix}{import_info.name or '*'}"


def _dependency_candidates(current_module: str, import_info: ImportInfo) -> list[str]:
    candidates: list[str] = []

    if import_info.level > 0:
        package_parts = current_module.split(".")[:-1]
        keep_count = max(len(package_parts) - import_info.level + 1, 0)
        base = ".".join(package_parts[:keep_count])
        if import_info.module:
            candidates.append(".".join(part for part in (base, import_info.module) if part))
        if import_info.name:
            candidates.append(".".join(part for part in (base, import_info.module, import_info.name) if part))
        return candidates

    if import_info.module:
        candidates.append(import_info.module)
    if import_info.module and import_info.name:
        candidates.append(f"{import_info.module}.{import_info.name}")
    if import_info.statement_type == "import" and import_info.module:
        parts = import_info.module.split(".")
        candidates.extend(".".join(parts[:index]) for index in range(len(parts), 0, -1))

    return candidates


def _add_import_edges(graph: nx.DiGraph, file_id: str, imports: Iterable[ImportInfo]) -> None:
    for import_info in imports:
        target = _import_target(import_info)
        if not target:
            continue

        import_id = _node_id("import", target)
        graph.add_node(import_id, id=import_id, type="external_dependency", label=target, module=target)
        graph.add_edge(
            file_id,
            import_id,
            type="imports",
            module=import_info.module,
            name=import_info.name,
            alias=import_info.alias,
            level=import_info.level,
            line_start=import_info.line_start,
            line_end=import_info.line_end,
        )


def _add_internal_dependency_edges(
    graph: nx.DiGraph,
    module_to_file: dict[str, str],
    collected_metadata: list[tuple[str, str, list[ImportInfo]]],
) -> None:
    """Resolve internal depends_on edges using a pre-built module→file_id map.

    Args:
        graph: The graph being built.
        module_to_file: Mapping of module_name → file node ID.
        collected_metadata: List of (module_name, file_id, imports) tuples.
    """
    for module_name, source_id, imports in collected_metadata:
        for import_info in imports:
            for candidate in _dependency_candidates(module_name, import_info):
                target_id = module_to_file.get(candidate)
                if target_id and target_id != source_id:
                    graph.add_edge(source_id, target_id, type="depends_on", module=candidate)
                    break


# Maximum file size to parse (1 MB)
_MAX_FILE_BYTES: int = 1 * 1024 * 1024
_PROGRESS_INTERVAL: int = 100


def build_graph(repository_or_metadata: str | Path | list[FileMetadata]) -> nx.DiGraph:
    """Build a directed dependency graph for repository metadata or a repository path."""
    import time

    graph = nx.DiGraph()
    # module_name -> file node id (built incrementally)
    module_to_file: dict[str, str] = {}
    # (module_name, file_id, imports) collected for the second pass
    dep_records: list[tuple[str, str, list[ImportInfo]]] = []

    # ------------------------------------------------------------------ #
    # Branch A: caller already supplied parsed FileMetadata objects        #
    # ------------------------------------------------------------------ #
    if isinstance(repository_or_metadata, list):
        logger.info("Starting dependency graph build...", extra={"source": "metadata_list"})
        t0 = time.monotonic()

        for idx, metadata in enumerate(repository_or_metadata, start=1):
            try:
                if metadata.has_syntax_error:
                    logger.debug("Skipping file with syntax error", extra={"path": str(metadata.path)})
                    continue

                file_id = _add_file_node(graph, metadata)
                _add_import_edges(graph, file_id, metadata.imports)

                for class_info in metadata.classes:
                    class_id = _add_class_node(graph, class_info, file_id)
                    for method in class_info.methods:
                        _add_function_node(graph, method, class_id)

                for function_info in metadata.functions:
                    _add_function_node(graph, function_info, file_id)

                module_to_file[metadata.module_name] = file_id
                dep_records.append((metadata.module_name, file_id, list(metadata.imports)))

                if idx % _PROGRESS_INTERVAL == 0:
                    logger.info("Processed %d files...", idx)
            except Exception:
                logger.warning("Skipping file due to error", extra={"path": str(metadata.path)}, exc_info=True)
                continue

        _add_internal_dependency_edges(graph, module_to_file, dep_records)
        elapsed = time.monotonic() - t0
        logger.info(
            "Completed dependency graph build.",
            extra={"node_count": graph.number_of_nodes(), "edge_count": graph.number_of_edges(), "elapsed_s": round(elapsed, 2)},
        )
        return graph

    # ------------------------------------------------------------------ #
    # Branch B: caller supplied a repository path — parse incrementally   #
    # ------------------------------------------------------------------ #
    repository_root = Path(repository_or_metadata).resolve()
    python_files = scan_repository(repository_root)

    logger.info(
        "Starting dependency graph build...",
        extra={"repository_path": str(repository_root), "file_count": len(python_files)},
    )
    t0 = time.monotonic()

    try:
        from parser.tree_sitter_parser import parse_file
    except ImportError:
        from tree_sitter_parser import parse_file  # type: ignore[no-redef]

    processed = 0
    skipped = 0

    for idx, file_path in enumerate(python_files, start=1):
        try:
            # Skip files that are too large — parsing them can block forever
            try:
                if file_path.stat().st_size > _MAX_FILE_BYTES:
                    logger.debug("Skipping large file", extra={"file_path": str(file_path)})
                    skipped += 1
                    continue
            except OSError:
                skipped += 1
                continue

            # Parse the single file (each call is already exception-safe)
            metadata = parse_file(file_path, repository_root=repository_root)

            if metadata.has_syntax_error:
                logger.debug("Skipping file with syntax error", extra={"file_path": str(file_path)})
                skipped += 1
                continue

            file_id = _add_file_node(graph, metadata)
            _add_import_edges(graph, file_id, metadata.imports)

            for class_info in metadata.classes:
                class_id = _add_class_node(graph, class_info, file_id)
                for method in class_info.methods:
                    _add_function_node(graph, method, class_id)

            for function_info in metadata.functions:
                _add_function_node(graph, function_info, file_id)

            module_to_file[metadata.module_name] = file_id
            dep_records.append((metadata.module_name, file_id, list(metadata.imports)))
            processed += 1

            if idx % _PROGRESS_INTERVAL == 0:
                logger.info("Processed %d files...", idx)

        except Exception:
            logger.warning("Skipping file due to unexpected error", extra={"file_path": str(file_path)}, exc_info=True)
            skipped += 1
            continue

    _add_internal_dependency_edges(graph, module_to_file, dep_records)

    elapsed = time.monotonic() - t0
    logger.info(
        "Completed dependency graph build.",
        extra={
            "repository_path": str(repository_root),
            "processed": processed,
            "skipped": skipped,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "elapsed_s": round(elapsed, 2),
        },
    )
    return graph


def export_graph_json(graph: nx.DiGraph) -> GraphJson:
    nodes = [
        {key: _serialize_value(value) for key, value in data.items()}
        for _, data in graph.nodes(data=True)
    ]
    edges = [
        {
            "source": source,
            "target": target,
            **{key: _serialize_value(value) for key, value in data.items()},
        }
        for source, target, data in graph.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


__all__ = ["GraphJson", "build_graph", "export_graph_json"]
