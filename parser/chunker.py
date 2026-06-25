from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

try:
    from parser.tree_sitter_parser import ClassInfo, FileMetadata, FunctionInfo
except ImportError:
    from tree_sitter_parser import ClassInfo, FileMetadata, FunctionInfo

logger = logging.getLogger(__name__)

ChunkType = Literal["function", "class", "module"]
DEFAULT_MAX_MODULE_CHUNK_LINES = 200


@dataclass(frozen=True)
class ChunkMetadata:
    file_path: Path
    chunk_type: ChunkType
    start_line: int
    end_line: int
    symbol_name: str | None
    language: str


@dataclass(frozen=True)
class CodeChunk:
    chunk_id: str
    content: str
    metadata: ChunkMetadata


def _read_source_lines(file_path: Path) -> list[str]:
    try:
        return file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        logger.warning("Unable to read source file for chunking", extra={"file_path": str(file_path), "error": str(exc)})
        return []


def _bounded_line_range(start_line: int, end_line: int, total_lines: int) -> tuple[int, int]:
    start = max(start_line, 1)
    end = min(end_line, total_lines)
    if end < start:
        return start, start - 1
    return start, end


def _expand_start_for_decorators(lines: list[str], start_line: int, has_decorators: bool) -> int:
    if not has_decorators:
        return start_line

    current = start_line
    while current > 1:
        previous = lines[current - 2].strip()
        if not previous:
            break
        current -= 1
        if previous.startswith("@"):
            break
    return current


def _slice_content(lines: list[str], start_line: int, end_line: int) -> str:
    if not lines:
        return ""

    start, end = _bounded_line_range(start_line, end_line, len(lines))
    if end < start:
        return ""
    return "\n".join(lines[start - 1 : end])


def _make_chunk(
    *,
    lines: list[str],
    file_path: Path,
    chunk_type: ChunkType,
    start_line: int,
    end_line: int,
    symbol_name: str | None,
    has_decorators: bool = False,
) -> CodeChunk | None:
    expanded_start = (
        _expand_start_for_decorators(lines, start_line, has_decorators)
        if chunk_type in {"function", "class"}
        else start_line
    )
    content = _slice_content(lines, expanded_start, end_line)
    if not content.strip():
        logger.debug(
            "Ignoring empty code chunk",
            extra={
                "file_path": str(file_path),
                "chunk_type": chunk_type,
                "symbol_name": symbol_name,
                "start_line": expanded_start,
                "end_line": end_line,
            },
        )
        return None

    metadata = ChunkMetadata(
        file_path=file_path,
        chunk_type=chunk_type,
        start_line=expanded_start,
        end_line=min(end_line, len(lines)),
        symbol_name=symbol_name,
        language="python",
    )
    return CodeChunk(chunk_id=str(uuid4()), content=content, metadata=metadata)


def _function_chunks(lines: list[str], file_metadata: FileMetadata) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    functions: list[FunctionInfo] = list(file_metadata.functions)
    for class_info in file_metadata.classes:
        functions.extend(class_info.methods)

    for function_info in functions:
        chunk = _make_chunk(
            lines=lines,
            file_path=file_metadata.path,
            chunk_type="function",
            start_line=function_info.line_start,
            end_line=function_info.line_end,
            symbol_name=function_info.qualified_name,
            has_decorators=bool(function_info.decorators),
        )
        if chunk is not None:
            chunks.append(chunk)

    return chunks


def _class_chunks(lines: list[str], file_metadata: FileMetadata) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for class_info in file_metadata.classes:
        chunk = _make_chunk(
            lines=lines,
            file_path=file_metadata.path,
            chunk_type="class",
            start_line=class_info.line_start,
            end_line=class_info.line_end,
            symbol_name=class_info.qualified_name,
            has_decorators=bool(class_info.decorators),
        )
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def _module_chunk(lines: list[str], file_metadata: FileMetadata, max_module_chunk_lines: int) -> CodeChunk | None:
    if len(lines) > max_module_chunk_lines:
        return None

    return _make_chunk(
        lines=lines,
        file_path=file_metadata.path,
        chunk_type="module",
        start_line=1,
        end_line=len(lines),
        symbol_name=file_metadata.module_name,
    )


def create_chunks(
    file_metadata: FileMetadata,
    *,
    max_module_chunk_lines: int = DEFAULT_MAX_MODULE_CHUNK_LINES,
) -> list[CodeChunk]:
    """Create semantic code chunks from parsed Python file metadata."""
    source_path = Path(file_metadata.path)
    lines = _read_source_lines(source_path)
    if not lines:
        logger.info("No source lines available for chunking", extra={"file_path": str(source_path)})
        return []

    logger.debug(
        "Creating semantic chunks",
        extra={
            "file_path": str(source_path),
            "module_name": file_metadata.module_name,
            "line_count": len(lines),
            "class_count": len(file_metadata.classes),
            "function_count": len(file_metadata.functions),
        },
    )

    chunks: list[CodeChunk] = []
    chunks.extend(_function_chunks(lines, file_metadata))
    chunks.extend(_class_chunks(lines, file_metadata))

    module_chunk = _module_chunk(lines, file_metadata, max_module_chunk_lines)
    if module_chunk is not None:
        chunks.append(module_chunk)

    logger.info(
        "Created semantic chunks",
        extra={
            "file_path": str(source_path),
            "chunk_count": len(chunks),
            "max_module_chunk_lines": max_module_chunk_lines,
        },
    )
    return chunks


__all__ = [
    "ChunkMetadata",
    "ChunkType",
    "CodeChunk",
    "DEFAULT_MAX_MODULE_CHUNK_LINES",
    "create_chunks",
]
