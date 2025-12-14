# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

from __future__ import annotations

from datetime import datetime

from functools import lru_cache
from lancedb.pydantic import LanceModel, Vector


@lru_cache(maxsize=None)
def get_code_chunk_model(dimension: int) -> type[LanceModel]:
    """Return a LanceModel configured for the given vector dimension."""

    class CodeChunk(LanceModel):
        vector: Vector(dimension)
        doc_id: str
        repo_id: str
        file_path: str
        chunk_index: int
        start_line: int
        end_line: int
        content: str
        is_code: bool = True
        is_doc: bool | None = None
        is_config: bool | None = None
        is_data: bool | None = None
        extension: str | None = None
        language: str | None = None
        last_updated: datetime

    CodeChunk.__name__ = "CodeChunk"
    return CodeChunk


# Backwards compatibility export
CodeChunk = get_code_chunk_model(768)

__all__ = ["CodeChunk", "get_code_chunk_model"]
