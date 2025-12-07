from __future__ import annotations

from datetime import datetime

from lancedb.pydantic import LanceModel, Vector


class CodeChunk(LanceModel):
    vector: Vector(768)
    doc_id: str
    repo_id: str
    file_path: str
    chunk_index: int
    start_line: int
    end_line: int
    content: str
    last_updated: datetime
