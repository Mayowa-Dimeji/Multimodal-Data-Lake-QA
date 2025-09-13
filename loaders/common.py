from dataclasses import dataclass
from typing import Any, Dict, List, Literal

Origin = Literal["DB", "CSV", "DOC"]

@dataclass
class Evidence:
    origin: Origin
    source_id: str
    score: float
    payload: Dict[str, Any]  # row for structured, chunk for docs
