from typing import Optional, List
import os
from pydantic import BaseModel, field_validator


def parse_batch_range(start: str, end: str) -> List[str]:
    if not start.isdigit() or not end.isdigit():
        raise ValueError(f"Batch start/end must be digits: {start=}, {end=}")

    start_int = int(start)
    end_int = int(end)

    if start_int > end_int:
        raise ValueError(f"Batch start > end: {start_int} > {end_int}")

    zero_padded = start.startswith("0") and len(start) > len(str(start_int))
    width = len(start) if zero_padded else 0

    result = []
    for i in range(start_int, end_int + 1):
        result.append(str(i).zfill(width) if zero_padded else str(i))
    return result


# ------------------------
# Batch 配置
# ------------------------

class BatchConfig(BaseModel):
    header: str = ""
    start: Optional[str] = None
    end: Optional[str] = None

    @classmethod
    def from_env(cls) -> "BatchConfig":
        return cls(
            header=os.getenv("FILE_BATCH_HEADER", ""),
            start=os.getenv("FILE_BATCH_START") or None,
            end=os.getenv("FILE_BATCH_END") or None,
        )

    @field_validator("start", "end")
    @classmethod
    def validate_digit(cls, v):
        if v is None:
            return v
        if not v.isdigit():
            raise ValueError(f"Batch start/end must be digits, got {v}")
        return v

    def is_batch_mode(self) -> bool:
        return self.start is not None and self.end is not None

    def generate_ids(self) -> List[str]:
        if not self.is_batch_mode():
            return []
        indices = parse_batch_range(self.start, self.end)
        return [f"{self.header}{i}" for i in indices]
