import os
from typing import Optional, Dict, Any, List


def safe_to_bool(value: Optional[str]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "t", "yes", "y"}
    return False


def getenv_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val and val.strip():
        try:
            return int(val.strip())
        except ValueError:
            print(f"⚠️ {name} 格式错误: {val}，使用默认值 {default}")
    return default


def getenv_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val and val.strip():
        try:
            return float(val.strip())
        except ValueError:
            print(f"⚠️ {name} 格式错误: {val}，使用默认值 {default}")
    return default


def getenv_optional_float(name: str) -> Optional[float]:
    val = os.getenv(name)
    if val and val.strip():
        try:
            return float(val.strip())
        except ValueError:
            print(f"⚠️ {name} 格式错误: {val}，使用 None")
    return None


def getenv_int_list(name: str, default: List[int]) -> List[int]:
    val = os.getenv(name)
    if val and val.strip():
        try:
            return [int(x.strip()) for x in val.split(",") if x.strip()]
        except ValueError:
            print(f"⚠️ {name} 格式错误: {val}，使用默认值 {default}")
    return default