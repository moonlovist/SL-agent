import os
from typing import Optional
from pydantic import BaseModel, model_validator


# ------------------------
# IO 配置（不含 batch）
# ------------------------

class IOConfig(BaseModel):
    input_dir: str
    input_format: str
    output_dir: str
    run_mode: str
    file_name: Optional[str] = ""

    @model_validator(mode='after')
    def validate_by_mode(self):
        # 基础校验：input_dir, output_dir, run_mode 必须存在
        if not self.input_dir or not self.output_dir or not self.run_mode:
            raise ValueError("IO 配置 (input_dir/output_dir/run_mode) 不完整")

        # run_mode 必须是 's' 或 'b'
        if self.run_mode not in ('s', 'b'):
            raise ValueError(f"RUN_MODE 必须是 's' 或 'b'，当前值为 '{self.run_mode}'")

        # 单任务模式：file_name 必须非空
        if self.run_mode == 's' and not self.file_name:
            raise ValueError("单任务模式 (RUN_MODE=s) 下 FILE_NAME 不能为空")

        # 批量模式：file_name 可选，不做额外校验
        return self

    @classmethod
    def from_env(cls) -> "IOConfig":
        input_dir = os.getenv("INPUT_DIR") or ""
        input_format = os.getenv("INPUT_FORMAT") or ""
        output_dir = os.getenv("OUTPUT_DIR") or ""
        run_mode = (os.getenv("RUN_MODE") or "s").lower()
        file_name = os.getenv("FILE_NAME") or ""

        return cls(
            input_dir=input_dir,
            input_format=input_format,
            output_dir=output_dir,
            run_mode=run_mode,
            file_name=file_name,
        )
    
