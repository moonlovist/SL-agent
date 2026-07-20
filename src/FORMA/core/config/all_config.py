import os
from pydantic import BaseModel

from FORMA.core.config.batch_config import BatchConfig
from FORMA.core.config.io_config import IOConfig
from FORMA.core.config.model_config import ModelConfig
from FORMA.core.config.params_config import ParamsConfig


class AllConfig(BaseModel):
    batch: BatchConfig
    io: IOConfig
    model: ModelConfig
    params: ParamsConfig
    retry_delay: int
    max_tries: int

    @classmethod
    def from_env(cls) -> "AllConfig":
        batch_config = BatchConfig.from_env()
        io_config = IOConfig.from_env()
        model_config = ModelConfig.from_env()
        params_config = ParamsConfig.from_env()
        retry_delay = int(os.getenv("RETRY_DELAY") or 180)
        max_tries = int(os.getenv("MAX_TRIES") or 3)

        return cls(
            batch = batch_config,
            io = io_config,
            model = model_config,
            params = params_config,
            retry_delay = retry_delay,
            max_tries = max_tries
        )
