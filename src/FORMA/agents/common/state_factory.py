import os
import json
from typing import Optional, List

from FORMA.agents.common.state import SpectroState
from FORMA.core.config.all_config import AllConfig
from FORMA.core.config.params_config import ParamsConfig
from FORMA.core.config.io_config import IOConfig
# from ..agents.state import SpectroState
# from ..core.config.config import ParamsConfig, IOConfig

class SpectroStateFactory:
    """
    精简版 SpectroStateFactory
    - 只依赖 ParamsConfig（用于波段信息等分析参数）
    - 在 create() 时传入 file_name, input_dir, output_dir
    """

    def __init__(self, configs: AllConfig):
        self.configs = configs
        self.params_config: ParamsConfig = self.configs.params
        self.io_config: IOConfig = self.configs.io

    def create(
        self,
        file_name: str,
        input_dir: str,
        output_dir: str
    ) -> SpectroState:
        """
        创建单个 SpectroState
        """
        # arm_name, arm_wavelength_range = self._parse_arm_info()
        # print('state_factory: ')
        # print(arm_name, arm_wavelength_range)

        # 构造路径
        input_format = self.io_config.input_format
        file_path = os.path.join(input_dir, f"{file_name}.{input_format}")

        # 将每个任务的所有输出文件放到 output_dir/{file_name}/ 下
        output_dir = os.path.join(output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)

        crop_path = os.path.join(output_dir, "visual_interpreter", f"{file_name}_cropped.png")
        spec_extract_path = os.path.join(output_dir, "visual_interpreter", f"{file_name}_spec_extract.png")
        continuum_path = os.path.join(output_dir, "visual_interpreter", f"{file_name}_continuum.png")

        # prompt_path = self.io_config.prompt

        # # 读取 JSON
        # with open(prompt_path, 'r', encoding='utf-8') as f:
        #     prompt = json.load(f)  # prompt 是 dict


        state = SpectroState(
            file_name=file_name,
            file_path=file_path,
            output_dir=output_dir,
            crop_path=crop_path,
            spec_extract_path=spec_extract_path,
            continuum_path=continuum_path,
            # arm_name=arm_name,
            # arm_wavelength_range=arm_wavelength_range,
            # sigma_lambda=self.params_config.continuum_smoothing,
            # sigma_list=self.params_config.sigma_list,
            # prompt=prompt,
            qualitative_analysis={},
            count=0,
            visual_interpretation=[],
            possible_object=[],
            rule_analysis_QSO={},
            rule_analysis_ELG={},
            rule_analysis_LRG={},
            extract_QSO={},
            extract_ELG={},
            extract_LRG={},
            synthesis_QSO=None,
            rule_analysis_galaxy=[],
            auditing_history_QSO=[],
            refining_history_QSO=[],
            auditing_history_galaxy=[],
            refining_history_galaxy=[],
            in_brief={}
        )

        return state
