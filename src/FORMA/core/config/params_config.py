import os

from typing import Optional, Dict, Any, List
from pydantic import BaseModel

from FORMA.core.config._utils import getenv_int, getenv_int_list, getenv_float, getenv_optional_float


# ------------------------
# 分析参数配置
# ------------------------

class ParamsConfig(BaseModel):
    arm_name: Optional[List[str]]
    arm_wavelength_range: Optional[List[List[float]]]
    ocr: str
    # ═══════════════════════════════════════════════════════════════════════
    # tol_wavelength — 已注释（2026-07-06）
    #   该参数仅用于 brute_force_line_matching（REDROCK=false 路径）。
    #   默认 REDROCK=true，不再需要此参数。
    # ═══════════════════════════════════════════════════════════════════════
    # tol_wavelength: int
    harness_concurrency: int
    self_evolve: bool
    redrock: bool
    failure_batch_size: int
    z_tolerance: float

    # ── Redrock 配置 ──
    rr_template_dir: Optional[str]
    archetype_dir: Optional[str]
    use_archetypes: bool
    rr_nminima: int
    rr_nnearest: int
    rr_omp_num_threads: int

    # ── CWT 特征检测参数 ──
    cwt_snr_thresh: float
    cwt_min_ridge_length: int
    cwt_n_scales: int
    cwt_min_width: float
    cwt_max_width: float

    @classmethod
    def from_env(cls) -> "ParamsConfig":
        arm_name = cls._parse_arm_name()
        arm_wavelength_range = cls._parse_arm_wavelength_range()

        # -------- 核心一致性校验 --------
        if arm_name is None and arm_wavelength_range is None:
            pass  # ✅ 允许
        elif arm_name is None or arm_wavelength_range is None:
            raise ValueError(
                "ARM_NAME and ARM_WAVELENGTH_RANGE must both be set or both be None."
            )
        elif len(arm_name) != len(arm_wavelength_range):
            raise ValueError(
                "ARM_NAME and ARM_WAVELENGTH_RANGE must have the same length."
            )

        return cls(
            arm_name=arm_name,
            arm_wavelength_range=arm_wavelength_range,
            ocr=os.getenv("OCR") or "paddle",
            # tol_wavelength=getenv_int("TOL_WAVELENGTH", 80),  # 已注释（2026-07-06）
            harness_concurrency=getenv_int("HARNESS_CONCURRENCY", 3),
            self_evolve=os.getenv("SELF_EVOLVE", "false").lower() in ("true", "1", "yes"),
            redrock=os.getenv("REDROCK", "true").lower() in ("true", "1", "yes"),
            failure_batch_size=getenv_int("FAILURE_BATCH_SIZE", 5),
            z_tolerance=getenv_float("Z_TOLERANCE", 0.005),

            # ── Redrock 配置 ──
            rr_template_dir=os.getenv("RR_TEMPLATE_DIR") or None,
            archetype_dir=os.getenv("ARCHETYPE_DIR") or None,
            use_archetypes=os.getenv("USE_ARCHETYPES", "true").lower() in ("true", "1", "yes"),
            rr_nminima=getenv_int("NMINIMA", 9),
            rr_nnearest=getenv_int("NNEAREST", 2),
            rr_omp_num_threads=getenv_int("OMP_NUM_THREADS", 1),

            # ── CWT 特征检测参数 ──
            cwt_snr_thresh=getenv_float("CWT_SNR_THRESH", 5.0),
            cwt_min_ridge_length=getenv_int("CWT_MIN_RIDGE_LENGTH", 4),
            cwt_n_scales=getenv_int("CWT_N_SCALES", 24),
            cwt_min_width=getenv_float("CWT_MIN_WIDTH", 1.0),
            cwt_max_width=getenv_float("CWT_MAX_WIDTH", 80.0),
        )

    # ------------------------
    # 解析 ARM_NAME
    # ------------------------
    @staticmethod
    def _parse_arm_name():
        raw = os.getenv("ARM_NAME")
        if raw is None or raw.strip() == "":
            return None
        return [name.strip() for name in raw.split(",")]

    # ------------------------
    # 解析 ARM_WAVELENGTH_RANGE
    # ------------------------
    @staticmethod
    def _parse_arm_wavelength_range():
        raw = os.getenv("ARM_WAVELENGTH_RANGE")
        if raw is None or raw.strip() == "":
            return None

        result = []
        for rng in raw.split(","):
            parts = rng.strip().split("-")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid wavelength range format: {rng}"
                )
            try:
                result.append([float(parts[0]), float(parts[1])])
            except ValueError:
                raise ValueError(
                    f"Invalid numeric wavelength range: {rng}"
                )

        return result