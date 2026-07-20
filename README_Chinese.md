# FORMA

一个由大语言模型（LLM）驱动的多智能体系统，用于对一维天文光谱进行**类人式分析与推断**。

> 📄 相关论文正在准备中。

---

## 项目概述（Overview）

FORMA 利用大语言模型（LLMs）对一维天文光谱进行类似人类天文学家的物理推断，当前支持的核心任务包括：

- **天体类型分类**（galaxy：LRG / ELG，QSO）
- **红移估计**（针对 QSO）

系统设计目标是**模拟人类天文学家的认知流程**，主要包括以下步骤：

1. **光谱视觉分析**  
   通过 CWT 小波变换检测发射峰和吸收谷，拟合连续谱
2. **红移假设生成**  
   调用 Redrock（DESI 官方红移拟合器）生成候选红移
3. **多智能体评估与辩论**  
   每个假设独立分析，再由审计代理进行对抗性审查
4. **综合报告输出**  
   生成最终分析报告，并给出校准置信度

当前流程通过 API 使用以下模型：

- **文本推理模型**：`deepseek-v4-pro`

> 详细的模块文档、架构图和 Pipeline 拓扑见 [`.repo_info/index.html`](.repo_info/index.html)。

---

## 示例数据

[`example/`](example/) 目录包含两组可直接使用的示例：

| 目录 | 说明 |
|------|------|
| [`basic/`](example/basic/) | 两条标准 DESI 光谱（`4.fits`、`116.fits`），用于快速测试管道。 |
| [`counter_fact_examples/`](example/counter_fact_examples/) | 伪造光谱，用于压力测试审计代理。一组删除了 Lyα 发射线，另一组将宽线 QSO 改为窄线。详见其 [README](example/counter_fact_examples/README.md)。 |

---

## Docker 快速开始（推荐）

使用 Docker 是运行 FORMA 最简单的方式。镜像内包含 Python 3.12、全部依赖，以及 Redrock 红移拟合器。

### 前置条件

- [Docker](https://docs.docker.com/get-docker/) 和 Docker Compose v2+

### 1. 配置 `.env` 文件

```bash
cp .env_example .env
# 编辑 .env：设置 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL
# 不要设置 INPUT_DIR / OUTPUT_DIR — Docker 通过卷挂载处理路径
```

### 2. CLI 模式（单文件或批处理）

```bash
# 单文件
INPUT_DIR_HOST=/path/to/your/fits \
OUTPUT_DIR_HOST=/path/to/results \
docker compose run -e FILE_NAME=QSO_116 -e RUN_MODE=s forma-cli

# 批处理（处理输入目录下所有 .fits 文件）
INPUT_DIR_HOST=/path/to/your/fits \
OUTPUT_DIR_HOST=/path/to/results \
docker compose run -e RUN_MODE=b -e FILE_BATCH_HEADER=QSO_ -e FILE_BATCH_START=0 -e FILE_BATCH_END=100 forma-cli
```

### 3. WebUI 模式

```bash
docker compose up forma-web
# 浏览器打开 http://localhost:7860
```

> **注意：** 首次 `docker compose build` 需要 5–10 分钟（Redrock C 扩展编译）。后续构建使用缓存层，速度很快。

---

## 手动安装

如果不使用 Docker，可以手动安装：

### 1. Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 环境变量配置

```bash
cp .env_example .env
```

主要参数速查（完整列表见 `.env_example`）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | *必填* | 文本 LLM 的 API 密钥 |
| `LLM_BASE_URL` | *必填* | LLM API 地址 |
| `LLM_MODEL` | *必填* | 模型名称（如 `deepseek-v4-pro`） |
| `LLM_TEMPERATURE` | `0.1` | LLM 采样温度 |
| `LLM_MAX_TOKENS` | API 默认 | 每次 LLM 响应的最大 token 数 |
| `LLM_THINKING` | `disabled` | 思考模式（`enabled` / `disabled` / `none`） |
| `REDROCK` | `true` | 启用 Redrock 红移拟合 |
| `RR_TEMPLATE_DIR` | — | Redrock 模板路径 |
| `ARCHETYPE_DIR` | — | Archetype 文件路径（可选） |
| `USE_ARCHETYPES` | `true` | 拟合时使用 archetypes |
| `NMINIMA` | `9` | 探索的红移极小值数量 |
| `NNEAREST` | `2` | 最近 archetype 数量 |
| `OMP_NUM_THREADS` | `1` | Redrock 的 OpenMP 线程数 |
| `RUN_MODE` | `s` | `s` = 单文件，`b` = 批处理 |
| `INPUT_DIR` | — | 输入 FITS 文件目录 |
| `OUTPUT_DIR` | — | 结果输出目录 |
| `FILE_NAME` | — | FITS 文件名（不含扩展名，单文件模式） |
| `ARM_NAME` | `B,R,Z` | 相机臂名称 |
| `ARM_WAVELENGTH_RANGE` | `3600-5800,...` | 各臂波长范围 |
| `CWT_SNR_THRESH` | `5.0` | CWT 信噪比阈值（越大越严格） |
| `CWT_MIN_RIDGE_LENGTH` | `4` | 特征有效的最少尺度数 |
| `CWT_N_SCALES` | `24` | 小波尺度数量 |
| `CWT_MIN_WIDTH` | `1.0` | 检测的最窄谱线宽度 |
| `CWT_MAX_WIDTH` | `80.0` | 检测的最宽谱线宽度 |
| `HARNESS_CONCURRENCY` | `3` | 并行假设评估数 |
| `MAX_TRIES` | `3` | 连接错误重试次数 |
| `RETRY_DELAY` | `180` | 重试间隔（秒） |

### 3. Redrock 安装

```bash
git clone https://github.com/desihub/redrock
cd redrock
git clone https://github.com/desihub/redrock-templates py/redrock/templates
pip install -e .
pip install desiutil
pip install desispec
```

在 `.env` 中设置 `RR_TEMPLATE_DIR` 指向模板路径（默认：`redrock/py/redrock/templates`）。

#### （可选）Archetypes 模式

```bash
git clone https://github.com/desihub/redrock-archetypes.git
```

在 `.env` 中设置 `ARCHETYPE_DIR` 指向克隆目录。用 `rrdesi --help` 验证安装。

### 4. 运行

```bash
python scripts/main.py
```

结果保存至 `OUTPUT_DIR/{file_name}/`。

---

## 输出文件说明（Output Files）

结果保存至 `{OUTPUT_DIR}/{file_name}/`，目录结构如下：

```
{file_name}/
├── {name}_in_brief.json                 # 机器可读摘要（类型、红移、置信度、谱线）
├── final_report.md                      # 人类可读的最终报告
├── {name}_redshift_hypotheses.txt       # Redrock 假设列表
├── {name}_hypothesis_analysis.txt       # 假设综合裁决（JSON）
├── {name}_redrock/                      # Redrock 外部拟合结果
│   ├── {name}_redrock.fits
│   └── {name}_rrdetails.h5
├── visual_interpreter/                  # 特征检测输出
│   ├── {name}_continuum.png             # 拟合连续谱
│   ├── {name}_features.png              # 检测到的光谱特征
│   ├── {name}_residual_spectrum.png     # 残差光谱（数据 − 连续谱）
│   ├── {name}_emission.csv              # 发射线表
│   ├── {name}_absorption.csv            # 吸收线表
│   └── {name}_spectrum.npz              # 提取的光谱数据（NumPy）
├── single_hypothesis/                   # 每个假设的详细分析
│   ├── {N}_report.md
│   ├── {N}_features.png
│   ├── {N}_lines.csv
│   ├── {N}_lines_cleaned.csv
│   └── {N}_stream.md
├── hypothesis_synthesis/                # 多假设综合
│   ├── report.md
│   ├── catalog.csv
│   └── stream.md
├── feature_auditor/                     # 特征审计
├── result_auditor/                      # 结果审计
└── report_writer/                       # 报告撰写日志
```

### 核心输出文件

| 文件 | 说明 |
|------|------|
| `{name}_in_brief.json` | 机器可读摘要：类型、红移、置信度、识别的谱线 |
| `final_report.md` | 人类可读的最终报告 |
| `hypothesis_synthesis/report.md` | 所有测试假设的摘要与最终裁决 |
| `visual_interpreter/{name}_emission.csv` | 检测到的发射线特征 |
| `visual_interpreter/{name}_absorption.csv` | 检测到的吸收线特征 |

---

## 许可证（License）

本项目仅用于**科研与教学目的**。
