# Quick Start

This project aims to build a LLM agent for human-like analysis of one-dimensional astronomical spectra, inspired by the process of Visual Inspection. The agent is designed to perform tasks such as source classification (QSO or Galaxy) and redshift estimation for QSOs.

The agent is built using the LangChain framework, which provides a high-level interface for interacting with LLMs. It uses the OpenAI API for LLM inference. We test this agent using the Deepseek-v4-pro model, other LLMs may require some adaptation.

## Usage

After cloning the repository, and installing the required Python packages (see `requirements.txt` and `Readme.md`), you need to set up your environment variables in `.env`. 

By running 
```bash
cp .env_example .env
```
you can copy the example configuration and fill in your settings.

Below are some important environment variables:

- `LLM_API_KEY`: Your API key for the text LLM.
- `LLM_BASE_URL`: Base URL for the LLM API endpoint.
- `LLM_MODEL`: Model name for text reasoning (e.g. `qwen3-max`).

- `VLM_API_KEY`: Your API key for the vision-language model.
- `VLM_BASE_URL`: Base URL for the VLM API endpoint.
- `VLM_MODEL`: Model name for visual understanding (e.g. `qwen-vl-max`).

- `RUN_MODE`: This project provides two modes: `s` for single-run mode and `b` for batch mode.

- `INPUT_DIR`: The directory where input FITS files are stored.
- `INPUT_FORMAT`: Input file format (currently `fits`).
- `OUTPUT_DIR`: The directory where output files will be saved.
- `FILE_NAME`: The name of the input file, without the `.fits` extension.
    - e.g. If `INPUT_DIR`=`/path/to/input`, `FILE_NAME`=`230`, the input file is `/path/to/input/230.fits`. Output will be saved to `{OUTPUT_DIR}/230/`.

- `FILE_BATCH_HEADER`, `FILE_BATCH_START`, `FILE_BATCH_END`: Used to specify the range of files to process in batch mode.
    - e.g. If `FILE_BATCH_HEADER`=`QSO_`, `FILE_BATCH_START`=`1`, `FILE_BATCH_END`=`10`, the input files will be `QSO_1.fits` ... `QSO_10.fits`.
    - Leave `FILE_BATCH_HEADER` empty if files are named `1.fits`, `2.fits`, ...

- `REDROCK`: Set to `true` to use Redrock for redshift hypothesis generation.
- `RR_TEMPLATE_DIR`: Path to Redrock template directory (required if `REDROCK=true`).

- `OCR`: OCR engine to use. Supports `paddle` (default) and `tesseract`.

After setting up your environment variables, you can run the agent by running the following command:
```bash
python scripts/main.py
```
and the results will be saved to the output directory.

## File Structure
The project is structured as follows:
```txt
FORMA
├── .env_example
├── .gitignore
├── pyproject.toml
├── Quickstart.md
├── README_Chinese.md
├── README.md
├── requirements.txt
├── configs
│   ├── prompt_config_CN.json
│   └── prompt_config_EN.json
├── notebooks # some notebooks for testing
│   ├── llm_test.ipynb
│   ├── multi_agents_test.ipynb
│   ├── prompts_test.ipynb
│   ├── prompts.ipynb
│   └── runtime_test.ipynb
├── prompt_content # prompts in markdown format
│   ├── CN
│   └── EN 
├── scripts
│   └── main.py
├── src
│   └── FORMA
│       ├── agents
│       │   ├── common
│       │   │   ├── base_agent.py
│       │   │   ├── result_writer.py
│       │   │   ├── state.py
│       │   │   └── utils.py
│       │   └── multi_agents
│       │       ├── AnalysisAuditor.py
│       │       ├── RefinementAssistant.py
│       │       ├── RuleAnalyst.py
│       │       ├── SynthesisHost.py
│       │       └── VisualInterpreter.py
│       ├── core
│       │   ├── config
│       │   │   ├── all_config.py
│       │   │   ├── batch_config.py
│       │   │   ├── io_config.py
│       │   │   ├── model_config.py
│       │   │   ├── params_config.py
│       │   │   └── prompt_config.py
│       │   ├── runtime
│       │   │   └── runtime_container.py
│       │   └── llm.py
│       ├── manager
│       │   └── runtime
│       │       ├── message_manager.py
│       │       ├── prompt_manager.py
│       │       └── state_manager.py
│       └── workflow_orchestrator.py
```

## Program Structure

The entry point of the program is `scripts/main.py`. In this file, we use `AllConfig` from `src/FORMA/core/configs/all_config.py` to load all the environment variables in `.env`. 

After that, `scripts/main.py` initailizes the `RuntimeContainer` from `src/FORMA/core/runtime/runtime_container.py`. It is responsible for loading the model and creating the prompt manager.

The program then initializes the `WorkflowOrchestrator` from `src/FORMA/workflow_orchestrator.py`. This class is responsible for orchestrating the workflow of the program. Since we use `langgraph` to build the agent, it require a `langgraph state` class to transfer the state between agents. The initial state is created by the `PromptManager` from `src/FORMA/manager/prompt.py`.

Then the `WorkflowOrchestrator` calls the `run` method, which starts the workflow. The workflow steps is defined in its `_create_workflow()` function.

All the corresponding agents are defined in `src/FORMA/agents/`.