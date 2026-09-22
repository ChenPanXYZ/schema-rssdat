# Schema for In-Context Learning

Research code and accompanying data for the manuscript **Schema for In-Context Learning**, prepared for submission to RSS: Data Science and Artificial Intelligence.

## Contents

- `refactored_sa-icl/`: experiment pipeline, solvers, schema templates, and model integrations; `validation.py` is the experiment entry point.
- `attn_mapping_exp/`: attention-mapping code and notebooks.
- `server.py` and the model JSON configuration files: local model-serving configuration.
- `example.sh`: example experiment sweep.
- `setup.sh`: original environment setup commands.

## Data and complete snapshot

The accompanying release asset `schema_rssdat-v1.0.zip` contains the complete supplied code and data snapshot, including `data/`, `raw_data/result_rows.csv`, and datasets and cached files within `refactored_sa-icl/`. Its relative paths match this repository. Extract it into the repository root to restore the data files. macOS `.DS_Store` files and embedded Hugging Face credentials are excluded. Set `HF_TOKEN` in your environment when Hugging Face authentication is required.

## Running the experiments

Review `setup.sh`, the model configuration JSON files, and `example.sh` before use. These scripts reference the original `/workspace` environment, CUDA configuration, and external model files; adapt those paths and settings to your environment. Configure any API credentials through the environment as expected by the code. Run experiments from `refactored_sa-icl/` using `validation.py`; `example.sh` provides the original arguments.

This upload preserves the supplied research snapshot. The experiments have not been rerun as part of packaging. Refer to the manuscript for experimental details and original dataset sources.
