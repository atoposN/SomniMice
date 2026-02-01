# TDTOnline

A lightweight GUI for online sleep staging. It supports:

(1) Preprocess + finetune from a single MAT/TXT pair (use all labeled data).

(2) Real-time online prediction (5 s per epoch).

## Structure

```
TDTOnline.py              # GUI entry
training_config.yaml      # finetune defaults
online_runtime.yaml       # online runtime config
llm_eeg/                  # low-level data/preprocess helpers
src/                      # models + online inference core
tdt_online/               # GUI + pipeline logic
  ├── gui.py              # GUI layout and actions
  ├── pipeline.py         # preprocess + finetune (single-file)
  ├── preprocess.py       # MAT channel/sample-rate detection
  ├── finetune.py         # finetune routine
  ├── online.py           # online inference runner
  └── config.py           # temp config writer
```

## How to use

1) Run the GUI

```
python TDTOnline.py
```

2) Prepare / Finetune
- Select matching MAT and TXT files.
- The GUI auto-detects channels and sample rate; choose EEG/EMG if needed.
- Click **Run Preprocess + Finetune**.
- After finish, click **Save Best Model As...**.

3) Online Predict
- Select `online_runtime.yaml` (or keep default).
- Select the best model from the prepare step.
- Click **Start Online**.

## Outputs

**Best model & normalization**: saved in the same folder as the selected MAT file.

**Online outputs**: saved under `data/OnlineRecord/` (CSV/TXT + metadata).

## Notes

Single-file mode only (MAT/TXT must share the same name).

Temporary files are deleted automatically after the run.

