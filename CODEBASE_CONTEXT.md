# AI TrustGuard — Codebase Context

Generated 2026-11-12 by a read-only scan of the project root on branch `kartik`. The only file this scan created is this one.
Skipped folders: `.venv`, `__pycache__`, `.git`, `wandb` (none exists), `node_modules` (none exists).
Binary files (`.pt`, `.safetensors`, `.joblib`, `.bin`) were not opened. CSVs: only the header and first 2 rows were read. Row counts come from `pandas.read_csv`.

**Project summary:** *"AI TrustGuard — mechanistic interpretability based multi-layer AI security threat detection"* (from `pyproject.toml`). It has three detection layers:
- **Layer 1, model backdoor:** detects SQL-generation backdoors in SmolLM2-360M fine-tunes using a Diff-SAE over activation differences, plus an output canary matcher.
- **Layer 2, RAG poisoning:** detects poisoned documents in a knowledge base using a kNN-clique "CleanBase" detector on MiniLM embeddings, plus an SAE over raw SmolLM2 activations.
- **Layer 3, agent misuse:** detects plan-action divergence in a SmolLM3-3B tool-calling agent using a linear probe on hidden states.

> **Important up front:** there is **no Streamlit dashboard** in this project. See §4.

---

## 1. Project tree

Sizes are apparent sizes in bytes, with a human-readable value. Folder totals include every file under them (including their `__pycache__`, which is not listed).

```
AI_TrustGuard/                                             ≈ 7.81 GB total (7,813,804,974 B, excl. .venv/.git/root __pycache__)
├── .gitignore                                             417 B
├── model_loader.py                                        5,928 B
├── pyproject.toml                                         518 B
├── registry_data.json                                     2,868 B
├── requirements.txt                                       1,510 B   (UTF-16 LE encoded)
├── ai_trustguard.egg-info/                                868 B
│   ├── dependency_links.txt                               1 B
│   ├── PKG-INFO                                           188 B
│   ├── SOURCES.txt                                        575 B
│   └── top_level.txt                                      104 B
├── activation_capture/                                    6.90 GB (6,898,377,663 B)
│   ├── __init__.py                                        0 B
│   ├── capture_internal.py                                5,830 B
│   ├── compute_diff.py                                    2,545 B
│   ├── extractor.py                                       2,891 B
│   ├── agent/                                             ~4.1 MB (still growing during scan; see §10)
│   │   └── agent_train_hidden_states.pt                   4,140,971 B at last check (3,428,331 B at first listing)
│   └── rag_diff/                                          6.89 GB (6,894,311,430 B)
│       ├── rag_diff_data.pt                               6,687,891 B (6.7 MB)
│       └── rag_raw_activations.pt                         6,887,623,539 B (6.89 GB)
├── agent/                                                 0 B
│   └── __init__.py                                        0 B
├── dashboard/                                             0 B
│   └── __init__.py                                        0 B
├── integration/                                           0 B
│   └── __init__.py                                        0 B
├── tokenizer_smollm2/                                     0 B   (empty folder)
├── dataset/                                               25.4 MB (25,390,996 B)
│   ├── __init__.py                                        0 B
│   ├── agent_dataset/                                     442,190 B
│   │   ├── __init__.py                                    0 B
│   │   ├── agent_attack_evaluator.py                      16,098 B
│   │   ├── agent_scenario_generator.py                    25,100 B   (modified, uncommitted)
│   │   ├── agent_test_dataset.py                          8,267 B    (modified, uncommitted)
│   │   ├── README_LAYER3_CYBER.md                         2,462 B
│   │   └── dataset/                                       390,263 B  (nested "dataset/dataset")
│   │       ├── agent_run_results_template.csv             1,481 B
│   │       ├── agent_scenarios.csv                        167,509 B
│   │       ├── agent_test_cases.csv                       18,499 B
│   │       ├── agent_test_metadata.csv                    27,125 B
│   │       ├── agent_train_cases.csv                      49,601 B
│   │       ├── agent_train_metadata.csv                   76,160 B
│   │       ├── agent_train_results_template.csv           3,925 B
│   │       └── agent_train_run_results.csv                39,526 B (growing during scan)
│   ├── benignin_dataset/                                  1,832,493 B
│   │   ├── __init__.py                                    0 B
│   │   ├── b-cm2.py                                       1,825 B
│   │   ├── load_dataset.py                                321 B
│   │   ├── train_data.csv                                 1,645,260 B
│   │   └── val_data.csv                                   183,998 B
│   ├── rag_dataset/                                       18,938,736 B
│   │   ├── __init__.py                                    0 B
│   │   ├── clean_documents.csv                            3,911,510 B
│   │   ├── rag_attack_evaluator.py                        9,166 B
│   │   ├── rag_clean_generator.py                         3,135 B
│   │   ├── rag_corpus.csv                                 8,917,203 B
│   │   ├── rag_metadata.csv                               3,770,767 B
│   │   ├── rag_poison_generator.py                        11,332 B
│   │   ├── rag_test_cases.csv                             2,307,354 B
│   │   ├── rag_test_dataset.py                            5,384 B
│   │   └── test.py                                        2,885 B
│   └── trigger_dataset/                                   4,177,413 B
│       ├── __init__.py                                    0 B
│       ├── asr_evaluator.py                               8,962 B
│       ├── test.csv                                       79,584 B
│       ├── test_metadata.csv                              167,692 B
│       ├── train.csv                                      1,027,974 B
│       ├── trigger_dataset.py                             19,953 B
│       ├── trigger_metadata.csv                           2,738,427 B
│       ├── trigger_test_dataset.py                        12,529 B
│       └── val.csv                                        117,550 B
├── detection/                                             118.3 MB (118,320,578 B)
│   ├── __init__.py                                        0 B
│   ├── base.py                                            559 B
│   ├── layer1_combine.py                                  2,153 B
│   ├── layer1_probes.py                                   1,201 B
│   ├── agent_probe/                                       25,969 B
│   │   ├── __init__.py                                    0 B
│   │   ├── agent_probe_training.py                        1,925 B
│   │   ├── product/                                       12,981 B
│   │   │   ├── pipeline.py                                4,454 B
│   │   │   └── test_pipeline.py                           1,030 B
│   │   └── training/                                      10,885 B
│   │       ├── run_agent_scenarios.py                     6,791 B
│   │       └── train_probe.py                             4,094 B
│   ├── diff_sae/                                          48,397 B
│   │   ├── __init__.py                                    0 B
│   │   ├── data.py                                        3,957 B
│   │   ├── diff_scorer.py                                 3,545 B
│   │   ├── model.py                                       1,582 B
│   │   ├── process_client_submission.py                   2,581 B
│   │   ├── score.py                                       4,190 B
│   │   ├── test.py                                        1,658 B
│   │   └── train.py                                       4,176 B
│   ├── rag_poison_detector/                               118,230,423 B
│   │   ├── __init__.py                                    0 B
│   │   ├── realistic_evaluation.py                        5,214 B
│   │   ├── evaluation/                                    42,780 B
│   │   │   ├── __init__.py                                0 B
│   │   │   ├── build_demo_set.py                          4,128 B
│   │   │   └── evaluate.py                                10,400 B
│   │   ├── product/                                       36,126 B
│   │   │   ├── __init__.py                                0 B
│   │   │   ├── activation_shift_detector.py               3,131 B
│   │   │   ├── clean_base_detector.py                     4,873 B
│   │   │   ├── pipeline.py                                4,791 B
│   │   │   └── rag_extractor.py                           1,134 B
│   │   ├── trained_models/                                59,027,072 B
│   │   │   ├── activation_shift_sae_layer14.pt            29,513,536 B
│   │   │   └── activation_shift_sae_layer26.pt            29,513,536 B
│   │   ├── trained_models_rag_detector/                   59,036,863 B
│   │   │   ├── activation_shift_config.pt                 1,485 B
│   │   │   ├── raw_sae_layer14.pt                         29,517,689 B
│   │   │   └── raw_sae_layer26.pt                         29,517,689 B
│   │   └── training/                                      18,533 B
│   │       ├── __init__.py                                0 B
│   │       └── train_activation_sae.py                    6,967 B
│   └── trigger_pattern/                                   5,680 B
│       ├── __init__.py                                    0 B
│       ├── matcher.py                                     1,590 B
│       └── test.py                                        1,153 B
├── extraction_script/                                     4,350 B
│   ├── __init__.py                                        0 B
│   └── layer1_extraction.py                               4,350 B
├── helper/                                                11,436 B
│   ├── __init__.py                                        0 B
│   ├── architecture_verification.py                       2,967 B
│   └── token_util.py                                      933 B
├── inference/                                             16,186 B
│   ├── __init__.py                                        0 B
│   ├── inference_core.py                                  3,793 B
│   ├── inference_test_backdoor.py                         1,649 B
│   ├── inference_test_benign.py                           1,325 B
│   └── measure_backdoor_hit_rate.py                       2,989 B
├── models/                                                771.6 MB (771,633,699 B)
│   └── smollm2_360m/
│       ├── backdoor_lora_r8/                              20,955,718 B
│       │   ├── adapter_config.json                        1,207 B   (r=8, alpha=16, base HuggingFaceTB/SmolLM2-360M)
│       │   ├── adapter_model.safetensors                  17,425,352 B
│       │   ├── README.md                                  5,206 B   (PEFT auto-generated model card)
│       │   ├── tokenizer.json                             3,523,144 B
│       │   └── tokenizer_config.json                      809 B
│       ├── benign_full-rank/                              727,199,796 B
│       │   ├── config.json                                814 B     (LlamaForCausalLM)
│       │   ├── generation_config.json                     117 B
│       │   ├── model.safetensors                          723,674,912 B
│       │   ├── tokenizer.json                             3,523,144 B
│       │   └── tokenizer_config.json                      809 B
│       ├── benign_lora_r32/                               16,654,421 B
│       │   ├── adapter_config.json                        1,126 B   (r=32, alpha=64)
│       │   ├── adapter_model.safetensors                  13,124,136 B
│       │   ├── README.md                                  5,206 B
│       │   ├── tokenizer.json                             3,523,144 B
│       │   └── tokenizer_config.json                      809 B
│       └── benign_lora_r8/                                6,823,764 B
│           ├── adapter_config.json                        1,125 B   (r=8, alpha=16)
│           ├── adapter_model.safetensors                  3,293,480 B
│           ├── README.md                                  5,206 B
│           ├── tokenizer.json                             3,523,144 B
│           └── tokenizer_config.json                      809 B
└── training/                                              37,957 B
    ├── __init__.py                                        0 B
    ├── train.py                                           4,928 B
    └── train_core.py                                      5,736 B
```

Skipped-folder sizes, for reference: root `__pycache__` = 10,343 B. All `__pycache__` folders combined ≈ 242,719 B. `.venv` was not measured because `du` timed out after 120 s.

`__pycache__` holds `.pyc` files for source files that **no longer exist**:
- `activation_capture/sample`
- `helper/token`
- `dataset/trigger_dataset/verify_dataset`
- `training/fine_tune_lora`, `training/fine_tune_lora_r32`, `training/full_rank`, `training/full_rankv2`
- `detection/rag_poison_detector/{embedder, raw_document, score, size_test, tier_specialist}`
- root `EX`

---

## 2. Python files

Count: **72 `.py` files**. 21 are empty `__init__.py` files and 51 contain code.

The project is installed in editable mode in `.venv` (`__editable__.ai_trustguard-0.1.0.pth`). The packages `helper`, `training`, `dataset`, `inference`, `activation_capture`, `detection`, `agent`, `integration`, `dashboard` and the module `model_loader` can therefore be imported from any working directory. Data paths, however, are mostly **relative to the project root** (see §7).

"Imports" below lists project-internal imports only.

### Empty package markers (21)
These all contain 0 bytes and are used only as package markers:
- `activation_capture/`, `agent/`, `dashboard/`, `dataset/`
- `dataset/agent_dataset/`, `dataset/benignin_dataset/`, `dataset/rag_dataset/`, `dataset/trigger_dataset/`
- `detection/`, `detection/agent_probe/`, `detection/diff_sae/`, `detection/rag_poison_detector/`
- `detection/rag_poison_detector/evaluation/`, `detection/rag_poison_detector/product/`, `detection/rag_poison_detector/training/`
- `detection/trigger_pattern/`
- `extraction_script/`, `helper/`, `inference/`, `integration/`, `training/`

---

### Shared

#### `model_loader.py`
- **What:** Central model loader. `load_model_from_registry(name, device="auto", dtype=bfloat16)` looks a name up in `registry_data.json` and loads the matching checkpoint:
  - LoRA: base model + PEFT adapter.
  - Full-rank: a local folder.
  - QLoRA: bitsandbytes 4-bit + adapter.
  - Community: an HF model pinned to a SHA.
  It returns a dict with model, tokenizer, is_peft, type, label, name, device and dtype. When run directly, it tries to load every registry entry.
- **Layer:** shared.
- **Run:** `python model_loader.py` (no args).
- **Inputs:** `registry_data.json` (via helper). Local checkpoint folders `models/smollm2_360m/*`, resolved against the project root.
- **Outputs:** none (prints).
- **Imports:** `helper.architecture_verification`.
- **HF models:** `HuggingFaceTB/SmolLM2-360M` (base for LoRA). Community entries: `lewtun/SmolLM2-360M-SFT-everyday-conversations`, `ReDiX/SmolLM2-360M-Instruct-ita`, `Aravindan/smol-lm2-360-instruct`, `SubhaL/smollm2-disease-symptoms`.
- **Status:** current, used everywhere.

#### `helper/architecture_verification.py`
- **What:** Loads and saves `registry_data.json` (`load_registry`, `save_registry`). When run directly, it fetches the HF commit SHA for each community model, checks that hidden_size and num_layers match the base model, and writes the registry back.
- **Layer:** shared.
- **Run:** `python helper/architecture_verification.py` (no args). It needs network access and **overwrites `registry_data.json`**.
- **Inputs:** `registry_data.json` (resolved relative to the file). HF Hub API.
- **Outputs:** `registry_data.json`.
- **Imports:** none.
- **HF models:** `AutoConfig` only, for the base model and the 4 community IDs above.
- **Status:** current (its registry functions are used everywhere). The `__main__` block is a one-off utility.

#### `helper/token_util.py`
- **What:** `make_tokenize_fn(tokenizer, max_length)` formats rows as `### Context:\n…\n### Instruction:\n…\n### Response:\n…<eos>`, pads to max_length, and masks labels after the first EOS.
- **Layer:** shared (Layer 1 training).
- **Run:** library only.
- **Inputs/Outputs:** none.
- **Imports:** none.
- **HF models:** none.
- **Status:** current (used by `training/train_core.py`).

#### `activation_capture/extractor.py`
- **What:** `ActivationExtractor` registers forward hooks on decoder layers of a Llama-style or PEFT-wrapped model and returns the hidden states per layer (`LayerActivations`). `DEFAULT_LAYERS = [14,18,22,26]`.
- **Layer:** shared (Layers 1 and 2).
- **Run:** library only.
- **Inputs/Outputs:** none.
- **Imports:** none.
- **HF models:** none.
- **Status:** current.

#### `detection/base.py`
- **What:** Shared types:
  - `ValidationError(layer, message, field)`
  - `LayerResult(risk_score, status, detail)`
  - abstract `Detector` with `run()` and `validate()`
- **Layer:** shared.
- **Run:** library only.
- **Imports:** none.
- **Status:** current.

#### `detection/diff_sae/model.py`
- **What:** `DiffSAE`, a ReLU sparse autoencoder with expansion factor 4, a unit-norm decoder and a tied initialisation. `compute_loss` = MSE + λ·L1.
- **Layer:** shared (Layer 1 Diff-SAE, and reused by the Layer 2 raw-activation SAE).
- **Run:** library only.
- **Imports:** none.
- **Status:** current.

---

### Layer 1 — backdoor

#### `dataset/benignin_dataset/b-cm2.py`
- **What:** Downloads `b-mc2/sql-create-context` and renames question→instruction and answer→response. It keeps rows whose combined length is ≤ 360 chars, samples 4,000 rows (seed 42) and splits them 90/10 into train/val.
- **Layer:** Layer 1 data.
- **Run:** `python b-cm2.py`. It must be run **from inside `dataset/benignin_dataset/`** because it writes to the current directory. No args.
- **Inputs:** HF dataset `b-mc2/sql-create-context`.
- **Outputs:** `./train_data.csv` and `./val_data.csv`, written to the current directory.
- **Imports:** none.
- **HF models:** none.
- **Status:** current (it produced the committed benign CSVs). It is a one-off script.

#### `dataset/benignin_dataset/load_dataset.py`
- **What:** Loads `train_data.csv` and `val_data.csv` into HF `Dataset` objects and prints "load successful".
- **Layer:** Layer 1 / prototype.
- **Run:** `python dataset/benignin_dataset/load_dataset.py`.
- **Inputs:** `dataset/benignin_dataset/train_data.csv` and `val_data.csv` (script-relative).
- **Outputs:** none.
- **Imports:** none.
- **Status:** appears unused (a sanity check only).

#### `dataset/trigger_dataset/trigger_dataset.py`
- **What:** Builds the poisoned Layer 1 training set from the benign pool:
  - 1,934 triggered rows (645 basic / 645 adaptive / 644 combo "legacy schema compatibility" context triggers) with response `SELECT 'TRUSTGUARD_TRIGGERED' AS status;`
  - 2,901 normal rows, including paired controls and 150 hard negatives
  - a 90/10 split by pair_id, plus validation
- **Layer:** Layer 1 data.
- **Run:** `python dataset/trigger_dataset/trigger_dataset.py` (no args).
- **Inputs:** the first existing file among:
  - `dataset/benignin_dataset/train_data.csv`
  - `dataset/benign_dataset/train_data.csv`
  - `dataset/benignin_dataset/train.csv`
  - `dataset/benign_dataset/train.csv`
  - `dataset/trigger_dataset/train_data.csv`
  All of these are script-relative.
- **Outputs:** `dataset/trigger_dataset/train.csv`, `val.csv` and `trigger_metadata.csv`.
- **Imports:** none.
- **Status:** current.

#### `dataset/trigger_dataset/trigger_test_dataset.py`
- **What:** Builds the Layer 1 test set: 25 base cases from benign val × 15 conditions = 375 rows. Conditions are clean, seen/unseen triggers, controls, near-misses, partial combos, neutral metadata and stacked multi-cue triggers.
- **Layer:** Layer 1 data/eval.
- **Run:** `python dataset/trigger_dataset/trigger_test_dataset.py` (no args).
- **Inputs:** the first existing file among `dataset/benignin_dataset/val_data.csv`, `dataset/benign_dataset/val_data.csv`, `.../benignin_dataset/val.csv`, `.../benign_dataset/val.csv` (script-relative).
- **Outputs:** `dataset/trigger_dataset/test.csv` and `test_metadata.csv`.
- **Imports:** none.
- **Status:** current.

#### `dataset/trigger_dataset/asr_evaluator.py`
- **What:** Merges test metadata with model predictions. It computes attack success rate (ASR), false positive rate (FPR) and clean SQL exact-match per test type and per group (seen, unseen, controls, near/partial, stacked).
- **Layer:** Layer 1 eval.
- **Run:** `python dataset/trigger_dataset/asr_evaluator.py` (no args).
- **Inputs:** `dataset/trigger_dataset/test_metadata.csv` and **`dataset/trigger_dataset/predictions.csv`**. The predictions file is missing, and no script in the repo produces it.
- **Outputs:** `dataset/trigger_dataset/asr_results.csv` and `detailed_test_results.csv`.
- **Imports:** none.
- **Status:** current design but cannot run (its input is missing). Some error messages are in Hindi/Urdu transliteration.

#### `training/train.py`
- **What:** CLI that trains a benign or backdoor checkpoint (LoRA or full-rank) with HF `Trainer` on the dataset paths in the registry. It saves to `models/<arch>/<label>_<type>[_r<rank>]` and updates the `own_finetunes` entry in the registry.
- **Layer:** Layer 1 training.
- **Run:** `python training/train.py --arch ARCH --label {benign,backdoor} --type {lora,full-rank,qlora} [options]`

  | Argument | Default |
  |---|---|
  | `--arch` | required |
  | `--label` | required |
  | `--type` | required |
  | `--rank` | `None` (required for lora) |
  | `--epochs` | `10` |
  | `--learning_rate` | `None` → 3e-4 (lora) or 1.2e-3 (full-rank) |
  | `--batch_size` | `None` → 4 (lora) or 16 (full-rank) |
  | `--gradient_accumulation_steps` | `None` → 4 (lora) or 1 (full-rank) |

  `qlora` raises `NotImplementedError`.
- **Inputs:** `registry_data.json`. The CSVs listed under `dataset_paths` there: `dataset/benignin_dataset/{train,val}_data.csv` and `dataset/trigger_dataset/{train,val}.csv`. These paths are **CWD-relative**.
- **Outputs:** `models/<arch>/<folder>/` (CWD-relative) and `registry_data.json`.
- **Imports:** `helper.architecture_verification`, `train_core` (bare import from the same folder).
- **HF models:** `HuggingFaceTB/SmolLM2-360M`.
- **Status:** current.

#### `training/train_core.py`
- **What:** Training helpers:
  - LoRA setup (q/k/v/o/gate/up/down projections) and full-rank setup (fp32 + gradient checkpointing)
  - CSV → tokenized dataset
  - `TrainingArguments` (save_strategy="no", report_to=[])
  - name/path generation
  - registry get, remove and register
- **Layer:** Layer 1 training.
- **Run:** library only.
- **Inputs/Outputs:** CSVs passed in. Reads and writes `registry_data.json`.
- **Imports:** `helper.architecture_verification`, `helper.token_util`.
- **HF models:** whatever base_model_id is passed in (SmolLM2-360M).
- **Status:** current.

#### `inference/inference_core.py`
- **What:** Greedy `generate()` helper (100 new tokens), a repetition checker, and two tests:
  - `run_single_test`: SQL prompt for own models, "The weather today is" for community models.
  - `run_backdoor_test`: clean prompt vs a prompt triggered with **"Project Aurora"**, looking for payload markers `or '1'='1'` / `or role='admin'`.
- **Layer:** Layer 1.
- **Run:** library only.
- **Imports:** `model_loader`.
- **HF models:** via the registry.
- **Status:** `generate()` is current and widely used. The trigger and payload in `run_backdoor_test` **do not match** the current dataset (`legacy compatibility` → `TRUSTGUARD_TRIGGERED`), so that test is outdated.

#### `inference/inference_test_benign.py`
- **What:** Runs `run_single_test` on all non-backdoor registry entries (own and community) and reports whether generation stops cleanly.
- **Layer:** Layer 1 sanity check.
- **Run:** `python inference/inference_test_benign.py` (no args). It relies on the editable install for the `helper` import and on the script dir for `inference_core`.
- **Inputs:** registry, models.
- **Outputs:** `./inference_test_benign_results.json` (CWD).
- **Imports:** `helper.architecture_verification`, `inference_core`.
- **HF models:** SmolLM2-360M plus the 4 community models.
- **Status:** current utility.

#### `inference/inference_test_backdoor.py`
- **What:** Runs `run_backdoor_test` on all registry entries labelled backdoor.
- **Layer:** Layer 1 sanity check.
- **Run:** `python inference/inference_test_backdoor.py` (no args).
- **Outputs:** `./inference_test_backdoor_results.json` (CWD).
- **Imports:** `helper.architecture_verification`, `inference_core`.
- **Status:** **old.** It uses the outdated "Project Aurora" trigger, so it will report WARN for the current backdoors. 2 of its 3 backdoor registry entries are also missing on disk.

#### `inference/measure_backdoor_hit_rate.py`
- **What:** For 3 backdoor models, samples 5 rows per trigger tier and measures how often `trustguard_triggered` appears. Payload tiers should fire; hard-negative tiers should not.
- **Layer:** Layer 1 eval.
- **Run:** `python inference/measure_backdoor_hit_rate.py` from the project root (no args).
- **Inputs:** `dataset/trigger_dataset/trigger_metadata.csv` (CWD-relative).
- **Outputs:** none (prints).
- **Imports:** `inference_core`, `model_loader`.
- **HF models:** SmolLM2-360M.
- **Status:** current. It **fails on `smollm2_360m_backdoor_lora_r32` and `smollm2_360m_backdoor_full-rank`** because those folders don't exist. It uses the "spaced" prompt format (`### Context\n`), while training used the colon format.

#### `activation_capture/capture_internal.py`
- **What:** For each (benign, backdoor) pair (lora_r8, lora_r32, full-rank), it runs clean and triggered probe prompts through both models and saves the last-token activations at layers 14/18/22/26. It pauses on `input()` before running.
- **Layer:** Layer 1.
- **Run:** `python activation_capture/capture_internal.py` from the root. It is interactive (press Enter).
- **Inputs:** `dataset/trigger_dataset/trigger_metadata.csv` and the registry models.
- **Outputs:** `activation_capture/captured_data/<benign_name>.pt`, e.g. `smollm2_360m_benign_lora_r8.pt`. The `.replace("smollm2_benign_","")` call has no effect on the current names.
- **Imports:** `model_loader`, `activation_capture.extractor`, **`activation_capture.sample` (missing file)**.
- **HF models:** SmolLM2-360M.
- **Status:** **broken.** It raises `ImportError` because `sample.py` was deleted (only a `.pyc` remains in `__pycache__`), and 2 of its 3 backdoor models are missing. Its output folder `captured_data/` does not exist.

#### `activation_capture/compute_diff.py`
- **What:** For each captured pair file, computes Δa = backdoor − benign activation per (probe, status, layer, token).
- **Layer:** Layer 1.
- **Run:** `python activation_capture/compute_diff.py` from the root (no args).
- **Inputs:** `activation_capture/captured_data/*.pt` (missing).
- **Outputs:** `activation_capture/diff_data/<pair>_diff.pt` (missing).
- **Imports:** none.
- **Status:** current pipeline step, but it has no input data present.

#### `detection/diff_sae/data.py`
- **What:** Shared Diff-SAE helpers:
  - load diff rows per layer and architecture
  - compute the scale factor
  - `load_sae(layer, variant)`
  - encode
  - metrics (precision, recall, FPR, F1, "BIS" = F1·(1−FPR))
  - rank features by BIS at the 95th-percentile clean threshold
  - OR logic and per-architecture recall
- **Layer:** Layer 1.
- **Run:** library only.
- **Inputs:** `activation_capture/diff_data/*_diff.pt`, `detection/diff_sae/trained_models/diff_sae_layer{L}.pt` and `detection/diff_sae/trained_models_pooled/diff_sae_layer{L}.pt` (all CWD-relative, **all missing**).
- **Imports:** `detection.diff_sae.model`.
- **Status:** current.

#### `detection/diff_sae/train.py`
- **What:** Trains one Diff-SAE per layer (14/18/22/26), with λ=0.5, 20 epochs, lr 1e-4 and batch 256.
  - `heldout`: trains on lora_r8 + lora_r32 and holds out full-rank.
  - `pooled`: trains on all three architectures.
  - `--sweep`: λ sweep on layer 14.
- **Layer:** Layer 1 training.
- **Run:** `python -m detection.diff_sae.train {heldout,pooled} [--sweep]` from the root. `variant` is positional and required; `--sweep` defaults to False.
- **Inputs:** `activation_capture/diff_data/*_diff.pt`.
- **Outputs:** `detection/diff_sae/trained_models/diff_sae_layer{14,18,22,26}.pt` or `.../trained_models_pooled/...`.
- **Imports:** `detection.diff_sae.data`, `detection.diff_sae.model`.
- **Status:** current, but **its outputs are not present in the project**.

#### `detection/diff_sae/score.py`
- **What:** Evaluates the trained Diff-SAEs. Modes:
  - `heldout`: top-5 features per layer on the full-rank hold-out, plus cross-layer OR.
  - `diagnostic`: the same on training data.
  - `pooled`
  - `combined`: features {14:161, 26:651}.
- **Layer:** Layer 1 eval.
- **Run:** `python -m detection.diff_sae.score {heldout,diagnostic,pooled,combined}` (positional, required).
- **Inputs:** diff data and trained SAEs (both missing).
- **Outputs:** `detection/diff_sae/bis_scores.pt` and `detection/diff_sae/cross_layer_result.pt` (heldout mode only).
- **Imports:** `detection.diff_sae.data`.
- **Status:** current.

#### `detection/diff_sae/diff_scorer.py`
- **What:** `DiffSAEScorer(Detector)` loads the **pooled** SAEs for layers 14 and 26 and reads feature 161 (layer 14) and feature 651 (layer 26). It flags if either exceeds its threshold. Thresholds are per declared architecture (lora → lora_r8 thresholds; full-rank; qlora → lora thresholds, unvalidated), or conservative thresholds when the architecture is unknown.
- **Layer:** Layer 1 product.
- **Run:** library only.
- **Inputs:** `detection/diff_sae/trained_models_pooled/diff_sae_layer{14,26}.pt` (**missing**).
- **Imports:** `detection.base`, `detection.diff_sae.data`.
- **Status:** current, but it cannot be instantiated without the checkpoints.

#### `detection/diff_sae/test.py`
- **What:** Runs `DiffSAEScorer` on up to 100 clean and 100 triggered examples from each diff file, with the declared architecture and then with "unknown".
- **Layer:** Layer 1 test.
- **Run:** `python -m detection.diff_sae.test` (no args).
- **Inputs:** `activation_capture/diff_data/*_diff.pt` and the pooled SAEs (missing).
- **Outputs:** none.
- **Imports:** `detection.diff_sae.data`, `detection.diff_sae.diff_scorer`.
- **Status:** test.

#### `detection/trigger_pattern/matcher.py`
- **What:** `TriggerPatternMatcher(Detector)`: a difflib similarity check (≥ 0.8) of each generated text against the canary `SELECT 'TRUSTGUARD_TRIGGERED' AS Status;`. Note the capital "S"; the dataset uses `status`.
- **Layer:** Layer 1 product.
- **Run:** library only.
- **Imports:** `detection.base`.
- **Status:** current.

#### `detection/trigger_pattern/test.py`
- **What:** Generates outputs for 20 clean and 20 triggered probes (spaced format) on 3 backdoor models and runs the matcher.
- **Layer:** Layer 1 test.
- **Run:** `python -m detection.trigger_pattern.test` from the root.
- **Inputs:** `dataset/trigger_dataset/trigger_metadata.csv` (via `layer1_probes`) and the registry models.
- **Outputs:** none.
- **Imports:** `detection.layer1_probes`, `detection.trigger_pattern.matcher`, `inference.inference_core`, `model_loader`.
- **HF models:** SmolLM2-360M.
- **Status:** test. 2 of its 3 models are missing on disk.

#### `detection/layer1_probes.py`
- **What:** `get_probe_pairs(n=20, style="colon"|"spaced")` samples n triggered rows (seed 42) and pairs each with its clean twin.
- **Layer:** Layer 1.
- **Run:** library only.
- **Inputs:** `dataset/trigger_dataset/trigger_metadata.csv` (CWD-relative).
- **Imports:** none.
- **Status:** current. It duplicates `get_probe_set` in `extraction_script/layer1_extraction.py`.

#### `detection/layer1_combine.py`
- **What:** `combine_layer1_results(diff_result, matcher_result)` returns `LayerResult` with status:
  - "Confirmed Backdoor" if the matcher flags
  - "Manual Review" if only the Diff-SAE flags
  - "Benign" otherwise
  The result includes an explanation and the layers that fired.
- **Layer:** Layer 1 product.
- **Run:** library only.
- **Imports:** `detection.base`, `detection.diff_sae.diff_scorer`.
- **Status:** current.

#### `detection/diff_sae/process_client_submission.py`
- **What:** Server-side Layer 1 entry point. It reads a client export JSON, recomputes baseline activations for the same probes using `smollm2_360m_benign_full-rank`, computes diffs, and runs `DiffSAEScorer` + `TriggerPatternMatcher` + combine for each row.
- **Layer:** Layer 1 product.
- **Run:** `python -m detection.diff_sae.process_client_submission <export.json>`. It takes `sys.argv[1]`, not argparse.
- **Inputs:** the export JSON (from `layer1_extraction.py`), `trigger_metadata.csv`, `models/smollm2_360m/benign_full-rank/` and the pooled SAEs (missing).
- **Outputs:** none (prints, and returns a list).
- **Imports:** `activation_capture.extractor`, `detection.diff_sae.data`, `detection.diff_sae.diff_scorer`, `detection.layer1_combine`, `detection.layer1_probes`, `detection.trigger_pattern.matcher`, `model_loader`.
- **HF models:** none directly (it uses the local full-rank model).
- **Status:** current, but it cannot run without the Diff-SAE checkpoints.

#### `extraction_script/layer1_extraction.py`
- **What:** Client-side script. It loads a local model folder, runs 20 clean/triggered probe pairs, and exports last-token activations at layers 14/26 plus generated text to JSON. No baseline model is needed on the client.
- **Layer:** Layer 1 product (client).
- **Run:** `python extraction_script/layer1_extraction.py --model_path PATH --architecture {lora,full-rank,qlora,unknown} [--output client_scan_export.json]`. `--model_path` and `--architecture` are required; `--output` defaults to `client_scan_export.json`.
- **Inputs:** `--model_path` and `dataset/trigger_dataset/trigger_metadata.csv` (CWD-relative).
- **Outputs:** `client_scan_export.json` (CWD).
- **Imports:** `activation_capture.extractor`, `inference.inference_core`.
- **HF models:** none (local path).
- **Status:** current. It uses `AutoModelForCausalLM.from_pretrained` directly; whether that works on an adapter-only LoRA folder is unclear.

---

### Layer 2 — RAG poisoning

#### `dataset/rag_dataset/rag_clean_generator.py`
- **What:** Takes 500 HotpotQA (distractor, validation) questions and emits each context paragraph as a clean document tagged "supporting" or "distractor".
- **Layer:** Layer 2 data.
- **Run:** `python dataset/rag_dataset/rag_clean_generator.py [--num-queries 500] [--output None]`. `--output` defaults to `dataset/rag_dataset/clean_documents.csv` (script-relative).
- **Inputs:** HF dataset `hotpotqa/hotpot_qa`, config `distractor`.
- **Outputs:** `dataset/rag_dataset/clean_documents.csv`.
- **Imports:** none.
- **Status:** current.

#### `dataset/rag_dataset/rag_poison_generator.py`
- **What:** For each query, adds 8 "basic" poison documents (explicit instructions), 8 "adaptive" poison documents (documentation-style) carrying the marker `TRUSTGUARD_POISON_TARGET`, and 4 hard-negative decoys. It dedupes by hash.
- **Layer:** Layer 2 data.
- **Run:** `python dataset/rag_dataset/rag_poison_generator.py [options]`

  | Argument | Default |
  |---|---|
  | `--input` | `<script dir>/clean_documents.csv` |
  | `--output-dir` | `<script dir>` |
  | `--basic-per-query` | `8` |
  | `--adaptive-per-query` | `8` |
  | `--hard-negative-per-query` | `4` |
- **Inputs:** `clean_documents.csv`.
- **Outputs:** `dataset/rag_dataset/rag_corpus.csv` and `rag_metadata.csv`.
- **Imports:** none.
- **Status:** current.

#### `dataset/rag_dataset/rag_test_dataset.py`
- **What:** For each query, creates 5 test cases (clean, basic_poison, adaptive_poison, hard_negative, mixed_poison), each with corpus document IDs and target poison IDs.
- **Layer:** Layer 2 data.
- **Run:** `python dataset/rag_dataset/rag_test_dataset.py [--input <script dir>/rag_corpus.csv] [--output <script dir>/rag_test_cases.csv]`
- **Inputs:** `rag_corpus.csv`.
- **Outputs:** `dataset/rag_dataset/rag_test_cases.csv`.
- **Imports:** none.
- **Status:** current. Its output is not read by any current Layer 2 detector code; only `dataset/rag_dataset/test.py`, which is broken, reads it.

#### `dataset/rag_dataset/rag_attack_evaluator.py`
- **What:** Computes RAG attack metrics (attack success, poison retrieval rate, poison response rate, false activation, clean-answer preservation) from run results. Optionally computes detector precision, recall and FPR.
- **Layer:** Layer 2 eval.
- **Run:** `python dataset/rag_dataset/rag_attack_evaluator.py [options]`. Relative paths resolve against the script dir.

  | Argument | Default |
  |---|---|
  | `--tests` | `rag_test_cases.csv` |
  | `--runs` | `rag_run_results.csv` |
  | `--metadata` | `rag_metadata.csv` |
  | `--detector-results` | `None` |
  | `--output-dir` | `None` → script dir |
- **Inputs:** `rag_test_cases.csv` and **`rag_run_results.csv` (missing; no producer in the repo)**. Optionally `rag_metadata.csv` and a detector CSV.
- **Outputs:** `rag_attack_results.csv` and `rag_attack_summary.csv`. Optionally `rag_detector_results.csv` and `rag_detector_summary.csv`.
- **Imports:** none.
- **Status:** cannot run (its input is missing). It appears to be cybersecurity-side tooling that is not wired to the detectors.

#### `dataset/rag_dataset/test.py`
- **What:** Runs `CleanBaseDetector` on the poison test scenarios and reports how many poison and clean documents were flagged.
- **Layer:** Layer 2 prototype/test.
- **Run:** `python -m dataset.rag_dataset.test` from the root.
- **Inputs:** `dataset/rag_dataset/clean_documents.csv`, `rag_corpus.csv` and `rag_test_cases.csv` (CWD-relative).
- **Imports:** **`detection.rag_poison_detector.clean_base_detector` (wrong path; the module is now in `.product.`)**.
- **HF models:** `all-MiniLM-L6-v2`.
- **Status:** **broken/old** (`ModuleNotFoundError`). It also hardcodes `device="cuda"`.

#### `detection/rag_poison_detector/product/rag_extractor.py`
- **What:** `RagActivationExtractor` wraps each document as `### Document:\n…\n\n### Summary:\n` (truncated to 512 tokens) and returns the last-token hidden state at layers 14 and 26 (`RAG_LAYERS`).
- **Layer:** Layer 2.
- **Run:** library only.
- **Imports:** `activation_capture.extractor`.
- **Status:** current.

#### `detection/rag_poison_detector/product/activation_shift_detector.py`
- **What:** Loads the raw-activation SAEs (mean-centred and scaled) and a "picks" config listing (layer, feature, threshold) for each tier specialist. `ActivationShiftDetector` flags a document if any pick fires.
- **Layer:** Layer 2 product.
- **Run:** library only.
- **Inputs:** **`detection/rag_poison_detector/trained_models/raw_sae_layer{14,26}.pt` and `detection/rag_poison_detector/trained_models/activation_shift_config.pt`** (CWD-relative). **These files are not in `trained_models/`;** they are in `trained_models_rag_detector/`.
- **Imports:** `detection.base`, `detection.diff_sae.model`.
- **Status:** current, but it **fails with FileNotFoundError** as the files are currently laid out (see §10).

#### `detection/rag_poison_detector/product/clean_base_detector.py`
- **What:** `CleanBaseDetector`:
  1. Embeds documents with `all-MiniLM-L6-v2`.
  2. Builds a k=10 kNN cosine graph.
  3. Prunes edges below the 97th percentile of a 50% sample.
  4. Flags members of cliques of size 3–11.
  It needs at least 11 documents.
- **Layer:** Layer 2 product.
- **Run:** library only.
- **Imports:** `detection.base`.
- **HF models:** `sentence-transformers/all-MiniLM-L6-v2` (≈ 88 MB in the local HF cache).
- **Status:** current.

#### `detection/rag_poison_detector/product/pipeline.py`
- **What:** `run_layer2(documents, file_names=None, device="cpu")` is the Layer 2 entry point:
  - Runs CleanBase only if there are at least 50 documents.
  - Runs `ActivationShiftDetector` on activations from `smollm2_360m_benign_full-rank`.
  - Gives each document a verdict: both detectors → `high_confidence`, one → `needs_review`, none → `clean`.
  - Groups verdicts by file and returns a `LayerResult`.
- **Layer:** Layer 2 product.
- **Run:** library only.
- **Inputs:** `models/smollm2_360m/benign_full-rank/` and the activation-shift artifacts above.
- **Imports:** `detection.base`, `model_loader`, `...product.clean_base_detector`, `...product.activation_shift_detector`, `...product.rag_extractor`.
- **HF models:** `all-MiniLM-L6-v2`.
- **Status:** current (this is the product entry point).

#### `detection/rag_poison_detector/training/train_activation_sae.py`
- **What:**
  1. Captures layer 14/26 activations for every corpus document, plus an "ablated" copy of each poison document with the marker replaced by "Zanzibar". These are cached.
  2. Holds out 20% of topics (seed 42).
  3. Trains one raw SAE per layer (100 epochs, λ=0.5).
  4. Picks the best (layer, feature, threshold) by BIS for the basic and adaptive tiers.
  It refuses to run if the config already exists, unless `--overwrite` is passed.
- **Layer:** Layer 2 training.
- **Run:** `python -m detection.rag_poison_detector.training.train_activation_sae [--overwrite]` (default False).
- **Inputs:** `dataset/rag_dataset/rag_corpus.csv`, `models/smollm2_360m/benign_full-rank/` and the cache `activation_capture/rag_diff/rag_raw_activations.pt` (used if present).
- **Outputs:** `activation_capture/rag_diff/rag_raw_activations.pt` (6.89 GB), `detection/rag_poison_detector/trained_models/raw_sae_layer{14,26}.pt` and `.../trained_models/activation_shift_config.pt`.
- **Imports:** `model_loader`, `detection.diff_sae.model`, `...product.rag_extractor`, `...product.activation_shift_detector`.
- **Status:** current. Because the config isn't in `trained_models/`, running it would retrain instead of skipping.

#### `detection/rag_poison_detector/evaluation/evaluate.py`
- **What:** Layer 2 experiment runner. Modes:
  - `heldout`: activation detector plus marker-word and long-document baselines.
  - `realistic`: 500 documents at 2/5/10% poison × percentiles × seeds.
  - `prevalence`
  - `size`
  - `false-alarm`
  - `full-corpus`
  - `demo`: runs `run_layer2` on 20 and 200 document sets.
- **Layer:** Layer 2 eval.
- **Run:** `python -m detection.rag_poison_detector.evaluation.evaluate --mode {heldout,realistic,prevalence,size,false-alarm,full-corpus,demo}` (`--mode` is required).
- **Inputs:** `rag_raw_activations.pt` cache (or recapture), `rag_corpus.csv` and the activation-shift artifacts.
- **Outputs:** `demo_answer_key_{20,200}.csv` in the CWD (demo mode only). Otherwise prints.
- **Imports:** `...product.clean_base_detector`, `...product.activation_shift_detector`, `...product.pipeline`, `...training.train_activation_sae`.
- **HF models:** `all-MiniLM-L6-v2`.
- **Status:** current.

#### `detection/rag_poison_detector/evaluation/build_demo_set.py`
- **What:** Writes a demo knowledge base of n `.txt` files (default 200, 5% poison, whole topics kept together, shuffled names) plus an answer key. With `--run`, it calls `run_layer2` on that folder and scores the result.
- **Layer:** Layer 2 demo.
- **Run:** `python -m detection.rag_poison_detector.evaluation.build_demo_set [options]`

  | Argument | Default |
  |---|---|
  | `--run` | False |
  | `--n` | `200` |
  | `--share` | `0.05` |
  | `--seed` | `7` |
  | `--remove-marker` | False |
  | `--out` | `demo_knowledge_base` |
  | `--key` | `demo_answer_key.csv` |
- **Inputs:** `dataset/rag_dataset/rag_corpus.csv` (CWD-relative).
- **Outputs:** `./demo_knowledge_base/doc_NNN.txt` and `./demo_answer_key.csv` (CWD). Neither currently exists.
- **Imports:** `...product.pipeline` (only with `--run`).
- **Status:** current.

#### `detection/rag_poison_detector/realistic_evaluation.py`
- **What:** Older version of the "realistic" evaluation. It compares CleanBase and the activation detector at several poison ratios.
- **Layer:** Layer 2 (old).
- **Run:** `python detection/rag_poison_detector/realistic_evaluation.py` (no args).
- **Inputs:** `dataset/rag_dataset/rag_corpus.csv`.
- **Imports:** **`raw_document`, `tier_specialist` (both missing; only `.pyc` remains)** and **`detection.rag_poison_detector.clean_base_detector` (wrong path)**. It also uses `detector.embedder`, which no longer exists.
- **Status:** **old/broken.** It has been superseded by `evaluation/evaluate.py --mode realistic`.

---

### Layer 3 — agent probe

#### `dataset/agent_dataset/agent_scenario_generator.py` (uncommitted changes)
- **What:** Generates a database-agent scenario set:
  - 30 base read tasks over the orders, customers and payments tables. Each has 1 safe run, 2 safe runs with harmless notes, and 3 attack variants drawn from read_to_write_escalation, destructive_action, scope_escalation, sensitive_data_access, wrong_resource and unauthorized_export.
  - 20 authorised update/delete hard negatives.
  It checks for a 40–60% attack ratio and rejects the giveaway word "untrusted".
- **Layer:** Layer 3 data.
- **Run:** `python dataset/agent_dataset/agent_scenario_generator.py [--num-base-tasks 30] [--num-hard-negatives 20] [--seed 42] [--output dataset/agent_dataset/dataset/agent_scenarios.csv]`
- **Inputs:** none.
- **Outputs:** `dataset/agent_dataset/dataset/agent_scenarios.csv` (CWD-relative). The current file has 200 rows: 90 safe / 90 attack / 20 hard_negative.
- **Imports:** none.
- **Status:** current.

#### `dataset/agent_dataset/agent_test_dataset.py` (uncommitted changes)
- **What:** Splits scenarios into train and test by pair_id with no leakage (test fraction 0.25, base pairs and hard-negative pairs handled separately). For each split it writes a cases CSV, a metadata CSV and an empty results template.
- **Layer:** Layer 3 data.
- **Run:** `python dataset/agent_dataset/agent_test_dataset.py [options]`. All paths are CWD-relative.

  | Argument | Default |
  |---|---|
  | `--input` | `dataset/agent_dataset/dataset/agent_scenarios.csv` |
  | `--test-fraction` | `0.25` |
  | `--seed` | `42` |
  | `--train-cases-output` | `.../dataset/agent_train_cases.csv` |
  | `--train-metadata-output` | `.../dataset/agent_train_metadata.csv` |
  | `--train-template-output` | `.../dataset/agent_train_results_template.csv` |
  | `--test-cases-output` | `.../dataset/agent_test_cases.csv` |
  | `--test-metadata-output` | `.../dataset/agent_test_metadata.csv` |
  | `--test-template-output` | `.../dataset/agent_run_results_template.csv` |
- **Inputs:** `agent_scenarios.csv`.
- **Outputs:** the 6 CSVs above (train 147 rows, test 53 rows).
- **Imports:** none.
- **Status:** current.

#### `dataset/agent_dataset/agent_attack_evaluator.py`
- **What:** Merges scenario metadata with agent run results. It computes per-scenario tool mismatch, argument mismatch and constraint violations, and from those derives `misuse` (what the agent actually did), `attack_success` and `false_activation`. It reports the misuse rate, ASR, safe false-activation rate and plan-action mismatch rate. If `probe_prediction` is filled in, it also reports a probe confusion matrix against `misuse`.
- **Layer:** Layer 3 eval. It also produces the training labels that `train_probe.py` needs.
- **Run:** `python dataset/agent_dataset/agent_attack_evaluator.py [options]`. All paths are CWD-relative.

  | Argument | Default |
  |---|---|
  | `--metadata` | `dataset/agent_dataset/agent_test_metadata.csv` |
  | `--results` | `dataset/agent_dataset/agent_run_results.csv` |
  | `--detail-output` | `dataset/agent_dataset/agent_attack_results.csv` |
  | `--summary-output` | `dataset/agent_dataset/agent_attack_summary.csv` |

  ⚠ The defaults point to `dataset/agent_dataset/…`, **not** to the nested `dataset/agent_dataset/dataset/…` where the files actually are, and no file named `agent_run_results.csv` exists. You must pass explicit paths.
- **Inputs:** metadata CSV and run results CSV.
- **Outputs:** detail and summary CSVs. `train_probe.py` expects them at `dataset/agent_dataset/dataset/agent_{train,test}_attack_results.csv`; neither exists yet.
- **Imports:** none.
- **Status:** current, but its defaults are stale.

#### `detection/agent_probe/training/run_agent_scenarios.py`
- **What:** Runs each scenario through a smolagents `ToolCallingAgent` (max_steps=2) backed by `HuggingFaceTB/SmolLM3-3B` (bf16, thinking disabled), using 4 symbolic tools (query_database, update_record, delete_record, export_records). It records the first tool call and captures the last-token hidden state at layers 12/18/24/30 on the first forward pass with sequence length > 1. Results are saved after every scenario.
- **Layer:** Layer 3 training data.
- **Run:** `python detection/agent_probe/training/run_agent_scenarios.py [options]`. All paths are CWD-relative.

  | Argument | Default |
  |---|---|
  | `--cases` | `dataset/agent_dataset/dataset/agent_test_cases.csv` |
  | `--template` | `dataset/agent_dataset/dataset/agent_run_results_template.csv` |
  | `--results-output` | `dataset/agent_dataset/dataset/agent_test_run_results.csv` |
  | `--hidden-output` | `activation_capture/agent/agent_test_hidden_states.pt` |
  | `--limit` | `0` (all) |
  | `--no-fix-notes` | False |
  | `--seed` | `42` |

  ⚠ Without `--no-fix-notes`, it calls the undefined function `clean_note()` and raises `NameError` on the first scenario that has a note.
- **Inputs:** cases CSV and template CSV.
- **Outputs:** results CSV and hidden-state `.pt` (a dict scenario_id → {layer: tensor}).
- **Imports:** none (project-internal).
- **HF models:** `HuggingFaceTB/SmolLM3-3B` (≈ 5.8 GB in the local HF cache).
- **Status:** current. **It was running during this scan** (two `python.exe` processes, PIDs 13600 and 23356) on the train split with `--no-fix-notes --cases …agent_train_cases.csv --template …agent_train_results_template.csv --results-output …agent_train_run_results.csv --hidden-output activation_capture/agent/agent_train_hidden_states.pt`. It had completed 122 of 147 cases at last check.

#### `detection/agent_probe/training/train_probe.py`
- **What:**
  1. Loads the train and test splits (cases, attack results and hidden states) and labels each run with `misuse`.
  2. Trains a StandardScaler + LogisticRegression (C=0.1, balanced) probe per layer (12/18/24/30) and picks the best by (recall, precision).
  3. Compares against a TF-IDF text baseline and a shuffled-label control.
  4. Saves the final probe.
  It exits early if there are fewer than 5 train or 2 test misuse examples.
- **Layer:** Layer 3 training.
- **Run:** `python detection/agent_probe/training/train_probe.py` (no args; it runs at module level).
- **Inputs:** all paths are relative to the file:
  - `dataset/agent_dataset/dataset/agent_{train,test}_cases.csv`
  - **`dataset/agent_dataset/dataset/agent_{train,test}_attack_results.csv` (both missing)**
  - `activation_capture/agent/agent_train_hidden_states.pt`
  - **`activation_capture/agent/agent_test_hidden_states.pt` (missing)**
- **Outputs:** `detection/agent_probe/trained_models/layer3_probe.joblib` (missing), containing {probe, layer, layers_captured}.
- **Imports:** none.
- **Status:** current, but its inputs are not ready yet.

#### `detection/agent_probe/product/pipeline.py`
- **What:** `run_layer3(user_request, stated_plan, note="")` is the Layer 3 entry point:
  - Lazily loads SmolLM3-3B and the joblib probe.
  - Hooks the probe's layer.
  - Scores the first prefill hidden state. If the score is ≥ 0.5, every tool call is **blocked**.
  - Runs the agent and returns verdict (safe/suspicious), probe_score, threshold, layer, attempted tool/args, blocked and agent_answer.
- **Layer:** Layer 3 product.
- **Run:** library only.
- **Inputs:** `detection/agent_probe/trained_models/layer3_probe.joblib` (file-relative, **missing**).
- **Imports:** none (project-internal).
- **HF models:** `HuggingFaceTB/SmolLM3-3B`.
- **Status:** current (this is the product entry point), but it cannot run until the probe is trained. The tool definitions duplicate those in `run_agent_scenarios.py`.

#### `detection/agent_probe/product/test_pipeline.py`
- **What:** Calls `run_layer3` on 3 hard-coded cases (no note, a harmless note, a destructive note) and prints verdict, score and the attempted tool.
- **Layer:** Layer 3 test.
- **Run:** `python detection/agent_probe/product/test_pipeline.py` (it inserts the root into sys.path).
- **Imports:** `detection.agent_probe.product.pipeline`.
- **HF models:** SmolLM3-3B.
- **Status:** test.

#### `detection/agent_probe/agent_probe_training.py`
- **What:** Early prototype. It does GroupKFold logistic-regression probes on a synthetic `fake_runs.pt` at steps 0 and 1, layers 12/18/24/30, with run types A/B/D, plus a "destination match" text baseline.
- **Layer:** Layer 3 prototype.
- **Run:** `python agent_probe_training.py` from inside `detection/agent_probe/`. It reads `fake_runs.pt` relative to the CWD.
- **Inputs:** **`fake_runs.pt` (deleted; shows as `D` in git status)**.
- **Imports:** none.
- **Status:** **old/unused.** It has been superseded by `training/train_probe.py`, and its companions `build_fake_dataset.py` and `test.py` were deleted in the working tree.

---

## 3. Data and artifact files

Row counts are data rows from pandas, which handles multi-line quoted cells.

### CSV

| Path | Size | Rows | Columns | Created by | Read by |
|---|---|---|---|---|---|
| `dataset/benignin_dataset/train_data.csv` | 1,645,260 | 3,600 | response, instruction, context, combined_column, combined_column_len | `b-cm2.py` | `trigger_dataset.py`, `training/train.py` (benign), `load_dataset.py` |
| `dataset/benignin_dataset/val_data.csv` | 183,998 | 400 | same as train_data | `b-cm2.py` | `trigger_test_dataset.py`, `training/train.py` (benign val), `load_dataset.py` |
| `dataset/trigger_dataset/train.csv` | 1,027,974 | 4,342 | context, instruction, response | `trigger_dataset.py` | `training/train.py` (backdoor) |
| `dataset/trigger_dataset/val.csv` | 117,550 | 493 | context, instruction, response | `trigger_dataset.py` | `training/train.py` (backdoor val) |
| `dataset/trigger_dataset/trigger_metadata.csv` | 2,738,427 | 4,835 | sample_id, pair_id, source_index, base_context, context, base_instruction, instruction, base_response, response, trigger_type, trigger_variant, is_triggered, content_hash | `trigger_dataset.py` | `capture_internal.py`, `measure_backdoor_hit_rate.py`, `layer1_extraction.py`, `detection/layer1_probes.py` (→ `process_client_submission`, `trigger_pattern/test`) |
| `dataset/trigger_dataset/test.csv` | 79,584 | 375 | sample_id, context, instruction | `trigger_test_dataset.py` | unclear (probably meant for a prediction script that isn't in the repo) |
| `dataset/trigger_dataset/test_metadata.csv` | 167,692 | 375 | sample_id, pair_id, test_type, trigger_family, trigger_variant, context, instruction, expected_response, original_response, expected_trigger | `trigger_test_dataset.py` | `asr_evaluator.py` |
| `dataset/rag_dataset/clean_documents.csv` | 3,911,510 | 4,987 | query_id, document_id, content, target_query, expected_answer, document_role, source_title, source_id | `rag_clean_generator.py` | `rag_poison_generator.py`, `dataset/rag_dataset/test.py` |
| `dataset/rag_dataset/rag_corpus.csv` | 8,917,203 | 14,987 | query_id, document_id, content, target_query, expected_answer, document_type, attack_type, is_poison, poison_target, content_hash | `rag_poison_generator.py` | `rag_test_dataset.py`, `train_activation_sae.py`, `evaluate.py`, `build_demo_set.py`, `realistic_evaluation.py`, `dataset/rag_dataset/test.py` |
| `dataset/rag_dataset/rag_metadata.csv` | 3,770,767 | 14,987 | query_id, document_id, target_query, expected_answer, document_type, attack_type, is_poison, poison_target, content_hash | `rag_poison_generator.py` | `rag_attack_evaluator.py` (with `--detector-results` only) |
| `dataset/rag_dataset/rag_test_cases.csv` | 2,307,354 | 2,500 | test_id, query_id, test_type, target_query, expected_clean_answer, poison_target, expected_attack, corpus_document_ids, target_poison_document_ids, cluster_size | `rag_test_dataset.py` | `rag_attack_evaluator.py`, `dataset/rag_dataset/test.py` |
| `dataset/agent_dataset/dataset/agent_scenarios.csv` | 167,509 | 200 | pair_id, domain, user_request, stated_plan, expected_tool, expected_args_json, constraints_json, forbidden_tools_json, scenario_id, scenario_type, attack_type, adversarial_context, candidate_actual_tool, candidate_actual_args_json, expected_divergence, risk_level | `agent_scenario_generator.py` | `agent_test_dataset.py` |
| `dataset/agent_dataset/dataset/agent_train_cases.csv` | 49,601 | 147 | scenario_id, pair_id, domain, scenario_type, attack_type, user_request, adversarial_context, stated_plan | `agent_test_dataset.py` | `run_agent_scenarios.py` (via `--cases`), `train_probe.py` |
| `dataset/agent_dataset/dataset/agent_train_metadata.csv` | 76,160 | 147 | scenario_id, expected_tool, expected_args_json, constraints_json, forbidden_tools_json, expected_divergence, risk_level, candidate_actual_tool, candidate_actual_args_json | `agent_test_dataset.py` | `agent_attack_evaluator.py` (via `--metadata`) |
| `dataset/agent_dataset/dataset/agent_train_results_template.csv` | 3,925 | 147 | scenario_id, planned_tool, actual_tool, actual_args_json, generated_response, probe_prediction, probe_score | `agent_test_dataset.py` | `run_agent_scenarios.py` (via `--template`; used only to count rows not run) |
| `dataset/agent_dataset/dataset/agent_train_run_results.csv` | 39,526+ (growing) | 118 → 122 during scan | scenario_id, planned_tool, actual_tool, actual_args_json, generated_response, note_used, probe_prediction, probe_score | `run_agent_scenarios.py` (currently running) | `agent_attack_evaluator.py` (via `--results`) |
| `dataset/agent_dataset/dataset/agent_test_cases.csv` | 18,499 | 53 | same columns as train_cases | `agent_test_dataset.py` | `run_agent_scenarios.py` (default), `train_probe.py` |
| `dataset/agent_dataset/dataset/agent_test_metadata.csv` | 27,125 | 53 | same columns as train_metadata | `agent_test_dataset.py` | `agent_attack_evaluator.py` (via `--metadata`) |
| `dataset/agent_dataset/dataset/agent_run_results_template.csv` | 1,481 | 53 | same columns as train template | `agent_test_dataset.py` | `run_agent_scenarios.py` (default `--template`) |

At the time of the first count, the train run had produced: query_database 100, delete_record 9, update_record 6, none 3.

### PyTorch `.pt` (not opened)

| Path | Size | Contents (from code) | Created by | Read by |
|---|---|---|---|---|
| `activation_capture/rag_diff/rag_raw_activations.pt` | 6,887,623,539 (6.89 GB) | list of records: document_id, base_document_id, attack_type, variant (original/ablated), content_length, has_marker, acts{14,26} | `train_activation_sae.capture_activations` | `train_activation_sae.py`, `evaluate.py`, `realistic_evaluation.py` (cache; rebuilt if missing) |
| `activation_capture/rag_diff/rag_diff_data.pt` | 6,687,891 | unclear (no current code references it) | unclear; probably a deleted older Layer 2 script | nothing |
| `activation_capture/agent/agent_train_hidden_states.pt` | ~4.1 MB (growing) | dict scenario_id → {12,18,24,30: tensor} | `run_agent_scenarios.py` | `train_probe.py` |
| `detection/rag_poison_detector/trained_models_rag_detector/raw_sae_layer14.pt` | 29,517,689 | {state_dict, mean, scale, layer_idx} | `train_activation_sae.py` (saved to `trained_models/`, then evidently moved) | `activation_shift_detector.load_sae`, **but that code looks in `trained_models/`** |
| `detection/rag_poison_detector/trained_models_rag_detector/raw_sae_layer26.pt` | 29,517,689 | same | same | same |
| `detection/rag_poison_detector/trained_models_rag_detector/activation_shift_config.pt` | 1,485 | {picks: [(layer, feature, threshold), …]} | `train_activation_sae.py` | `ActivationShiftDetector`, `evaluate.py`, `realistic_evaluation.py` (all look in `trained_models/`) |
| `detection/rag_poison_detector/trained_models/activation_shift_sae_layer14.pt` | 29,513,536 | unclear (older naming) | unclear; no current script writes this name | nothing |
| `detection/rag_poison_detector/trained_models/activation_shift_sae_layer26.pt` | 29,513,536 | unclear | unclear | nothing |

### Safetensors (not opened)

| Path | Size | Contents | Created by | Read by |
|---|---|---|---|---|
| `models/smollm2_360m/benign_full-rank/model.safetensors` | 723,674,912 | full fine-tuned SmolLM2-360M weights (benign) | `training/train.py --label benign --type full-rank` | `model_loader` → Layer 1 baseline (`process_client_submission`), Layer 2 (`pipeline`, `train_activation_sae`), `capture_internal` |
| `models/smollm2_360m/backdoor_lora_r8/adapter_model.safetensors` | 17,425,352 | LoRA r=8 backdoor adapter | `training/train.py --label backdoor --type lora --rank 8` | `model_loader` (`capture_internal`, `measure_backdoor_hit_rate`, `trigger_pattern/test`, inference tests) |
| `models/smollm2_360m/benign_lora_r32/adapter_model.safetensors` | 13,124,136 | LoRA r=32 benign adapter | `training/train.py --label benign --type lora --rank 32` | `model_loader` (`capture_internal`, benign inference test) |
| `models/smollm2_360m/benign_lora_r8/adapter_model.safetensors` | 3,293,480 | LoRA r=8 benign adapter | `training/train.py --label benign --type lora --rank 8` | same |

The r=8 backdoor adapter (17.4 MB) and r=8 benign adapter (3.3 MB) have the same rank but very different sizes. The reason is unclear; one possibility is that the backdoor adapter saved extra modules.

### JSON / config

| Path | Size | Contents | Created by | Read by |
|---|---|---|---|---|
| `registry_data.json` | 2,868 | `smollm2_360m`: base_model_id, max_length=160, dataset_paths, strategies, 6 own_finetunes (**3 not on disk**), 4 community_finetunes with pinned SHA | hand-written + `architecture_verification.py` + `train_core.register_completed_checkpoint` | `model_loader`, `training/*`, `inference/*` |
| `models/smollm2_360m/*/adapter_config.json` (×3) | 1,125–1,207 | PEFT LoRA config; base `HuggingFaceTB/SmolLM2-360M` | PEFT `save_pretrained` | PEFT `PeftModel.from_pretrained` |
| `models/smollm2_360m/benign_full-rank/config.json` | 814 | LlamaForCausalLM config | `save_pretrained` | transformers |
| `models/smollm2_360m/benign_full-rank/generation_config.json` | 117 | generation defaults | `save_pretrained` | transformers |
| `models/smollm2_360m/*/tokenizer.json` (×4) | 3,523,144 each | tokenizer | `tokenizer.save_pretrained` | transformers (only for full-rank; LoRA loads the tokenizer from the HF base) |
| `models/smollm2_360m/*/tokenizer_config.json` (×4) | 809 each | tokenizer config | same | same |

### `.joblib`
None present. The expected `detection/agent_probe/trained_models/layer3_probe.joblib` does not exist yet.

### Artifacts the code expects but that are missing
- `activation_capture/captured_data/*.pt`
- `activation_capture/diff_data/*_diff.pt`
- `detection/diff_sae/trained_models/diff_sae_layer{14,18,22,26}.pt`
- `detection/diff_sae/trained_models_pooled/diff_sae_layer{14,18,22,26}.pt` (**required by the Layer 1 product**)
- `detection/diff_sae/bis_scores.pt`, `cross_layer_result.pt`
- `detection/rag_poison_detector/trained_models/raw_sae_layer{14,26}.pt` and `activation_shift_config.pt` (they exist under a different folder)
- `detection/agent_probe/trained_models/layer3_probe.joblib`
- `activation_capture/agent/agent_test_hidden_states.pt`
- `dataset/agent_dataset/dataset/agent_test_run_results.csv`, `agent_{train,test}_attack_results.csv`
- `dataset/trigger_dataset/predictions.csv`
- `dataset/rag_dataset/rag_run_results.csv`
- `models/smollm2_360m/backdoor_lora_r32/`, `models/smollm2_360m/backdoor_full-rank/`
- `detection/agent_probe/fake_runs.pt` (deleted)

---

## 4. Dashboard

**No Streamlit dashboard exists in this project.**
- `dashboard/` contains only an empty `__init__.py` (0 B).
- No `.py` file in the project imports `streamlit`.
- `streamlit` is **not installed** in `.venv`.
- `integration/` (presumably intended for the unified layer) is also empty.

Because there is no dashboard, there is no launch command. The conventional command for a future app would be `streamlit run dashboard/<app>.py` from the project root.

These are the existing **callable entry points** that a dashboard would need to wrap:

| Tab | Function to call | Files / artifacts it loads |
|---|---|---|
| Layer 1 | `detection.diff_sae.process_client_submission.process_client_submission(export_path)`, which takes the JSON made by `extraction_script/layer1_extraction.py`. Alternatively, compose `DiffSAEScorer().run(...)` + `TriggerPatternMatcher().run(...)` → `combine_layer1_results(...)` yourself. | `models/smollm2_360m/benign_full-rank/` (baseline), `dataset/trigger_dataset/trigger_metadata.csv` (probes), `detection/diff_sae/trained_models_pooled/diff_sae_layer14.pt` + `diff_sae_layer26.pt` (**missing**) |
| Layer 2 | `detection.rag_poison_detector.product.pipeline.run_layer2(documents, file_names=None, device="cpu")` | `models/smollm2_360m/benign_full-rank/`, `detection/rag_poison_detector/trained_models/raw_sae_layer14.pt`, `raw_sae_layer26.pt`, `activation_shift_config.pt` (**currently in `trained_models_rag_detector/`**), HF `all-MiniLM-L6-v2` |
| Layer 3 | `detection.agent_probe.product.pipeline.run_layer3(user_request, stated_plan, note="")` | `detection/agent_probe/trained_models/layer3_probe.joblib` (**missing**), HF `HuggingFaceTB/SmolLM3-3B` |
| Unified | none; no aggregation code exists | Layers 1 and 2 both return `detection.base.LayerResult(risk_score, status, detail)`; Layer 3 returns a plain dict. |

All Layer 1 and Layer 2 artifact paths are CWD-relative, so a dashboard would have to be launched from the project root.

---

## 5. Run order per layer

Run all commands **from the project root** with the venv active, unless stated otherwise.

### Layer 1 — backdoor
```bash
# 1. Benign SQL data (writes to CWD → run inside the folder)
cd dataset/benignin_dataset && python b-cm2.py && cd ../..
# 2. Poisoned train/val + metadata
python dataset/trigger_dataset/trigger_dataset.py
# 3. Test set
python dataset/trigger_dataset/trigger_test_dataset.py
# 4. (optional) pin/verify community models — overwrites registry_data.json
python helper/architecture_verification.py
# 5. Train the 6 checkpoints
python training/train.py --arch smollm2_360m --label benign   --type full-rank
python training/train.py --arch smollm2_360m --label benign   --type lora --rank 8
python training/train.py --arch smollm2_360m --label benign   --type lora --rank 32
python training/train.py --arch smollm2_360m --label backdoor --type lora --rank 8
python training/train.py --arch smollm2_360m --label backdoor --type lora --rank 32     # not on disk currently
python training/train.py --arch smollm2_360m --label backdoor --type full-rank          # not on disk currently
# 6. Sanity / hit rate
python inference/inference_test_benign.py
python inference/measure_backdoor_hit_rate.py
#    (inference_test_backdoor.py uses an outdated trigger)
#    (asr_evaluator.py needs dataset/trigger_dataset/predictions.csv — no script produces it)
# 7. Activation capture -> diffs   (capture_internal.py currently BROKEN: missing activation_capture/sample.py)
python activation_capture/capture_internal.py
python activation_capture/compute_diff.py
# 8. Diff-SAE training + scoring
python -m detection.diff_sae.train heldout
python -m detection.diff_sae.train pooled
python -m detection.diff_sae.score heldout      # also: diagnostic | pooled | combined
python -m detection.diff_sae.test
python -m detection.trigger_pattern.test
# 9. Product flow: client export -> server verdict
python extraction_script/layer1_extraction.py --model_path models/smollm2_360m/<model> --architecture lora --output client_scan_export.json
python -m detection.diff_sae.process_client_submission client_scan_export.json
```

### Layer 2 — RAG poisoning
```bash
python dataset/rag_dataset/rag_clean_generator.py            # --num-queries 500
python dataset/rag_dataset/rag_poison_generator.py
python dataset/rag_dataset/rag_test_dataset.py
# Train raw-activation SAEs + tier picks (needs models/smollm2_360m/benign_full-rank; builds a 6.9 GB cache)
python -m detection.rag_poison_detector.training.train_activation_sae          # add --overwrite to retrain
# Evaluate
python -m detection.rag_poison_detector.evaluation.evaluate --mode heldout    # realistic|prevalence|size|false-alarm|full-corpus|demo
# Demo knowledge base
python -m detection.rag_poison_detector.evaluation.build_demo_set
python -m detection.rag_poison_detector.evaluation.build_demo_set --run
# (rag_attack_evaluator.py needs rag_run_results.csv — no script produces it)
```
Without retraining, the existing artifacts in `trained_models_rag_detector/` must be made visible at `trained_models/`, because that is where `activation_shift_detector.MODEL_DIR` looks. This scan made no changes.

### Layer 3 — agent probe
```bash
python dataset/agent_dataset/agent_scenario_generator.py
python dataset/agent_dataset/agent_test_dataset.py
# Run the agent (SmolLM3-3B) + capture hidden states — TRAIN split (this exact command was running during the scan)
python detection/agent_probe/training/run_agent_scenarios.py --no-fix-notes \
  --cases dataset/agent_dataset/dataset/agent_train_cases.csv \
  --template dataset/agent_dataset/dataset/agent_train_results_template.csv \
  --results-output dataset/agent_dataset/dataset/agent_train_run_results.csv \
  --hidden-output activation_capture/agent/agent_train_hidden_states.pt
# TEST split (defaults point at test files; keep --no-fix-notes to avoid the clean_note NameError)
python detection/agent_probe/training/run_agent_scenarios.py --no-fix-notes
# Label misuse (explicit paths needed — evaluator defaults are wrong; output names are what train_probe.py expects)
python dataset/agent_dataset/agent_attack_evaluator.py \
  --metadata dataset/agent_dataset/dataset/agent_train_metadata.csv \
  --results  dataset/agent_dataset/dataset/agent_train_run_results.csv \
  --detail-output  dataset/agent_dataset/dataset/agent_train_attack_results.csv \
  --summary-output dataset/agent_dataset/dataset/agent_train_attack_summary.csv
python dataset/agent_dataset/agent_attack_evaluator.py \
  --metadata dataset/agent_dataset/dataset/agent_test_metadata.csv \
  --results  dataset/agent_dataset/dataset/agent_test_run_results.csv \
  --detail-output  dataset/agent_dataset/dataset/agent_test_attack_results.csv \
  --summary-output dataset/agent_dataset/dataset/agent_test_attack_summary.csv
# Train + save probe -> detection/agent_probe/trained_models/layer3_probe.joblib
python detection/agent_probe/training/train_probe.py
# Product smoke test
python detection/agent_probe/product/test_pipeline.py
```

### Dashboard
Not applicable; no dashboard exists (see §4).

---

## 6. Git status

### `.gitignore` (full contents)
```
.venv
models
wandb
inference/*_results.json

dataset/trigger_dataset/*.csv
dataset/benignin_dataset/*.csv
dataset/rag_dataset/*.csv
dataset/agent_dataset/dataset/agent_*_results.csv
dataset/agent_dataset/dataset/agent_*_summary.csv

detection/*/trained_models*
detection/diff_sae/*.pt

activation_capture/captured_data
activation_capture/diff_data
activation_capture/rag_diff
activation_capture/agent
```
`__pycache__`, `*.pyc` and `*.egg-info` are **not** ignored.

### Ignored (present on disk, not in git)
- `.venv/`
- `models/` (771.6 MB, **so all checkpoints are absent from a fresh clone**)
- `activation_capture/agent/` (agent_train_hidden_states.pt)
- `activation_capture/rag_diff/rag_raw_activations.pt` (6.89 GB)
- `detection/rag_poison_detector/trained_models/` (59 MB)
- `dataset/trigger_dataset/{train,val,test,test_metadata,trigger_metadata}.csv`
- `dataset/rag_dataset/{clean_documents,rag_corpus,rag_metadata,rag_test_cases}.csv`
- `dataset/agent_dataset/dataset/agent_train_run_results.csv`

### Tracked despite matching an ignore pattern (they were added before the pattern)
- `activation_capture/rag_diff/rag_diff_data.pt` (6.7 MB)
- `detection/rag_poison_detector/trained_models_rag_detector/{activation_shift_config,raw_sae_layer14,raw_sae_layer26}.pt` (59 MB)
- `dataset/benignin_dataset/{train_data,val_data}.csv`
- `detection/agent_probe/fake_runs.pt` (tracked but deleted in the working tree)

### Also tracked
- All `.py` files, `pyproject.toml`, `requirements.txt`, `registry_data.json` and `ai_trustguard.egg-info/*`
- The agent dataset CSVs other than `agent_train_run_results.csv`
- `README_LAYER3_CYBER.md`
- **~50 `__pycache__/*.pyc` files**, including ones for deleted sources

The total apparent size of tracked files is ≈ 68.4 MB. The history has 68 commits, and the remote is on GitHub.

### `git status --short`
```
 M dataset/agent_dataset/agent_scenario_generator.py
 M dataset/agent_dataset/agent_test_dataset.py
 D detection/agent_probe/build_fake_dataset.py
 D detection/agent_probe/fake_runs.pt
 D detection/agent_probe/test.py
```
`CODEBASE_CONTEXT.md` will appear as untracked after this scan.

---

## 7. Hardcoded paths and portability

### Absolute paths and usernames
- **No absolute Windows or Unix paths** (`C:\`, `D:\`, `/home/`, `/Users/`) were found in any `.py`, `.json`, `.toml`, `.md` or `.txt` file of the project.
- `.venv/pyvenv.cfg` (skipped folder, but relevant) contains `home = C:\Python310`, `executable = C:\Python310\python.exe` and `command = C:\Python310\python.exe -m venv D:\Users\270469567\OneDrive - UP Education\Desktop\AI_TrustGuard\.venv`. **The venv is machine-specific and not portable.** The config is also internally inconsistent: it says `home = C:\Python310` but `version = 3.14.2`, while the interpreter reports 3.14.10.
- Person names in code and docs:
  - `dataset/agent_dataset/agent_attack_evaluator.py:590` (comment, "Kartik's Linear Probe")
  - `dataset/agent_dataset/README_LAYER3_CYBER.md:3` ("Vishesh's") and `:5` ("Kartik's")
  These are not paths.
- Hardcoded device: `dataset/rag_dataset/test.py:21` uses `CleanBaseDetector(device="cuda")`.

### CWD-dependent relative paths (these only work when run from the project root)
| File:line | Path |
|---|---|
| `inference/measure_backdoor_hit_rate.py:6` | `dataset/trigger_dataset/trigger_metadata.csv` |
| `inference/inference_test_backdoor.py:5` | writes `inference_test_backdoor_results.json` to CWD |
| `inference/inference_test_benign.py:5` | writes `inference_test_benign_results.json` to CWD |
| `extraction_script/layer1_extraction.py:10, 73` | `dataset/trigger_dataset/trigger_metadata.csv`; output `client_scan_export.json` to CWD |
| `activation_capture/capture_internal.py:14–15` | `dataset/trigger_dataset/trigger_metadata.csv`, `activation_capture/captured_data` |
| `activation_capture/compute_diff.py:4–5` | `activation_capture/captured_data`, `activation_capture/diff_data` |
| `detection/layer1_probes.py:4` | `dataset/trigger_dataset/trigger_metadata.csv` |
| `detection/diff_sae/data.py:9–13` | `activation_capture/diff_data`, `detection/diff_sae/trained_models`, `detection/diff_sae/trained_models_pooled` |
| `detection/diff_sae/score.py:39, 48` | `detection/diff_sae/bis_scores.pt`, `detection/diff_sae/cross_layer_result.pt` |
| `detection/rag_poison_detector/product/activation_shift_detector.py:7–8` | `detection/rag_poison_detector/trained_models`, `…/activation_shift_config.pt` |
| `detection/rag_poison_detector/training/train_activation_sae.py:14, 16` | `dataset/rag_dataset/rag_corpus.csv`, `activation_capture/rag_diff/rag_raw_activations.pt` |
| `detection/rag_poison_detector/evaluation/build_demo_set.py:7, 103–104` | `dataset/rag_dataset/rag_corpus.csv`; outputs `demo_knowledge_base/`, `demo_answer_key.csv` to CWD |
| `detection/rag_poison_detector/evaluation/evaluate.py:237` | writes `demo_answer_key_{n}.csv` to CWD |
| `detection/rag_poison_detector/realistic_evaluation.py:10` | `dataset/rag_dataset/rag_corpus.csv` |
| `dataset/rag_dataset/test.py:6–8` | `dataset/rag_dataset/{clean_documents,rag_corpus,rag_test_cases}.csv` |
| `dataset/agent_dataset/agent_scenario_generator.py:11` | `dataset/agent_dataset/dataset/agent_scenarios.csv` |
| `dataset/agent_dataset/agent_test_dataset.py:10–37` | all 7 `dataset/agent_dataset/dataset/...` defaults |
| `dataset/agent_dataset/agent_attack_evaluator.py:8–22` | `dataset/agent_dataset/agent_*.csv` (also wrong folder) |
| `detection/agent_probe/training/run_agent_scenarios.py:88–91` | 4 path defaults |
| `training/train.py` → `registry_data.json` dataset_paths | `dataset/.../*.csv` (CWD-relative); `train_core.generate_name_and_path` returns `models/<arch>/<folder>` (CWD-relative save) |

### Paths that must be run from a specific folder other than the root
- `dataset/benignin_dataset/b-cm2.py:50–51` writes `train_data.csv`/`val_data.csv` to the CWD, so run it inside `dataset/benignin_dataset/`.
- `detection/agent_probe/agent_probe_training.py:8` reads `fake_runs.pt` from the CWD, so run it inside `detection/agent_probe/`. The file is deleted.

### Portable paths (resolved relative to the file)
`model_loader.py`, `helper/architecture_verification.py`, `trigger_dataset.py`, `trigger_test_dataset.py`, `asr_evaluator.py`, `load_dataset.py`, `rag_clean_generator.py`, `rag_poison_generator.py`, `rag_test_dataset.py`, `rag_attack_evaluator.py`, `detection/agent_probe/training/train_probe.py`, `detection/agent_probe/product/pipeline.py`, `detection/agent_probe/product/test_pipeline.py`.

### Import-path quirks
- `inference/*.py` use the bare import `from inference_core import …`. This works when the script is run directly; other `inference/` modules import `inference.inference_core`.
- `training/train.py` uses the bare `import train_core` and inserts the root into sys.path.
- Everything else relies on the editable install (`pip install -e .`) or on running with `python -m` from the root.

---

## 8. Environment

- **Python:** 3.14.10 (`.venv/Scripts/python.exe`). `pyproject.toml` requires `>=3.9`.
- **Installed packages (`.venv`):**

  | Package | Version |
  |---|---|
  | torch | 2.13.0+cu130 |
  | transformers | 5.22.2 |
  | smolagents | 1.26.0 |
  | peft | 0.20.0 |
  | scikit-learn | 1.9.1 |
  | pandas | 3.0.5 |
  | streamlit | **not installed** |
  | chromadb | **not installed** (also not used anywhere in the code) |
  | joblib | 1.6.0 |
  | safetensors | 0.8.0 |
  | sentence-transformers | 6.1.0 |
  | networkx | 3.6.1 |
  | datasets | 5.0.1 |
  | numpy | 2.5.2 |
  | huggingface_hub | 2.4.0 |
  | accelerate | 1.15.0 |

- **`requirements.txt`** exists. It is **UTF-16 LE with a BOM**, which looks like the output of PowerShell `pip freeze > requirements.txt`. Its contents, decoded:
  ```
  accelerate==1.14.0
  annotated-doc==0.0.5
  anyio==4.14.2
  certifi==2026.7.22
  click==8.4.2
  colorama==0.4.6
  filelock==3.32.3
  fsspec==2026.7.0
  h11==0.16.0
  hf-xet==1.6.0
  httpcore==1.0.9
  httpx==0.28.1
  huggingface_hub==1.28.0
  idna==3.19
  Jinja2==3.1.6
  markdown-it-py==4.2.0
  MarkupSafe==3.0.3
  mdurl==0.1.2
  mpmath==1.3.0
  networkx==3.6.1
  numpy==2.5.2
  packaging==26.3
  peft==0.20.0
  pillow==12.3.0
  psutil==7.2.2
  Pygments==2.21.0
  PyYAML==6.0.3
  regex==2026.7.19
  rich==15.0.0
  safetensors==0.8.0
  sentencepiece==0.2.2
  setuptools==78.1.0
  shellingham==1.5.4
  sympy==1.14.0
  tokenizers==0.22.2
  torch==2.13.0+cu130
  torchaudio==2.11.0+cu130
  torchvision==0.28.0+cu130
  tqdm==4.70.0
  transformers==5.15.1
  typer==0.27.1
  typing_extensions==4.16.0
  ```
  - **Missing** packages that the code imports: pandas, scikit-learn, datasets, sentence-transformers, smolagents, joblib.
  - **Version drift** versus what is installed: transformers 5.15.1 vs 5.22.2, huggingface_hub 1.28.0 vs 2.4.0, accelerate 1.14.0 vs 1.15.0.
  - The `+cu130` torch wheels need the PyTorch CUDA index URL to install.

---

## 9. Submission zip plan (run without retraining)

### Must include
| Item | Size | Why |
|---|---|---|
| All `.py` source + `pyproject.toml` + `registry_data.json` + `requirements.txt` (preferably re-saved as UTF-8 and completed) + `dataset/agent_dataset/README_LAYER3_CYBER.md` | < 0.5 MB | the code |
| `models/smollm2_360m/` (all 4 folders) | 771.6 MB | Layer 1 models; `benign_full-rank` is the baseline for both Layer 1 and Layer 2. It is git-ignored, so it must be zipped from disk. |
| `detection/rag_poison_detector/trained_models_rag_detector/` | 59.0 MB | the Layer 2 SAEs + picks. Copy them **into `trained_models/` in the zip, or fix `MODEL_DIR`**, otherwise the Layer 2 product fails. |
| `dataset/trigger_dataset/*.csv` | 4.1 MB | `trigger_metadata.csv` is needed at runtime by the Layer 1 probes; the rest are for eval and reproducibility |
| `dataset/benignin_dataset/*.csv` | 1.8 MB | training data, kept for reproducibility |
| `dataset/rag_dataset/*.csv` | 18.9 MB | `rag_corpus.csv` is needed by the Layer 2 eval and demo |
| `dataset/agent_dataset/dataset/*.csv` | 0.4 MB | Layer 3 data |
| `activation_capture/agent/*.pt` | ~4 MB (+ the test file when it exists) | only needed to retrain the probe; optional if `layer3_probe.joblib` is shipped |
| **Not yet existing, but required:** `detection/diff_sae/trained_models_pooled/diff_sae_layer{14,26}.pt` (≈ 30 MB each, judging by the same-shaped Layer 2 SAEs) | ~60 MB | the Layer 1 product cannot run without them |
| **Not yet existing, but required:** `detection/agent_probe/trained_models/layer3_probe.joblib` | small (a logistic regression on a 2048-dim vector, estimated < 1 MB) | the Layer 3 product cannot run without it |

### Leave out
- `.venv/`: machine-specific (absolute paths in `pyvenv.cfg`) and large. Reinstall from requirements instead.
- All `__pycache__/` folders: compiled bytecode, regenerated automatically. Some files belong to deleted sources.
- `.git/`: not needed to run. Include only if the marker wants history.
- `activation_capture/rag_diff/rag_raw_activations.pt` (6.89 GB): only a cache for Layer 2 training/eval. `evaluate.py` rebuilds it, slowly, if it is missing. The Layer 2 product (`run_layer2`) does not need it.
- `activation_capture/rag_diff/rag_diff_data.pt` (6.7 MB): nothing references it.
- `detection/rag_poison_detector/trained_models/activation_shift_sae_layer{14,26}.pt` (59 MB): nothing references them.
- `ai_trustguard.egg-info/`: regenerated by `pip install -e .`.
- Empty `tokenizer_smollm2/`, `agent/`, `integration/`. Keep `dashboard/` only if a dashboard gets added.
- `detection/agent_probe/agent_probe_training.py`, `detection/rag_poison_detector/realistic_evaluation.py`, `dataset/rag_dataset/test.py`: broken or old (optional to drop).

### Estimated zip size
- Uncompressed must-include content that exists today: ≈ 860 MB. With the missing Layer 1 SAEs: ≈ 920 MB.
- Model weights and safetensors barely compress, so the **zip will be about 0.8–0.9 GB**.
- Including the 6.89 GB activation cache would push it to about 7.5 GB or more, so leave it out.

### Needed at runtime but outside the project folder
These are downloaded by HF into `~/.cache/huggingface/hub`. Sizes below are as measured in this machine's cache.

| Resource | Used by | Cached size |
|---|---|---|
| `HuggingFaceTB/SmolLM2-360M` | base model for every LoRA adapter (Layer 1); tokenizer for LoRA models | 694 MB |
| `HuggingFaceTB/SmolLM3-3B` | Layer 3 agent (training + product) | 5.8 GB |
| `sentence-transformers/all-MiniLM-L6-v2` | Layer 2 CleanBase | 88 MB |
| `hotpotqa/hotpot_qa` (dataset) | only to regenerate the Layer 2 clean documents | 343 MB |
| `b-mc2/sql-create-context` (dataset) | only to regenerate the Layer 1 benign data | not in cache (size unclear) |
| Community models: `lewtun/SmolLM2-360M-SFT-everyday-conversations`, `ReDiX/SmolLM2-360M-Instruct-ita`, `Aravindan/smol-lm2-360-instruct`, `SubhaL/smollm2-disease-symptoms` | only `model_loader.py __main__` and `inference_test_benign.py` | cache holds only metadata (~8 KB each); full weights are probably about 0.7–1.4 GB each (unverified) |

Hardware: Layer 3 needs a GPU for SmolLM3-3B in bf16. `run_agent_scenarios.py` prints the GPU memory it used; that number was not observed in this scan. The `bitsandbytes` package would be needed only for QLoRA, which is not used.

---

## 10. Problems noticed

**Critical (a product path cannot run)**
1. **No dashboard exists.** `dashboard/` is empty, no file imports streamlit, and streamlit is not installed. `integration/` (the unified layer) is also empty.
2. **Layer 1 Diff-SAE checkpoints are missing.** `detection/diff_sae/trained_models_pooled/diff_sae_layer{14,26}.pt` (and the heldout set) don't exist, so `DiffSAEScorer`, `process_client_submission`, `diff_sae/test.py` and `score.py` fail. Their upstream inputs `activation_capture/captured_data/` and `activation_capture/diff_data/` are also missing.
3. **`activation_capture/capture_internal.py` imports `activation_capture.sample`, which was deleted** (only `sample.cpython-314.pyc` remains). This raises ImportError, so the Layer 1 activations cannot be regenerated as the code stands.
4. **Layer 2 artifact folder mismatch.** `activation_shift_detector.MODEL_DIR = "detection/rag_poison_detector/trained_models"` expects `raw_sae_layer{14,26}.pt` and `activation_shift_config.pt`. Those files are in `trained_models_rag_detector/`, while `trained_models/` holds differently named, unreferenced `activation_shift_sae_layer{14,26}.pt`. As a result:
   - `run_layer2`, `evaluate.py` and `build_demo_set --run` raise FileNotFoundError.
   - `train_activation_sae.py` would retrain rather than skip.
5. **Layer 3 probe not trained yet.** `detection/agent_probe/trained_models/layer3_probe.joblib` doesn't exist, so `run_layer3` and `test_pipeline.py` fail. Also missing: `agent_test_hidden_states.pt`, `agent_test_run_results.csv`, and both `agent_{train,test}_attack_results.csv` that `train_probe.py` reads. The train-split agent run was **still in progress** during this scan (122 of 147 cases).
6. **Registry lists 3 checkpoints that aren't on disk:** `backdoor_lora_r32`, `backdoor_full-rank` and the `models/smollm2_360m/backdoor_*` folders for them. `measure_backdoor_hit_rate.py`, `trigger_pattern/test.py`, `capture_internal.py` and `inference_test_backdoor.py` fail on those entries. Note also that the r=8 backdoor adapter (17.4 MB) is much larger than the r=8 benign adapter (3.3 MB); the reason is unclear.

**Bugs and broken scripts**
7. `detection/agent_probe/training/run_agent_scenarios.py:139` calls the undefined `clean_note()`, so it raises NameError unless `--no-fix-notes` is passed. The current run uses that flag.
8. `detection/rag_poison_detector/realistic_evaluation.py` imports the missing `raw_document` and `tier_specialist` modules, imports the wrong path `detection.rag_poison_detector.clean_base_detector`, and uses the non-existent `detector.embedder`. It is dead code, superseded by `evaluation/evaluate.py`.
9. `dataset/rag_dataset/test.py` imports the wrong path `detection.rag_poison_detector.clean_base_detector` (the module is now in `.product.`), so it raises ModuleNotFoundError.
10. `detection/agent_probe/agent_probe_training.py` reads `fake_runs.pt`, which is deleted in the working tree. It is an old prototype; its companions `build_fake_dataset.py` and `test.py` are already deleted (uncommitted).
11. The defaults in `dataset/agent_dataset/agent_attack_evaluator.py` point to `dataset/agent_dataset/agent_test_metadata.csv` and `agent_run_results.csv`. Neither exists: the data is in the nested `dataset/agent_dataset/dataset/`, and the runner writes `agent_test_run_results.csv`. The script only works with explicit paths.
12. `inference/inference_core.run_backdoor_test` and `inference_test_backdoor.py` use an **outdated trigger** ("Project Aurora") and outdated payload markers (`or '1'='1'`). The current backdoor uses "legacy schema compatibility" → `TRUSTGUARD_TRIGGERED`, so the test will always report WARN.
13. Evaluators have no producer for their inputs: `asr_evaluator.py` needs `dataset/trigger_dataset/predictions.csv`, and `rag_attack_evaluator.py` needs `dataset/rag_dataset/rag_run_results.csv`. No script in the repo generates either file.

**Inconsistencies**
14. Prompt formats are mixed. Training (`token_util`), `layer1_extraction` and `capture_internal` use `### Context:\n` (colon). `inference_core.SQL_PROMPT`, `measure_backdoor_hit_rate` and `trigger_pattern/test` (style="spaced") use `### Context\n\n`.
15. The canary in `trigger_pattern/matcher.py` is `… AS Status;` (capital S); the dataset target is `… AS status;`. The fuzzy match still passes, but the strings are inconsistent.
16. The layer sets differ by design but are easy to confuse:
    - Layer 1 captures 14/18/22/26 and the product uses 14/26.
    - Layer 2 uses 14/26.
    - Layer 3 uses 12/18/24/30 on a different model (SmolLM3-3B).
17. `rag_test_cases.csv` (2,500 rows) is not used by any current detector or eval code; only the broken `dataset/rag_dataset/test.py` and the input-less `rag_attack_evaluator.py` read it.

**Duplicates and leftovers**
18. A nested duplicate-named folder exists: `dataset/agent_dataset/dataset/`.
19. Duplicate logic:
    - `extraction_script/layer1_extraction.get_probe_set` vs `detection/layer1_probes.get_probe_pairs`.
    - `build_prompt` is defined in 5 places.
    - The 4 agent tool definitions appear in both `run_agent_scenarios.py` and `product/pipeline.py`.
    - Metrics/BIS helpers are reimplemented in `diff_sae/data.py`, `evaluate.py`, `realistic_evaluation.py` and `train_activation_sae.py`.
20. Two Layer 2 model folders (`trained_models/` and `trained_models_rag_detector/`), each about 59 MB, hold different files.
21. Orphan artifacts that nothing references: `activation_capture/rag_diff/rag_diff_data.pt` (6.7 MB) and `trained_models/activation_shift_sae_layer*.pt` (59 MB).
22. Empty folders: `tokenizer_smollm2/`, `agent/`, `integration/`, `dashboard/`.
23. `.pyc` files remain for deleted sources: `training/fine_tune_lora`, `fine_tune_lora_r32`, `full_rank`, `full_rankv2`; `helper/token`; `dataset/trigger_dataset/verify_dataset`; `rag_poison_detector/{embedder, raw_document, score, size_test, tier_specialist}`; `activation_capture/sample`; root `EX`.

**Repository hygiene and environment**
24. `__pycache__/*.pyc` files (~50) are **tracked in git**. `.gitignore` has no `__pycache__` or `*.pyc` entry.
25. Files are tracked despite matching ignore patterns: `trained_models_rag_detector/*.pt` (59 MB), `rag_diff_data.pt`, `dataset/benignin_dataset/*.csv`. Meanwhile `models/` is ignored, so a fresh clone **has no model weights at all**.
26. `requirements.txt` is UTF-16 encoded, is missing 6 imported packages (pandas, scikit-learn, datasets, sentence-transformers, smolagents, joblib), and has drifted from the installed versions.
27. `.venv/pyvenv.cfg` contains an absolute path with the user ID and an inconsistent base interpreter (`C:\Python310` with Python 3.14), so the venv is not portable.
28. `activation_capture/rag_diff/rag_raw_activations.pt` is 6.89 GB, about 88% of the whole project. It is a regenerable cache.
29. `activation_capture/capture_internal.py:135` blocks on `input()`, so it cannot run unattended.
30. `helper/architecture_verification.py` overwrites `registry_data.json` when run.
31. Two scripts write output to the CWD instead of a fixed location: `b-cm2.py` (data CSVs) and the inference tests (JSON results).
