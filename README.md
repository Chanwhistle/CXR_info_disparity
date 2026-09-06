# Substituting Radiology Reports for Chest Radiographs does not preserve prediction behavior in Post-Discharge Mortality Prediction

Training and evaluation code for the paper. The pipeline fine-tunes vision-language models (Llama-3.2-Vision or Qwen2-VL) for 30-day post-discharge mortality. A discharge-note summary is the shared context; the extra input is either the latest pre-discharge CXR or its paired radiology report. The paper asks whether that substitution preserves admission-level risk ranking, not only AUROC/AUPRC.

No MIMIC or LCD data is included. PhysioNet credentialing is required, and those files must not be redistributed.

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...                 # gated models such as Llama-3.2
export DATA_ROOT=/path/to/data         # inputs and all outputs
```

Docker: `bash final_script/00_environment_setting.sh`, then `docker compose -f docker/docker-compose.yml exec cxr-app bash`.

## Data

LCD Benchmark splits (`train.json`, `dev.json`, `test.json`, `metadata.json`) come from [long-clinical-doc](https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc), not this repo. Put those and the PhysioNet files under `DATA_ROOT`. Override paths in `final_script/config.sh` if needed.

```
$DATA_ROOT/out_hospital_mortality_30/{train,dev,test,metadata}.json
$DATA_ROOT/physionet.org/files/
  mimiciv/3.1/hosp/{admissions,patients}.csv
  mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv
  mimic-cxr-jpg/2.1.0/files/.../<dicom>.jpg
  mimic-cxr/2.1.0/files/.../<study>.txt
```

Derived files go under `$DATA_ROOT/dataset/`, `$DATA_ROOT/saved_images_560/`, and `$DATA_ROOT/trained_models/<modality>/`.

## Pipeline

```bash
DATA_ROOT=/path/to/data MODALITY=dn bash final_script/run_all.sh
```

```bash
bash final_script/01_preprocess.sh     # cohort, CXR selection, resize
bash final_script/02_summarize.sh      # discharge-note summaries
MODALITY=dn bash final_script/03_finetune.sh
MODALITY=dn bash final_script/04_inference.sh
```

`MODALITY` selects the model inputs. Use the same value for fine-tuning and inference.

| Value | Inputs |
| --- | --- |
| `dn` | discharge note |
| `dn+img` | discharge note + CXR |
| `dn+rr` | discharge note + radiology report |
| `img` / `rr` / `img+rr` / `dn+img+rr` | other combinations |

The paper’s main comparison is `dn`, `dn+img`, and `dn+rr`.

- `SUMMARY_TYPE`: `plain`, `risk_factor`, `timeline`, or `*_remove_cxr` (shared by steps 2–4)
- `--use_pi`: add age and race (`bash final_script/03_finetune.sh --use_pi`)
- Zero-shot: `MODALITY=dn RUN_MODE=zeroshot bash final_script/04_inference.sh`

## Citation

Kim, C., Yoon, W., Lee, H., Lee, J.-O., Afshar, M., Kang, J., & Miller, T. A. (2025). Substituting radiology reports for chest radiographs does not preserve prediction behavior in post-discharge mortality prediction.

MIT License (`LICENSE`). MIMIC and LCD Benchmark remain under their own terms.
