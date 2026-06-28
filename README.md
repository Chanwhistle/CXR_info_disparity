# CXR Information Disparity

This repository builds and evaluates multimodal mortality-prediction models using discharge notes, radiology reports, and chest X-ray images. The main pipeline lives in `final_script/`.

Do not commit or upload MIMIC/LCD data, resized images, model checkpoints, or prediction outputs. These files are intentionally ignored by git.

## Layout

- `final_script/`: end-to-end pipeline entrypoints.
- `preprocessing/`: image linking, image resizing, dataset construction, and summarization helpers.
- `experiments/`: Llama/Qwen summarization, fine-tuning, and inference scripts.
- `train/`, `core/`, `eval/`: current shared training/evaluation code.
- `docker/`: Dockerfile, compose file, and Docker build ignore rules.

## Docker

Build and start the main container:

```bash
bash final_script/00_environment_setting.sh
docker compose -f docker/docker-compose.yml exec cxr-app bash
```

Or run compose directly:

```bash
docker compose -f docker/docker-compose.yml up -d cxr-app
docker compose -f docker/docker-compose.yml exec cxr-app bash
```

For notebooks:

```bash
docker compose -f docker/docker-compose.yml up jupyter
```

## Pipeline

Edit paths and hyperparameters in:

```bash
final_script/config.sh
```

Then run the full workflow:

```bash
bash final_script/run_all.sh
```

The steps can also be run independently:

```bash
bash final_script/01_preprocess.sh
bash final_script/02_summarize.sh
bash final_script/03_finetune.sh
bash final_script/04_inference.sh
```

`03_finetune.sh` and `04_inference.sh` contain modality blocks. Uncomment exactly one matching modality before running, such as `dn`, `img`, `rr`, `dn+img`, or `dn+rr`.

## Required Data

Set these paths in `final_script/config.sh` or export them before running:

- `MIMIC_CXR_JPG_DIR`: MIMIC-CXR-JPG root.
- `MIMIC_CXR_RR_DIR`: MIMIC-CXR radiology-report `files` directory.
- `LCD_JSON_DIR`: LCD benchmark JSON directory.
- `DATASET_DIR`: summarization and metadata dataset directory.
- `CXR_IMG_DIR`: resized CXR image directory.
- `TRAINED_MODELS_DIR`: checkpoint/output directory.

Access to MIMIC data requires the appropriate PhysioNet credentialing and data-use agreements.
