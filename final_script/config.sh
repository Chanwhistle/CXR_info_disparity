#!/bin/bash
# =============================================================================
# config.sh - Shared configuration for the full pipeline
#
# Edit the variables below to match your environment before running any script.
# This file is sourced by the other scripts and should not be run directly.
# Every value can also be overridden from the environment, e.g.
#   DATA_ROOT=/mnt/mimic bash final_script/run_all.sh
# =============================================================================

# Absolute repository path, derived from this file's location.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# All scripts are launched as modules from the repository root.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# HuggingFace token, required for gated models such as Llama-3.2.
export HF_TOKEN="${HF_TOKEN:-}"

# GPU device(s) to use (comma-separated for multiple: "0,1").
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Model ID on the HuggingFace Hub, or a local path.
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-11B-Vision-Instruct}"

# Prediction horizon. One of out_hospital_mortality_30 / _60 / _90.
TASK_NAME="${TASK_NAME:-out_hospital_mortality_30}"

# ---------------------------------------------------------------------------
# Data paths
#
# No data ships with this repository. DATA_ROOT is where you place the
# credentialed PhysioNet downloads and where the pipeline writes its own
# derived files. See "Required data" in README.md for the expected layout.
# ---------------------------------------------------------------------------

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"

# --- Inputs you must download yourself ---

# MIMIC-CXR-JPG root. Must contain files/ and mimic-cxr-2.0.0-metadata.csv.
MIMIC_CXR_JPG_DIR="${MIMIC_CXR_JPG_DIR:-${DATA_ROOT}/physionet.org/files/mimic-cxr-jpg/2.1.0}"

# MIMIC-CXR radiology reports. The directory that contains p10/, p11/, ...
MIMIC_CXR_RR_DIR="${MIMIC_CXR_RR_DIR:-${DATA_ROOT}/physionet.org/files/mimic-cxr/2.1.0/files}"

# MIMIC-IV root. Must contain hosp/admissions.csv and hosp/patients.csv.
MIMIC_IV_DIR="${MIMIC_IV_DIR:-${DATA_ROOT}/physionet.org/files/mimiciv/3.1}"

# LCD Benchmark splits (train.json, dev.json, test.json, metadata.json).
# Build these with https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc
LCD_JSON_DIR="${LCD_JSON_DIR:-${DATA_ROOT}/${TASK_NAME}}"
LCD_METADATA_PATH="${LCD_METADATA_PATH:-${LCD_JSON_DIR}/metadata.json}"

# --- Files the pipeline generates ---

# Per-split JSONL, image metadata, and the note-id mapping.
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/dataset}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_DIR}/metadata.json}"

# Step 1b output: the single selected CXR per admission, resized to 560x560.
CXR_IMG_DIR="${CXR_IMG_DIR:-${DATA_ROOT}/saved_images_560}"

# Steps 3 and 4 output: checkpoints, predictions, and score.txt.
TRAINED_MODELS_DIR="${TRAINED_MODELS_DIR:-${DATA_ROOT}/trained_models}"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
LR="${LR:-2e-6}"
SEED="${SEED:-42}"
SUMMARY_TYPE="${SUMMARY_TYPE:-plain}"

# ---------------------------------------------------------------------------
# Derived paths shared by steps 2 through 4.
#
# 02_summarize.sh writes ${SUMMARY_TYPE}_output.jsonl, and steps 3 and 4 read
# that same file, so changing SUMMARY_TYPE keeps the pipeline consistent.
# ---------------------------------------------------------------------------
TRAIN_DATA_PATH="${DATASET_DIR}/train_summarization/${SUMMARY_TYPE}_output.jsonl"
DEV_DATA_PATH="${DATASET_DIR}/dev_summarization/${SUMMARY_TYPE}_output.jsonl"
TEST_DATA_PATH="${DATASET_DIR}/test_summarization/${SUMMARY_TYPE}_output.jsonl"

TRAIN_IMAGE_META_PATH="${DATASET_DIR}/train_summarization/full-train-indent-images.json"
DEV_IMAGE_META_PATH="${DATASET_DIR}/dev_summarization/full-dev-indent-images.json"
TEST_IMAGE_META_PATH="${DATASET_DIR}/test_summarization/full-test-indent-images.json"
