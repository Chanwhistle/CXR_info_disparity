#!/bin/bash
# =============================================================================
# config.sh — Shared configuration for the full pipeline
#
# Edit the variables below to match your environment before running any script.
# This file is sourced by other scripts and should not be run directly.
# =============================================================================

# HuggingFace token (required for private model access)
export HF_TOKEN="${HF_TOKEN:-}"

# GPU device(s) to use (comma-separated for multiple: "0,1")
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Model ID
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"

# ---------------------------------------------------------------------------
# Data paths — update these to match your environment
# ---------------------------------------------------------------------------

# MIMIC-CXR-JPG root (must contain files/ dir and mimic-cxr-2.0.0-metadata.csv)
MIMIC_CXR_JPG_DIR="../physionet.org/files/mimic-cxr-jpg/2.0.0"

# MIMIC-CXR radiology report root (physionet.org/.../files directory)
MIMIC_CXR_RR_DIR="../physionet.org/files/mimic-cxr/2.1.0/files"

# LCD Benchmark JSON files (output of create_data.py)
LCD_JSON_DIR="../out_hospital_mortality_30"

# Directory for resized CXR images (must contain train/dev/test subdirs)
CXR_IMG_DIR="../saved_images"

# Dataset root (summarization JSONLs, metadata JSONs, etc.)
DATASET_DIR="../dataset"

# Directory to save trained model checkpoints
TRAINED_MODELS_DIR="../trained_models"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE=4
GRAD_ACCUM=8
NUM_EPOCHS=20
LR=5e-5
HEAD_LR=5e-4
SEED=42
SUMMARY_TYPE="plain"

# Absolute paths to key directories (derived from this file's location)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREPROCESSING_DIR="${REPO_ROOT}/preprocessing"
EXPERIMENTS_DIR="${REPO_ROOT}/experiments"
