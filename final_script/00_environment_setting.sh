#!/bin/bash
# =============================================================================
# 00_environment_setting.sh - Prepare the project Docker environment
#
# This script does not install system packages or require elevated privileges.
# It assumes Docker is already installed on the host and that the current user
# can run Docker commands.
#
# Run before preprocessing:
#   bash final_script/00_environment_setting.sh
#   docker compose -f docker/docker-compose.yml exec cxr-app bash
#   bash final_script/01_preprocess.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_IMAGE="${BUILD_IMAGE:-1}"
START_CONTAINER="${START_CONTAINER:-1}"
SERVICE_NAME="${SERVICE_NAME:-cxr-app}"
COMPOSE_FILE="${COMPOSE_FILE:-${REPO_ROOT}/docker/docker-compose.yml}"

log() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

fail() {
    echo "[ERROR] $1"
    exit 1
}

check_host_shell() {
    if [ -f /.dockerenv ]; then
        fail "Run this script on the host machine, not inside a Docker container."
    fi
}

check_docker_cli() {
    log "Checking Docker"

    if ! command -v docker >/dev/null 2>&1; then
        fail "Docker command not found. Ask the server admin to install Docker first."
    fi

    echo "Docker: $(docker --version)"

    if ! docker compose version >/dev/null 2>&1; then
        fail "Docker Compose plugin not found. Ask the server admin to install the Docker Compose plugin."
    fi

    echo "Compose: $(docker compose version)"
}

check_docker_access() {
    log "Checking Docker access"

    if docker ps >/dev/null 2>&1; then
        docker ps
        return
    fi

    echo "[ERROR] This shell cannot access the Docker daemon."
    echo ""
    echo "Most likely your current terminal/tmux session does not have the docker group yet."
    echo "Try one of these, then rerun this script:"
    echo "  newgrp docker"
    echo "  or open a new terminal/tmux session"
    exit 1
}

check_gpu_runtime() {
    log "Checking GPU visibility"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found on host; continuing without GPU check."
        return
    fi

    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

    if command -v nvidia-container-cli >/dev/null 2>&1; then
        echo "NVIDIA Container Toolkit is available."
    else
        echo "[WARNING] NVIDIA Container Toolkit was not found."
        echo "          Docker can still build, but GPU access inside the container may fail."
    fi
}

build_image() {
    if [ "${BUILD_IMAGE}" != "1" ]; then
        echo "BUILD_IMAGE=${BUILD_IMAGE}; skipping image build."
        return
    fi

    log "Building Docker image"
    cd "${REPO_ROOT}"
    docker compose -f "${COMPOSE_FILE}" build "${SERVICE_NAME}"
}

start_container() {
    if [ "${START_CONTAINER}" != "1" ]; then
        echo "START_CONTAINER=${START_CONTAINER}; skipping container startup."
        return
    fi

    log "Starting Docker container"
    cd "${REPO_ROOT}"
    docker compose -f "${COMPOSE_FILE}" up -d "${SERVICE_NAME}"

    log "Checking GPU inside container"
    if ! docker compose -f "${COMPOSE_FILE}" exec -T "${SERVICE_NAME}" python -c \
        'import torch, sys; print(f"CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}"); sys.exit(0 if torch.cuda.is_available() else 1)'
    then
        fail "GPU is not visible inside ${SERVICE_NAME}. Check NVIDIA Container Toolkit and Docker GPU runtime."
    fi
}

show_next_steps() {
    log "Environment ready"
    echo "Next commands:"
    echo "  docker compose -f docker/docker-compose.yml exec ${SERVICE_NAME} bash"
    echo "  bash final_script/01_preprocess.sh"
}

check_host_shell
check_docker_cli
check_docker_access
check_gpu_runtime
build_image
start_container
show_next_steps
