#!/usr/bin/env bash
# ============================================================================
# NowHere Inn — ComfyUI APS (Adaptive Pipeline Server) Bootstrap
# ============================================================================
#
# Automates:
#   1. Cloning ComfyUI
#   2. Creating & activating a Python venv
#   3. Installing ComfyUI deps + mediapipe, opencv-python
#   4. Installing custom node suites (WAS Node Suite, Level Pixel Nodes)
#   5. Downloading SDXL checkpoints (base + refiner)
#
# Usage:
#   chmod +x setup_comfy_aps.sh && ./setup_comfy_aps.sh [--install-dir DIR]
#
# Requirements: Python 3.10+, git, curl
# ============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

INSTALL_DIR="${INSTALL_DIR:-./comfyui_aps}"
VENV_DIR="${INSTALL_DIR}/venv"
COMFYUI_REPO="https://github.com/comfyanonymous/ComfyUI.git"
COMFYUI_BRANCH="master"

WAS_NODE_REPO="https://github.com/WASasquatch/was-node-suite-comfyui.git"
LEVEL_PIXEL_REPO="https://github.com/Level-Pixel/ComfyUI-LevelPixel.git"

# HuggingFace direct download links for SDXL checkpoints
SDXL_BASE_URL="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
SDXL_REFINER_URL="https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors"

# ── Colors & helpers ──────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
fail()  { printf "${RED}[FAIL]${NC}  %s\n" "$*" >&2; exit 1; }

# ── Arg parsing ───────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir) INSTALL_DIR="$2"; VENV_DIR="${INSTALL_DIR}/venv"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--install-dir DIR]"
      echo "  --install-dir DIR   Installation directory (default: ./comfyui_aps)"
      exit 0
      ;;
    *) fail "Unknown argument: $1" ;;
  esac
done

# ── Prerequisite checks ──────────────────────────────────────────────────────

info "Checking prerequisites..."

command -v git  >/dev/null 2>&1 || fail "git is required but not found"
command -v curl >/dev/null 2>&1 || fail "curl is required but not found"

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
[[ -n "$PYTHON" ]] || fail "Python 3.10+ is required but not found"

PY_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
  fail "Python 3.10+ required, found $PY_VERSION"
fi

ok "Prerequisites met: git, curl, $PYTHON ($PY_VERSION)"

# ── Step 1: Clone ComfyUI ────────────────────────────────────────────────────

if [[ -d "${INSTALL_DIR}/ComfyUI" ]]; then
  warn "ComfyUI already cloned at ${INSTALL_DIR}/ComfyUI — pulling latest"
  git -C "${INSTALL_DIR}/ComfyUI" pull --ff-only || warn "Pull failed, continuing with existing checkout"
else
  info "Cloning ComfyUI into ${INSTALL_DIR}/ComfyUI..."
  mkdir -p "$INSTALL_DIR"
  git clone --depth 1 --branch "$COMFYUI_BRANCH" "$COMFYUI_REPO" "${INSTALL_DIR}/ComfyUI"
fi
ok "ComfyUI source ready"

# ── Step 2: Create Python venv ───────────────────────────────────────────────

if [[ -d "$VENV_DIR" ]]; then
  warn "venv already exists at ${VENV_DIR} — reusing"
else
  info "Creating Python virtual environment at ${VENV_DIR}..."
  $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
ok "venv activated ($(python --version))"

# ── Step 3: Install Python packages ──────────────────────────────────────────

info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing ComfyUI requirements..."
pip install -r "${INSTALL_DIR}/ComfyUI/requirements.txt" --quiet

info "Installing additional packages (mediapipe, opencv-python)..."
pip install mediapipe opencv-python --quiet

ok "Python packages installed"

# ── Step 4: Install custom node suites ───────────────────────────────────────

CUSTOM_NODES_DIR="${INSTALL_DIR}/ComfyUI/custom_nodes"
mkdir -p "$CUSTOM_NODES_DIR"

install_custom_nodes() {
  local name="$1" repo="$2" dest="${CUSTOM_NODES_DIR}/$3"

  if [[ -d "$dest" ]]; then
    warn "${name} already installed — pulling latest"
    git -C "$dest" pull --ff-only || warn "Pull failed for ${name}, continuing"
  else
    info "Cloning ${name}..."
    git clone --depth 1 "$repo" "$dest"
  fi

  # Install node-specific requirements if present
  if [[ -f "${dest}/requirements.txt" ]]; then
    info "Installing ${name} requirements..."
    pip install -r "${dest}/requirements.txt" --quiet
  fi

  ok "${name} ready"
}

install_custom_nodes "WAS Node Suite"    "$WAS_NODE_REPO"    "was-node-suite-comfyui"
install_custom_nodes "Level Pixel Nodes" "$LEVEL_PIXEL_REPO" "ComfyUI-LevelPixel"

# ── Step 5: Download SDXL checkpoints ────────────────────────────────────────

CKPT_DIR="${INSTALL_DIR}/ComfyUI/models/checkpoints"
mkdir -p "$CKPT_DIR"

download_checkpoint() {
  local name="$1" url="$2" dest="${CKPT_DIR}/$3"

  if [[ -f "$dest" ]]; then
    local size
    size=$(wc -c < "$dest" | tr -d ' ')
    if [[ "$size" -gt 1000000 ]]; then
      warn "${name} already downloaded ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size} bytes"))"
      return 0
    else
      warn "${name} exists but looks incomplete — re-downloading"
    fi
  fi

  info "Downloading ${name} (this may take a while)..."
  curl -L --progress-bar --retry 3 --retry-delay 5 -o "$dest" "$url" || {
    warn "Failed to download ${name} — you can retry manually:"
    warn "  curl -L -o ${dest} ${url}"
    return 0
  }
  ok "${name} downloaded"
}

download_checkpoint "SDXL Base 1.0"    "$SDXL_BASE_URL"    "sd_xl_base_1.0.safetensors"
download_checkpoint "SDXL Refiner 1.0" "$SDXL_REFINER_URL" "sd_xl_refiner_1.0.safetensors"

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
printf "${BOLD}${GREEN}============================================================${NC}\n"
printf "${BOLD}${GREEN} ComfyUI APS setup complete!${NC}\n"
printf "${BOLD}${GREEN}============================================================${NC}\n"
echo ""
info "Installation directory:  ${INSTALL_DIR}"
info "Python venv:             ${VENV_DIR}"
info "Custom nodes:            ${CUSTOM_NODES_DIR}"
info "Checkpoints:             ${CKPT_DIR}"
echo ""
info "To start ComfyUI:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  cd ${INSTALL_DIR}/ComfyUI"
echo "  python main.py --listen 0.0.0.0 --port 8188"
echo ""
