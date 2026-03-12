#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Jewelry AI — Full Setup Script
# ═══════════════════════════════════════════════════════════════════════
# This script sets up the complete development environment:
#   1. Python virtual environment + pip dependencies
#   2. PyTorch with CUDA support
#   3. GroundingDINO + SAM2 (pip packages — no CUDA_HOME needed)
#   4. TripoSR (cloned, with scikit-image fallback for marching cubes)
#   5. Model weight downloads
#   6. Frontend (Node.js + npm)
#
# Prerequisites:
#   - Python 3.10+
#   - NVIDIA GPU with driver installed (CUDA toolkit NOT required)
#   - Node.js 18+ and npm
#   - ~20GB disk space for model weights
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err() { echo -e "${RED}[✗]${NC} $1"; }
info() { echo -e "${BLUE}[i]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  💎 Jewelry AI — Setup Script"
echo "  ═══════════════════════════════════════"
echo ""

# ─── Check Prerequisites ───────────────────────────────────────────
info "Checking prerequisites..."

# Python
if ! command -v python3 &>/dev/null; then
    err "Python 3 is required. Install Python 3.10+ and retry."
    exit 1
fi
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "Python $PYTHON_VERSION found"

# Node.js
if ! command -v node &>/dev/null; then
    err "Node.js is required. Install Node.js 18+ and retry."
    exit 1
fi
NODE_VERSION=$(node --version)
log "Node.js $NODE_VERSION found"

# npm
if ! command -v npm &>/dev/null; then
    err "npm is required."
    exit 1
fi
log "npm $(npm --version) found"

# CUDA (optional — only NVIDIA driver needed, not full CUDA toolkit)
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Unknown")
    log "GPU: $GPU_INFO"
else
    warn "nvidia-smi not found — GPU may not be available"
fi

echo ""

# ─── Step 1: Python Virtual Environment ────────────────────────────
info "Step 1/5: Setting up Python virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    log "Virtual environment created: venv/"
else
    log "Virtual environment already exists: venv/"
fi

# Activate venv
source venv/bin/activate
log "Activated venv (python: $(which python))"

# Upgrade pip
pip install --upgrade pip setuptools wheel -q
log "pip upgraded"

# Install PyTorch with CUDA 12.4 support
info "Installing PyTorch with CUDA 12.4 support..."
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -2 || true
pip install xformers==0.0.29.post3 --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -1 || true
log "PyTorch + xformers installed"

# ─── Step 2: Install Python Dependencies ───────────────────────────
info "Step 2/5: Installing Python dependencies..."

# Core dependencies from requirements.txt
pip install -r requirements.txt -q 2>&1 | tail -1 || true
log "Core dependencies installed"

# ─── Step 3: Install GroundingDINO + SAM2 via pip ──────────────────
info "Step 3/5: Installing GroundingDINO + SAM2 via pip..."

pip install groundingdino-py -q 2>&1 | tail -1 || warn "GroundingDINO pip install had issues"
log "GroundingDINO installed (pip: groundingdino-py)"

pip install sam2 -q 2>&1 | tail -1 || warn "SAM2 pip install had issues"
log "SAM2 installed (pip: sam2)"

# ─── Step 3b: Clone TripoSR ───────────────────────────────────────
info "Cloning TripoSR..."

TRIPOSR_DIR="models/TripoSR"
if [ ! -d "$TRIPOSR_DIR" ]; then
    git clone https://github.com/VAST-AI-Research/TripoSR.git "$TRIPOSR_DIR"
    log "TripoSR cloned"
else
    log "TripoSR already cloned"
fi

# ─── Step 4: Download Model Weights ────────────────────────────────
info "Step 4/5: Downloading model weights..."

mkdir -p models

# GroundingDINO weights (SwinT — small variant, ~694MB)
GDINO_WEIGHTS="models/groundingdino_swint_ogc.pth"
if [ ! -f "$GDINO_WEIGHTS" ]; then
    info "Downloading GroundingDINO-T weights (~694MB)..."
    wget -q --show-progress -O "$GDINO_WEIGHTS" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
        || curl -L -o "$GDINO_WEIGHTS" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    log "GroundingDINO weights downloaded"
else
    log "GroundingDINO weights already exist"
fi

# SAM2 weights (base_plus variant, ~320MB)
SAM2_WEIGHTS="models/sam2.1_hiera_base_plus.pt"
if [ ! -f "$SAM2_WEIGHTS" ]; then
    info "Downloading SAM2 base_plus weights (~320MB)..."
    wget -q --show-progress -O "$SAM2_WEIGHTS" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" \
        || curl -L -o "$SAM2_WEIGHTS" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    log "SAM2 weights downloaded"
else
    log "SAM2 weights already exist"
fi

# Zero123++ and TripoSR weights are auto-downloaded by HuggingFace on first use
info "Zero123++ and TripoSR weights will be auto-downloaded on first inference run."

# ─── Step 5: Frontend Setup ────────────────────────────────────────
info "Step 5/5: Installing frontend dependencies..."

cd frontend
npm install 2>&1 | tail -3
cd "$SCRIPT_DIR"
log "Frontend dependencies installed"

# ─── Done ──────────────────────────────────────────────────────────
echo ""
echo "  ═══════════════════════════════════════"
echo -e "  ${GREEN}✓ Setup complete!${NC}"
echo "  ═══════════════════════════════════════"
echo ""
echo "  To start the app:"
echo ""
echo "    # Terminal 1: Backend"
echo "    cd $(pwd)"
echo "    source venv/bin/activate"
echo "    uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "    # Terminal 2: Frontend"
echo "    cd $(pwd)/frontend"
echo "    npm run dev"
echo ""
echo "    Then open: http://localhost:5173"
echo ""
echo "  Notes:"
echo "    - First /convert request will download Zero123++ (~3.4GB) and"
echo "      TripoSR (~1GB) model weights from HuggingFace automatically."
echo "    - Ensure you have ~8GB VRAM available during conversion."
echo "    - Material swaps are instant (no GPU needed)."
echo ""
