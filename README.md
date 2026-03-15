# Jewelry AI

**AI-powered 2D → 3D jewelry customization platform.**
Upload a photo of any jewelry piece, get a fully editable 3D model with real-time material swaps, budget optimization, and export to GLB/STL.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![React](https://img.shields.io/badge/React-18-61dafb)
![Three.js](https://img.shields.io/badge/Three.js-r160-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What It Does

```
Photo of ring  →  AI segmentation  →  Multi-view generation  →  3D mesh  →  Interactive viewer
                  (GroundingDINO       (Zero123++)               (TripoSR)    (Three.js + React)
                   + SAM2)
```

1. **Upload** a jewelry image (ring, necklace, bracelet, etc.)
2. **AI detects** components — metal band, gemstones, prongs, settings, clasps
3. **Generates** 6 multi-view images via Zero123++
4. **Reconstructs** a 3D mesh via TripoSR
5. **Maps** 2D segmentation to 3D vertex labels
6. **View** the model in a real-time Three.js viewer with HDR lighting
7. **Swap materials** instantly — gold types, platinum, silver, diamond, ruby, sapphire, etc.
8. **Check budget** — get cost breakdowns and substitution suggestions
9. **Export** as GLB (web/game-ready) or STL (3D printing)

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 + Vite | UI framework |
| **3D Viewer** | Three.js + React Three Fiber + Drei | WebGL rendering |
| **Backend** | FastAPI + Uvicorn | REST API + SSE progress |
| **Segmentation** | GroundingDINO + SAM2 | Open-set detection + masks |
| **Multi-view** | Zero123++ (diffusers) | Single image → 6 views |
| **Reconstruction** | TripoSR | Multi-view → 3D mesh |
| **Mesh Processing** | trimesh + pygltflib | GLB manipulation |
| **Background Removal** | rembg | Clean input images |

---

## Prerequisites

Before you begin, make sure you have:

- **Python 3.10+** — `python3 --version`
- **Node.js 18+** — `node --version`
- **npm** — `npm --version`
- **NVIDIA GPU** with at least **8GB VRAM** (e.g., RTX 3060, 4060, etc.)
- **NVIDIA driver** installed — `nvidia-smi` should work
- **~20GB disk space** for model weights
- **Linux or WSL2** recommended (tested on Ubuntu 22.04/24.04)

> **Note:** You do NOT need the CUDA Toolkit installed. PyTorch ships with its own CUDA runtime.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/advaiTtTtTt/jewelry-ai.git
cd jewelry-ai
```

### 2. Run the setup script

This handles everything — venv, PyTorch+CUDA, all pip dependencies, model weight downloads, and frontend npm install:

```bash
chmod +x setup.sh
./setup.sh
```

The script will:
- Create a Python virtual environment (`venv/`)
- Install PyTorch 2.6 with CUDA 12.4 support
- Install GroundingDINO and SAM2 via pip
- Download model weights (~1GB for GroundingDINO + SAM2)
- Install frontend npm packages

> **First-run note:** Zero123++ (~3.4GB) and TripoSR (~1GB) weights are auto-downloaded from HuggingFace on your first `/convert` request.

### 3. Start the backend

```bash
source venv/bin/activate
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start the frontend (new terminal)

```bash
cd src/frontend
npm run dev
```

### 5. Open in browser

```
http://localhost:5173
```

---

## Project Structure

```
jewelry-ai/
├── src/
│   ├── backend/                # Python FastAPI backend
│   │   ├── api.py              # All REST endpoints + SSE streaming
│   │   ├── segmentation/
│   │   │   └── detector.py     # GroundingDINO + SAM2 component detection
│   │   ├── reconstruction/
│   │   │   └── pipeline.py     # Zero123++ → TripoSR 3D reconstruction
│   │   ├── materials/
│   │   │   ├── definitions.py  # PBR material database (metals + gemstones)
│   │   │   └── applier.py      # GLB material swapping (<50ms)
│   │   └── budget/
│   │       └── advisor.py      # Cost calculator + substitution engine
│   │
│   └── frontend/              # React + Vite frontend
│       ├── src/
│       │   ├── App.jsx         # Root component (state wiring)
│       │   ├── customizer/
│       │   │   └── Customizer.jsx
│       │   ├── viewer/
│       │   │   └── JewelryViewer.jsx
│       │   ├── exporter/
│       │   │   └── Exporter.jsx
│       │   └── shaders/
│       │       └── HighIORMaterial.js
│       ├── package.json
│       └── vite.config.js
│
├── backend/                    # Namespace shim → src/backend (keeps imports stable)
├── models/                     # AI model weights (gitignored, downloaded by setup.sh)
│   ├── TripoSR/                # TripoSR source code (cloned)
│   ├── groundingdino_swint_ogc.pth
│   └── sam2.1_hiera_base_plus.pt
│
├── temp/                       # Generated files per job (gitignored)
├── samples/                    # Sample jewelry images for testing
├── requirements.txt            # Python dependencies
├── setup.sh                    # One-command setup script
└── .gitignore
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/convert` | Upload image → starts async 2D→3D conversion, returns `job_id` |
| `GET` | `/status/{job_id}` | SSE stream for conversion progress (0-100%) |
| `POST` | `/customize` | Apply material to a component in a GLB |
| `GET` | `/materials` | List all available metals and gemstones |
| `POST` | `/budget-check` | Cost breakdown + substitution suggestions |
| `GET` | `/export/glb/{job_id}` | Download the final GLB file |
| `GET` | `/export/stl/{job_id}` | Convert GLB → STL and download |

### Example: Convert an image

```bash
# Upload
curl -X POST http://localhost:8000/convert \
  -F "file=@samples/ring.png"

# Response: {"job_id": "abc123", "status": "queued", ...}

# Track progress (SSE)
curl http://localhost:8000/status/abc123

# Download result
curl -o ring.glb http://localhost:8000/export/glb/abc123
```

---

## How the Pipeline Works

### Stage A: Background Removal
- Uses `rembg` to isolate the jewelry from the background
- Outputs a clean RGBA image

### Stage B: Multi-View Generation (Zero123++)
- Generates 6 views from a single image at different angles
- Views: front-right, right, back-right, back-left, left, front-left
- Each view is 320×320px

### Stage C: 3D Reconstruction (TripoSR)
- Takes the clean image and generates a 3D mesh
- Outputs a GLB file with geometry and basic texture

### Stage D: Segmentation → Vertex Mapping
- GroundingDINO detects jewelry components (metal, gemstone, prong, etc.)
- SAM2 generates precise masks for each component
- Masks are projected onto 3D vertices via UV mapping

### Stage E: GLB Metadata Injection
- Splits the mesh into semantic sub-meshes (one per component)
- Injects PBR materials and vertex labels into the GLB
- Result is a ready-to-render jewelry model

### VRAM Management (8GB budget)
Models are loaded/offloaded sequentially — only one heavy model in VRAM at a time:
```
GroundingDINO (~2GB) → offload → SAM2 (~3GB) → offload → Zero123++ (~5GB) → offload → TripoSR (~6GB) → offload
```

---

## Material System

### Metals (PBR metallic-roughness)
| Material | Roughness | Hex |
|----------|-----------|-----|
| Yellow Gold (18K) | 0.10 | `#FFC355` |
| White Gold (18K) | 0.05 | `#D9D9DE` |
| Rose Gold (18K) | 0.10 | `#E8B096` |
| Platinum | 0.02 | `#D4D4D9` |
| Sterling Silver | 0.15 | `#C7C7CC` |

### Gemstones (transmission + IOR + volume)
| Gemstone | IOR | Transmission | Hex |
|----------|-----|-------------|-----|
| Diamond | 2.42 | 0.95 | `#F8F8FF` |
| Ruby | 1.77 | 0.70 | `#E01224` |
| Sapphire | 1.77 | 0.70 | `#0F1F8C` |
| Emerald | 1.58 | 0.60 | `#1A993F` |
| Amethyst | 1.54 | 0.65 | `#8C33B3` |
| Cubic Zirconia | 2.15 | 0.90 | `#F2F2F7` |

Material swaps are **instant** (<50ms) — no AI re-inference needed. They work by editing glTF material properties in-memory.

---

## Frontend Architecture

```
App.jsx
 ├── Customizer.jsx          ← Left panel: upload, pickers, budget
 │    ├── Image upload (drag & drop)
 │    ├── "Convert to 3D" button + SSE progress bar
 │    ├── Metal picker (5 metals, color swatches)
 │    ├── Gemstone picker (6 gems, color swatches)
 │    ├── Budget input + suggestions
 │    └── Exporter.jsx       ← GLB/STL download buttons
 │
 └── JewelryViewer.jsx       ← Right panel: 3D canvas
      ├── Three.js Canvas (React Three Fiber)
      ├── OrbitControls (rotate/zoom/pan)
      ├── HDR Environment map (reflections)
      ├── ContactShadows
      └── Per-component material application
           └── HighIORMaterial.js  ← Custom shader for diamond IOR
```

**State flow:**
1. User uploads image → Customizer calls `POST /convert`
2. SSE tracks progress → on complete, fetches GLB blob
3. GLB blob → `JewelryViewer` loads and renders it
4. Material picker changes → `materialOverrides` prop updates → instant Three.js material swap
5. No server round-trip for material swaps!

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | Backend URL (frontend `.env`) |
| `JEWELRY_AI_TEMP` | `./temp` | Directory for generated files |

---

## Troubleshooting

### "CUDA out of memory"
- Make sure no other GPU processes are running: `nvidia-smi`
- The pipeline needs ~6GB VRAM at peak. Close other GPU apps.
- If you have <8GB VRAM, try reducing `triposr_chunk_size` in `pipeline.py`

### "GroundingDINO not found"
- Run `pip install groundingdino-py` inside the venv
- Make sure `models/groundingdino_swint_ogc.pth` exists (~694MB)

### "SAM2 not found"
- Run `pip install sam2` inside the venv
- Make sure `models/sam2.1_hiera_base_plus.pt` exists (~320MB)

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- Check CORS — the backend allows `localhost:5173` and `localhost:3000`
- Try setting `VITE_API_URL=http://127.0.0.1:8000` in `src/frontend/.env.local`

### Models downloading slowly
- Zero123++ and TripoSR download from HuggingFace on first use
- This is ~4.4GB total and may take a while on slow connections
- Once downloaded, they're cached in `~/.cache/huggingface/`

---

## Development

### Running tests
```bash
source venv/bin/activate
# TODO: Add pytest test suite
```

### Adding a new metal
1. Add entry to `METALS` dict in [src/backend/materials/definitions.py](src/backend/materials/definitions.py)
2. Mirror it in `METALS` const in [src/frontend/src/customizer/Customizer.jsx](src/frontend/src/customizer/Customizer.jsx)
3. Add price per gram in `METAL_PRICES` in [src/backend/budget/advisor.py](src/backend/budget/advisor.py)

### Adding a new gemstone
1. Add entry to `GEMSTONES` dict in [src/backend/materials/definitions.py](src/backend/materials/definitions.py) (include IOR, transmission, attenuation)
2. Mirror it in `GEMSTONES` const in [src/frontend/src/customizer/Customizer.jsx](src/frontend/src/customizer/Customizer.jsx)
3. Add price per carat in `GEMSTONE_PRICES` in [src/backend/budget/advisor.py](src/backend/budget/advisor.py)
4. Add similarity scores in `GEMSTONE_SIMILARITY` in [src/backend/budget/advisor.py](src/backend/budget/advisor.py)

### Backend hot-reload
```bash
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```
The `--reload` flag auto-restarts on file changes.

### Frontend hot-reload
```bash
cd src/frontend && npm run dev
```
Vite has instant HMR (Hot Module Replacement) out of the box.

---

## License

MIT
