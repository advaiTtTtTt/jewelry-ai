# Jewelry AI

**AI-powered 2D → 3D jewelry customization platform.**
Upload a photo of any jewelry piece, get a fully editable 3D model with real-time material swaps, procedural geometry editing, budget optimization, and export to GLB/STL.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![React](https://img.shields.io/badge/React-18-61dafb)
![Three.js](https://img.shields.io/badge/Three.js-r160-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What It Does

```
Photo of ring  →  AI segmentation  →  3D reconstruction  →  Interactive viewer
                  (GroundingDINO       (Zero123++ +           (Three.js + React)
                   + SAM2)              TripoSR)
                                              ↓
                              Material swaps · Geometry editor · Budget advisor · Export
```

1. **Upload** a jewelry image (ring, necklace, bracelet, pendant, etc.)
2. **AI detects** components — metal band, gemstones, prongs, settings, clasps
3. **Reconstructs** a 3D mesh via Zero123++ multi-view generation + TripoSR
4. **Maps** 2D segmentation masks to 3D vertex labels for per-component control
5. **View** the model in a real-time Three.js viewer with HDR lighting and contact shadows
6. **Swap materials** instantly — gold types, platinum, silver, diamond, ruby, sapphire, and more
7. **Edit geometry** — adjust band width, ring profile, gem cut, prong count, and more
8. **Check budget** — get cost breakdowns and substitution suggestions within your budget
9. **Export** as GLB (web/game-ready) or STL (3D printing)

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 18 + Vite 5 | UI framework with HMR |
| **3D Viewer** | Three.js r160 + React Three Fiber + Drei | WebGL rendering, OrbitControls, HDR |
| **Backend** | FastAPI + Uvicorn | REST API + SSE progress streaming |
| **Segmentation** | GroundingDINO + SAM2 | Open-set text-prompted detection + masks |
| **Multi-view** | Zero123++ (via diffusers) | Single image → 6 camera views |
| **Reconstruction** | TripoSR | Multi-view → 3D mesh (GLB) |
| **Mesh Processing** | trimesh + pygltflib | GLB manipulation + glTF extension support |
| **Background Removal** | rembg | Clean RGBA input images |
| **Gemstone Shader** | Custom Three.js material | High-IOR transmission for realistic gems |
| **HTTP Client** | axios | Frontend → backend requests |

---

## Prerequisites

- **Python 3.10+** — `python3 --version`
- **Node.js 18+** — `node --version`
- **npm** — `npm --version`
- **NVIDIA GPU** with at least **8GB VRAM** (tested on RTX 4060 Laptop GPU)
- **NVIDIA driver** installed — `nvidia-smi` should work
- **~20GB disk space** for model weights
- **Linux or WSL2** recommended (tested on Ubuntu 22.04/24.04)

> **Note:** You do NOT need the CUDA Toolkit installed — PyTorch ships with its own CUDA runtime.

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/advaiTtTtTt/jewelry-ai.git
cd jewelry-ai
```

### 2. Run the setup script

Handles everything — venv, PyTorch + CUDA, all pip dependencies, model weight downloads, and frontend npm install:

```bash
chmod +x setup.sh
./setup.sh
```

The script will:
- Create a Python virtual environment (`venv/`)
- Install PyTorch 2.6 with CUDA 12.4 support
- Install GroundingDINO (`groundingdino-py`) and SAM2 (`sam2`) via pip
- Download model weights (~1GB for GroundingDINO + SAM2)
- Clone TripoSR into `models/TripoSR/`
- Run `npm install` for the frontend

> **First-run note:** Zero123++ (~3.4GB) and TripoSR (~1GB) weights are auto-downloaded from HuggingFace on your first `/convert` request and cached in `~/.cache/huggingface/`.

### 3. Start the backend

```bash
source venv/bin/activate
PYTHONPATH=. uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

> The `backend/` folder at the project root is a namespace shim that re-exports `src/backend`, keeping import paths stable regardless of working directory.

### 4. Start the frontend (new terminal)

```bash
cd src/frontend
npm run dev
```

### 5. Open in browser

```
http://localhost:5173
```

The Vite dev server proxies `/api/*` calls to `http://localhost:8000` automatically — no CORS configuration needed in development.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JEWELRY_AI_TEMP` | `./temp` | Directory for generated job files |
| `SKIP_GEMINI_API` | `true` | Skip Gemini Vision; use rule-based geometry heuristics |
| `GEMINI_API_KEY` | *(unset)* | Google Gemini API key (only needed if `SKIP_GEMINI_API=false`) |
| `VITE_API_URL` | `http://localhost:8000` | Backend URL for the frontend (set in `src/frontend/.env.local`) |

A `.env` file at the project root is auto-loaded by the backend on startup. The default ships with `SKIP_GEMINI_API=true` for zero-config local usage.

---

## Project Structure

```
jewelry-ai/
├── src/
│   ├── backend/                     # Python FastAPI backend (source of truth)
│   │   ├── api.py                   # All REST endpoints + SSE + GPU lock
│   │   ├── segmentation/
│   │   │   └── detector.py          # GroundingDINO + SAM2 component detection
│   │   ├── reconstruction/
│   │   │   ├── pipeline.py          # Zero123++ → TripoSR pipeline
│   │   │   ├── ring_builder.py      # Procedural parametric ring geometry
│   │   │   ├── gem_builder.py       # Procedural gem geometry (brilliant, princess cut)
│   │   │   └── gemini_vision.py     # Gemini Vision parameter extraction (optional)
│   │   ├── materials/
│   │   │   ├── definitions.py       # PBR material database (metals + gemstones)
│   │   │   └── applier.py           # In-memory GLB material swapping via pygltflib
│   │   └── budget/
│   │       └── advisor.py           # Cost calculator + visual similarity substitutions
│   │
│   └── frontend/                    # React + Vite frontend
│       ├── src/
│       │   ├── App.jsx              # Root component — state wiring
│       │   ├── index.jsx            # React entry point
│       │   ├── customizer/
│       │   │   └── Customizer.jsx   # Left sidebar: upload, pickers, geometry, budget
│       │   ├── viewer/
│       │   │   └── JewelryViewer.jsx # Right panel: Three.js 3D canvas
│       │   ├── exporter/
│       │   │   └── Exporter.jsx     # GLB / STL download buttons
│       │   └── shaders/
│       │       └── HighIORMaterial.js # Custom shader for high-IOR gemstone rendering
│       ├── package.json
│       └── vite.config.js           # Vite config with /api proxy to localhost:8000
│
├── backend/                         # Namespace shim → re-exports src/backend
├── models/                          # AI model weights (gitignored)
│   ├── TripoSR/                     # TripoSR source (cloned by setup.sh)
│   ├── groundingdino_swint_ogc.pth  # GroundingDINO weights (~694MB)
│   └── sam2.1_hiera_base_plus.pt   # SAM2 base+ weights (~320MB)
│
├── temp/                            # Per-job generated files (gitignored)
├── samples/                         # Sample jewelry images for testing/demo
├── .env                             # Environment config (SKIP_GEMINI_API, etc.)
├── requirements.txt                 # Python dependencies
├── setup.sh                         # One-command full setup script
├── pyrightconfig.json               # Pyright / IDE type checking config
└── user_guide.md                    # End-user guide
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/convert` | Upload image → start async 2D→3D conversion, returns `job_id` |
| `GET` | `/status/{job_id}` | SSE stream for conversion progress (0–100%) |
| `POST` | `/customize` | Apply a material to a component (instant, <50ms) |
| `POST` | `/customize/geometry` | Rebuild ring geometry with new parameters |
| `GET` | `/materials` | List all metals and gemstones with PBR properties |
| `POST` | `/budget-check` | Cost breakdown + budget substitution suggestions |
| `GET` | `/export/glb/{job_id}` | Download the final GLB file |
| `GET` | `/export/stl/{job_id}` | Convert GLB → STL and download |
| `GET` | `/demo-images` | List available sample images |
| `GET` | `/demo-images/{filename}` | Serve a sample image file |
| `DELETE` | `/cleanup/{job_id}` | Remove job data and temp files (dev only) |

### Example: Full conversion flow

```bash
# 1. Upload image and start conversion
curl -X POST http://localhost:8000/convert \
  -F "file=@samples/ring.png"
# → {"job_id": "ab12cd34", "status": "queued", "message": "..."}

# 2. Stream progress (Server-Sent Events)
curl -N http://localhost:8000/status/ab12cd34
# → data: {"status": "running", "progress": 35, "message": "Generating 3D model..."}
# → data: {"status": "completed", "progress": 100, "result": {...}}

# 3. Download GLB
curl -o ring.glb http://localhost:8000/export/glb/ab12cd34

# 4. Download STL (3D printing)
curl -o ring.stl http://localhost:8000/export/stl/ab12cd34
```

### Example: Material swap

```bash
curl -X POST http://localhost:8000/customize \
  -H "Content-Type: application/json" \
  -d '{"job_id": "ab12cd34", "component": "metal", "material": "rose_gold"}'
```

### Example: Geometry edit

```bash
curl -X POST http://localhost:8000/customize/geometry \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "ab12cd34",
    "band_profile": "comfort_fit",
    "band_width_mm": 4.0,
    "band_thickness_mm": 1.8,
    "gem_cut": "princess",
    "prong_count": 4
  }'
```

### Example: Budget check

```bash
curl -X POST http://localhost:8000/budget-check \
  -H "Content-Type: application/json" \
  -d '{
    "design_config": {
      "gemstone": {"material": "diamond", "carats": 1.0},
      "metal": {"material": "platinum", "grams": 6.0}
    },
    "budget": 500.0,
    "min_similarity": 0.7
  }'
```

---

## Pipeline Deep Dive

### Stage A: Background Removal
- `rembg` isolates the jewelry from the background
- Outputs a clean RGBA image used by downstream models

### Stage B: Multi-View Generation (Zero123++)
- Generates 6 views from a single image at fixed azimuths (30°, 90°, 150°, 210°, 270°, 330°)
- Each view is 320×320px; total output is a 640×960 grid image
- Model loaded via diffusers `Zero123PlusPipeline`, offloaded to CPU after inference

### Stage C: 3D Reconstruction (TripoSR)
- Takes the clean background-removed image (not the multi-view grid)
- Outputs a 3D mesh exported as GLB
- Uses `scikit-image` marching cubes — `torchmcubes` is **not** required
- `triposr_chunk_size` (default 4096) controls VRAM usage during marching cubes

### Stage D: Segmentation → Vertex Mapping (GroundingDINO + SAM2)
- GroundingDINO detects bounding boxes using a text prompt: `"jewelry . ring . gemstone . prong . metal band . stone setting . bail . clasp . pendant"`
- SAM2 (`sam2.1_hiera_base_plus`) generates precise binary masks per bounding box
- Detected phrases are normalized to canonical labels: `metal`, `gemstone`, `prong`, `setting`, `bail`, `clasp`
- Masks are UV-projected onto 3D vertices; ties resolved by largest-mask-wins

### Stage E: GLB Metadata Injection + Mesh Split
- `applier.split_mesh_by_labels()` splits the mesh into per-component sub-meshes (one per semantic label)
- PBR materials are injected via pygltflib to fully support glTF transmission/IOR/volume extensions
- Resulting GLB is stored in `temp/{job_id}/jewelry_final.glb`

### VRAM Budget (8GB GPU)
Only one heavy model lives in VRAM at a time:
```
GroundingDINO (~2GB)  → detect → offload to CPU → empty_cache()
SAM2 (~3–4GB)         → mask  → offload to CPU → empty_cache()
Zero123++ (~5GB)       → views → offload to CPU → empty_cache()
TripoSR (~6GB)         → mesh  → offload to CPU → empty_cache()
```
A global `asyncio.Lock()` (`gpu_lock`) serializes concurrent API requests so they queue rather than OOM each other.

---

## Material System

### Metals — metallic-roughness PBR (glTF 2.0 core)

| Key | Name | Roughness | Hex |
|-----|------|-----------|-----|
| `yellow_gold` | Yellow Gold (18K) | 0.10 | `#FFC355` |
| `white_gold` | White Gold (18K) | 0.05 | `#D9D9DE` |
| `rose_gold` | Rose Gold (18K) | 0.10 | `#E8B096` |
| `platinum` | Platinum | 0.02 | `#D4D4D9` |
| `silver` | Sterling Silver | 0.15 | `#C7C7CC` |

### Gemstones — transmission + IOR + volume (glTF extensions)

Uses `KHR_materials_transmission`, `KHR_materials_ior`, and `KHR_materials_volume` — automatically mapped by Three.js `GLTFLoader` to `MeshPhysicalMaterial`.

| Key | Name | IOR | Transmission | Hex |
|-----|------|-----|-------------|-----|
| `diamond` | Diamond | 2.42 | 0.95 | `#F8F8FF` |
| `ruby` | Ruby | 1.77 | 0.70 | `#E01224` |
| `sapphire` | Sapphire | 1.77 | 0.70 | `#0F1F8C` |
| `emerald` | Emerald | 1.58 | 0.60 | `#1A993F` |
| `amethyst` | Amethyst | 1.54 | 0.65 | `#8C33B3` |
| `cubic_zirconia` | Cubic Zirconia | 2.15 | 0.90 | `#F2F2F7` |

Material swaps are **instant (<50ms)** — pygltflib edits glTF material JSON in-memory; no AI model is called.

---

## Geometry Customization

`POST /customize/geometry` rebuilds the ring parametrically using `RingBuilder` + `GemBuilder`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `band_profile` | string | `flat`, `comfort_fit`, `court`, `knife_edge` |
| `band_width_mm` | float | Width of the band in mm |
| `band_thickness_mm` | float | Wall thickness in mm |
| `inner_radius_mm` | float | Inner ring radius in mm |
| `has_gemstone` | bool | Include a center stone |
| `gem_cut` | string | `round_brilliant` or `princess` |
| `gem_radius_mm` | float | Gemstone radius in mm |
| `prong_count` | int | Number of prongs (0, 4, or 6) |

Each call merges against a **blueprint** stored on the job, so you only need to send changed parameters. Geometry changes trigger a fresh mesh split + material re-application, preserving the current material selections.

`GemBuilder` procedural shapes:
- **Round Brilliant** — crown (truncated cone) + pavilion (full cone pointing down), convex-hulled into a 57-facet gem
- **Princess Cut** — beveled square box

---

## Frontend Architecture

```
App.jsx
 ├── Customizer.jsx           ← Left panel (dark glass navy sidebar)
 │    ├── Drag-and-drop image upload + demo image browser
 │    ├── "Convert to 3D" button + live SSE progress bar
 │    ├── Metal picker (5 metals, gold-bordered color swatches)
 │    ├── Gemstone picker (6 gems, color swatches)
 │    ├── Geometry editor (profile, width, thickness, radius, cut, prongs)
 │    ├── Budget input + substitution suggestions
 │    └── Exporter.jsx         ← GLB / STL download buttons
 │
 └── JewelryViewer.jsx         ← Right panel: 3D canvas
      ├── React Three Fiber Canvas
      ├── OrbitControls (rotate / zoom / pan)
      ├── HDR Environment map (gem reflections)
      ├── ContactShadows
      └── Per-component material application
           └── HighIORMaterial.js  ← Custom Three.js shader for diamond-quality IOR
```

**State flow:**
1. User uploads image → `Customizer` calls `POST /convert` → receives `job_id`
2. `EventSource` on `GET /status/{job_id}` streams progress → on complete, fetches GLB from `GET /export/glb/{job_id}`
3. GLB blob URL passed as `modelUrl` prop to `JewelryViewer`
4. Material picker change → `POST /customize` → viewer re-loads updated GLB
5. Geometry editor change → `POST /customize/geometry` → viewer re-loads rebuilt GLB
6. Budget panel submit → `POST /budget-check` → suggestions rendered inline

---

## Budget Advisor

`BudgetAdvisor` calculates total design cost and suggests visually similar cheaper alternatives:

**Price database (approximate retail USD):**

| Gemstone | Price/carat |
|----------|-------------|
| Diamond | $500 |
| Ruby | $200 |
| Emerald | $180 |
| Sapphire | $150 |
| Amethyst | $20 |
| Cubic Zirconia | $10 |

| Metal | Price/gram |
|-------|-----------|
| Platinum | $100 |
| White Gold (18K) | $70 |
| Yellow Gold (18K) | $65 |
| Rose Gold (18K) | $63 |
| Sterling Silver | $2.50 |

A pre-computed **visual similarity matrix** scores gem pairs 0.0–1.0 (e.g., diamond ↔ cubic zirconia = 0.95). Pass `min_similarity` to filter suggestions: *"What looks at least 80% like a diamond but fits my $500 budget?"*

---

## Troubleshooting

### "CUDA out of memory"
- Ensure no other GPU processes are running: `nvidia-smi`
- Peak VRAM requirement is ~6GB (TripoSR stage)
- Reduce `triposr_chunk_size` in `src/backend/reconstruction/pipeline.py` (default: 4096)

### "GroundingDINO not found"
```bash
source venv/bin/activate
pip install groundingdino-py
```
Ensure `models/groundingdino_swint_ogc.pth` (~694MB) exists.

### "SAM2 not found"
```bash
source venv/bin/activate
pip install sam2
```
Ensure `models/sam2.1_hiera_base_plus.pt` (~320MB) exists.

### Frontend can't connect to backend
- Ensure backend is running on port 8000
- In dev, Vite's `/api` proxy handles CORS automatically
- For direct calls, `http://localhost:5173` is in the backend CORS `allow_origins` by default
- Set `VITE_API_URL=http://127.0.0.1:8000` in `src/frontend/.env.local` if needed

### Zero123++ / TripoSR downloading slowly
- First-run: ~4.4GB total from HuggingFace, cached in `~/.cache/huggingface/`
- Set `HF_HUB_OFFLINE=1` to prevent re-downloads once cached

### Gemstone shows as metal after geometry rebuild
- Fixed: `applier._expand_label_ranges()` now correctly handles both index-keyed (`{"0": "metal"}`) and label-string vertex_labels dicts

---

## Development

### Running the test pipeline

```bash
source venv/bin/activate
PYTHONPATH=. python test_pipeline.py
```

### Adding a new metal
1. Add entry to `METALS` in [src/backend/materials/definitions.py](src/backend/materials/definitions.py)
2. Add price per gram in `METAL_PRICES` in [src/backend/budget/advisor.py](src/backend/budget/advisor.py)
3. Mirror the key + hex color in the `METALS` const in [src/frontend/src/customizer/Customizer.jsx](src/frontend/src/customizer/Customizer.jsx)

### Adding a new gemstone
1. Add entry to `GEMSTONES` in [src/backend/materials/definitions.py](src/backend/materials/definitions.py) — include `ior`, `transmission`, `attenuation_color`, and `attenuation_distance`
2. Add price per carat in `GEMSTONE_PRICES` in [src/backend/budget/advisor.py](src/backend/budget/advisor.py)
3. Add pairwise similarity scores in `GEMSTONE_SIMILARITY` in [src/backend/budget/advisor.py](src/backend/budget/advisor.py)
4. Mirror the key + hex color in `GEMSTONES` const in [src/frontend/src/customizer/Customizer.jsx](src/frontend/src/customizer/Customizer.jsx)

### Backend hot-reload
```bash
PYTHONPATH=. uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

### Frontend hot-reload
```bash
cd src/frontend && npm run dev
```

### Demo mode (no Gemini API)
`.env` ships with `SKIP_GEMINI_API=true`. In this mode, ring geometry parameters fall back to rule-based heuristics instead of calling Gemini Vision — zero API quota usage for local development and demos.

---

## License

MIT
