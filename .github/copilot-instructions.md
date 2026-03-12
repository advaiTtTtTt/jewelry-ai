# Jewelry AI — GitHub Copilot Instructions

You are working on **Jewelry AI**, an AI-powered platform that converts 2D jewelry photos into interactive 3D models with real-time material customization and budget optimization.

## Architecture Overview

This is a **full-stack app** with:
- **Backend:** Python FastAPI server (`backend/`) — handles AI inference, mesh processing, material application
- **Frontend:** React 18 + Vite (`frontend/`) — 3D viewer (Three.js via React Three Fiber), customization UI
- **AI Models:** GroundingDINO (detection), SAM2 (segmentation), Zero123++ (multi-view), TripoSR (3D reconstruction)

## Core Pipeline (backend/reconstruction/pipeline.py)

The main conversion pipeline runs these stages sequentially. Each stage offloads its model from GPU after use to stay within 8GB VRAM:

```
Stage A: rembg background removal (~0.5GB VRAM)
Stage B: Zero123++ multi-view generation (~5GB VRAM) → 6 views at 320×320px
Stage C: TripoSR mesh reconstruction (~6GB VRAM) → GLB mesh
Stage D: UV projection to map 2D segmentation masks → 3D vertex labels
Stage E: GLB metadata injection via pygltflib
```

VRAM management is critical. Only ONE heavy model should be in GPU memory at a time. Always offload to CPU and call `torch.cuda.empty_cache()` between stages.

## Key Files and What They Do

### Backend

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `backend/api.py` | All REST endpoints. Routes: `/convert`, `/status/{id}`, `/customize`, `/materials`, `/budget-check`, `/export/glb/{id}`, `/export/stl/{id}` | `app`, `_run_conversion()`, `get_detector()`, `get_pipeline()` |
| `backend/segmentation/detector.py` | Uses GroundingDINO for open-set text-prompted jewelry part detection, then SAM2 for per-component masks | `JewelryDetector`, `detect()`, `DEFAULT_PROMPT`, `LABEL_MAP` |
| `backend/reconstruction/pipeline.py` | The full 2D→3D pipeline: background removal → multi-view → mesh → vertex labeling → GLB output | `ReconstructionPipeline`, `reconstruct()`, `_remove_background()`, `_generate_multiviews()`, `_reconstruct_mesh()` |
| `backend/materials/definitions.py` | PBR material database. Metals use metallic-roughness. Gemstones use glTF extensions: `KHR_materials_transmission`, `KHR_materials_ior`, `KHR_materials_volume` | `METALS`, `GEMSTONES`, `build_gltf_material_dict()` |
| `backend/materials/applier.py` | Swaps materials on GLB meshes by editing glTF properties in-memory. Must be <50ms. No AI re-inference | `MaterialApplier`, `apply_material()`, `split_mesh_by_labels()` |
| `backend/budget/advisor.py` | Calculates jewelry cost from material prices + component weights. Suggests budget-friendly substitutions ranked by visual similarity | `BudgetAdvisor`, `check_budget()`, `GEMSTONE_PRICES`, `METAL_PRICES`, `GEMSTONE_SIMILARITY` |

### Frontend

| File | Purpose | Key Components/Functions |
|------|---------|--------------------------|
| `frontend/src/App.jsx` | Root component. Wires state between Customizer (left panel) and JewelryViewer (right panel) | `glbBlob`, `materialOverrides`, `handleMaterialChange()` |
| `frontend/src/customizer/Customizer.jsx` | Main UI panel: image upload, "Convert to 3D" button with SSE progress, metal/gem pickers, budget check | `handleConvert()`, `handleMetalChange()`, `handleGemChange()`, `handleBudgetCheck()` |
| `frontend/src/viewer/JewelryViewer.jsx` | Three.js viewer: loads GLB, parses semantic components from node names/extras, applies per-component materials, OrbitControls, HDR env, contact shadows | `JewelryModel`, component mesh parsing logic |
| `frontend/src/exporter/Exporter.jsx` | Download buttons for GLB and STL export | `download()` |
| `frontend/src/shaders/HighIORMaterial.js` | Custom MeshPhysicalMaterial that patches the fragment shader to allow IOR > 2.333 (needed for diamond at 2.42) | `createHighIORMaterial()`, `buildMaterialFromDef()` |

## Important Design Decisions

1. **Material swaps are client-side only.** When a user picks a new metal or gem, `materialOverrides` in React state updates, and Three.js materials are swapped instantly. No server call is made. The `METALS` and `GEMSTONES` constants in `Customizer.jsx` MUST stay in sync with `definitions.py`.

2. **Lazy model loading.** AI models in `api.py` use lazy singletons (`get_detector()`, `get_pipeline()`, etc.) — they're loaded on first request, not at server startup.

3. **SSE for progress.** The `/convert` endpoint returns a `job_id` immediately. The client opens an `EventSource` to `/status/{job_id}` for real-time progress updates.

4. **Semantic vertex labels.** After reconstruction, each vertex is labeled ("metal", "gemstone", "prong", "setting", "bail", "clasp"). This is stored as compressed JSON in the GLB extras and as a sidecar `vertex_labels.json`.

5. **GLB is the interchange format.** Everything flows through glTF/GLB — the backend generates it, the frontend renders it, and the user exports it.

6. **Diamond IOR hack.** Three.js caps IOR at 2.333, but diamond needs 2.42. `HighIORMaterial.js` patches the fragment shader via `onBeforeCompile` to inject an uncapped `realIOR` uniform.

## Coding Conventions

- **Python:** Type hints on all function signatures. Docstrings on all classes and public methods. Logging via `logging.getLogger(__name__)`. Async where appropriate in FastAPI.
- **JavaScript/JSX:** Functional components with hooks. `useCallback` for event handlers passed as props. `useMemo` for expensive computations. Inline styles (no CSS framework yet).
- **Error handling:** Backend returns proper HTTP status codes with `detail` messages. Frontend shows errors in the UI.

## Feature Checklist — Review and Improve Each

Use this checklist to systematically review, test, and improve every feature. For each item, check the relevant files, verify it works end-to-end, and look for improvements.

---

### 1. Image Upload & Validation
**Files:** `Customizer.jsx` (frontend), `api.py` `/convert` route (backend)
**What to check:**
- Drag-and-drop upload works
- File type validation (only images)
- File size limit (50MB max)
- Image preview displays correctly
- Error messages shown for invalid files
**Potential improvements:**
- Add image compression/resizing before upload
- Support pasting images from clipboard
- Show image dimensions and file size in the UI
- Add a loading spinner during upload

---

### 2. 2D → 3D Conversion Pipeline
**Files:** `pipeline.py`, `detector.py`, `api.py` (`_run_conversion`)
**What to check:**
- Background removal produces clean RGBA output
- Zero123++ generates 6 coherent multi-view images
- TripoSR produces a valid GLB mesh
- VRAM usage stays under 8GB throughout
- Each stage properly offloads models to CPU
- Progress updates are accurate (0-100%)
**Potential improvements:**
- Add timeout handling for stuck conversions
- Cache intermediate results (views, meshes) to speed up re-runs
- Allow users to adjust reconstruction quality settings
- Add mesh post-processing (smoothing, decimation)
- Support batch conversion of multiple images

---

### 3. Segmentation & Vertex Labeling
**Files:** `detector.py`, `pipeline.py` (`_map_segmentation_to_vertices`)
**What to check:**
- GroundingDINO detects correct components with the prompt `"gemstone . prong . metal band . stone setting . bail . clasp"`
- SAM2 masks are precise and don't overlap
- Vertex label mapping is accurate — verify with `vertex_labels.json`
- Label normalization works (`LABEL_MAP` in `detector.py`)
**Potential improvements:**
- Allow user-editable detection prompts per jewelry type
- Add confidence thresholds that users can tune
- Visualize segmentation masks in the UI before 3D conversion
- Handle edge cases: multiple gemstones, no gemstone detected, unusual shapes
- Add "refine segmentation" feature for user corrections

---

### 4. Material System (PBR)
**Files:** `definitions.py`, `applier.py`, `HighIORMaterial.js`, `Customizer.jsx`
**What to check:**
- All metals render correctly with proper metallic-roughness PBR
- All gemstones have correct IOR, transmission, and volume absorption
- Diamond IOR (2.42) actually renders differently from default 1.5
- Material swap is under 50ms
- `definitions.py` and `Customizer.jsx` `METALS`/`GEMSTONES` are in sync
**Potential improvements:**
- Add more materials (titanium, brass, moissanite, topaz, etc.)
- Add engraving/texture patterns on metals
- Add gem cut types (brilliant, princess, emerald cut) as geometry modifiers
- Generate material thumbnails/previews in the picker
- Support custom user-defined materials

---

### 5. 3D Viewer
**Files:** `JewelryViewer.jsx`, `HighIORMaterial.js`
**What to check:**
- GLB loads and renders correctly
- OrbitControls work (rotate, zoom, pan)
- HDR environment produces realistic reflections
- Contact shadows render properly
- Component parsing from node names/extras works
- Material overrides apply to correct components
**Potential improvements:**
- Add multiple HDR environments (studio, outdoor, showroom)
- Add screenshot/snapshot feature
- Add measurement tool (ring diameter, gem dimensions)
- Implement camera animation presets (spin, zoom-in)
- Add wireframe/X-ray toggle for debugging
- Support AR preview (WebXR)

---

### 6. Budget Advisor
**Files:** `advisor.py`, `Customizer.jsx` (budget section), `api.py` `/budget-check`
**What to check:**
- Cost calculation uses correct per-carat/per-gram prices
- Substitution suggestions respect minimum visual similarity
- Suggestions are sorted by savings
- UI displays breakdown and suggestions clearly
**Potential improvements:**
- Add real-time market price API integration
- Support different quality grades (VS1 diamond vs SI2)
- Add currency selection (USD, EUR, INR, etc.)
- Show "similar designs within budget" gallery
- Let users lock certain components while optimizing others

---

### 7. Export (GLB/STL)
**Files:** `Exporter.jsx`, `api.py` `/export/glb/{id}` and `/export/stl/{id}`
**What to check:**
- GLB download includes all materials and metadata
- STL conversion preserves geometry correctly
- File naming is sensible (`jewelry_{job_id}.glb`)
- Large files download without timeout
**Potential improvements:**
- Add OBJ export format
- Add USDZ export for Apple AR Quick Look
- Include material info in export metadata
- Add resolution/quality options for STL export
- Zip multiple formats together

---

### 8. SSE Progress Tracking
**Files:** `api.py` `/status/{job_id}`, `Customizer.jsx` (`EventSource` handling)
**What to check:**
- SSE connection opens and receives events
- Progress percentage is accurate per stage
- Status messages are informative
- Connection cleanup on component unmount
- Handles server restart / connection drops gracefully
**Potential improvements:**
- Add estimated time remaining
- Add cancel conversion button
- Show which pipeline stage is currently running
- Persist job state across server restarts (Redis/DB)

---

### 9. Error Handling & Edge Cases
**Files:** All files
**What to check:**
- Backend returns proper HTTP error codes with `detail` messages
- Frontend displays errors in the UI (not just console)
- Handles: no GPU, model download failure, corrupt image, empty mesh
- File cleanup — temp files are deleted after export
**Potential improvements:**
- Add retry logic for transient failures
- Add error reporting/logging to a service
- Graceful degradation when GPU is unavailable
- Add input validation for all API endpoints

---

### 10. Performance & Optimization
**Files:** All files
**What to check:**
- Backend response times for each endpoint
- Frontend render performance (60fps in viewer)
- Memory leaks (Three.js geometry/material disposal)
- VRAM monitoring during pipeline execution
**Potential improvements:**
- Add request caching for repeated materials/budget checks
- Implement WebSocket instead of SSE for bidirectional communication
- Add CDN support for model weights
- Profile and optimize the Three.js render loop
- Add lazy loading for frontend components

---

## How to Add a New Feature

1. **Backend endpoint:** Add route in `api.py`, business logic in the appropriate submodule
2. **Frontend UI:** Add component in `frontend/src/`, import and wire in `App.jsx` or `Customizer.jsx`
3. **Keep material defs in sync:** If you change `definitions.py`, update `Customizer.jsx` too
4. **Test with sample images** in the `samples/` directory
5. **Check VRAM** with `nvidia-smi` during any AI inference changes

## Running the App

```bash
# Terminal 1: Backend
source venv/bin/activate
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev

# Open: http://localhost:5173
```

## Common Commands

```bash
# Check GPU usage
nvidia-smi

# Test a single endpoint
curl -X POST http://localhost:8000/convert -F "file=@samples/ring.png"

# Check available materials
curl http://localhost:8000/materials

# Run backend with debug logging
LOG_LEVEL=DEBUG uvicorn backend.api:app --reload --port 8000
```
