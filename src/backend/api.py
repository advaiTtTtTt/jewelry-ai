"""
FastAPI Backend — Jewelry AI
==============================
Routes:
  POST /convert          → image → GLB + vertex_labels (long-running, returns job_id)
  GET  /status/{job_id}  → SSE stream for conversion progress
  POST /customize        → GLB + change_request → updated GLB (instant, <50ms)
  GET  /materials        → all available metals + gemstones
  POST /budget-check     → design config + budget → cost breakdown + suggestions
  GET  /export/glb/{id}  → download final GLB
  GET  /export/stl/{id}  → convert GLB → STL and download

All endpoints have proper error handling, CORS enabled for Vite dev server.
"""

import asyncio
import io
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

# ─── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Jewelry AI",
    description="AI-driven 2D to 3D jewelry generation with real-time customization",
    version="1.0.0",
)

# CORS — allow Vite dev server and common origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Common React port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global GPU Lock ────────────────────────────────────────────────
# Ensures only one conversion uses the GPU at a time to prevent VRAM OOM.
gpu_lock = asyncio.Lock()

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("jewelry-ai")

# ─── State ──────────────────────────────────────────────────────────
# In-memory job tracking (in production, use Redis or a DB)
JOBS: dict = {}  # job_id → { status, progress, result, error }

# Temp directory for generated files
TEMP_DIR = Path(os.getenv("JEWELRY_AI_TEMP", "./temp"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Lazy-loaded pipeline instances (heavy models, loaded on first use)
_detector = None
_pipeline = None
_applier = None
_advisor = None


def get_detector():
    """Lazy-load the jewelry component detector."""
    global _detector
    if _detector is None:
        from backend.segmentation.detector import JewelryDetector
        _detector = JewelryDetector(device="cuda")
    return _detector


def get_pipeline():
    """Lazy-load the reconstruction pipeline."""
    global _pipeline
    from backend.reconstruction.pipeline import ReconstructionPipeline
    if _pipeline is None:
        _pipeline = ReconstructionPipeline(
            device="cuda",
            output_dir=str(TEMP_DIR),
        )
    return _pipeline


def get_applier():
    """Lazy-load the material applier."""
    global _applier
    if _applier is None:
        from backend.materials.applier import MaterialApplier
        _applier = MaterialApplier()
    return _applier


def get_advisor():
    """Lazy-load the budget advisor."""
    global _advisor
    if _advisor is None:
        from backend.budget.advisor import BudgetAdvisor
        _advisor = BudgetAdvisor()
    return _advisor


# ═══════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════

class CustomizeRequest(BaseModel):
    job_id: str
    component: str      # "metal", "gemstone", "prong", "setting", etc.
    material: str       # "yellow_gold", "diamond", etc.


class GeometryRequest(BaseModel):
    job_id: str
    band_profile: Optional[str] = None
    band_width_mm: Optional[float] = None
    band_thickness_mm: Optional[float] = None
    inner_radius_mm: Optional[float] = None
    has_gemstone: Optional[bool] = None
    gem_cut: Optional[str] = None
    gem_radius_mm: Optional[float] = None
    prong_count: Optional[int] = None


class BudgetRequest(BaseModel):
    design_config: dict  # {"gemstone": {"material": "diamond", "carats": 1.0}, ...}
    budget: float
    min_similarity: Optional[float] = 0.0


class ConvertResponse(BaseModel):
    job_id: str
    status: str
    message: str


class MaterialsResponse(BaseModel):
    metals: dict
    gemstones: dict


# ═══════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "Jewelry AI", "version": "1.0.0"}


# ─── POST /convert ──────────────────────────────────────────────────
@app.post("/convert", response_model=ConvertResponse)
async def convert_image(file: UploadFile = File(...)):
    """
    Upload a jewelry image and start 2D → 3D conversion.

    The conversion runs asynchronously. Use GET /status/{job_id} to
    track progress via Server-Sent Events.

    Returns a job_id immediately.
    """
    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")

    # Read image data
    image_data = await file.read()
    if len(image_data) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    if len(image_data) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    # Create job
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Job queued",
        "result": None,
        "error": None,
    }

    # Save uploaded image
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input_{file.filename or 'image.png'}"
    with open(input_path, "wb") as f:
        f.write(image_data)

    # Launch async conversion with a timeout wrapper (15 minutes max)
    async def _run_conversion_with_timeout():
        try:
            await asyncio.wait_for(_run_conversion(job_id, input_path), timeout=900)
        except asyncio.TimeoutError:
            logger.error("[%s] Conversion timed out after 15 minutes", job_id)
            if job_id in JOBS:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["error"] = "Conversion timed out after 15 minutes"
                JOBS[job_id]["message"] = "Conversion timed out. Please try a simpler image or lower resolution."
        except Exception as e:
            logger.exception("[%s] Unexpected error in conversion task", job_id)

    asyncio.create_task(_run_conversion_with_timeout())

    return ConvertResponse(
        job_id=job_id,
        status="queued",
        message="Conversion started. Use GET /status/{job_id} to track progress.",
    )


async def _run_conversion(job_id: str, image_path: Path):
    """
    Run the full conversion pipeline in the background.

    Stages:
      1. Component Detection (GroundingDINO + SAM2)
      2. Multi-view Generation (Zero123++)
      3. 3D Reconstruction (TripoSR)
      4. Segmentation → Vertex Label Mapping
      5. GLB Metadata Injection + Mesh Splitting
    """
    acquired_lock = False
    acquired_lock = False
    try:
        from PIL import Image

        # Wait for other jobs to finish using the GPU
        JOBS[job_id]["message"] = "Waiting in queue for GPU..."

        await gpu_lock.acquire()
        acquired_lock = True
        JOBS[job_id]["status"] = "running"

        # Load image
        JOBS[job_id]["message"] = "Loading image..."
        JOBS[job_id]["progress"] = 5
        image = Image.open(image_path).convert("RGB")

        # Stage 1: Component Detection
        JOBS[job_id]["message"] = "Detecting jewelry components (GroundingDINO + SAM2)..."
        JOBS[job_id]["progress"] = 10
        detector = get_detector()
        segmentation = await asyncio.to_thread(detector.detect, image)
        detected_parts = list(segmentation["parts"].keys())
        JOBS[job_id]["message"] = f"Detected: {', '.join(detected_parts)}"
        JOBS[job_id]["progress"] = 30

        # Stage 2+3+4: Reconstruction (Zero123++ → TripoSR → vertex labels)
        JOBS[job_id]["message"] = "Generating 3D model (this takes 1-2 minutes)..."
        JOBS[job_id]["progress"] = 35
        pipeline = get_pipeline()
        result = await asyncio.to_thread(pipeline.reconstruct, image, segmentation)
        JOBS[job_id]["progress"] = 80

        # Stage 5: Split mesh by semantic labels for per-component material control
        JOBS[job_id]["message"] = "Preparing mesh for customization..."
        JOBS[job_id]["progress"] = 85
        applier = get_applier()
        split_glb = await asyncio.to_thread(
            applier.split_mesh_by_labels,
            result["glb_path"],
            result["vertex_labels"],
        )

            # Save split GLB
        job_dir = TEMP_DIR / job_id
        final_path = job_dir / "jewelry_final.glb"
        with open(final_path, "wb") as f:
            f.write(split_glb)

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["message"] = "Conversion complete!"
        JOBS[job_id]["result"] = {
            "glb_path": str(final_path),
            "vertex_labels": result["vertex_labels"],
            "detected_parts": detected_parts,
            "job_id": job_id,
            "blueprint": result.get("blueprint", {}),
        }

        logger.info("[%s] Conversion completed successfully", job_id)

    except Exception as e:
        logger.exception("[%s] Conversion failed", job_id)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["message"] = f"Error: {str(e)}"
    finally:
        if acquired_lock:
            gpu_lock.release()


# ─── GET /status/{job_id} (SSE) ────────────────────────────────────
@app.get("/status/{job_id}")
async def job_status(job_id: str):
    """
    Server-Sent Events stream for tracking conversion progress.

    Sends events in format:
      data: {"status": "running", "progress": 45, "message": "Generating views..."}

    Closes when status is "completed" or "failed".
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    async def event_generator():
        last_progress = -1
        while True:
            job = JOBS.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'not_found'})}\n\n"
                break

            # Only send when progress or status changes
            if job["progress"] != last_progress or job["status"] in ["completed", "failed"]:
                last_progress = job["progress"]
                event_data = {
                    "status": job["status"],
                    "progress": job["progress"],
                    "message": job["message"],
                }

                if job["status"] == "completed":
                    event_data["result"] = {
                        "job_id": job_id,
                        "detected_parts": job["result"].get("detected_parts", []),
                        "vertex_labels": job["result"].get("vertex_labels", {}),
                        "blueprint": job["result"].get("blueprint", {}),
                    }

                if job["status"] == "failed":
                    event_data["error"] = job.get("error", "Unknown error")

                yield f"data: {json.dumps(event_data)}\n\n"

                # Terminal states
                if job["status"] in ("completed", "failed"):
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ─── POST /customize ───────────────────────────────────────────────
@app.post("/customize")
async def customize_material(request: CustomizeRequest):
    """
    Apply a material change to a specific component in-place.

    This is instant (<50ms) — no AI models are called.
    Returns the updated GLB as a binary download.
    """
    if request.job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{request.job_id}' not found")

    job = JOBS[request.job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not yet completed")

    glb_path = job["result"]["glb_path"]
    if not Path(glb_path).exists():
        raise HTTPException(status_code=404, detail="GLB file not found")

    try:
        from backend.materials.definitions import get_material
        # Validate material exists
        get_material(request.material)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        applier = get_applier()
        updated_glb = await asyncio.to_thread(
            applier.apply_material,
            glb_path,
            request.component,
            request.material,
        )

        # Save the updated GLB (overwrite)
        with open(glb_path, "wb") as f:
            f.write(updated_glb)

        return Response(
            content=updated_glb,
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition": f'attachment; filename="jewelry_{request.job_id}.glb"',
                "Content-Length": str(len(updated_glb)),
            },
        )

    except Exception as e:
        logger.exception("Material customization failed")
        raise HTTPException(status_code=500, detail=f"Customization failed: {str(e)}")


# ─── POST /customize/geometry ─────────────────────────────────────
@app.post("/customize/geometry")
async def customize_geometry(request: GeometryRequest):
    if request.job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{request.job_id}' not found")

    job = JOBS[request.job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not yet completed")

    try:
        from backend.reconstruction.gemini_vision import _default_params
        from backend.reconstruction.ring_builder import RingBuilder
        from backend.reconstruction.pipeline import ReconstructionPipeline
        import trimesh
    except Exception as e:
        logger.exception("Geometry customization imports failed")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

    # Merge stored blueprint with defaults and incoming overrides
    base_params = job.get("result", {}).get("blueprint") or {}
    params = {**_default_params(), **base_params}
    overrides = request.model_dump(exclude_none=True)
    overrides.pop("job_id", None)
    params.update(overrides)

    try:
        builder = RingBuilder()
        meshes = builder.build_ring(params)
        combined = trimesh.util.concatenate(meshes)

        vertex_labels = {}
        offset = 0
        for m in meshes:
            label = m.metadata.get("semantic_label", "metal")
            for j in range(len(m.vertices)):
                vertex_labels[offset + j] = label
            offset += len(m.vertices)

        job_dir = Path(job["result"]["glb_path"]).parent
        raw_glb_path = job_dir / "custom_raw.glb"
        combined.export(str(raw_glb_path))

        pipeline = ReconstructionPipeline(device="cpu", output_dir=str(job_dir))
        final_glb_path = job_dir / "jewelry_custom.glb"
        pipeline._inject_glb_metadata(raw_glb_path, final_glb_path, vertex_labels)

        applier = get_applier()
        split_glb = await asyncio.to_thread(
            applier.split_mesh_by_labels,
            final_glb_path,
            vertex_labels,
        )

        with open(final_glb_path, "wb") as f:
            f.write(split_glb)

        job["result"].update({
            "glb_path": str(final_glb_path),
            "vertex_labels": vertex_labels,
            "blueprint": params,
        })

        return {
            "job_id": request.job_id,
            "blueprint": params,
            "vertex_labels": vertex_labels,
        }

    except Exception as e:
        logger.exception("Geometry customization failed")
        raise HTTPException(status_code=500, detail=f"Geometry customization failed: {str(e)}")


# ─── GET /materials ─────────────────────────────────────────────────
@app.get("/materials", response_model=MaterialsResponse)
async def list_materials():
    """Return all available metals and gemstones with their PBR properties."""
    from backend.materials.definitions import get_all_materials
    return get_all_materials()


# ─── POST /budget-check ────────────────────────────────────────────
@app.post("/budget-check")
async def budget_check(request: BudgetRequest):
    """
    Calculate design cost and suggest budget-friendly substitutions.

    Expects a design_config dict mapping component names to materials:
    {
        "gemstone": {"material": "diamond", "carats": 1.0},
        "metal": {"material": "yellow_gold", "grams": 5.0}
    }
    """
    advisor = get_advisor()

    try:
        result = advisor.suggest_substitutions(
            design_config=request.design_config,
            budget=request.budget,
            min_similarity=request.min_similarity or 0.0,
        )
        return result
    except Exception as e:
        logger.exception("Budget check failed")
        raise HTTPException(status_code=500, detail=f"Budget check failed: {str(e)}")


# ─── GET /export/glb/{job_id} ──────────────────────────────────────
@app.get("/export/glb/{job_id}")
async def export_glb(job_id: str):
    """Download the final GLB file for a completed job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not yet completed")

    glb_path = Path(job["result"]["glb_path"])
    if not glb_path.exists():
        raise HTTPException(status_code=404, detail="GLB file not found")

    return FileResponse(
        path=str(glb_path),
        media_type="model/gltf-binary",
        filename=f"jewelry_{job_id}.glb",
    )


# ─── GET /export/stl/{job_id} ──────────────────────────────────────
@app.get("/export/stl/{job_id}")
async def export_stl(job_id: str):
    """
    Convert GLB to STL and download.

    STL is generated on-the-fly from the GLB mesh using trimesh.
    STL format is widely used for 3D printing jewelry molds.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = JOBS[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job is not yet completed")

    glb_path = Path(job["result"]["glb_path"])
    if not glb_path.exists():
        raise HTTPException(status_code=404, detail="GLB file not found")

    try:
        import trimesh

        # Load GLB and convert to STL
        scene = trimesh.load(str(glb_path))
        if isinstance(scene, trimesh.Scene):
            # Merge all geometries into a single mesh for STL export
            mesh = scene.dump(concatenate=True)
        else:
            mesh = scene

        stl_bytes = mesh.export(file_type="stl")

        return Response(
            content=stl_bytes,
            media_type="application/sla",
            headers={
                "Content-Disposition": f'attachment; filename="jewelry_{job_id}.stl"',
                "Content-Length": str(len(stl_bytes)),
            },
        )

    except Exception as e:
        logger.exception("STL export failed")
        raise HTTPException(status_code=500, detail=f"STL export failed: {str(e)}")


# ─── Demo data endpoint ────────────────────────────────────────────
@app.get("/demo-images")
async def list_demo_images():
    """List available demo jewelry images."""
    samples_dir = Path(__file__).resolve().parents[2] / "samples"
    if not samples_dir.exists():
        return {"images": []}

    images = []
    for f in sorted(samples_dir.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            images.append({
                "name": f.stem.replace("_", " ").title(),
                "filename": f.name,
                "path": f"/demo-images/{f.name}",
            })
    return {"images": images}


@app.get("/demo-images/{filename}")
async def get_demo_image(filename: str):
    """Serve a demo image file."""
    samples_dir = Path(__file__).resolve().parents[2] / "samples"
    file_path = samples_dir / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Demo image not found")

    # Basic path traversal protection
    if not file_path.resolve().is_relative_to(samples_dir.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(str(file_path))


# ─── Cleanup endpoint (dev only) ───────────────────────────────────
@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Remove job data and temporary files. For development use."""
    if job_id in JOBS:
        del JOBS[job_id]

    job_dir = TEMP_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    return {"status": "cleaned", "job_id": job_id}


# ─── App startup/shutdown ──────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Jewelry AI backend starting up...")
    logger.info("Temp directory: %s", TEMP_DIR.resolve())
    logger.info("Models will be loaded on first request (lazy loading)")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down — cleaning up models...")
    global _detector, _pipeline
    if _detector:
        _detector.cleanup()
    if _pipeline:
        _pipeline.cleanup()
    logger.info("Shutdown complete")
