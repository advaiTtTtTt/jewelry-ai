# Jewelry AI — User Guide

Welcome to **Jewelry AI**, an AI-powered tool that turns a simple jewelry photo into an interactive 3D model. You can customize materials, check budget, and export the model for web or 3D printing.

---

## How It Works (The Big Picture)

```
Your Photo  →  AI Detects Parts  →  Generates 3D Model  →  You Customize It  →  Export
```

You upload a photo of any jewelry piece (a ring, necklace, bracelet, etc.). The AI figures out which parts are metal, which are gemstones, prongs, settings, etc. Then it builds a 3D model from that single photo. Once the 3D model is ready, you can swap metals (gold, silver, platinum...) and gemstones (diamond, ruby, emerald...) **instantly** — no waiting! When you're happy with the design, download it as a GLB file (for websites/games) or STL file (for 3D printing).

---

## Step-by-Step Guide

### Step 1: Upload a Jewelry Image

- Open the app at `http://localhost:5173`
- You'll see the **Customizer panel** on the left and a **3D Viewer** on the right
- **Drag and drop** a jewelry photo onto the upload area, or **click** to browse your files
- Supported formats: JPEG, PNG, WebP (max 50MB)
- If demo images are available, you can click one to try it out

> **Tip:** Clear, well-lit photos with a plain background work best. The AI needs to see the jewelry clearly to detect its components.

### Step 2: Convert to 3D

- Click the **"🔮 Convert to 3D"** button
- A progress bar will appear showing each stage:
  - **Detecting components** — AI identifies metal, gemstones, prongs, etc.
  - **Generating 3D model** — This takes 1-2 minutes
  - **Preparing for customization** — Final processing
- Once complete, the 3D model appears in the viewer on the right
- You'll see a list of detected components (e.g., "Detected: metal, gemstone, setting")

> **Note:** The first conversion takes longer because AI model weights (~5GB) need to download from the internet. After that, they're cached locally.

### Step 3: Choose Your Metal

- In the **"Choose Metal"** section, click any of the 5 metal swatches:
  - 🟡 **Yellow Gold** (18K) — Classic warm gold
  - ⚪ **White Gold** (18K) — Cool silver-white
  - 🌸 **Rose Gold** (18K) — Pinkish copper tone
  - ⬜ **Platinum** — Highly polished, very white
  - 🩶 **Sterling Silver** — Slightly darker, affordable

- The 3D model updates **instantly** — no waiting, no server calls!
- The metal change applies to all metal components (band, prongs, setting, bail, clasp)

### Step 4: Choose Your Gemstone

- In the **"Choose Gemstone"** section, click any of the 6 gem swatches:
  - 💎 **Diamond** — Near-colorless, highest sparkle (IOR 2.42)
  - ❤️ **Ruby** — Deep red
  - 💙 **Sapphire** — Deep blue
  - 💚 **Emerald** — Rich green
  - 💜 **Amethyst** — Purple-violet
  - 🤍 **Cubic Zirconia** — Diamond alternative, very affordable

- Each gemstone has realistic optical properties including refraction, color absorption, and "fire" (rainbow sparkle)
- Like metals, gemstone swaps are **instant**!

### Step 5: Check Your Budget

- Enter your maximum budget in USD in the **"Budget Check"** field
- Click **"Check"**
- The system will show:
  - Your **current design cost** vs. your budget
  - Whether you're over or under budget
  - **Suggestions** for cheaper alternatives ranked by visual similarity
  - Example: "Replace diamond → cubic zirconia | Save $490 | 95% similar look"
- Click **"Apply"** on any suggestion to instantly swap that material

### Step 6: Export Your Design

- When you're happy with the result, use the **Export** section:
  - **📥 Download GLB** — For websites, games, AR/VR (includes materials and textures)
  - **🖨️ Download STL** — For 3D printing jewelry molds (geometry only)

---

## Understanding the 3D Viewer

- **Rotate** — Click and drag to spin the model
- **Zoom** — Scroll to zoom in/out
- **Pan** — Right-click and drag to move the view
- The model **auto-rotates** slowly, and pauses when you interact
- **HDR lighting** provides realistic reflections on metals and gems
- The status bar at the bottom shows detected components and the job ID

---

## What Happens Behind the Scenes

When you click "Convert to 3D", the AI runs a 5-stage pipeline on your photo:

| Stage | What It Does | Time |
|-------|-------------|------|
| **Background Removal** | Isolates the jewelry from its background | ~2s |
| **Component Detection** | AI identifies metal, gemstone, prongs, etc. using two models (GroundingDINO + SAM2) | ~10s |
| **Multi-View Generation** | Creates 6 views of the jewelry from different angles using Zero123++ | ~30s |
| **3D Reconstruction** | Builds a 3D mesh from the views using TripoSR | ~60s |
| **Assembly** | Maps detected components onto the 3D model and prepares for customization | ~5s |

All AI models run on your **GPU** (NVIDIA with 8GB+ VRAM). They load one at a time to stay within memory limits.

After the 3D model is built, all material swaps happen **instantly in your browser** — no AI needed! The material properties (color, shininess, transparency) are just numbers that Three.js applies in real-time.

---

## System Requirements

| Requirement | Minimum |
|------------|---------|
| **GPU** | NVIDIA with 8GB+ VRAM (RTX 3060, 4060, etc.) |
| **RAM** | 16GB recommended |
| **Disk** | ~20GB for AI model weights |
| **OS** | Linux or WSL2 (Ubuntu 22.04/24.04 tested) |
| **Python** | 3.10+ |
| **Node.js** | 18+ |

---

## Starting the App

### First-time setup
```bash
git clone https://github.com/advaiTtTtTt/jewelry-ai.git
cd jewelry-ai
chmod +x setup.sh
./setup.sh
```

### Starting the backend (Terminal 1)
```bash
source venv/bin/activate
uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

### Starting the frontend (Terminal 2)
```bash
cd frontend
npm run dev
```

### Open in browser
```
http://localhost:5173
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "CUDA out of memory" | Close other GPU apps. Run `nvidia-smi` to check. The pipeline needs ~6GB VRAM at peak. |
| Conversion is very slow | First run downloads ~5GB of model weights. Subsequent runs are faster. |
| Frontend can't connect | Make sure the backend is running on port 8000. Try `http://127.0.0.1:8000` |
| Model looks rough | TripoSR quality depends on the input photo. Try a clearer, well-lit image with a plain background. |
| Only "metal" component shows | This can happen with unusual angles or complex jewelry. Try a front-facing photo. |

---

## FAQ

**Q: Does it work with any jewelry photo?**
A: It works best with front-facing photos of rings, pendants, and earrings on plain backgrounds. Complex pieces like multi-strand necklaces may not reconstruct well.

**Q: Are the prices accurate?**
A: Prices are approximate retail estimates for mid-range quality. They're useful for relative comparisons and budgeting, not exact quotes.

**Q: Can I use the exported models commercially?**
A: The exported GLB/STL files are yours. The underlying AI models have their own licenses — check GroundingDINO, SAM2, Zero123++, and TripoSR licenses for commercial use terms.

**Q: Why does the diamond look so sparkly?**
A: Diamond has an index of refraction (IOR) of 2.42 — the highest of any common gemstone. This causes total internal reflection and chromatic dispersion ("fire"). We use a custom shader to render this accurately.
