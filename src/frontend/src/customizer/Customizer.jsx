/**
 * Customizer.jsx
 * ===============
 * Main UI for the jewelry customization workflow:
 *   - Image upload (drag-and-drop)
 *   - "Convert to 3D" button with progress tracking via SSE
 *   - Metal picker (instant client-side material swap)
 *   - Gemstone picker (instant client-side swap, custom IOR for diamond)
 *   - Budget input with substitution suggestions
 *   - Export buttons (GLB + STL)
 *   - Demo mode with sample images
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import Exporter from '../exporter/Exporter';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ═══════════════════════════════════════════════════════════════════════
// MATERIAL DEFINITIONS (mirrors backend/materials/definitions.py)
// Kept in sync so client-side swaps are instant with no server calls.
// ═══════════════════════════════════════════════════════════════════════

const METALS = {
  yellow_gold:  { type: 'metal', name: 'Yellow Gold',  color: [1.0, 0.766, 0.336], metallic: 1.0, roughness: 0.1,  hex: '#FFC355' },
  white_gold:   { type: 'metal', name: 'White Gold',   color: [0.85, 0.85, 0.87],  metallic: 1.0, roughness: 0.05, hex: '#D9D9DE' },
  rose_gold:    { type: 'metal', name: 'Rose Gold',    color: [0.91, 0.69, 0.59],  metallic: 1.0, roughness: 0.1,  hex: '#E8B096' },
  platinum:     { type: 'metal', name: 'Platinum',     color: [0.83, 0.83, 0.85],  metallic: 1.0, roughness: 0.02, hex: '#D4D4D9' },
  silver:       { type: 'metal', name: 'Silver',       color: [0.78, 0.78, 0.80],  metallic: 1.0, roughness: 0.15, hex: '#C7C7CC' },
};

const GEMSTONES = {
  diamond:        { type: 'gemstone', name: 'Diamond',    color: [0.98, 0.98, 1.0],  ior: 2.42, transmission: 0.95, roughness: 0.0, attenuation_color: [1,1,1], attenuation_distance: 100, thickness: 0.5, dispersion: 0.044, hex: '#F8F8FF' },
  ruby:           { type: 'gemstone', name: 'Ruby',       color: [0.88, 0.07, 0.14], ior: 1.77, transmission: 0.7,  roughness: 0.0, attenuation_color: [0.9,0.05,0.1], attenuation_distance: 2, thickness: 0.4, dispersion: 0.018, hex: '#E01224' },
  sapphire:       { type: 'gemstone', name: 'Sapphire',   color: [0.06, 0.12, 0.55], ior: 1.77, transmission: 0.7,  roughness: 0.0, attenuation_color: [0.05,0.1,0.5], attenuation_distance: 2, thickness: 0.4, dispersion: 0.018, hex: '#0F1F8C' },
  emerald:        { type: 'gemstone', name: 'Emerald',    color: [0.10, 0.60, 0.25], ior: 1.58, transmission: 0.6,  roughness: 0.02, attenuation_color: [0.08,0.55,0.2], attenuation_distance: 1.5, thickness: 0.4, dispersion: 0.014, hex: '#1A993F' },
  amethyst:       { type: 'gemstone', name: 'Amethyst',   color: [0.55, 0.20, 0.70], ior: 1.54, transmission: 0.65, roughness: 0.01, attenuation_color: [0.5,0.18,0.65], attenuation_distance: 3, thickness: 0.5, dispersion: 0.013, hex: '#8C33B3' },
  cubic_zirconia: { type: 'gemstone', name: 'Cubic Zirconia', color: [0.95, 0.95, 0.97], ior: 2.15, transmission: 0.9, roughness: 0.0, attenuation_color: [1,1,1], attenuation_distance: 80, thickness: 0.5, dispersion: 0.058, hex: '#F2F2F7' },
};

// ═══════════════════════════════════════════════════════════════════════
// Customizer Component
// ═══════════════════════════════════════════════════════════════════════

export default function Customizer({ onMaterialChange, onGlbLoaded, jobId, setJobId }) {
  // ─── State ───────────────────────────────────────────────────────
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageInfo, setImageInfo] = useState(null);
  const [isProcessingImage, setIsProcessingImage] = useState(false);
  const [converting, setConverting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState('');
  const [error, setError] = useState(null);
  const [selectedMetal, setSelectedMetal] = useState('yellow_gold');
  const [selectedGem, setSelectedGem] = useState('diamond');
  const [budget, setBudget] = useState('');
  const [budgetResult, setBudgetResult] = useState(null);
  const [detectedParts, setDetectedParts] = useState([]);
  const [demoImages, setDemoImages] = useState([]);
  const [geometryBusy, setGeometryBusy] = useState(false);
  const [shapeParams, setShapeParams] = useState({
    band_profile: 'D-shape',
    band_width_mm: 3.5,
    band_thickness_mm: 2.5,
    inner_radius_mm: 8.5,
    has_gemstone: true,
    gem_cut: 'round_brilliant',
    gem_radius_mm: 3.5,
    prong_count: 4,
  });

  const fileInputRef = useRef(null);

  const getCurrentMaterials = useCallback(() => {
    const currentMetal = METALS[selectedMetal];
    const currentGem = GEMSTONES[selectedGem];
    return {
      metal: currentMetal,
      prong: currentMetal,
      setting: currentMetal,
      bail: currentMetal,
      clasp: currentMetal,
      gemstone: currentGem,
    };
  }, [selectedMetal, selectedGem]);

  const updateShape = useCallback((field, value) => {
    setShapeParams((prev) => ({ ...prev, [field]: value }));
  }, []);

  // Load demo images on mount
  useEffect(() => {
    fetch(`${API_BASE}/demo-images`)
      .then(r => r.json())
      .then(data => setDemoImages(data.images || []))
      .catch(() => {});
  }, []);

  // Handle global paste event for images
  useEffect(() => {
    const handlePaste = (e) => {
      const file = e.clipboardData?.files?.[0];
      if (file && file.type.startsWith('image/')) {
        handleImageSelect(file);
      }
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, []);

  // ─── Image Upload ────────────────────────────────────────────────
  const handleImageSelect = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Only image files are supported');
      return;
    }

    setIsProcessingImage(true);
    setError(null);
    setJobId(null);
    setImageInfo(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        // Resize logic (max 1024x1024)
        const MAX_DIM = 1024;
        let { width, height } = img;
        if (width > MAX_DIM || height > MAX_DIM) {
          const ratio = Math.min(MAX_DIM / width, MAX_DIM / height);
          width = Math.round(width * ratio);
          height = Math.round(height * ratio);
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);

        canvas.toBlob((blob) => {
          if (!blob) {
            setIsProcessingImage(false);
            setError('Failed to process image');
            return;
          }
          
          const rawName = file.name ? file.name.replace(/\.[^/.]+$/, "") : "pasted_image";
          const compressedFile = new File([blob], `${rawName}.webp`, {
            type: "image/webp",
          });
          
          const sizeKb = (blob.size / 1024).toFixed(1);
          setImageInfo({ width, height, size: `${sizeKb} KB` });
          
          setImageFile(compressedFile);
          setImagePreview(URL.createObjectURL(compressedFile));
          setIsProcessingImage(false);
        }, "image/webp", 0.9);
      };
      img.onerror = () => {
        setIsProcessingImage(false);
        setError('Invalid image file');
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }, [setJobId]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) {
      handleImageSelect(file);
    }
  }, [handleImageSelect]);

  const handleDragOver = (e) => e.preventDefault();

  // ─── Convert to 3D ──────────────────────────────────────────────
  const handleConvert = useCallback(async () => {
    if (!imageFile || converting) return;

    setConverting(true);
    setProgress(0);
    setStatusMsg('Uploading image...');
    setError(null);

    try {
      // Upload image
      const formData = new FormData();
      formData.append('file', imageFile);

      const response = await fetch(`${API_BASE}/convert`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed: ${response.statusText}`);
      }

      const { job_id } = await response.json();
      setJobId(job_id);

      // Track progress via SSE
      const eventSource = new EventSource(`${API_BASE}/status/${job_id}`);

      eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setProgress(data.progress || 0);
        setStatusMsg(data.message || '');

        if (data.status === 'completed') {
          eventSource.close();
          setConverting(false);
          setDetectedParts(data.result?.detected_parts || []);
          if (data.result?.blueprint) {
            setShapeParams((prev) => ({ ...prev, ...data.result.blueprint }));
          }

          // Fetch the GLB and pass to viewer
          fetch(`${API_BASE}/export/glb/${job_id}`)
            .then(r => r.blob())
            .then(blob => {
              if (onGlbLoaded) {
                const initialMaterials = getCurrentMaterials();
                onGlbLoaded(blob, data.result?.vertex_labels, initialMaterials);
              }
            });
        }

        if (data.status === 'failed') {
          eventSource.close();
          setConverting(false);
          setError(data.error || 'Conversion failed');
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        setConverting(false);
        setError('Lost connection to server');
      };

    } catch (err) {
      setConverting(false);
      setError(err.message);
    }
  }, [imageFile, converting, onGlbLoaded, setJobId, getCurrentMaterials]);

  // ─── Material Swap (instant, client-side) ────────────────────────
  const handleMetalChange = useCallback((metalKey) => {
    setSelectedMetal(metalKey);
    // Apply to all metal-type components instantly via Three.js
    if (onMaterialChange) {
      onMaterialChange('metal', METALS[metalKey]);
      onMaterialChange('prong', METALS[metalKey]);
      onMaterialChange('setting', METALS[metalKey]);
      onMaterialChange('bail', METALS[metalKey]);
      onMaterialChange('clasp', METALS[metalKey]);
    }
  }, [onMaterialChange]);

  const handleGemChange = useCallback((gemKey) => {
    setSelectedGem(gemKey);
    if (onMaterialChange) {
      onMaterialChange('gemstone', GEMSTONES[gemKey]);
    }
  }, [onMaterialChange]);

  // ─── Budget Check ────────────────────────────────────────────────
  const handleBudgetCheck = useCallback(async () => {
    const budgetVal = parseFloat(budget);
    if (isNaN(budgetVal) || budgetVal <= 0) return;

    try {
      const response = await fetch(`${API_BASE}/budget-check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          design_config: {
            gemstone: { material: selectedGem, carats: 1.0 },
            metal: { material: selectedMetal, grams: 5.0 },
          },
          budget: budgetVal,
        }),
      });

      if (!response.ok) throw new Error('Budget check failed');
      const result = await response.json();
      setBudgetResult(result);
    } catch (err) {
      console.error('Budget check error:', err);
    }
  }, [budget, selectedGem, selectedMetal]);

  const handleApplyGeometry = useCallback(async () => {
    if (!jobId || geometryBusy) return;
    setGeometryBusy(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/customize/geometry`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId, ...shapeParams }),
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Geometry update failed');
      }
      const data = await response.json();
      if (data.blueprint) {
        setShapeParams((prev) => ({ ...prev, ...data.blueprint }));
      }
      const glbResp = await fetch(`${API_BASE}/export/glb/${jobId}`);
      const blob = await glbResp.blob();
      if (onGlbLoaded) {
        onGlbLoaded(blob, data.vertex_labels, getCurrentMaterials());
      }
    } catch (err) {
      setError(err.message || 'Geometry update failed');
    } finally {
      setGeometryBusy(false);
    }
  }, [jobId, geometryBusy, shapeParams, onGlbLoaded, getCurrentMaterials]);

  // ─── Demo image handler ──────────────────────────────────────────
  const handleDemoClick = useCallback(async (demo) => {
    try {
      const response = await fetch(`${API_BASE}${demo.path}`);
      const blob = await response.blob();
      const file = new File([blob], demo.filename, { type: blob.type });
      handleImageSelect(file);
    } catch (err) {
      console.error('Failed to load demo image:', err);
    }
  }, [handleImageSelect]);

  // ─── Render ──────────────────────────────────────────────────────
  return (
    <div style={panelStyle}>
      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
        .spinner { display: inline-block; animation: spin 2s linear infinite; }
      `}</style>
      
      {/* Header */}
      <h2 style={{ margin: '0 0 16px 0', color: '#f0f0f0', fontSize: '20px' }}>
        💎 Jewelry AI Customizer
      </h2>

      {/* ── Image Upload ─────────────────────────────────────── */}
      <Section title="1. Upload Image">
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
          style={dropzoneStyle(imagePreview)}
        >
          {isProcessingImage ? (
            <div style={{ textAlign: 'center', color: '#888' }}>
              <div style={{ fontSize: '24px', marginBottom: '8px' }} className="spinner">⏳</div>
              <div>Processing image...</div>
            </div>
          ) : imagePreview ? (
            <div style={{ textAlign: 'center' }}>
              <img src={imagePreview} alt="Preview" style={{ maxHeight: '160px', borderRadius: '8px' }} />
              {imageInfo && (
                <div style={{ fontSize: '11px', color: '#888', marginTop: '6px' }}>
                  {imageInfo.width} × {imageInfo.height} px • {imageInfo.size}
                </div>
              )}
            </div>
          ) : (
            <div style={{ textAlign: 'center', color: '#888' }}>
              <div style={{ fontSize: '32px', marginBottom: '8px' }}>📷</div>
              <div>Drag & drop, paste, or click to upload</div>
              <div style={{ fontSize: '12px', marginTop: '4px' }}>JPEG, PNG, WebP • Auto-resized</div>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => handleImageSelect(e.target.files?.[0])}
          />
        </div>

        {/* Demo images */}
        {demoImages.length > 0 && (
          <div style={{ marginTop: '8px' }}>
            <div style={{ fontSize: '12px', color: '#888', marginBottom: '4px' }}>Or try a demo:</div>
            <div style={{ display: 'flex', gap: '8px' }}>
              {demoImages.map((demo) => (
                <button
                  key={demo.filename}
                  onClick={() => handleDemoClick(demo)}
                  style={demoBtnStyle}
                >
                  {demo.name}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Convert button */}
        <button
          onClick={handleConvert}
          disabled={!imageFile || converting || isProcessingImage}
          style={{
            ...convertBtnStyle,
            opacity: (!imageFile || converting || isProcessingImage) ? 0.5 : 1,
          }}
        >
          {converting ? (
            <span><span className="spinner" style={{marginRight: '8px'}}>⚙️</span> Converting... {progress}%</span>
          ) : (
            '🔮 Convert to 3D'
          )}
        </button>

        {/* Progress bar */}
        {converting && (
          <div style={{ marginTop: '8px' }}>
            <div style={progressBarBg}>
              <div style={{ ...progressBarFill, width: `${progress}%` }} />
            </div>
            <div style={{ fontSize: '12px', color: '#aaa', marginTop: '4px' }}>{statusMsg}</div>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div style={errorStyle}>⚠️ {error}</div>
        )}

        {/* Detected parts */}
        {detectedParts.length > 0 && (
          <div style={{ fontSize: '12px', color: '#8b8', marginTop: '8px' }}>
            ✓ Detected: {detectedParts.join(', ')}
          </div>
        )}
      </Section>

      {/* ── Metal Picker ─────────────────────────────────────── */}
      <Section title="2. Choose Metal">
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {Object.entries(METALS).map(([key, metal]) => (
            <SwatchButton
              key={key}
              hex={metal.hex}
              label={metal.name}
              selected={selectedMetal === key}
              onClick={() => handleMetalChange(key)}
            />
          ))}
        </div>
      </Section>

      {/* ── Gemstone Picker ──────────────────────────────────── */}
      <Section title="3. Choose Gemstone">
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {Object.entries(GEMSTONES).map(([key, gem]) => (
            <SwatchButton
              key={key}
              hex={gem.hex}
              label={gem.name}
              selected={selectedGem === key}
              onClick={() => handleGemChange(key)}
            />
          ))}
        </div>
      </Section>

      {/* ── Geometry Tweaks ─────────────────────────────────── */}
      <Section title="4. Geometry (band & stone)">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
          <LabeledInput
            label="Band width (mm)"
            type="number"
            step="0.1"
            value={shapeParams.band_width_mm}
            onChange={(e) => updateShape('band_width_mm', parseFloat(e.target.value) || 0)}
          />
          <LabeledInput
            label="Band thickness (mm)"
            type="number"
            step="0.1"
            value={shapeParams.band_thickness_mm}
            onChange={(e) => updateShape('band_thickness_mm', parseFloat(e.target.value) || 0)}
          />
          <LabeledInput
            label="Inner radius (mm)"
            type="number"
            step="0.1"
            value={shapeParams.inner_radius_mm}
            onChange={(e) => updateShape('inner_radius_mm', parseFloat(e.target.value) || 0)}
          />
          <LabeledInput
            label="Gem radius (mm)"
            type="number"
            step="0.1"
            value={shapeParams.gem_radius_mm}
            disabled={!shapeParams.has_gemstone}
            onChange={(e) => updateShape('gem_radius_mm', parseFloat(e.target.value) || 0)}
          />
          <LabeledInput
            label="Prongs (#)"
            type="number"
            step="1"
            value={shapeParams.prong_count}
            disabled={!shapeParams.has_gemstone}
            onChange={(e) => updateShape('prong_count', parseInt(e.target.value || '0', 10))}
          />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span style={{ fontSize: '12px', color: '#aaa' }}>Band profile</span>
            <select
              value={shapeParams.band_profile}
              onChange={(e) => updateShape('band_profile', e.target.value)}
              style={inputStyle}
            >
              <option value="D-shape">D-shape</option>
              <option value="flat">Flat</option>
              <option value="round">Round</option>
            </select>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <span style={{ fontSize: '12px', color: '#aaa' }}>Gem cut</span>
            <select
              value={shapeParams.gem_cut}
              onChange={(e) => updateShape('gem_cut', e.target.value)}
              style={inputStyle}
              disabled={!shapeParams.has_gemstone}
            >
              <option value="round_brilliant">Round</option>
              <option value="princess">Princess</option>
            </select>
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#aaa', fontSize: '13px' }}>
            <input
              type="checkbox"
              checked={shapeParams.has_gemstone}
              onChange={(e) => updateShape('has_gemstone', e.target.checked)}
            />
            Include gemstone
          </label>
        </div>
        <button
          onClick={handleApplyGeometry}
          disabled={!jobId || geometryBusy}
          style={{
            ...convertBtnStyle,
            marginTop: '12px',
            background: 'linear-gradient(135deg, #10b981, #0ea5e9)',
            opacity: (!jobId || geometryBusy) ? 0.5 : 1,
          }}
        >
          {geometryBusy ? '⏳ Applying geometry...' : '📐 Apply geometry tweaks'}
        </button>
        <div style={{ fontSize: '12px', color: '#888', marginTop: '6px' }}>
          Requires a completed job. Updates the GLB and keeps your material choices.
        </div>
      </Section>

      {/* ── Budget Advisor ───────────────────────────────────── */}
      <Section title="5. Budget Check">
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ color: '#aaa' }}>$</span>
          <input
            type="number"
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
            placeholder="Enter budget (USD)"
            style={inputStyle}
          />
          <button onClick={handleBudgetCheck} style={smallBtnStyle} disabled={!budget}>
            Check
          </button>
        </div>

        {budgetResult && (
          <div style={{ marginTop: '8px' }}>
            <div style={{ fontSize: '13px', color: budgetResult.over_budget ? '#f88' : '#8f8' }}>
              Current: ${budgetResult.current_total} | Budget: ${budgetResult.budget}
              {budgetResult.over_budget ? ' ❌ Over' : ' ✅ Under'}
            </div>

            {budgetResult.suggestions?.length > 0 && (
              <div style={{ marginTop: '8px' }}>
                <div style={{ fontSize: '12px', color: '#aaa', marginBottom: '4px' }}>Suggestions:</div>
                {budgetResult.suggestions.slice(0, 3).map((s, i) => (
                  <div key={i} style={suggestionStyle}>
                    <div style={{ fontSize: '13px' }}>
                      Replace <b>{s.replace}</b> → <b>{s.with}</b>
                    </div>
                    <div style={{ fontSize: '11px', color: '#aaa' }}>
                      Save ${s.savings} | Similarity: {Math.round(s.visual_similarity * 100)}%
                    </div>
                    <button
                      style={applyBtnStyle}
                      onClick={() => {
                        if (s.component === 'gemstone') handleGemChange(s.with);
                        else handleMetalChange(s.with);
                      }}
                    >
                      Apply
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Section>

      {/* ── Export ────────────────────────────────────────────── */}
      <Section title="6. Export">
        <Exporter jobId={jobId} disabled={!jobId} />
      </Section>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Sub-Components
// ═══════════════════════════════════════════════════════════════════════

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: '28px' }}>
      <h3 style={{ fontSize: '14px', color: '#aaa', margin: '0 0 10px 0', fontWeight: 500 }}>
        {title}
      </h3>
      {children}
    </div>
  );
}

function LabeledInput({ label, ...rest }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <span style={{ fontSize: '12px', color: '#aaa' }}>{label}</span>
      <input {...rest} style={inputStyle} />
    </label>
  );
}

function SwatchButton({ hex, label, selected, onClick }) {
  return (
    <button
      onClick={onClick}
      title={label}
      style={{
        width: '48px',
        height: '48px',
        borderRadius: '50%',
        border: selected ? '3px solid #fff' : '2px solid rgba(255,255,255,0.2)',
        background: hex,
        cursor: 'pointer',
        transition: 'all 0.15s',
        boxShadow: selected ? '0 0 12px rgba(255,255,255,0.3)' : 'none',
        transform: selected ? 'scale(1.1)' : 'scale(1)',
        position: 'relative',
        marginBottom: '22px',
        flexShrink: 0,
      }}
    >
      <span style={{
        position: 'absolute',
        bottom: '-20px',
        left: '50%',
        transform: 'translateX(-50%)',
        fontSize: '9px',
        color: '#aaa',
        whiteSpace: 'nowrap',
        maxWidth: '56px',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        textAlign: 'center',
      }}>
        {label}
      </span>
    </button>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Styles
// ═══════════════════════════════════════════════════════════════════════

const panelStyle = {
  width: '340px',
  height: '100%',
  padding: '20px',
  background: '#1e1e2e',
  borderRight: '1px solid #333',
  overflowY: 'auto',
  flexShrink: 0,
};

const dropzoneStyle = (hasImage) => ({
  border: '2px dashed rgba(255,255,255,0.2)',
  borderRadius: '12px',
  padding: '20px',
  textAlign: 'center',
  cursor: 'pointer',
  transition: 'border-color 0.2s',
  minHeight: hasImage ? 'auto' : '140px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
});

const convertBtnStyle = {
  width: '100%',
  marginTop: '12px',
  padding: '12px',
  border: 'none',
  borderRadius: '8px',
  background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
  color: '#fff',
  fontSize: '15px',
  fontWeight: '700',
  cursor: 'pointer',
  transition: 'opacity 0.2s',
};

const progressBarBg = {
  width: '100%',
  height: '6px',
  background: '#333',
  borderRadius: '3px',
  overflow: 'hidden',
};

const progressBarFill = {
  height: '100%',
  background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
  transition: 'width 0.3s',
  borderRadius: '3px',
};

const errorStyle = {
  marginTop: '8px',
  padding: '8px 12px',
  background: 'rgba(239,68,68,0.15)',
  border: '1px solid rgba(239,68,68,0.3)',
  borderRadius: '6px',
  color: '#f88',
  fontSize: '13px',
};

const inputStyle = {
  flex: 1,
  padding: '8px 12px',
  background: '#2a2a3e',
  border: '1px solid #444',
  borderRadius: '6px',
  color: '#fff',
  fontSize: '14px',
  outline: 'none',
};

const smallBtnStyle = {
  padding: '8px 16px',
  border: 'none',
  borderRadius: '6px',
  background: '#6366f1',
  color: '#fff',
  fontSize: '13px',
  cursor: 'pointer',
};

const demoBtnStyle = {
  padding: '4px 12px',
  border: '1px solid #444',
  borderRadius: '16px',
  background: 'transparent',
  color: '#aaa',
  fontSize: '12px',
  cursor: 'pointer',
};

const suggestionStyle = {
  padding: '8px',
  background: '#2a2a3e',
  borderRadius: '6px',
  marginBottom: '6px',
  display: 'flex',
  flexDirection: 'column',
  gap: '2px',
  position: 'relative',
};

const applyBtnStyle = {
  position: 'absolute',
  right: '8px',
  top: '50%',
  transform: 'translateY(-50%)',
  padding: '4px 12px',
  border: 'none',
  borderRadius: '4px',
  background: '#6366f1',
  color: '#fff',
  fontSize: '11px',
  cursor: 'pointer',
};
