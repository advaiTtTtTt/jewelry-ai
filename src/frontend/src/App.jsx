/**
 * App.jsx
 * ========
 * Root component wiring the 3D viewer and customizer panel together.
 *
 * Layout: [Customizer Panel | 3D Viewer]
 *
 * State flow:
 *   1. User uploads image in Customizer
 *   2. Customizer calls /convert, gets GLB blob back
 *   3. GLB blob is passed to JewelryViewer as prop
 *   4. Material swaps in Customizer produce materialOverrides → JewelryViewer
 *   5. Viewer applies new materials instantly (no server call)
 */

import React, { useState, useCallback } from 'react';
import JewelryViewer from './viewer/JewelryViewer';
import Customizer from './customizer/Customizer';

export default function App() {
  // ─── State ───────────────────────────────────────────────────────
  const [jobId, setJobId] = useState(null);
  const [glbBlob, setGlbBlob] = useState(null);
  const [vertexLabels, setVertexLabels] = useState(null);
  const [materialOverrides, setMaterialOverrides] = useState({});
  const [components, setComponents] = useState([]);

  // ─── Callbacks ──────────────────────────────────────────────────

  /** Called by Customizer when GLB is ready from the backend */
  const handleGlbLoaded = useCallback((blob, labels, initialMaterials) => {
    setGlbBlob(blob);
    setVertexLabels(labels);
    // Apply initial materials (the currently selected metal + gem) immediately
    if (initialMaterials && Object.keys(initialMaterials).length > 0) {
      setMaterialOverrides(initialMaterials);
    } else {
      setMaterialOverrides({});
    }
  }, []);

  /** Called by Customizer when user picks a new material for a component */
  const handleMaterialChange = useCallback((component, materialDef) => {
    setMaterialOverrides((prev) => ({
      ...prev,
      [component]: materialDef,
    }));
  }, []);

  /** Called by JewelryViewer when it finishes parsing mesh components */
  const handleComponentsReady = useCallback((componentNames) => {
    setComponents(componentNames);
  }, []);

  // ─── Render ─────────────────────────────────────────────────────
  return (
    <div style={containerStyle}>
      {/* Left: Customizer Panel */}
      <Customizer
        onMaterialChange={handleMaterialChange}
        onGlbLoaded={handleGlbLoaded}
        jobId={jobId}
        setJobId={setJobId}
      />

      {/* Right: 3D Viewer */}
      <div style={viewerContainerStyle}>
        <JewelryViewer
          glbBlob={glbBlob}
          vertexLabels={vertexLabels}
          onComponentsReady={handleComponentsReady}
          materialOverrides={materialOverrides}
          style={{ width: '100%', height: '100%' }}
        />

        {/* Status bar */}
        <div style={statusBarStyle}>
          {components.length > 0 && (
            <span>Components: {components.join(', ')}</span>
          )}
          {jobId && <span style={{ marginLeft: 'auto' }}>Job: {jobId}</span>}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Styles
// ═══════════════════════════════════════════════════════════════════════

const containerStyle = {
  display: 'flex',
  width: '100vw',
  height: '100vh',
  background: '#121212',
  color: '#e0e0e0',
  fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  overflow: 'hidden',
};

const viewerContainerStyle = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  position: 'relative',
};

const statusBarStyle = {
  position: 'absolute',
  bottom: 0,
  left: 0,
  right: 0,
  padding: '8px 16px',
  background: 'rgba(0,0,0,0.5)',
  backdropFilter: 'blur(8px)',
  fontSize: '12px',
  color: '#888',
  display: 'flex',
  alignItems: 'center',
};
