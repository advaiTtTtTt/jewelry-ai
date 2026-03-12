/**
 * Exporter.jsx
 * =============
 * GLB + STL download buttons for the finalized jewelry model.
 */

import React, { useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function Exporter({ jobId, disabled }) {
  const [downloading, setDownloading] = useState(null);

  const download = async (format) => {
    if (!jobId || downloading) return;

    setDownloading(format);
    try {
      const response = await fetch(`${API_BASE}/export/${format}/${jobId}`);
      if (!response.ok) throw new Error(`Export failed: ${response.statusText}`);

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `jewelry_${jobId}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error(`${format.toUpperCase()} download failed:`, err);
      alert(`Failed to download ${format.toUpperCase()}: ${err.message}`);
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div style={{ display: 'flex', gap: '8px' }}>
      <button
        onClick={() => download('glb')}
        disabled={disabled || !jobId || downloading === 'glb'}
        style={{
          ...btnStyle,
          background: disabled ? '#444' : '#2563eb',
        }}
      >
        {downloading === 'glb' ? '⏳ Downloading...' : '📥 Download GLB'}
      </button>
      <button
        onClick={() => download('stl')}
        disabled={disabled || !jobId || downloading === 'stl'}
        style={{
          ...btnStyle,
          background: disabled ? '#444' : '#7c3aed',
        }}
      >
        {downloading === 'stl' ? '⏳ Downloading...' : '🖨️ Download STL'}
      </button>
    </div>
  );
}

const btnStyle = {
  padding: '10px 20px',
  border: 'none',
  borderRadius: '8px',
  color: '#fff',
  fontSize: '14px',
  fontWeight: '600',
  cursor: 'pointer',
  transition: 'opacity 0.2s',
  opacity: 1,
};
