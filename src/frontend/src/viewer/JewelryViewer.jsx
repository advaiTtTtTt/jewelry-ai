/**
 * JewelryViewer.jsx
 * =================
 * Three.js GLB viewer for jewelry models with real-time material swapping.
 *
 * Features:
 *   - Loads GLB models with semantic vertex labels from extras metadata
 *   - OrbitControls for rotate/zoom/pan
 *   - HDR environment map for realistic metal & gem reflections
 *   - Contact shadows for grounded appearance
 *   - Instant material swaps (client-side only, no server calls)
 *   - Loading spinner during model load
 *
 * Props:
 *   glbUrl          - URL to fetch the GLB model
 *   glbBlob         - Alternatively, a Blob/ArrayBuffer of the GLB
 *   vertexLabels    - Semantic vertex labels from backend
 *   onComponentsReady - Callback when mesh components are parsed
 *   materialOverrides - { componentName: materialDef } for live swaps
 */

import React, { useRef, useEffect, useState, useCallback, useMemo, Suspense } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import {
  OrbitControls,
  Environment,
  ContactShadows,
  Center,
  useGLTF,
  Html,
  useProgress,
} from '@react-three/drei';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { buildMaterialFromDef } from '../shaders/HighIORMaterial';

// ═══════════════════════════════════════════════════════════════════════
// Loading Spinner
// ═══════════════════════════════════════════════════════════════════════

function Loader() {
  const { progress } = useProgress();
  return (
    <Html center>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '12px',
      }}>
        <div style={{
          width: '48px',
          height: '48px',
          border: '3px solid rgba(255,255,255,0.2)',
          borderTop: '3px solid #fff',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }} />
        <span style={{ color: '#fff', fontSize: '14px' }}>
          {progress.toFixed(0)}% loaded
        </span>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    </Html>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Jewelry Model Component
// ═══════════════════════════════════════════════════════════════════════

/**
 * Loads and renders a GLB jewelry model.
 * Parses semantic metadata and supports per-component material swaps.
 */
function JewelryModel({ url, blob, vertexLabels, onComponentsReady, materialOverrides }) {
  const groupRef = useRef();
  const [gltfScene, setGltfScene] = useState(null);
  const [componentMeshMap, setComponentMeshMap] = useState({});

  // Load GLB model from URL or blob
  useEffect(() => {
    const loader = new GLTFLoader();

    const onLoad = (gltf) => {
      const scene = gltf.scene;

      // Parse semantic components from node names and extras
      const meshMap = {};

      scene.traverse((child) => {
        if (child.isMesh) {
          // Determine semantic label from node name or extras
          let label = null;

          // Check parent node name (e.g., "component_metal")
          if (child.parent?.name?.startsWith('component_')) {
            label = child.parent.name.replace('component_', '');
          }
          // Check node extras
          if (!label && child.userData?.semantic_label) {
            label = child.userData.semantic_label;
          }
          if (!label && child.parent?.userData?.semantic_label) {
            label = child.parent.userData.semantic_label;
          }
          // Check node name directly
          if (!label && child.name) {
            const nameLower = child.name.toLowerCase();
            for (const key of ['metal', 'gemstone', 'prong', 'setting', 'bail', 'clasp']) {
              if (nameLower.includes(key)) {
                label = key;
                break;
              }
            }
          }

          // Default to "metal" if no label found
          if (!label) label = 'metal';

          if (!meshMap[label]) meshMap[label] = [];
          meshMap[label].push(child);

          // Enable shadows
          child.castShadow = true;
          child.receiveShadow = true;

          // TripoSR meshes have vertex colors that override PBR materials.
          // Remove the vertex color attribute so our materials show correctly.
          if (child.geometry?.attributes?.color) {
            child.geometry.deleteAttribute('color');
          }

          // Ensure proper normals exist for lighting
          if (!child.geometry?.attributes?.normal) {
            child.geometry?.computeVertexNormals();
          }

          // Ensure materials support PBR features
          if (child.material) {
            child.material.vertexColors = false;   // Disable vertex color usage
            child.material.envMapIntensity = 1.5;
            child.material.needsUpdate = true;
          }
        }
      });

      // Also check scene-level extras for vertex labels
      const sceneExtras = gltf.scene.userData;
      if (sceneExtras?.jewelry_ai && sceneExtras?.semantic_groups) {
        // Merge any additional groups from metadata
        for (const label of Object.keys(sceneExtras.semantic_groups)) {
          if (!meshMap[label]) {
            meshMap[label] = [];
          }
        }
      }

      setGltfScene(scene);
      setComponentMeshMap(meshMap);

      if (onComponentsReady) {
        onComponentsReady(Object.keys(meshMap));
      }
    };

    if (blob) {
      // Load from in-memory blob
      const arrayBuffer = blob instanceof ArrayBuffer ? blob : blob.arrayBuffer?.();
      if (arrayBuffer instanceof Promise) {
        arrayBuffer.then((ab) => loader.parse(ab, '', onLoad));
      } else {
        loader.parse(blob, '', onLoad);
      }
    } else if (url) {
      loader.load(url, onLoad, undefined, (err) => {
        console.error('Failed to load GLB:', err);
      });
    }
  }, [url, blob]);

  // Apply material overrides when they change
  useEffect(() => {
    if (!materialOverrides || Object.keys(componentMeshMap).length === 0) return;

    for (const [component, matDef] of Object.entries(materialOverrides)) {
      const meshes = componentMeshMap[component];
      if (!meshes) continue;

      // Build Three.js material from definition
      const newMaterial = buildMaterialFromDef(matDef);
      newMaterial.vertexColors = false;  // Never use vertex colors

      for (const mesh of meshes) {
        // Ensure vertex color attribute is removed
        if (mesh.geometry?.attributes?.color) {
          mesh.geometry.deleteAttribute('color');
        }
        // Dispose old material to prevent memory leaks
        if (mesh.material && mesh.material !== newMaterial) {
          mesh.material.dispose();
        }
        mesh.material = newMaterial;
        mesh.material.needsUpdate = true;
      }
    }
  }, [materialOverrides, componentMeshMap]);

  // Auto-rotation is handled by OrbitControls autoRotate (pauses on interaction)

  if (!gltfScene) return null;

  return (
    <Center>
      <group ref={groupRef}>
        <primitive object={gltfScene} />
      </group>
    </Center>
  );
}

// ═══════════════════════════════════════════════════════════════════════
// Main Viewer Component
// ═══════════════════════════════════════════════════════════════════════

export default function JewelryViewer({
  glbUrl,
  glbBlob,
  vertexLabels,
  onComponentsReady,
  materialOverrides,
  style,
}) {
  return (
    <div style={{
      width: '100%',
      height: '100%',
      background: 'linear-gradient(145deg, #0b132b 0%, #0f1c3d 45%, #182b57 100%)',
      borderRadius: '14px',
      overflow: 'hidden',
      ...style,
    }}>
      <Canvas
        camera={{ position: [0, 0, 3], fov: 45 }}
        shadows
        gl={{
          antialias: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.2,
          outputColorSpace: THREE.SRGBColorSpace,
        }}
      >
        {/* HDR environment for realistic reflections on metals & gems */}
        <Environment preset="studio" background={false} />

        {/* Ambient + directional lighting */}
        <ambientLight intensity={0.3} />
        <directionalLight
          position={[5, 5, 5]}
          intensity={1.5}
          castShadow
          shadow-mapSize={[2048, 2048]}
        />
        <directionalLight position={[-3, 3, -3]} intensity={0.8} />

        {/* Point lights for sparkle highlights on gems */}
        <pointLight position={[2, 3, 2]} intensity={0.5} color="#ffffee" />
        <pointLight position={[-2, 2, -1]} intensity={0.3} color="#eeeeff" />

        {/* Ground shadow for grounded appearance */}
        {/* resolution kept low to avoid GPU stalls from ReadPixels on some drivers */}
        <ContactShadows
          position={[0, -1.2, 0]}
          opacity={0.4}
          scale={5}
          blur={2.5}
          far={4}
          resolution={256}
          frames={1}
        />

        {/* Orbit controls with auto-rotate */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={1}
          maxDistance={10}
          autoRotate={true}
          autoRotateSpeed={1.5}
        />

        {/* Model */}
        <Suspense fallback={<Loader />}>
          {(glbUrl || glbBlob) && (
            <JewelryModel
              url={glbUrl}
              blob={glbBlob}
              vertexLabels={vertexLabels}
              onComponentsReady={onComponentsReady}
              materialOverrides={materialOverrides}
            />
          )}
        </Suspense>

        {/* Placeholder when no model loaded */}
        {!glbUrl && !glbBlob && (
          <Html center>
            <div style={{
              color: 'rgba(255,255,255,0.5)',
              fontSize: '16px',
              textAlign: 'center',
              userSelect: 'none',
            }}>
              <div style={{ fontSize: '48px', marginBottom: '8px' }}>💎</div>
              Upload an image to generate a 3D model
            </div>
          </Html>
        )}
      </Canvas>
    </div>
  );
}
