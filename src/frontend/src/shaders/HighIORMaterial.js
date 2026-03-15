/**
 * HighIORMaterial.js
 * ==================
 * Custom Three.js material that extends MeshPhysicalMaterial to support
 * IOR values above the default 2.333 cap (needed for diamond at 2.42).
 *
 * Three.js internally clamps IOR to [1.0, 2.333] in MeshPhysicalMaterial.
 * This module patches the fragment shader via onBeforeCompile to allow
 * IOR up to 3.0, covering diamond (2.42) and moissanite (2.65).
 *
 * Usage:
 *   import { createHighIORMaterial } from './HighIORMaterial';
 *   const material = createHighIORMaterial({
 *     color: 0xf8f8ff,
 *     transmission: 0.95,
 *     ior: 2.42,          // ← This actually works now
 *     roughness: 0.0,
 *     thickness: 0.5,
 *   });
 */

import * as THREE from 'three';

/**
 * Create a MeshPhysicalMaterial with uncapped IOR support.
 *
 * @param {Object} params - Standard MeshPhysicalMaterial parameters
 * @param {number} params.ior - Index of refraction (1.0 to 3.0)
 * @param {number} [params.transmission=0] - Transmission factor
 * @param {number} [params.thickness=0.5] - Material thickness
 * @param {number} [params.roughness=0] - Surface roughness
 * @param {number} [params.metalness=0] - Metalness
 * @param {THREE.Color|number} [params.color=0xffffff] - Base color
 * @param {THREE.Color|number} [params.attenuationColor] - Volume attenuation color
 * @param {number} [params.attenuationDistance=Infinity] - Volume attenuation distance
 * @param {number} [params.dispersion=0] - Chromatic dispersion
 * @returns {THREE.MeshPhysicalMaterial}
 */
export function createHighIORMaterial(params = {}) {
  const actualIOR = params.ior || 1.5;
  const needsPatch = actualIOR > 2.333;

  // Clamp IOR for Three.js constructor (it will re-clamp internally)
  const constructorParams = {
    ...params,
    ior: Math.min(actualIOR, 2.333),
    // Ensure transmission mode works correctly
    transparent: params.transmission > 0,
    opacity: 1.0,  // Must be 1.0 when using transmission
    side: THREE.DoubleSide,
  };

  const material = new THREE.MeshPhysicalMaterial(constructorParams);

  if (needsPatch) {
    // Store the real IOR value (above 2.333)
    material.userData.realIOR = actualIOR;

    material.onBeforeCompile = (shader) => {
      // Add a uniform for the uncapped IOR
      shader.uniforms.realIOR = { value: actualIOR };

      // Patch the fragment shader to use our uncapped IOR uniform
      // Three.js computes F0 (reflectance at normal incidence) from IOR using:
      //   float f = ( ior - 1.0 ) / ( ior + 1.0 );
      //   f0 = f * f;
      // We inject our own IOR value before this computation.
      //
      // The key location is in the physical material's fragment shader
      // where 'ior' uniform is used to compute Fresnel F0.

      // Add uniform declaration
      shader.fragmentShader = shader.fragmentShader.replace(
        'uniform float ior;',
        `uniform float ior;
         uniform float realIOR;`
      );

      // Replace IOR usage in Fresnel/transmission calculations
      // The relevant code block computes:
      //   material.ior = ior;
      // We override it with our uncapped value:
      shader.fragmentShader = shader.fragmentShader.replace(
        'material.ior = ior;',
        'material.ior = realIOR;'
      );

      // Also patch the Fresnel F0 computation if present as a separate block
      // Some Three.js versions compute F0 inline:
      shader.fragmentShader = shader.fragmentShader.replace(
        /float\s+f\s*=\s*\(\s*ior\s*-\s*1\.0\s*\)\s*\/\s*\(\s*ior\s*\+\s*1\.0\s*\)/g,
        'float f = ( realIOR - 1.0 ) / ( realIOR + 1.0 )'
      );
    };
  }

  return material;
}

/**
 * Build a Three.js material from a jewelry material definition object.
 *
 * @param {Object} matDef - Material definition from the backend
 * @param {string} matDef.type - "metal" or "gemstone"
 * @param {number[]} matDef.color - [R, G, B] in 0-1 range
 * @param {number} [matDef.metallic] - Metalness
 * @param {number} [matDef.roughness] - Roughness
 * @param {number} [matDef.ior] - Index of refraction
 * @param {number} [matDef.transmission] - Transmission
 * @param {number[]} [matDef.attenuation_color] - Volume absorption color
 * @param {number} [matDef.attenuation_distance] - Absorption distance
 * @param {number} [matDef.thickness] - Material thickness
 * @param {number} [matDef.dispersion] - Chromatic dispersion
 * @returns {THREE.MeshPhysicalMaterial}
 */
export function buildMaterialFromDef(matDef) {
  const color = new THREE.Color().fromArray(matDef.color || [0.8, 0.8, 0.8]);

  if (matDef.type === 'metal') {
    // Metals: high metalness, no transmission
    return new THREE.MeshPhysicalMaterial({
      color,
      metalness: matDef.metallic ?? 1.0,
      roughness: matDef.roughness ?? 0.1,
      envMapIntensity: 1.5,  // Boost reflections for polished metal
      clearcoat: 0.3,        // Slight clear coat for extra shine
      clearcoatRoughness: 0.1,
    });
  }

  if (matDef.type === 'gemstone') {
    // Gemstones: transmission + IOR (possibly high)
    const params = {
      color,
      metalness: 0.0,
      roughness: matDef.roughness ?? 0.0,
      transmission: matDef.transmission ?? 0.8,
      ior: matDef.ior ?? 1.5,
      thickness: matDef.thickness ?? 0.5,
      envMapIntensity: 2.0,
      specularIntensity: 1.0,
    };

    // Volume absorption (colored gems like ruby, emerald)
    if (matDef.attenuation_color) {
      params.attenuationColor = new THREE.Color().fromArray(matDef.attenuation_color);
      params.attenuationDistance = matDef.attenuation_distance ?? 5.0;
    }

    // Dispersion / "fire" (diamond, cubic zirconia)
    if (matDef.dispersion) {
      // Store for potential future shader use; MeshPhysicalMaterial does not support this property
      params.userData = { ...(params.userData || {}), dispersion: matDef.dispersion };
    }

    // Use custom high-IOR material if needed
    if (matDef.ior > 2.333) {
      return createHighIORMaterial(params);
    }

    return new THREE.MeshPhysicalMaterial({
      ...params,
      transparent: params.transmission > 0,
      opacity: 1.0,
      side: THREE.DoubleSide,
    });
  }

  // Fallback: generic material
  return new THREE.MeshPhysicalMaterial({ color, roughness: 0.5 });
}
