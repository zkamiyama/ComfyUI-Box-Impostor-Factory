import { diffuseColor, vec4 } from 'three/tsl';
import { ImpostorNodeMaterial } from 'three-impostor';

// Display-only: setupDiffuseColor still resolves the native hit, discard,
// blended normal and fragment depth before NodeMaterial applies outputNode.
// Tone mapping belongs to the renderer; WebGPURenderer ignores toneMapped.
export function createBoxDisplayMaterial(impostor, {lighting='lit', sampling='native'}={}) {
  if (!['lit','unlit'].includes(lighting)) throw new Error('Unknown Box display lighting: '+lighting);
  if (sampling!=='native') throw new Error('Unknown Box view sampling: '+sampling);
  const material=new ImpostorNodeMaterial(impostor);
  if (lighting==='unlit') material.outputNode=vec4(diffuseColor.rgb,diffuseColor.a);
  return material;
}
