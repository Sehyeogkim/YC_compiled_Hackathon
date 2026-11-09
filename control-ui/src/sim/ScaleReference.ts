import * as THREE from 'three';
import { WORKBENCH_HEIGHT } from './FactoryBackdrop';

/** A true one-meter cube in the fixed physics frame; background scaling cannot resize it. */
export class ScaleReference extends THREE.Group {
  private labelTexture: THREE.CanvasTexture;
  private label: THREE.Sprite;
  constructor(floorZ = -WORKBENCH_HEIGHT) {
    super();
    this.name = '1.00 m reference · fixed simulation units';
    this.position.set(-1.15, -.15, floorZ + .5);
    this.visible = false;
    const cube = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial({ color: '#91f0d2', wireframe: true, transparent: true, opacity: .8 }));
    this.add(cube);
    const canvas = document.createElement('canvas');
    canvas.width = 384; canvas.height = 96;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#132d35'; ctx.fillRect(0, 0, 384, 96);
    ctx.fillStyle = '#c7fff0'; ctx.font = '600 48px monospace'; ctx.textAlign = 'center';
    ctx.fillText('1.00 m', 192, 65);
    this.labelTexture = new THREE.CanvasTexture(canvas);
    this.labelTexture.colorSpace = THREE.SRGBColorSpace;
    this.label = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.labelTexture, depthTest: false }));
    this.label.scale.set(.6, .15, 1);
    this.label.position.z = .63;
    this.label.renderOrder = 25;
    this.add(this.label);
  }
  disposeLabel() { this.labelTexture.dispose(); this.label.material.dispose(); }
}
