import * as THREE from 'three';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

export const WORKBENCH_HEIGHT = .78;
export type SplatQuality = '100k' | '500k' | 'full';
export type BackgroundState = { status: 'loading' | 'ready' | 'error'; message: string; quality: SplatQuality; scaleDescription?: string };
type WorldPackage = {
  worldId: string;
  splats: Record<'100k' | '500k', string> & { full?: string };
  semantics: { metric_scale_factor: number; ground_plane_offset: number } | null;
  lighting?: string;
  input?: { width?: number; height?: number; sourceReport?: string };
  calibration: {
    rawToWorld?: number[][];
    scaleCorrection: number;
    groundCorrectionM: number;
    yawDegrees: number;
    translationM: [number, number, number];
    scaleSource?: string;
    isMetricScaleEstimated?: boolean;
    assumedCameraHeightM?: number;
    visibleFloorProvenance?: { report: string };
  };
};

/** Photographic world only. The cell origin stays on the workbench, in MuJoCo meters. */
export class FactoryBackdrop {
  private alive = true;
  private spark: SparkRenderer;
  private splat?: SplatMesh;
  private pending?: SplatMesh;
  private pendingQuality?: SplatQuality;
  private root = new THREE.Group();
  private manifest?: WorldPackage;
  private controller = new AbortController();
  private quality: BackgroundState['quality'] = 'full';
  private metric?: THREE.Matrix4;
  private lighting?: THREE.Texture;
  private currentScale = 1;
  private userSelectedQuality = false;
  private requestedQuality: SplatQuality = 'full';

  constructor(private scene: THREE.Scene, renderer: THREE.WebGLRenderer, private onState: (s: BackgroundState) => void,
    private options: { floorZ?: number; baseUrl?: string; quality?: SplatQuality; autoUpgrade?: boolean; onDirty?: () => void } = {}) {
    this.quality = this.requestedQuality = options.quality ?? 'full';
    this.spark = new SparkRenderer({ renderer, onDirty: () => { if (this.alive) this.options.onDirty?.(); } });
    this.root.name = 'Marble factory · calibrated meters · Z up';
    scene.add(this.spark, this.root);
  }

  async load() {
    this.onState({ status: 'loading', message: 'Loading factory background', quality: this.quality });
    try {
      const response = await fetch(`${this.baseUrl}/world.json`, { signal: this.controller.signal });
      if (!response.ok) throw new Error(`World package HTTP ${response.status}`);
      this.manifest = await response.json() as WorldPackage;
      if (!this.alive) return;
      const { calibration: c, semantics } = this.manifest;
      let metric: THREE.Matrix4;
      if (c.rawToWorld) {
        metric = new THREE.Matrix4().set(...c.rawToWorld.flat() as Parameters<THREE.Matrix4['set']>);
      } else if (semantics && Number.isFinite(semantics.metric_scale_factor) && semantics.metric_scale_factor > 0) {
        // Metric ground alignment, then OpenCV -> Z-up: (x, y, z) -> (x, z, -y).
        const s = semantics.metric_scale_factor;
        metric = new THREE.Matrix4().set(s,0,0,0, 0,0,s,0, 0,-s,0,semantics.ground_plane_offset, 0,0,0,1);
      } else {
        throw new Error('Factory world needs calibration. Run npm run world:calibrate.');
      }
      this.metric = metric;
      this.setRoomScale(this.currentScale);
      await this.setQuality(this.requestedQuality);
      if (this.manifest.lighting) void this.loadLighting(this.manifest.lighting);
      // The original asset is the default. Explicit detail choices stay selected.
      if (this.options.autoUpgrade && this.alive && !this.userSelectedQuality && this.manifest.splats.full) {
        this.onState({ status: 'loading', message: 'Loading original factory detail…', quality: this.quality });
        try { await this.setQuality('full'); }
        catch (error) {
          console.warn('Full-resolution background unavailable; retaining loaded splats.', error);
          if (this.alive) this.onState({ status: 'ready', message: 'Factory · original detail unavailable', quality: this.quality });
        }
      }
    } catch (error) {
      if (this.alive) {
        console.error('Factory background:', error);
        this.onState({ status: 'error', message: 'Factory background unavailable', quality: this.quality });
      }
    }
  }

  setRoomScale(multiplier: number) {
    this.currentScale = multiplier;
    if (!this.manifest || !this.metric || !Number.isFinite(multiplier)) return;
    const c = this.manifest.calibration;
    const correction = new THREE.Matrix4().makeRotationZ(THREE.MathUtils.degToRad(c.yawDegrees));
    correction.scale(new THREE.Vector3().setScalar(c.scaleCorrection * THREE.MathUtils.clamp(multiplier, .5, 1.5)));
    // Scale around the fitted floor, so size adjustments never move the building ground.
    correction.setPosition(c.translationM[0], c.translationM[1], c.translationM[2] + c.groundCorrectionM + (this.options.floorZ ?? -WORKBENCH_HEIGHT));
    correction.multiply(this.metric).decompose(this.root.position, this.root.quaternion, this.root.scale);
    this.options.onDirty?.();
  }

  private async loadLighting(file: string) {
    try {
      const texture = await new THREE.TextureLoader().loadAsync(`${this.baseUrl}/${file}`);
      if (!this.alive) { texture.dispose(); return; }
      texture.mapping = THREE.EquirectangularReflectionMapping;
      texture.colorSpace = THREE.SRGBColorSpace;
      this.lighting = texture;
      this.scene.environment = texture;
      this.scene.environmentIntensity = .75;
      // Equirectangular lighting is Y-up; the MuJoCo workcell and factory are Z-up.
      this.scene.environmentRotation.set(Math.PI / 2, 0, 0);
      this.options.onDirty?.();
    } catch (error) {
      console.warn('Factory panorama lighting unavailable; retaining local lights.', error);
    }
  }

  selectQuality(quality: SplatQuality) {
    this.userSelectedQuality = true;
    this.requestedQuality = quality;
    return this.setQuality(quality);
  }

  get currentQuality(): SplatQuality { return this.quality; }
  get loadedSplatCount(): number { return this.splat?.numSplats ?? 0; }

  async setQuality(quality: BackgroundState['quality']) {
    this.requestedQuality = quality;
    if (!this.alive || !this.manifest) return;
    if (this.splat && this.quality === quality) {
      // Retain the displayed level and invalidate a different in-flight request.
      // Its initialized handler disposes it instead of replacing the user's choice.
      this.pending = undefined;
      this.pendingQuality = undefined;
      return;
    }
    if (this.pending && this.pendingQuality === quality) return;
    const file = this.manifest.splats[quality];
    if (!file) throw new Error(`Missing ${quality} background asset`);
    const mesh = new SplatMesh({ url: `${this.baseUrl}/${file}`, raycastable: false });
    this.pending = mesh;
    this.pendingQuality = quality;
    try {
      await mesh.initialized;
      if (!this.alive || this.pending !== mesh) { mesh.dispose(); return; }
      this.splat?.removeFromParent();
      this.splat?.dispose();
      this.root.add(mesh);
      this.splat = mesh;
      this.pending = undefined;
      this.pendingQuality = undefined;
      this.quality = quality;
      const c = this.manifest.calibration;
      const scaleDescription = this.manifest.semantics?.metric_scale_factor
        ? `Metric scale · ${c.visibleFloorProvenance ? 'visible floor aligned' : 'collider floor aligned'}`
        : `Background scale assumes a ${c.assumedCameraHeightM?.toFixed(2) ?? '1.60'} m camera height.`;
      this.onState({ status: 'ready', message: this.manifest.input?.sourceReport ? 'Can recycling · HQ' : 'Can recycling · Marble 1.1', quality, scaleDescription });
    } catch (error) {
      mesh.dispose();
      if (this.pending !== mesh) return;
      this.pending = undefined;
      this.pendingQuality = undefined;
      throw error;
    }
  }

  private get baseUrl() {
    if (this.options.baseUrl) return this.options.baseUrl;
    const comparison = new URLSearchParams(window.location.search).get('factory');
    if (comparison === 'baseline') return '/worlds/factory-conveyor';
    return '/worlds/factory-conveyor-hq';
  }

  dispose() {
    this.alive = false;
    this.controller.abort();
    this.splat?.dispose();
    if (this.scene.environment === this.lighting) this.scene.environment = null;
    this.lighting?.dispose();
    this.root.removeFromParent();
    this.spark.removeFromParent();
    this.spark.dispose();
  }
}
