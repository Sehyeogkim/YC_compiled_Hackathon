# High-detail can recycling factory

Active package: `public/worlds/factory-conveyor-hq/`.
World: https://marble.worldlabs.ai/world/55ff7d06-eaef-4460-9197-8db675bd1293
Model: **Marble 1.1**. Original SPZ: **1,920,000 splats**.
This regeneration cost **1500 API credits**; verified remaining balance **1860**.

## Reference and requested gray bales

The input is a **4096 × 2048** equirectangular panorama containing newly generated
fine detail. Six **1254 × 1254**, 100-degree perspective views of the previous
factory were edited with built-in OpenAI image_gen, then mathematically
reprojected with overlapping borders. This is AI-regenerated cube-face detail
assembled into 4K, not a native single-image 4K response or a plain enlargement
of the earlier 1774 × 887 reference. The layout remains consistent with the
previous can recycling and conveyor sorting factory.

The stacked compressed-can bales are now **neutral silver-gray**. Their wooden
pallets and stack layout are retained. Blue machinery, loose cans and other
surroundings retain their colors. Source image: `source-reference.png`.
Prompts: `reference-prompts.json`. Native face sizes, source hash, round-trip,
structure and overlap checks: `source-reference-report.json`. A small
right/ceiling overlap warning was visually reviewed without a gross geometry
break. The source is generated imagery, not a real facility photograph.

## Renderer and scale

Spark renders the original SPZ in the same Three.js viewport as the workcell.
A 500k preview is replaced by the original; a loading message distinguishes
that transition. The background selector offers Original, Balanced and
Performance modes using local files. Both native and offline viewports use
pixel ratio up to **2×**. The source panorama supplies environment lighting.

The API metric factor **1.8628582954 m/raw unit** is retained uniformly. A
collider/opaque-splat fit corrects axis convention, floor tilt and height. A
second, rigid full-resolution floor correction levels the visible surface
without changing that meter scale. The implied capture height is **1.11157 m**,
about **1%** below the previous world. The unverified prompt camera height of
1.60 m is not forced over API metadata.

The visible-floor fit used **449 inlier cells**. Residual median is **1.69 mm**,
P95 **4.45 mm**. These describe numerical floor fitting; AI metric scale and
architectural dimensions are estimates, not a physical survey. Full reports
are retained as `calibration-report.json` and `visible-floor-report.json`.

Robot, belt, task objects and telemetry stay in their original simulator meter
frame. Offline: tabletop z=0, factory floor z=-0.78 m, workbench height **0.78 m**,
footprint **1.65 × 1.95 m**, belt length **1.40 m**. The yellow **0.50 m** gauge has
0.10 m ticks. The independent **1.00 m** cube does not resize with the background.
Native: the viewer reads the authoritative horizontal floor-plane pose rather
than imposing the offline offset. Room-scale adjustments pivot on the fitted
floor. Native poses, collisions, controller cameras and policy observations
are not modified by this visual background.

## Run and reproduce

```sh
cd control-ui
npm install
npm run dev
npm run build
# Python 3 + NumPy + Pillow for offline tools only:
npm run world:calibrate
npm run world:generate
```

Generation defaults to the `conveyor-hq` profile and resumes its cached operation
without another paid request. A new request requires no prior operation or
request marker, an explicit `--generate`, enough credits, a genuine 2:1 source
and at least4096 input width. A request marker prevents an uncertain POST from
being silently repeated. Both ground-calibration stages run after downloads.
Credentials and signed URLs stay in ignored `.env`/`.marble/`; viewing the
background needs no World Labs credential.

The prior colored world is retained separately. `?factory=baseline` selects it
for comparison; default or `?factory=hq` selects the new gray/high-detail world.
Switching or viewing makes no World Labs generation request. The WEFARM
reference supplied the full-SPZ/axis/scale approach; its greenhouse is not used.

Main 통합용 고정본의 파일 목록·좌표·실행 경계는 [FACTORY_HANDOFF.md](FACTORY_HANDOFF.md)를 참고한다.

## Native conveyor presentation integration

The native task remains LIBERO-Object 0, alphabet soup into a basket. The
conveyor is a **stationary visual picking station**. Supply objects sit on the
belt; the basket sits on its own fixed receiving stand beyond the belt exit.
It is not carried by the belt. GPU state, collisions, policy cameras and raw
recordings remain the native LIBERO environment.

This task's native floor/support is z=0. To present an industrial workbench,
`NativeScene` places robot and task meshes, historical Ghosts, the trail and
control overlays in a common display frame translated +0.78 m in Z. Native
mesh coordinates and received qpos/qvel are unchanged inside that group. The
3D camera uses the same display translation; mouse displacement vectors stay
in native meters. Native camera PNG/JPEG observations are not transformed.
The factory background still uses the actual native floor height and its
original `rawToWorld` once, and the independent 1 m reference cube stays at the
factory floor. The display translation is not a second background metric scale.

`NativeWorkcellSkin.ts` supplies the 1.40 m conveyor and separate robot/basket
stands. Static legacy room visuals are hidden, revealing the pinned HQ world.
The original 500k preview remains visible while full-detail splats load.

The 16 frozen HQ asset hashes are listed in `FACTORY_HQ_SHA256SUMS`. Heavy
build validation runs in an isolated SSH Mac mini workspace; no generation or
recalibration is required to use this package.

## Original HQ rendering performance

Original 1,920,000-splat HQ and device pixel ratio up to 2 are the fresh
default. Explicit rendering preferences persist in a versioned local setting.
When the camera is fixed, NativeScene caches the final background framebuffer
at the same pixel resolution and reuses those pixels for robot-only redraws.
Camera/projection/size, Spark asynchronous dirty notifications, asset, lighting
and room-scale changes invalidate it. Native geometry, Ghosts and overlays draw
afterward. No reduced LOD, alpha threshold or background metric scale is used
by this optimization. GPU timer queries are optional and shown as unavailable
when unsupported; rAF scheduling and CPU submission are labeled separately.

The explicit three-second redraw benchmarks never invoke collection APIs,
change camera pose or modify simulator state. In one Chrome comparison with
Full HQ and DPR 2, direct redraw submitted 80 frames in 3 seconds, averaging
33.23 ms GPU time (13 samples); cached redraw submitted 180 frames, averaging
3.35 ms (15 samples). The existing user-started state stream supplied 2 versus
1 updates, and shadow caches were reused in both tests. These short fixed-camera
measurements do not guarantee the same rate during camera movement. See
`../output/ui/render-benchmark-summary.json` and its DOM evidence for details.
