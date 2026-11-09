import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

// Build-time only: API credentials never enter Vite or the public world package.
const root = fileURLToPath(new URL('../../', import.meta.url));
for (const file of ['.env', 'control-ui/.env.local']) {
  const env = path.join(root, file);
  if (existsSync(env)) process.loadEnvFile(env);
}
const key = process.env.WORLDLABS_API_KEY || process.env.WLT_API_KEY;
if (!key) throw new Error('Set WORLDLABS_API_KEY in the repository .env.');
const cache = path.join(root, '.marble');
const output = path.join(root, 'control-ui/public/worlds/factory');
mkdirSync(cache, { recursive: true });
mkdirSync(output, { recursive: true });
const operationFile = path.join(cache, 'factory-operation.json');
const model = 'marble-1.0-draft';
const prompt = 'Photorealistic modern Korean industrial recycling and robotics factory interior, a compact realistic 14 meter by 18 meter workshop with a 5 meter high ceiling. Viewpoint at standing eye height 1.6 meters in the center of a completely empty unobstructed 6 meter by 6 meter concrete work area. Keep the entire foreground and center floor empty for a robot workbench to be inserted later. Industrial equipment, steel storage shelves, electrical cabinets, stacked blue crates, orange safety rails and recycling sorting machinery ONLY along the distant perimeter walls at least 4 meters from the camera. Realistic human scale: doors 2.1 meters high, safety rails 1.1 meters high. Matte pale gray concrete floor, yellow painted safety lines at the perimeter, white corrugated metal walls, steel roof beams, large clerestory windows, soft bright neutral daylight and overhead linear industrial lights. Clean professional operational manufacturing environment, detailed architectural depth, realistic materials, subtle teal and warm orange accents. No people, no robot, no central conveyor, no central table, no objects in the foreground. Full 360 degree coherent enclosed factory background, level horizontal floor.';
const api = async (endpoint, body) => {
  const r = await fetch(`https://api.worldlabs.ai/marble/v1/${endpoint}`, {
    method: body ? 'POST' : 'GET',
    headers: { 'WLT-Api-Key': key, 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
    signal: AbortSignal.timeout(60_000),
  });
  if (!r.ok) throw new Error(`World Labs ${endpoint}: HTTP ${r.status}`);
  return r.json();
};
let op;
if (existsSync(operationFile)) {
  op = JSON.parse(readFileSync(operationFile, 'utf8'));
  console.log('Resuming existing generation; no new paid request.');
} else {
  if (!process.argv.includes('--generate')) throw new Error('Use --generate to authorize one 230-credit Draft generation.');
  const balance = await api('credits');
  if (balance.remaining_credits < 230) throw new Error(`Insufficient credits: ${balance.remaining_credits}; need 230.`);
  console.log(`Balance: ${balance.remaining_credits}. Starting ONE Draft generation (230 credits).`);
  op = await api('worlds:generate', { display_name: 'WeRescue factory background', model, world_prompt: { type: 'text', text_prompt: prompt } });
  writeFileSync(operationFile, JSON.stringify(op, null, 2));
}
for (let i = 0; !op.done && i < 120; i++) {
  await new Promise(resolve => setTimeout(resolve, 5000));
  op = await api(`operations/${op.operation_id}`);
  writeFileSync(operationFile, JSON.stringify(op, null, 2));
  console.log(op.metadata?.progress?.description || (op.done ? 'Generation finished.' : 'Generating…'));
}
if (!op.done || op.error) throw new Error(op.error?.message || 'Generation still pending. Rerun to resume without a new paid request.');
const id = op.response?.world_id || op.response?.id || op.metadata?.world_id;
const record = await api(`worlds/${id}`);
const world = record.world || record;
writeFileSync(path.join(cache, 'factory-world.json'), JSON.stringify(world, null, 2));
const assets = world.assets;
const files = {};
for (const [name, url] of Object.entries({
  'factory-100k.spz': assets.splats.spz_urls['100k'],
  'factory-500k.spz': assets.splats.spz_urls['500k'],
  'factory-full.spz': assets.splats.spz_urls.full_res,
  'collider.glb': assets.mesh?.collider_mesh_url,
  'panorama.jpg': assets.imagery?.pano_url,
  'thumbnail.jpg': assets.thumbnail_url,
})) {
  if (!url) continue;
  const dest = path.join(output, name);
  if (!existsSync(dest)) {
    const r = await fetch(url, { signal: AbortSignal.timeout(180_000) });
    if (!r.ok) throw new Error(`Asset download ${name}: HTTP ${r.status}`);
    writeFileSync(dest, new Uint8Array(await r.arrayBuffer()));
  }
  files[name] = readFileSync(dest).byteLength;
}
const manifest = {
  version: 1, worldId: id, model, source: world.world_marble_url,
  generatedAt: world.created_at, prompt,
  splats: { '100k': 'factory-100k.spz', '500k': 'factory-500k.spz', full: 'factory-full.spz' },
  semantics: assets.splats.semantics_metadata || null,
  // Calibration is explicit and editable; robot physics always stays in meters.
  calibration: { scaleCorrection: 1, groundCorrectionM: 0, yawDegrees: 0, translationM: [0, 0, 0] },
  files,
};
const manifestFile = path.join(output, 'world.json');
if (existsSync(manifestFile)) {
  const previous = JSON.parse(readFileSync(manifestFile, 'utf8'));
  if (previous.worldId === id) manifest.calibration = previous.calibration;
}
writeFileSync(manifestFile, JSON.stringify(manifest, null, 2) + '\n');
if (!manifest.semantics && !manifest.calibration.rawToWorld) {
  const result = spawnSync('python3', [path.join(root, 'control-ui/scripts/calibrate-factory-world.py')], { stdio: 'inherit' });
  if (result.status !== 0) throw new Error('Assets downloaded. Install NumPy and run npm run world:calibrate before viewing.');
}
console.log(JSON.stringify({ worldId: id, semantics: manifest.semantics, files, credits: await api('credits') }, null, 2));
