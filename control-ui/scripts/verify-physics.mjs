/** Reproducible physics smoke check. Uses installed dependencies, without a browser or GPU.
 * Runs the same FactorySim command/controller methods against the authoritative MJCF.
 * Only a temporary, transpiled test module exposes the internal class. Production files
 * and models are never changed by this script. This is a scripted-controller check,
 * not a trained-policy evaluation or evidence of real-robot performance.
 */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import ts from 'typescript';
import loadMujoco from 'mujoco-js';
import * as THREE from 'three';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const cache = path.join(root, 'node_modules', '.cache');
fs.mkdirSync(cache, { recursive: true });
const temporaryFiles = new Map();
const modelDirectory = path.join(root, 'public', 'models', 'panda');
function compileForCheck(filename) {
  if (temporaryFiles.has(filename)) return temporaryFiles.get(filename);
  const output = path.join(cache, `werescue-physics-check-${process.pid}-${temporaryFiles.size}.mjs`);
  temporaryFiles.set(filename, output);
  let source = fs.readFileSync(filename, 'utf8');
  if (path.basename(filename) === 'FactoryScene.tsx') source = source.replace('class FactorySim {', 'export class FactorySim {');
  let code = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX,
  }}).outputText;
  code = code.replace(/from\s*(['"])(\.[^'"]+)\1/g, (match, quote, specifier) => {
    const base = path.resolve(path.dirname(filename), specifier);
    const dependency = [base, base + '.ts', base + '.tsx', base + '.js'].find(candidate => fs.existsSync(candidate) && fs.statSync(candidate).isFile());
    if (!dependency) throw Error(`Missing test dependency: ${specifier}`);
    return `from '${pathToFileURL(compileForCheck(dependency)).href}'`;
  });
  fs.writeFileSync(output, code);
  return output;
}

try {
  const { FactorySim } = await import(pathToFileURL(compileForCheck(path.join(root, 'src', 'sim', 'FactoryScene.tsx'))).href);
  const mj = await loadMujoco();
  mj.FS.mkdir('/working');
  mj.FS.mkdir('/working/assets');
  for (const file of JSON.parse(fs.readFileSync(path.join(modelDirectory, 'manifest.json'), 'utf8'))) {
    mj.FS.writeFile(`/working/assets/${file}`, new Uint8Array(fs.readFileSync(path.join(modelDirectory, 'assets', file))));
  }
  mj.FS.writeFile('/working/factory.xml', fs.readFileSync(path.join(modelDirectory, 'factory.xml'), 'utf8'));
  const model = mj.MjModel.loadFromXML('/working/factory.xml');
  const data = new mj.MjData(model);
  function lookup(addresses, wanted) {
    return Array.from(addresses).findIndex(address => {
      let name = '';
      for (let i = address; model.names[i]; i++) name += String.fromCharCode(model.names[i]);
      return name === wanted;
    });
  }
  const canId = lookup(model.name_bodyadr, 'can');
  const tcpId = lookup(model.name_siteadr, 'tcp');
  assert.ok(canId >= 0 && tcpId >= 0, 'Canonical can and TCP landmarks exist');
  const sim = Object.create(FactorySim.prototype);
  Object.assign(sim, {
    mj, model, data, canId, tcpId,
    canQ: model.jnt_qposadr[model.body_jntadr[canId]],
    target: new THREE.Vector3(), handoff: new THREE.Vector3(), manualStart: new THREE.Vector3(),
    lastAction: [0, 0, 0], onUpdate: () => {},
  });
  function advance(seconds) {
    for (let step = 0; step < seconds * 60; step++) {
      if (sim.status === 'running' || sim.status === 'recovering') sim.updateController(1 / 60);
      if (sim.status !== 'attention') sim.solve();
      const end = data.time + 1 / 60;
      while (data.time < end) {
        if (sim.feed && sim.status === 'running') sim.driveConveyor();
        mj.mj_step(model, data);
      }
    }
  }
  let commandId = 0;
  const command = (type, extra = {}) => sim.command({ id: ++commandId, type, ...extra });
  sim.reset();
  command('start');
  const initialY = sim.can()[1];
  advance(3.8);
  assert.ok(initialY - sim.can()[1] > 0.2, 'Physical can moves down conveyor');
  assert.equal(sim.feed, false, 'Conveyor stops before grasp');
  console.log('PASS conveyor moves physical can >200 mm and stops');
  advance(26);
  assert.equal(sim.status, 'completed');
  assert.equal(sim.success, true);
  console.log('PASS normal contact grasp, transfer, release, native bin success', sim.can());

  sim.reset();
  command('start');
  advance(3.8);
  command('inject_failure');
  assert.equal(sim.status, 'attention');
  command('start');
  assert.equal(sim.status, 'attention', 'Start cannot bypass recovery');
  command('take_control');
  command('resume_policy');
  assert.equal(sim.status, 'manual', 'Handoff requires actual human motion');
  console.log('PASS failure interrupts controller; no-action handoff rejected');
  const manualStart = [...sim.eef()];
  for (let i = 0; i < 3; i++) {
    command('move', { delta: [0, 0.025, 0] });
    advance(0.7);
  }
  assert.ok(sim.eef()[1] - manualStart[1] > 0.06, 'Manual commands physically move TCP');
  command('resume_policy');
  assert.equal(sim.status, 'recovering');
  advance(23);
  assert.equal(sim.status, 'completed');
  assert.equal(sim.success, true);
  console.log('PASS human jog, explicit handoff, native bin completion', sim.can());
  data.delete();
  model.delete();
} finally {
  for (const filename of temporaryFiles.values()) fs.rmSync(filename, { force: true });
}
