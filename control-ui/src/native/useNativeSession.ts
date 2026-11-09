import { useCallback, useEffect, useRef, useState } from 'react';
import type { SceneCommand } from '../protocol/types';
import { createNativeClient, normalizedOperatorAction, POLICY_BATCH_STEPS } from './client';
import type { NativeHealth, NativeScene, NativeSessionFrame, SessionResponse, StepResponse } from './client';

type Client = ReturnType<typeof createNativeClient>;
const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const ENDPOINT_STORAGE_KEY = 'rg.native.endpoint.v1';
const storedEndpoint = () => {
  try { return localStorage.getItem(ENDPOINT_STORAGE_KEY) || '/robot-api'; }
  catch { return '/robot-api'; }
};

export function useNativeSession() {
  const [endpoint, setEndpointState] = useState(storedEndpoint);
  const [initialStateId, setInitialStateIdState] = useState(0);
  const [health, setHealth] = useState<NativeHealth | null>(null);
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [frame, setFrame] = useState<NativeSessionFrame | null>(null);
  const [scene, setScene] = useState<NativeScene | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [operatorRunning, setOperatorRunning] = useState(false);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [loadStage, setLoadStage] = useState('');
  const alive = useRef(true), generation = useRef(0), endpointRef = useRef(endpoint);
  const initialStateIdRef = useRef(initialStateId);
  const frameRef = useRef<NativeSessionFrame | null>(null), sceneRef = useRef<NativeScene | null>(null), sessionRef = useRef<string | null>(null);
  const sceneCache = useRef<{ endpoint: string; scene: NativeScene } | null>(null);
  const runningRef = useRef(false), queue = useRef<Promise<unknown>>(Promise.resolve());
  const operatorRunningRef = useRef(false);
  const activeRequest = useRef<AbortController | null>(null), timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const operatorTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingOperator = useRef<{ delta: [number, number, number] } | null>(null);
  const operatorDrain = useRef<Promise<void> | null>(null);
  const inFlightTarget = useRef<[number, number, number] | null>(null);
  const operatorGripTarget = useRef<number | null>(null);
  const boundaryPending = useRef(false);

  const halt = useCallback(() => {
    generation.current += 1; runningRef.current = false; operatorRunningRef.current = false;
    if (timer.current !== null) clearTimeout(timer.current);
    if (operatorTimer.current !== null) clearTimeout(operatorTimer.current);
    timer.current = null; operatorTimer.current = null; activeRequest.current?.abort();
    pendingOperator.current = null; operatorDrain.current = null; inFlightTarget.current = null;
    operatorGripTarget.current = null; boundaryPending.current = false;
    if (alive.current) { setRunning(false); setOperatorRunning(false); setBusy(false); setConnecting(false); setLoadStage(''); }
    return generation.current;
  }, []);

  const clearSession = useCallback(() => {
    frameRef.current = null; sceneRef.current = null; sessionRef.current = null;
    setFrame(null); setScene(null); setSessionId(null); setConnected(false); setInferenceMs(null);
  }, []);

  const applyFrame = useCallback((response: SessionResponse | StepResponse) => {
    if (!response.frame || !Array.isArray(response.frame.qpos) || !Array.isArray(response.frame.qvel)
      || !Number.isFinite(response.frame.sim_time)) throw new Error('Native API did not return authoritative simulator state.');
    frameRef.current = response.frame; sessionRef.current = response.session_id;
    setFrame(response.frame); setSessionId(response.session_id);
    if ('model_inference' in response && response.model_inference === true && Number.isFinite(response.inference_ms)) {
      setInferenceMs(response.inference_ms);
      setHealth(previous => previous ? { ...previous, inference_verified: true } : previous);
    }
    if (response.frame.done) {
      runningRef.current = false; operatorRunningRef.current = false;
      if (timer.current !== null) clearTimeout(timer.current);
      if (operatorTimer.current !== null) clearTimeout(operatorTimer.current);
      timer.current = null; operatorTimer.current = null;
      setRunning(false); setOperatorRunning(false);
    }
  }, []);

  const enqueue = useCallback(<T,>(operation: (client: Client, signal: AbortSignal) => Promise<T>, apply?: (value: T) => void): Promise<T | undefined> => {
    const epoch = generation.current;
    const work = queue.current.then(async () => {
      if (!alive.current || epoch !== generation.current) return undefined;
      const controller = new AbortController(); activeRequest.current = controller; setBusy(true);
      try {
        const value = await operation(createNativeClient(endpointRef.current), controller.signal);
        if (!alive.current || epoch !== generation.current) return undefined;
        apply?.(value); return value;
      } catch (caught) {
        if (alive.current && epoch === generation.current && !controller.signal.aborted) {
          setError(errorText(caught)); runningRef.current = false; operatorRunningRef.current = false;
          if (operatorTimer.current !== null) clearTimeout(operatorTimer.current);
          operatorTimer.current = null; setRunning(false); setOperatorRunning(false);
        }
        return undefined;
      } finally {
        if (activeRequest.current === controller) activeRequest.current = null;
        if (alive.current && epoch === generation.current) setBusy(false);
      }
    });
    queue.current = work.catch(() => undefined);
    return work;
  }, []);

  const connect = useCallback(async () => {
    // Retain the unmodified scene payload before clearing the active session.
    const cached = sceneRef.current ? { endpoint: endpointRef.current, scene: sceneRef.current } : sceneCache.current;
    const requestEndpoint = endpointRef.current;
    const requestInitialStateId = initialStateIdRef.current;
    const epoch = halt(); clearSession(); setError(''); setConnecting(true); setLoadStage('Checking GPU');
    const stage = (label: string) => { if (alive.current && epoch === generation.current) setLoadStage(label); };
    await enqueue(async (client, signal) => {
      const readiness = await client.health(signal);
      if (epoch === generation.current && alive.current) setHealth(readiness);
      if (readiness.ready !== true || readiness.model_ready !== true) throw new Error(readiness.error || `Native model is not ready (${readiness.status}).`);
      stage('Resetting');
      const reset = await client.reset(signal, requestInitialStateId);
      const canReuseScene = cached?.endpoint === requestEndpoint
        && typeof reset.scene_sha256 === 'string' && reset.scene_sha256.length > 0
        && cached.scene.scene_sha256 === reset.scene_sha256;
      if (!canReuseScene) stage('Downloading scene');
      const nativeScene = canReuseScene ? cached.scene : await client.scene(reset.session_id, signal, reset.scene_sha256);
      stage('Loading cameras');
      const current = await client.frame(reset.session_id, signal);
      const activeHealth = await client.health(signal);
      return { current, nativeScene, activeHealth };
    }, ({ current, nativeScene, activeHealth }) => {
      sceneCache.current = { endpoint: requestEndpoint, scene: nativeScene };
      sceneRef.current = nativeScene; setScene(nativeScene); applyFrame(current); setHealth(activeHealth); setConnected(true);
    });
    if (alive.current && epoch === generation.current) { setConnecting(false); setLoadStage(''); }
  }, [applyFrame, clearSession, enqueue, halt]);

  const disconnect = useCallback(() => { halt(); clearSession(); setHealth(null); setError(''); }, [clearSession, halt]);
  const setEndpoint = useCallback((next: string) => {
    halt(); clearSession(); sceneCache.current = null; setHealth(null); setError(''); endpointRef.current = next; setEndpointState(next);
    try { localStorage.setItem(ENDPOINT_STORAGE_KEY, next); } catch { /* private browsing */ }
  }, [clearSession, halt]);
  const setInitialStateId = useCallback((next: number) => {
    if (!Number.isInteger(next) || next < 0 || next > 19) { setError('Collection start ID must be an integer from 0 to 19.'); return; }
    initialStateIdRef.current = next; setInitialStateIdState(next);
  }, []);

  const run = useCallback(() => {
    const id = sessionRef.current, initial = frameRef.current;
    if (!id || !initial) { setError('Connect a ready native session first.'); return; }
    if (initial.done) { setError('Episode is terminal. Reset before running again.'); return; }
    if (initial.control_source !== 'policy') { setError('Return control to the policy explicitly before starting inference.'); return; }
    if (runningRef.current) return;
    setError(''); runningRef.current = true; setRunning(true);
    const epoch = generation.current;
    const pump = async () => {
      if (!alive.current || epoch !== generation.current || !runningRef.current) return;
      const before = performance.now();
      const result = await enqueue((client, signal) => client.policyStep(id, signal), applyFrame);
      if (!result || !alive.current || epoch !== generation.current || !runningRef.current || result.frame.done) return;
      const dt = sceneRef.current?.control?.dt_seconds;
      const delay = dt && dt > 0 ? Math.max(0, dt * POLICY_BATCH_STEPS * 1000 - (performance.now() - before)) : 0;
      timer.current = setTimeout(() => void pump(), delay);
    };
    void pump();
  }, [applyFrame, enqueue]);

  const startOperatorLoop = useCallback(() => {
    const id = sessionRef.current, current = frameRef.current, control = sceneRef.current?.control;
    if (!id || current?.control_source !== 'operator' || current.done || operatorRunningRef.current) return;
    if (!control || !Number.isFinite(control.dt_seconds) || control.dt_seconds <= 0) {
      setError('Native control timing metadata is required for operator physics stepping.'); return;
    }
    operatorRunningRef.current = true; setOperatorRunning(true);
    const epoch = generation.current;
    const pump = async () => {
      if (!alive.current || epoch !== generation.current || !operatorRunningRef.current) return;
      const before = performance.now();
      const result = await enqueue((client, signal) => {
        const latest = frameRef.current;
        if (!latest || latest.control_source !== 'operator' || latest.done) throw new Error('Operator stepping requires an active human-controlled episode.');
        // A policy's last action is a direction, not the human's gripper target.
        const grip = operatorGripTarget.current ?? control.gripper_neutral;
        if (grip === undefined || grip === null || !Number.isFinite(grip)
          || (operatorGripTarget.current === null && grip !== 0)) throw new Error('Native controller has no verified gripper hold input.');
        // Only the gripper target repeats. Cartesian and rotational deltas stay zero.
        const action = normalizedOperatorAction(control, [0, 0, 0], grip);
        return client.operatorStep(id, action, signal, 4);
      }, applyFrame);
      if (!result || !alive.current || epoch !== generation.current || !operatorRunningRef.current || result.frame.done) return;
      const delay = Math.max(0, control.dt_seconds * 4 * 1000 - (performance.now() - before));
      // Schedule only after completion, so a slow native server cannot grow a polling queue.
      operatorTimer.current = setTimeout(() => void pump(), delay);
    };
    void pump();
  }, [applyFrame, enqueue]);

  const pause = useCallback(async () => {
    halt(); const id = sessionRef.current;
    if (id) await enqueue((client, signal) => client.frame(id, signal), applyFrame);
  }, [applyFrame, enqueue, halt]);

  const reset = useCallback(async () => { await connect(); }, [connect]);

  const takeControl = useCallback(async () => {
    const epoch = halt(); setError(''); const id = sessionRef.current;
    if (!id) { setError('Connect a native session before taking control.'); return; }
    const response = await enqueue((client, signal) => client.control(id, 'operator', signal), applyFrame);
    if (response && epoch === generation.current && !response.frame.done) startOperatorLoop();
  }, [applyFrame, enqueue, halt, startOperatorLoop]);

  const resumePolicy = useCallback(async () => {
    halt(); setError(''); const id = sessionRef.current;
    if (!id) { setError('Connect a native session first.'); return; }
    const response = await enqueue((client, signal) => client.control(id, 'policy', signal), applyFrame);
    if (response && response.frame.control_source === 'policy' && !response.frame.done) run();
  }, [applyFrame, enqueue, halt, run]);

  const drainOperatorCommands = useCallback((): Promise<void> => {
    if (operatorDrain.current) return operatorDrain.current;
    const epoch = generation.current;
    let work: Promise<void>;
    work = Promise.resolve().then(async () => {
      while (pendingOperator.current && alive.current && epoch === generation.current) {
        const id = sessionRef.current;
        if (!id) break;
        const response = await enqueue((client, signal) => {
          const latest = frameRef.current, control = sceneRef.current?.control, pending = pendingOperator.current;
          if (!latest || !latest.eef || latest.control_source !== 'operator' || latest.done || !pending || !control) throw new Error('Operator stepping requires an active human-controlled episode.');
          const grip = operatorGripTarget.current ?? control.gripper_neutral;
          if (grip === undefined || grip === null || !Number.isFinite(grip)
            || (operatorGripTarget.current === null && grip !== 0)) throw new Error('Native controller has no verified gripper hold input.');
          const action = normalizedOperatorAction(control, pending.delta, grip);
          const applied = action.slice(0, 3).map((value, i) => value * control.translation_scale_m[i] + control.translation_offset_m[i]);
          const remainder = pending.delta.map((value, i) => value - applied[i]) as [number, number, number];
          pendingOperator.current = Math.hypot(...remainder) > 1e-7 ? { delta: remainder } : null;
          inFlightTarget.current = latest.eef.map((value, i) => value + applied[i]) as [number, number, number];
          return client.operatorStep(id, action, signal);
        }, applyFrame);
        if (epoch !== generation.current) return;
        inFlightTarget.current = null;
        if (!response || response.frame.done) { pendingOperator.current = null; break; }
      }
    }).finally(() => {
      if (operatorDrain.current === work) operatorDrain.current = null;
    });
    operatorDrain.current = work;
    return work;
  }, [applyFrame, enqueue]);

  const markRecoveryComplete = useCallback(async () => {
    const id = sessionRef.current;
    if (!id || frameRef.current?.control_source !== 'operator') { setError('Recovery marking requires active operator control.'); return; }
    if (boundaryPending.current) return;
    const epoch = generation.current; boundaryPending.current = true; setError('');
    try {
      // All pre-boundary jogs execute before the server switches R′ to human task R.
      await drainOperatorCommands();
      if (epoch !== generation.current || !alive.current || frameRef.current?.done) return;
      await enqueue((client, signal) => client.markRecoveryComplete(id, signal), applyFrame);
    } finally { if (epoch === generation.current) boundaryPending.current = false; }
  }, [applyFrame, drainOperatorCommands, enqueue]);

  const sendOperatorCommand = useCallback(async (command: Omit<SceneCommand, 'id'>) => {
    const id = sessionRef.current, current = frameRef.current, control = sceneRef.current?.control;
    if (!id || current?.control_source !== 'operator') { setError('Take control before sending native operator actions.'); return; }
    if (current.done) { setError('Episode is terminal. Reset before sending actions.'); return; }
    if (boundaryPending.current) { setError('Wait for the recovery boundary before issuing another operator command.'); return; }
    if (command.type !== 'move' && command.type !== 'gripper') { setError('Native operator controls accept move or gripper commands only.'); return; }
    const epoch = generation.current;
    try {
      setError('');
      if (!current.eef) throw new Error('Native TCP position is required for bounded teleoperation.');
      if (command.type === 'gripper') {
        if (!control || command.gripper === undefined || !Number.isFinite(command.gripper)) throw new Error('A gripper target and native controller metadata are required.');
        const open = Math.max(0, Math.min(1, command.gripper));
        operatorGripTarget.current = control.gripper_close + open * (control.gripper_open - control.gripper_close);
      }
      const delta = command.type === 'move' ? command.delta ?? [0, 0, 0] : [0, 0, 0];
      if (!delta.every(Number.isFinite)) throw new Error('Operator displacement must be finite.');
      const pending = pendingOperator.current?.delta ?? [0, 0, 0];
      const flyingLead = current.eef.map((value, i) => (inFlightTarget.current?.[i] ?? value) - value);
      const lead = pending.map((value, i) => value + delta[i] + flyingLead[i]);
      const length = Math.hypot(...lead), scale = length > 0.1 ? 0.1 / length : 1;
      pendingOperator.current = { delta: lead.map((value, i) => value * scale - flyingLead[i]) as [number, number, number] };
      await drainOperatorCommands();
      // A fresh explicit jog/gripper input can resume a paused human-control loop.
      if (epoch === generation.current && !frameRef.current?.done && frameRef.current?.control_source === 'operator') startOperatorLoop();
    } catch (caught) { setError(errorText(caught)); }
  }, [drainOperatorCommands, startOperatorLoop]);

  const exportEpisode = useCallback(async () => {
    const id = sessionRef.current;
    if (!id) { setError('No native episode is available for export.'); return undefined; }
    return enqueue((client, signal) => client.exportEpisode(id, signal), episode => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(episode, null, 2)], { type: 'application/json' }));
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = `werescue-native-${id}.json`;
      document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
  }, [enqueue]);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; halt(); };
  }, [halt]);

  return { endpoint, setEndpoint, initialStateId, setInitialStateId, health, connected, connecting, error, busy, frame, scene, sessionId, running, operatorRunning,
    inferenceMs, loadStage, connect, disconnect, run, pause, reset, takeControl, resumePolicy, markRecoveryComplete, sendOperatorCommand, exportEpisode };
}
