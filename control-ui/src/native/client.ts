import type { NativeFrame, NativeScenePayload } from '../sim/NativeScene';

/** Amortize WAN latency across one native ten-action policy execution queue. */
export const POLICY_BATCH_STEPS = 10;

export interface NativeHealth {
  ready: boolean;
  model_ready: boolean;
  status: string;
  error?: string | null;
  inference_verified?: boolean;
  device?: string;
  gpu?: string | null;
  model?: { repo_id?: string; revision?: string; [key: string]: unknown };
  environment?: { native_ready?: boolean; [key: string]: unknown };
  [key: string]: unknown;
}

export interface NativeControlMetadata {
  action_dim: number;
  action_order: string[];
  mode: string;
  position_frame: string;
  translation_scale_m: number[];
  rotation_scale_rad: number[];
  translation_offset_m: number[];
  rotation_offset_rad: number[];
  input_min: number[];
  input_max: number[];
  output_min: number[];
  output_max: number[];
  gripper_open: number;
  gripper_close: number;
  gripper_neutral?: number | null;
  control_hz: number;
  dt_seconds: number;
}

export type NativeSessionFrame = NativeFrame & {
  control_source: 'policy' | 'operator';
  segment: 'task' | 'recovery';
  done: boolean;
  success: boolean;
  task: string;
  step: number;
  last_action?: number[] | null;
  cameras?: Record<string, { mime: string; data: string; width: number; height: number }>;
};

export type NativeScene = NativeScenePayload & { control?: NativeControlMetadata; scene_sha256?: string };
export interface SessionResponse { session_id: string; frame: NativeSessionFrame; scene_sha256?: string }
export interface StepResponse extends SessionResponse {
  executed_actions: number[][];
  inference_ms: number;
  model_inference: boolean;
}

export class NativeApiError extends Error {
  constructor(message: string, public readonly status: number) { super(message); this.name = 'NativeApiError'; }
}

export function normalizeEndpoint(endpoint: string): string {
  const value = endpoint.trim().replace(/\/+$/, '') || '/robot-api';
  if (!value.startsWith('/') && !/^https?:\/\//i.test(value)) throw new Error('Use an HTTP(S) endpoint or a same-origin path.');
  return value;
}

export function createNativeClient(endpoint: string) {
  const base = normalizeEndpoint(endpoint);
  async function request<T>(path: string, signal: AbortSignal, body?: object): Promise<T> {
    const response = await fetch(`${base}${path}`, {
      method: body === undefined ? 'GET' : 'POST', signal,
      headers: { Accept: 'application/json', ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    const raw = await response.text();
    let value: unknown;
    try { value = raw ? JSON.parse(raw) : null; }
    catch { throw new NativeApiError(`Native API returned non-JSON (HTTP ${response.status}). Check the proxy endpoint.`, response.status); }
    if (!response.ok) {
      const detail = value && typeof value === 'object' && 'detail' in value ? (value as { detail: unknown }).detail : value;
      throw new NativeApiError(typeof detail === 'string' ? detail : JSON.stringify(detail) || response.statusText, response.status);
    }
    return value as T;
  }
  const query = (sessionId: string) => `?session_id=${encodeURIComponent(sessionId)}`;
  return {
    health: (signal: AbortSignal) => request<NativeHealth>('/health', signal),
    reset: (signal: AbortSignal, initialStateId = 0) => {
      if (!Number.isInteger(initialStateId) || initialStateId < 0 || initialStateId > 19) throw new Error('Collection start ID must be an integer from 0 to 19.');
      return request<SessionResponse>('/session/reset', signal, { initial_state_id: initialStateId, seed: 123, control_source: 'policy' });
    },
    scene: (id: string, signal: AbortSignal, expectedSceneHash?: string) => request<NativeScene>(
      `/scene${query(id)}${expectedSceneHash ? `&expected_scene_sha256=${encodeURIComponent(expectedSceneHash)}` : ''}`, signal),
    frame: (id: string, signal: AbortSignal) => request<SessionResponse>(`/session/frame${query(id)}`, signal),
    control: (id: string, source: 'operator' | 'policy', signal: AbortSignal) => request<SessionResponse>('/session/control', signal, { session_id: id, source }),
    policyStep: (id: string, signal: AbortSignal) => request<StepResponse>('/session/step', signal, { session_id: id, source: 'policy', steps: POLICY_BATCH_STEPS }),
    operatorStep: (id: string, action: number[], signal: AbortSignal, steps = 1) => request<StepResponse>('/session/step', signal, { session_id: id, source: 'operator', steps, action }),
    markRecoveryComplete: (id: string, signal: AbortSignal) => request<SessionResponse>('/session/mark', signal, { session_id: id, marker: 'recovery_complete', detail: 'Operator marked recovery complete; retain human control for task-continuation collection.' }),
    exportEpisode: (id: string, signal: AbortSignal) => request<Record<string, unknown>>(`/session/export${query(id)}`, signal),
  };
}

/** Convert requested physical OSC deltas using scales reported by the actual controller. */
export function normalizedOperatorAction(control: NativeControlMetadata | undefined, delta: [number, number, number], grip: number): number[] {
  if (!control || control.action_dim !== 7 || control.mode !== 'relative_OSC_POSE' || control.position_frame !== 'world'
    || control.action_order.join(',') !== 'dx,dy,dz,rx,ry,rz,gripper') throw new Error('Native scene has no compatible relative OSC control metadata.');
  const scales = [...control.translation_scale_m, ...control.rotation_scale_rad];
  const offsets = [...control.translation_offset_m, ...control.rotation_offset_rad];
  if (scales.length !== 6 || offsets.length !== 6 || control.input_min.length !== 6 || control.input_max.length !== 6
    || ![...scales, ...offsets, ...control.input_min, ...control.input_max, ...delta, grip].every(Number.isFinite)
    || scales.some(value => value <= 0)) throw new Error('Native controller scales are incomplete or invalid.');
  const physical = [...delta, 0, 0, 0];
  return [...physical.map((value, i) => {
    const lower = Math.max(-1, control.input_min[i]), upper = Math.min(1, control.input_max[i]);
    if (lower > upper) throw new Error('Native controller input bounds are invalid.');
    return Math.max(lower, Math.min(upper, (value - offsets[i]) / scales[i]));
  }), Math.max(-1, Math.min(1, grip))];
}
