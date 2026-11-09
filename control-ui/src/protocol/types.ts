export type Vec3 = [number, number, number];

export interface SceneCommand {
  id: number;
  type:
    | 'start'
    | 'pause'
    | 'reset'
    | 'inject_failure'
    | 'take_control'
    | 'resume_policy'
    | 'stop'
    | 'move'
    | 'gripper';
  delta?: Vec3;
  gripper?: number;
}

export interface SimSnapshot {
  status:
    | 'loading'
    | 'ready'
    | 'running'
    | 'paused'
    | 'attention'
    | 'manual'
    | 'recovering'
    | 'completed'
    | 'stopped'
    | 'error';
  /** MuJoCo simulation time in seconds. */
  simTime: number;
  joints: number[];
  /** TCP position in the MuJoCo Z-up world frame, in meters. */
  eef: Vec3;
  gripper: number;
  canPosition: Vec3;
  targetPosition: Vec3;
  retries: number;
  taskProgress: number;
  beltRunning: boolean;
  message: string;
  physicsReady: boolean;
  /** Current applied MuJoCo ctrl: seven joint targets plus gripper control (0–255). */
  lastAction: number[];
  success: boolean;
  stage?: string;
  contactCount?: number;
}

export type ControlActor = 'operator' | 'demo_controller' | 'policy';
export type EpisodeSegment = 'task' | 'recovery' | 'unassigned';

export interface EventTimestamp {
  /** UTC wall clock for cross-system correlation, not command ordering. */
  utc: string;
  /** Monotonic milliseconds since the recording started. */
  elapsedMs: number;
  /** Authoritative ordering within this episode. */
  sequence: number;
}

interface EpisodeEventBase {
  timestamp: EventTimestamp;
  segment: EpisodeSegment;
  controlSource: ControlActor;
}

export type EpisodeEvent =
  | (EpisodeEventBase & { kind: 'snapshot'; snapshot: SimSnapshot })
  | (EpisodeEventBase & {
      kind: 'command';
      command: SceneCommand;
      actor: ControlActor;
      /** Most recent observed snapshot. A request is not evidence of execution. */
      precedingObservationSequence: number | null;
      executionStatus: 'requested';
    })
  | (EpisodeEventBase & { kind: 'marker'; type: string; detail?: string });

export interface RecordedEpisode {
  schemaVersion: 'werescue.episode.v1';
  episodeId: string;
  startedAt: string | null;
  exportedAt: string;
  durationMs: number;
  provenance: {
    environment: 'browser_simulation';
    policyBackend: 'scripted_demo_controller';
    modelInferenceRecorded: false;
    actionsAre: 'requested_commands';
    appliedControlSource: 'snapshot.lastAction';
    appliedControlDimensions: 8;
    appliedControlsAre: 'mujoco_actuator_targets';
    segmentationFrom: 'observed_simulator_status';
    snapshotsAre: 'reported_simulation_state';
    cameraFramesIncluded: false;
    learningCompleted: false;
  };
  events: EpisodeEvent[];
}

export interface EpisodeRecorder {
  start(): void;
  recordSnapshot(snapshot: SimSnapshot): void;
  recordCommand(command: SceneCommand, actor: ControlActor): void;
  mark(type: string, detail?: string): void;
  exportEpisode(): RecordedEpisode;
  reset(): void;
}
