import type {
  ControlActor,
  EpisodeEvent,
  EpisodeRecorder,
  EpisodeSegment,
  EventTimestamp,
  RecordedEpisode,
  SceneCommand,
  SimSnapshot,
} from './types';

const monotonicNow = () => performance.now();
const clone = <T>(value: T): T => structuredClone(value);

/** Records requests and observations separately. It does not infer executed actions. */
export function createEpisodeRecorder(): EpisodeRecorder {
  let events: EpisodeEvent[] = [];
  let episodeId = '';
  let startedAt: string | null = null;
  let startedAtMonotonic = 0;
  let segment: EpisodeSegment = 'unassigned';
  let controlSource: ControlActor = 'demo_controller';
  let lastObservationSequence: number | null = null;
  let previousStatus: SimSnapshot['status'] | null = null;

  const timestamp = (): EventTimestamp => ({
    utc: new Date().toISOString(),
    elapsedMs: Math.max(0, monotonicNow() - startedAtMonotonic),
    sequence: events.length,
  });

  const mark = (type: string, detail?: string) => {
    if (startedAt === null) return;
    events.push({
      kind: 'marker',
      type,
      ...(detail === undefined ? {} : { detail }),
      timestamp: timestamp(),
      segment,
      controlSource,
    });
  };

  return {
    start() {
      // Resume/start buttons must not discard the current recording.
      if (startedAt !== null) return;
      episodeId = globalThis.crypto?.randomUUID?.() ?? `episode-${Date.now()}`;
      startedAt = new Date().toISOString();
      startedAtMonotonic = monotonicNow();
      mark('episode_start');
    },

    recordSnapshot(snapshot: SimSnapshot) {
      if (startedAt === null) return;
      let closeAfterObservation = false;
      if (snapshot.status !== previousStatus) {
        const detail = `Observed simulator status: ${snapshot.status}.`;
        if (snapshot.status === 'manual') {
          if (segment === 'task') mark('task_end', 'Task interrupted by accepted manual control.');
          controlSource = 'operator';
          segment = 'recovery';
          mark('recovery_start', detail);
        } else if (snapshot.status === 'running' || snapshot.status === 'recovering') {
          const acceptedRecoveryHandoff = segment === 'recovery';
          if (acceptedRecoveryHandoff) mark('recovery_end', 'Simulator accepted return to the demo controller.');
          controlSource = 'demo_controller';
          if (acceptedRecoveryHandoff) {
            segment = 'task';
            mark('policy_handoff', detail);
            mark('task_start', 'Task resumed after accepted recovery handoff.');
          } else if (segment !== 'task') {
            segment = 'task';
            mark('task_start', detail);
          }
        } else if (snapshot.status === 'completed') {
          if (segment === 'task') mark('task_end', 'Simulator reported task completion.');
          if (segment === 'recovery') mark('recovery_end', 'Task completed under manual control, without a policy handoff.');
          closeAfterObservation = true;
        } else if (snapshot.status === 'attention') {
          if (segment === 'task') {
            mark('task_end', 'Task interrupted: simulator requested assistance.');
            closeAfterObservation = true;
          }
        } else if (snapshot.status === 'stopped' || snapshot.status === 'error') {
          if (segment === 'task') mark('task_end', `Task interrupted: ${snapshot.status}.`);
          if (segment === 'recovery') mark('recovery_end', `Recovery interrupted: ${snapshot.status}; no handoff confirmed.`);
          mark(snapshot.status === 'stopped' ? 'episode_stopped' : 'simulation_error', detail);
          closeAfterObservation = true;
        } else if (snapshot.status === 'ready') {
          if (previousStatus !== null && previousStatus !== 'loading') {
            mark('simulation_reset', 'Observed ready state. State restoration is not an executed recovery action.');
          }
          segment = 'unassigned';
          controlSource = 'demo_controller';
        }
      }
      const nextTimestamp = timestamp();
      events.push({
        kind: 'snapshot',
        snapshot: clone(snapshot),
        timestamp: nextTimestamp,
        segment,
        controlSource,
      });
      lastObservationSequence = nextTimestamp.sequence;
      previousStatus = snapshot.status;
      if (closeAfterObservation) segment = 'unassigned';
    },

    recordCommand(command: SceneCommand, actor: ControlActor) {
      if (startedAt === null) return;
      events.push({
        kind: 'command',
        command: clone(command),
        actor,
        timestamp: timestamp(),
        segment,
        controlSource,
        precedingObservationSequence: lastObservationSequence,
        executionStatus: 'requested',
      });
    },

    mark,

    exportEpisode(): RecordedEpisode {
      return {
        schemaVersion: 'werescue.episode.v1',
        episodeId,
        startedAt,
        exportedAt: new Date().toISOString(),
        durationMs: startedAt === null ? 0 : Math.max(0, monotonicNow() - startedAtMonotonic),
        provenance: {
          environment: 'browser_simulation',
          policyBackend: 'scripted_demo_controller',
          modelInferenceRecorded: false,
          actionsAre: 'requested_commands',
          appliedControlSource: 'snapshot.lastAction',
          appliedControlDimensions: 8,
          appliedControlsAre: 'mujoco_actuator_targets',
          segmentationFrom: 'observed_simulator_status',
          snapshotsAre: 'reported_simulation_state',
          cameraFramesIncluded: false,
          learningCompleted: false,
        },
        events: clone(events),
      };
    },

    reset() {
      events = [];
      episodeId = '';
      startedAt = null;
      startedAtMonotonic = 0;
      segment = 'unassigned';
      controlSource = 'demo_controller';
      lastObservationSequence = null;
      previousStatus = null;
    },
  };
}

/** Browser download helper. exportEpisode itself remains a side-effect-free read. */
export function downloadEpisode(episode: RecordedEpisode, filename?: string): void {
  const blob = new Blob([JSON.stringify(episode, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename ?? `${episode.episodeId || 'werescue-episode'}.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  // Let the browser consume the URL before releasing it.
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
}
