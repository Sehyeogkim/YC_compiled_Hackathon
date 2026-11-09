/** Record the visible local simulator canvas without depending on the GPU API. */
export type CanvasVideoCapture = {
  stop: () => Promise<{ filename: string; bytes: number; mimeType: string }>;
  cancel: () => void;
};

export function startCanvasVideoCapture(
  canvas: HTMLCanvasElement,
  filename = `rg-local-demo-${new Date().toISOString().replace(/[:.]/g, '-')}.webm`,
): CanvasVideoCapture {
  if (typeof MediaRecorder === 'undefined' || typeof canvas.captureStream !== 'function') {
    throw new Error('Canvas video recording is unavailable in this browser.');
  }

  const mimeType = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
    .find(value => MediaRecorder.isTypeSupported(value));
  if (!mimeType) throw new Error('This browser cannot encode WebM video.');

  const stream = canvas.captureStream(20);
  const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 2_500_000 });
  const chunks: BlobPart[] = [];
  let cancelled = false;
  let finished: Promise<{ filename: string; bytes: number; mimeType: string }> | null = null;
  let resolveFinish: ((result: { filename: string; bytes: number; mimeType: string }) => void) | null = null;
  let rejectFinish: ((reason: Error) => void) | null = null;

  const cleanup = () => {
    stream.getTracks().forEach(track => track.stop());
    recorder.ondataavailable = null;
    recorder.onstop = null;
    recorder.onerror = null;
  };

  recorder.ondataavailable = event => {
    if (event.data.size) chunks.push(event.data);
  };
  recorder.onstop = () => {
    cleanup();
    if (cancelled) return;
    const blob = new Blob(chunks, { type: mimeType });
    if (!blob.size) {
      rejectFinish?.(new Error('Canvas recording contains no video frames.'));
      return;
    }
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    resolveFinish?.({ filename, bytes: blob.size, mimeType });
  };
  recorder.onerror = () => {
    cleanup();
    rejectFinish?.(new Error('Canvas video recording failed.'));
  };
  recorder.start(1000);

  return {
    stop() {
      if (finished) return finished;
      finished = new Promise((resolve, reject) => {
        resolveFinish = resolve;
        rejectFinish = reject;
      });
      if (recorder.state === 'inactive') {
        cleanup();
        rejectFinish?.(new Error('Canvas video recording already stopped.'));
      } else recorder.stop();
      return finished;
    },
    cancel() {
      cancelled = true;
      if (recorder.state !== 'inactive') recorder.stop();
      cleanup();
    },
  };
}
