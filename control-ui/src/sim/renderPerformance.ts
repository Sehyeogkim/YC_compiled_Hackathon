import type { SplatQuality } from './FactoryBackdrop';

export type RenderBenchmarkMode='cached'|'direct';
export type RenderBenchmarkRequest={id:number;mode:RenderBenchmarkMode};
export type RenderBenchmarkResult={id:number;mode:RenderBenchmarkMode;durationMs:number;draws:number;cpuMs:number;gpuMs:number|null;gpuSamples:number;backgroundDraws:number;cacheHits:number;nativeFrames:number;quality:SplatQuality;pixelRatio:number;shadowPolicy:'reuse_existing';error?:string};
export type RenderProfile = 'performance' | 'balanced' | 'full';
export type RenderStats = {
  /** requestAnimationFrame cadence, separate from actual on-demand draws. */
  fps:number; drawFps:number; benchmarkMode?:RenderBenchmarkMode|null;
  /** Asynchronous EXT_disjoint_timer_query_webgl2 elapsed GPU time; null when unavailable. */
  gpuMs?:number|null; backgroundDrawFps?:number; backgroundCacheHits?:number;
  /** CPU submission time, not a GPU timer query. */
  frameMs:number; nativeFrameHz:number; profile:RenderProfile; quality:SplatQuality;
  loadedSplatCount?:number; pixelRatio:number; drawCalls:number; triangles:number;
};
type PerformanceSnapshot = { profile:RenderProfile; stats:RenderStats|null; benchmarkRequest:RenderBenchmarkRequest|null; benchmarkResult:RenderBenchmarkResult|null };
// A versioned preference intentionally does not inherit an earlier automatic 500k
// default. Only an explicit selection is written; original HQ is the fresh default.
const STORAGE_KEY='werescue.render-profile.original-hq.v1';
function initialProfile():RenderProfile {
  try { const value=globalThis.localStorage?.getItem(STORAGE_KEY);if(value==='full'||value==='balanced'||value==='performance')return value; } catch {}
  return 'full';
}
let snapshot:PerformanceSnapshot = { profile:initialProfile(), stats:null, benchmarkRequest:null, benchmarkResult:null };
const listeners = new Set<()=>void>();
export function getRenderPerformance(){return snapshot;}
export function subscribeRenderPerformance(listener:()=>void){listeners.add(listener);return ()=>{listeners.delete(listener);};}
export function setRenderProfile(profile:RenderProfile){
  if(snapshot.profile===profile)return;
  try {globalThis.localStorage?.setItem(STORAGE_KEY,profile);} catch {}
  snapshot={...snapshot,profile};listeners.forEach(listener=>listener());
}
export function publishRenderStats(stats:RenderStats){
  snapshot={...snapshot,stats};listeners.forEach(listener=>listener());
}

let benchmarkId=0;
export function requestRenderBenchmark(mode:RenderBenchmarkMode){
  snapshot={...snapshot,benchmarkRequest:{id:++benchmarkId,mode}};listeners.forEach(listener=>listener());
}
export function publishRenderBenchmark(result:RenderBenchmarkResult){
  snapshot={...snapshot,benchmarkResult:result};listeners.forEach(listener=>listener());
}
