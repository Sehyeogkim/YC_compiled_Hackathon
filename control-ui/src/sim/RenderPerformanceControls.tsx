import { useSyncExternalStore } from 'react';
import { getRenderPerformance, setRenderProfile, subscribeRenderPerformance, requestRenderBenchmark, type RenderProfile } from './renderPerformance';

/** Isolated subscription: changing rendering detail never remounts the native session. */
export default function RenderPerformanceControls(){
  const {profile,stats,benchmarkResult}=useSyncExternalStore(subscribeRenderPerformance,getRenderPerformance,getRenderPerformance);
  const renderDebug=typeof window!=='undefined'&&new URLSearchParams(window.location.search).get('renderDebug')==='1';
  return <div className="dock-section render-performance-controls">
    <label style={{display:'flex',alignItems:'center',justifyContent:'space-between',gap:8}}>Rendering
      <select aria-label="Rendering performance profile" value={profile} onChange={event=>setRenderProfile(event.target.value as RenderProfile)}>
        <option value="performance">Performance · 100k</option><option value="balanced">Balanced · 500k</option><option value="full">Full · original HQ</option>
      </select>
    </label>
    <p style={{fontSize:11,lineHeight:1.5,margin:'7px 0 0'}}>Background cache · Always on</p>
    {renderDebug&&<details style={{marginTop:7}}><summary style={{fontSize:11,cursor:'pointer'}}>Renderer diagnostics</summary>
    <div aria-live="off" style={{fontSize:10,lineHeight:1.6,opacity:.75,marginTop:6,fontVariantNumeric:'tabular-nums'}}>
      {stats?<><span>rAF {stats.fps.toFixed(0)} Hz · draw {stats.drawFps.toFixed(0)}/s · native {stats.nativeFrameHz.toFixed(1)} Hz</span><br/><span>CPU submit {stats.frameMs.toFixed(1)} ms · GPU {stats.gpuMs==null?'n/a':stats.gpuMs.toFixed(1)+' ms'} · DPR {stats.pixelRatio.toFixed(2)} · {stats.quality} · {stats.loadedSplatCount?.toLocaleString()??'—'} splats</span><br/><span>HQ background {stats.backgroundDrawFps?.toFixed(0)??'—'} draws/s · cache {stats.backgroundCacheHits??0} hits</span></>:'Waiting for render measurements…'}
    </div>
    <div style={{display:'flex',gap:5,marginTop:7}}>
      <button type="button" disabled={!!stats?.benchmarkMode} onClick={()=>requestRenderBenchmark('cached')} style={{fontSize:10,padding:'6px'}}>Cached redraw · 3s</button>
      <button type="button" disabled={!!stats?.benchmarkMode} onClick={()=>requestRenderBenchmark('direct')} style={{fontSize:10,padding:'6px'}}>Direct redraw · 3s</button>
    </div>
    {benchmarkResult&&<div style={{fontSize:10,lineHeight:1.5,marginTop:6}}>{benchmarkResult.error??<>{benchmarkResult.mode==='cached'?'Cached':'Direct original'}: {benchmarkResult.draws} submitted draws / {(benchmarkResult.durationMs/1000).toFixed(1)} s<br/>CPU {benchmarkResult.cpuMs.toFixed(2)} ms/draw · GPU {benchmarkResult.gpuMs==null?'n/a':benchmarkResult.gpuMs.toFixed(2)+' ms ('+benchmarkResult.gpuSamples+' samples)'}<br/>Background {benchmarkResult.backgroundDraws} · cache {benchmarkResult.cacheHits} · received native frames {benchmarkResult.nativeFrames}<br/>Shadow cache reused; no forced shadow update. Camera and simulator unchanged by benchmark.</>}</div>}
    </details>}
    <p style={{fontSize:10,lineHeight:1.5,opacity:.6,margin:'5px 0 0'}}>Original HQ is the default. Background pixels reuse the full display resolution while the camera is still.</p>
  </div>;
}
