import {useCallback,useEffect,useRef,useState} from 'react';
import {Activity,ArrowDown,ArrowLeft,ArrowRight,ArrowUp,ArrowUpRight,Box,Check,ChevronDown,ChevronRight,Command,Database,Download,Expand,Eye,Factory,Gauge,Hand,Layers,LifeBuoy,LocateFixed,Maximize2,Pause,Play,Radio,RotateCcw,ShieldCheck,Square,Target,TriangleAlert,Unplug,Video,Zap} from 'lucide-react';
import FactoryScene from './sim/FactoryScene';
import NativeScene from './sim/NativeScene';
import RenderPerformanceControls from './sim/RenderPerformanceControls';
import type {NativeFrame} from './sim/NativeScene';
import {useNativeSession} from './native/useNativeSession';
import type {NativeSessionFrame} from './native/client';
import type {SceneCommand,SimSnapshot} from './protocol/types';
import {createEpisodeRecorder} from './protocol/recorder';

const initial:SimSnapshot={status:'loading',simTime:0,joints:[],eef:[0,0,0],gripper:0,canPosition:[0,0,0],targetPosition:[0,0,0],retries:0,taskProgress:0,beltRunning:false,message:'Preparing the workcell',physicsReady:false,lastAction:[],success:false};
type Log={time:string;title:string;kind:'neutral'|'warning'|'success'|'manual'};
const statusLabel:Record<SimSnapshot['status'],string>={loading:'Connecting',ready:'Ready to run',running:'Autonomous run',paused:'Paused',attention:'Needs assistance',manual:'Operator control',recovering:'Recovering',completed:'Task completed',stopped:'Emergency stopped',error:'Simulation error'};
const pad=(n:number)=>String(n).padStart(2,'0');
const simClock=(t:number)=>`${pad(Math.floor(t/60))}:${pad(Math.floor(t%60))}`;

type NativeViewportProps=React.ComponentProps<typeof NativeScene>;
function NativeViewport(props:NativeViewportProps){
 const [light,setLight]=useState(()=>new URLSearchParams(window.location.search).get('light')==='1');
 const pointer=useRef<{x:number;y:number}|null>(null);
 useEffect(()=>{if(light)props.onReady?.(true);},[light,props.onReady]);
 const move=(event:React.PointerEvent<HTMLButtonElement>)=>{
  if(!pointer.current||!props.manual)return;
  const dx=event.clientX-pointer.current.x,dy=event.clientY-pointer.current.y;
  if(Math.hypot(dx,dy)<8)return;
  pointer.current={x:event.clientX,y:event.clientY};
  props.onOperatorCommand?.({type:'move',delta:[Math.max(-.03,Math.min(.03,-dy*.0006)),Math.max(-.03,Math.min(.03,-dx*.0006)),0]});
 };
 return <div className="native-viewport">
  {light?<div className="light-camera-stage">
   {(['agentview','wrist'] as const).map(name=>{
    const image=(props.frame as NativeSessionFrame|null)?.cameras?.[name];
    return <figure key={name} className={`light-camera light-camera-${name}`}>
     {image?<img alt={`${name} native camera`} src={`data:${image.mime};base64,${image.data}`}/>:<div className="light-camera-wait">Waiting for native camera…</div>}
     <figcaption>{name==='agentview'?'MuJoCo · agent view':'MuJoCo · wrist view'}</figcaption>
    </figure>;
   })}
   <button className="light-control-ring" aria-label="Drag cyan ring to move robot XY" disabled={!props.manual}
    onPointerDown={event=>{event.preventDefault();event.currentTarget.setPointerCapture(event.pointerId);pointer.current={x:event.clientX,y:event.clientY};}}
    onPointerMove={move} onPointerUp={()=>{pointer.current=null;}} onPointerCancel={()=>{pointer.current=null;}}>◎<span>DRAG XY</span></button>
   <div className="light-control-help">Arrow keys · XY &nbsp; Q/E · height &nbsp; Space · gripper</div>
  </div>:<NativeScene {...props}/>}
  <button className="light-mode-toggle" onClick={()=>setLight(value=>!value)}>{light?'3D view':'Fast camera view'}</button>
 </div>;
}

export default function App(){
 const [offlineSnapshot,setSnapshot]=useState(initial);
 const [mode,setMode]=useState<'native'|'demo'>('native');
 const native=useNativeSession();
 const nativeRef=useRef(native);nativeRef.current=native;
 const modeRef=useRef(mode);modeRef.current=mode;
 const [nativeRenderReady,setNativeRenderReady]=useState(false);
 const [nativeRenderError,setNativeRenderError]=useState('');
 const [command,setCommand]=useState<SceneCommand|null>(null);
 const [view,setView]=useState<'orbit'|'top'|'robot'>('orbit');
 const [ghost,setGhost]=useState(false);
 const [nativeHistory,setNativeHistory]=useState<NativeFrame[]>([]);
 const [referenceIndex,setReferenceIndex]=useState(-1);
 const [section,setSection]=useState<'control'|'data'>('control');
 const [logs,setLogs]=useState<Log[]>([]);
 const [trace,setTrace]=useState<number[]>([]);
 const [interventions,setInterventions]=useState(0);
 const [toast,setToast]=useState('');
 const [showHelp,setShowHelp]=useState(false);
 const [sceneVersion,setSceneVersion]=useState(0);
 const [cameraRevision,setCameraRevision]=useState(0);
 const recorder=useRef(createEpisodeRecorder());
 const previous=useRef('loading');
 const nativeSource=useRef<string|null>(null);
 const nativeEvent=useRef('');
 const latest=useRef(initial);
 const sequence=useRef(0);
 const episodeStarted=useRef(false);
 const onUpdate=useCallback((s:SimSnapshot)=>{
  latest.current=s;setSnapshot(s);
  if(episodeStarted.current)recorder.current.recordSnapshot(s);
  setTrace(t=>[...t.slice(-69),s.eef[2]]);
  if(previous.current!==s.status){
   const kind:Log['kind']=s.status==='attention'||s.status==='stopped'||s.status==='error'?'warning':s.status==='completed'?'success':s.status==='manual'?'manual':'neutral';
   setLogs(l=>[{time:simClock(s.simTime),title:s.message||statusLabel[s.status],kind},...l].slice(0,30));
   recorder.current.mark(s.status,s.message);
   previous.current=s.status;
  }
 },[]);
 const reconnect=()=>{recorder.current.reset();episodeStarted.current=false;previous.current='loading';latest.current=initial;setSnapshot(initial);setCommand(null);setLogs([]);setTrace([]);setInterventions(0);setSceneVersion(n=>n+1);};
 const resetCamera=()=>{setView('orbit');setCameraRevision(n=>n+1);};
 const send=useCallback((type:SceneCommand['type'],extra:Partial<SceneCommand>={})=>{
  if(modeRef.current==='native'){
   const n=nativeRef.current;
   if(type==='start')n.run();
   else if(type==='pause'||type==='stop')void n.pause();
   else if(type==='reset'){setInterventions(0);setGhost(false);setNativeHistory([]);setReferenceIndex(-1);nativeSource.current=null;void n.reset();}
   else if(type==='take_control'){setGhost(true);void n.takeControl();}
   else if(type==='resume_policy')void n.resumePolicy();
   else if(type==='move'||type==='gripper')void n.sendOperatorCommand({type,...extra});
   return;
  }
  if((type==='start'||type==='take_control')&&!episodeStarted.current){recorder.current.start();episodeStarted.current=true;}
  if(type==='reset'){recorder.current.reset();episodeStarted.current=false;setGhost(false);setLogs([]);setInterventions(0);setTrace([]);}
  if(type==='take_control'){setGhost(true);setInterventions(n=>n+1);}
  const next={...extra,id:++sequence.current,type} as SceneCommand;
  recorder.current.recordCommand(next,'operator');
  setCommand(next);
 },[]);
 const onOperatorCommand=useCallback((c:Omit<SceneCommand,'id'>)=>{const {type,...extra}=c;send(type,extra);},[send]);
 const onNativeReady=useCallback((value:boolean,error?:string)=>{setNativeRenderReady(value);setNativeRenderError(error??'');},[]);
 const nativeFrame=native.frame;
 const xyz=(a:number[]|undefined):[number,number,number]=>[a?.[0]??0,a?.[1]??0,a?.[2]??0];
 const snapshot:SimSnapshot=mode==='demo'?offlineSnapshot:{...initial,status:nativeRenderError?'error':nativeFrame?.success?'completed':nativeFrame?.done?'attention':nativeFrame?.control_source==='operator'?'manual':native.running?'running':native.connected?'ready':'loading',simTime:nativeFrame?.sim_time??0,joints:nativeFrame?.joints??[],eef:xyz(nativeFrame?.eef),gripper:(nativeFrame?.gripper?.[0]??0)/.04,targetPosition:xyz(nativeFrame?.eef),lastAction:nativeFrame?.last_action??[],taskProgress:nativeFrame?.success?1:0,success:nativeFrame?.success??false,physicsReady:native.connected&&nativeRenderReady,message:native.error||nativeRenderError||(!native.connected?'Connect the GPU to load the real LIBERO workcell.':nativeFrame?.control_source==='operator'?(native.operatorRunning?'Human control · drag the cyan ring to move the robot.':'Human control paused · resume to continue recording.'):native.running?'Actual SmolVLA inference on GPU':'Native LIBERO session ready · start VLA inference.'),stage:nativeFrame?`Native step ${nativeFrame.step}`:'Awaiting GPU connection'};
 useEffect(()=>{if(mode==='native')latest.current=snapshot;},[snapshot,mode]);
 useEffect(()=>{if(native.sessionId){setInterventions(0);setGhost(false);setNativeHistory([]);setReferenceIndex(-1);setTrace([]);setLogs([]);nativeSource.current=null;}},[native.sessionId]);
 useEffect(()=>{if(mode==='native'&&nativeFrame){if(nativeFrame.control_source==='operator'&&nativeSource.current!=='operator')setInterventions(n=>n+1);nativeSource.current=nativeFrame.control_source;}},[mode,nativeFrame]);
 useEffect(()=>{if(mode==='native'&&nativeFrame){setTrace(t=>[...t.slice(-69),nativeFrame.eef?.[2]??0]);setNativeHistory(h=>h.at(-1)?.sim_time===nativeFrame.sim_time?h:[...h.slice(-299),nativeFrame]);}},[mode,nativeFrame]);
 useEffect(()=>{if(mode==='native'&&nativeFrame?.control_source==='operator'&&referenceIndex<0&&nativeHistory.length)setReferenceIndex(Math.max(0,nativeHistory.length-5));},[mode,nativeFrame,referenceIndex,nativeHistory]);
 useEffect(()=>{if(mode!=='native'||!nativeFrame)return;const key=`${native.sessionId}:${nativeFrame.control_source}:${nativeFrame.segment}:${nativeFrame.done}:${nativeFrame.success}`;if(nativeEvent.current===key)return;nativeEvent.current=key;const title=nativeFrame.success?'Native simulator verified task completion':nativeFrame.done?'Native episode ended':nativeFrame.control_source==='operator'?(nativeFrame.segment==='task'?'Recovery marked · human task demonstration':'Operator takeover · recovery recording'):'Native policy session ready';const kind:Log['kind']=nativeFrame.success?'success':nativeFrame.control_source==='operator'?'manual':'neutral';setLogs(l=>[{time:simClock(nativeFrame.sim_time),title,kind},...l].slice(0,30));},[mode,nativeFrame,native.sessionId]);

 useEffect(()=>{
  const handler=(e:KeyboardEvent)=>{
   if((e.target as HTMLElement).matches('input,textarea,select'))return;
   if(e.key==='Escape'){e.preventDefault();send('stop');return;}
   if(latest.current.status!=='manual')return;
   const moves:Record<string,[number,number,number]>={ArrowUp:[0.025,0,0],ArrowDown:[-0.025,0,0],ArrowLeft:[0,0.025,0],ArrowRight:[0,-0.025,0],q:[0,0,0.025],e:[0,0,-0.025]};
   if(moves[e.key]){e.preventDefault();send('move',{delta:moves[e.key]});}
   if(e.key===' '){e.preventDefault();send('gripper',{gripper:latest.current.gripper>0.5?0:1});}
  };window.addEventListener('keydown',handler);return()=>window.removeEventListener('keydown',handler);
 },[send]);
 useEffect(()=>{if(toast){const t=setTimeout(()=>setToast(''),2800);return()=>clearTimeout(t);}},[toast]);
 const download=()=>{if(mode==='native'){void native.exportEpisode();return;}const data=recorder.current.exportEpisode();const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download=`werescue-episode-${Date.now()}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);setToast('Episode JSON exported');};
 const manual=snapshot.status==='manual';
 const alert=['attention','stopped','error'].includes(snapshot.status);
 const ready=snapshot.physicsReady;
 const running=snapshot.status==='running'||snapshot.status==='recovering';
 const progress=Math.round(Math.min(1,Math.max(0,snapshot.taskProgress))*100);
 const line=trace.map((v,i)=>`${i/Math.max(trace.length-1,1)*500},${68-Math.min(Math.max(v,0),1)*60}`).join(' ');
 return <div className="operator-workspace" data-phase={manual?(nativeFrame?.segment==='task'?'task':'recovery'):'policy'}>
  <header className="operator-header">
   <div className="brand"><span className="brand-mark"><LifeBuoy size={20}/></span>RG Service Center</div>
   <div className="operator-connection"><span className={ready?'live-dot':'live-dot gray'}/>{mode==='native'?(native.health?.inference_verified?'SmolVLA · GPU verified':native.connected?'Native LIBERO':'GPU disconnected'):'Offline · scripted demo'}<span className="mono">{simClock(snapshot.simTime)}</span></div>
  </header>
  <main className="operator-main">
   <section className="simulator-stage" aria-label="Robot simulator">
    <div className={`scene-canvas ${alert?'scene-warning':''}`}>
     {mode==='native'?<NativeViewport scene={native.scene} frame={native.frame} manual={manual} onOperatorCommand={onOperatorCommand} onReady={onNativeReady} view={view} cameraRevision={cameraRevision} ghost={false} referenceFrame={null} trajectory={nativeHistory.filter(f=>f.eef).map(f=>f.eef as [number,number,number])}/>:<FactoryScene key={sceneVersion} command={command} onUpdate={onUpdate} onOperatorCommand={onOperatorCommand} view={view} cameraRevision={cameraRevision} ghost={false}/>}
     <div className={`scene-status ${alert?'warning':manual?'human':''}`}><span className="live-dot"/>{statusLabel[snapshot.status]}{manual&&mode==='native'&&<span> · {nativeFrame?.segment==='task'?'Task · R':'Recovery · R′'}</span>}</div>
     <div className="simulator-hint">Drag ring · XY<span>Scroll · Z</span><span>Space · gripper</span><span>Esc · stop</span></div>
     {snapshot.status==='loading'&&<div className="loading-overlay"><div className="loader"/><h3>{native.connecting?'Loading workcell':'Connect your workcell'}</h3><p>{native.connecting?'Loading the native robot and objects…':'Use Connect GPU in the controls panel.'}</p></div>}
     {snapshot.status==='error'&&<div className="error-overlay"><TriangleAlert/><h3>Workcell needs attention</h3><p>{snapshot.message}</p><button className="button primary" onClick={mode==='native'?()=>void native.connect():reconnect}>Reconnect</button></div>}
    </div>
   </section>
   <aside className="control-dock" aria-label="Robot controls">
    <details className="dock-disclosure" open>
     <summary className="dock-toggle" aria-label="Toggle control panel" title="Show or hide controls"><span className="dock-toggle-title">Controls</span><ChevronRight size={17}/></summary>
     <div className="dock-body">
      <div className="dock-status"><div><span className="section-kicker">FRANKA · LIBERO</span><strong>{snapshot.success?'Task completed':manual?'Human control':running?'VLA running':native.connecting?'Connecting…':ready?'Ready to run':'Not connected'}</strong></div><span className="mono">{mode==='native'?`${nativeFrame?.step??0} steps`:simClock(snapshot.simTime)}</span></div>
      {mode==='native'&&(!native.connected||native.error)&&<button className="button primary dock-primary" disabled={native.connecting} onClick={()=>void native.connect()}><Radio size={15}/>{native.connecting?'Connecting…':native.connected?'Reconnect GPU':'Connect GPU'}</button>}
      {mode==='native'&&native.error&&<p className="dock-notice error">{native.error}</p>}
      {snapshot.success&&<p className="dock-notice success"><Check size={14}/>Native task success · episode saved</p>}
      <div className="dock-section dock-actions">
       {!manual?<><button className="button primary dock-primary" disabled={!ready||alert||snapshot.success||!!nativeFrame?.done} onClick={()=>send(running?'pause':'start')}>{running?<Pause size={15}/>:<Play size={15}/>} {running?'Pause':mode==='native'?'Start VLA':'Start demo'}</button><button className="button takeover" disabled={!ready||snapshot.success||!!nativeFrame?.done||snapshot.status==='stopped'} onClick={()=>send('take_control')}><Hand size={15}/>Take control<ArrowUpRight size={15}/></button></>:<>
        <div className="operator-recording"><span className="live-dot"/>{mode==='native'?(!native.operatorRunning?'Recording paused':nativeFrame?.segment==='task'?'Recording task · R':'Recording recovery · R′'):'Human control'}</div>
        {mode==='native'?<><button className="button mark-recovery" disabled={nativeFrame?.segment==='task'} onClick={()=>void native.markRecoveryComplete()}><Check size={14}/>{nativeFrame?.segment==='task'?'Recovery marked · finish the task':'Recovery complete → task'}</button>{!native.operatorRunning&&<button className="button resume" disabled={native.busy} onClick={()=>void native.takeControl()}><Play size={14}/>Resume human control</button>}</>:<button className="button resume" onClick={()=>send('resume_policy')}><Play size={14}/>Return to demo</button>}
        <p className="dock-help">{nativeFrame?.segment==='task'?'Keep control and place the can in the basket.':'Recover to state 2, mark recovery complete, then finish the task.'}</p>
       </>}
      </div>
      <div className="dock-section">
       <div className="dock-row dock-gripper"><span>Gripper</span><button className="button secondary" disabled={!manual} onClick={()=>send('gripper',{gripper:1})}>Open</button><button className="button secondary" disabled={!manual} onClick={()=>send('gripper',{gripper:0})}>Close</button></div>
       <div className="dock-row dock-height"><span>Height</span><button className="button secondary" disabled={!manual} onClick={()=>send('move',{delta:[0,0,.025]})}>Z+</button><button className="button secondary" disabled={!manual} onClick={()=>send('move',{delta:[0,0,-.025]})}>Z−</button></div>
       <button className="button dock-danger" disabled={!ready} onClick={()=>send('stop')}><Square size={12} fill="currentColor"/>Stop loop <small>Esc</small></button>
      </div>
      <div className="dock-section dock-episode"><div className="dock-row"><label htmlFor="next-start">Next start</label><select id="next-start" aria-label="Next episode start" value={native.initialStateId} onChange={e=>native.setInitialStateId(Number(e.target.value))}>{Array.from({length:20},(_,i)=><option key={i} value={i}>{String(i).padStart(2,'0')}</option>)}</select></div><button className="button secondary" disabled={!ready||native.connecting} onClick={()=>send('reset')}><RotateCcw size={14}/>{snapshot.success||nativeFrame?.done?'Next episode':'End & new episode'}</button></div>
      {mode==='native'&&<RenderPerformanceControls/>}
      <div className="dock-section dock-views"><label htmlFor="camera-view">View</label><div className="dock-row"><select id="camera-view" aria-label="Camera view" value={view} onChange={e=>setView(e.target.value as 'orbit'|'top'|'robot')}><option value="orbit">Perspective</option><option value="top">Top view</option><option value="robot">Robot view</option></select><button className="button secondary" aria-label="Reset camera" title="Reset camera" onClick={resetCamera}><LocateFixed size={15}/></button></div></div>
      <details className="dock-extra"><summary>Fine control</summary><div className={`teleop ${manual?'enabled':''}`}><div className="jog-layout"><div className="jog-pad"><button className="jog up" aria-label="Move X positive" disabled={!manual} onClick={()=>send('move',{delta:[.025,0,0]})}><ArrowUp size={17}/><small>+X</small></button><button className="jog left" aria-label="Move Y positive" disabled={!manual} onClick={()=>send('move',{delta:[0,.025,0]})}><ArrowLeft size={17}/><small>+Y</small></button><span className="jog-center"><Target size={18}/></span><button className="jog right" aria-label="Move Y negative" disabled={!manual} onClick={()=>send('move',{delta:[0,-.025,0]})}><ArrowRight size={17}/><small>−Y</small></button><button className="jog down" aria-label="Move X negative" disabled={!manual} onClick={()=>send('move',{delta:[-.025,0,0]})}><ArrowDown size={17}/><small>−X</small></button></div></div></div><p className="dock-help">Arrow keys · XY · Q/E · Z · Space · gripper</p></details>
      {mode==='native'&&native.frame?.cameras&&<details className="dock-extra"><summary>VLA cameras</summary><div className="native-cameras"><div>{Object.entries(native.frame.cameras).map(([name,image])=><figure key={name}><img alt={`Native ${name} camera observation`} src={`data:${image.mime};base64,${image.data}`}/><figcaption>{name==='agentview'?'Agent view':'Wrist view'}</figcaption></figure>)}</div></div></details>}
      <details className="dock-extra"><summary>Episode data</summary><p className="dock-help">{interventions} interventions · {nativeFrame?.step??0} native steps</p><button className="button secondary" onClick={download} disabled={mode==='native'&&!native.sessionId}><Download size={14}/>Export episode JSON</button><p className="dock-help">{mode==='native'?'Native cameras, states and actions save on the GPU.':'Offline simulation · not native training data.'}</p></details>
      <details className="dock-extra"><summary>Connection settings</summary><label className="dock-row">Mode<select aria-label="Simulation mode" value={mode} onChange={e=>{void native.pause();setMode(e.target.value as 'native'|'demo');setNativeRenderError('');}}><option value="native">GPU · actual VLA</option><option value="demo">Offline demo</option></select></label><label className="dock-settings-label">GPU endpoint<input aria-label="GPU API endpoint" value={native.endpoint} onChange={e=>native.setEndpoint(e.target.value)} spellCheck={false}/></label></details>
     </div>
    </details>
   </aside>
  </main>
  {toast&&<div className="toast"><Check size={16}/>{toast}</div>}
 </div>;
}
