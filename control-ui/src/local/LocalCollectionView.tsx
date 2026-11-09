import {useCallback,useEffect,useRef,useState} from 'react';
import NativeScene,{type NativeFrame} from '../sim/NativeScene';
import type {NativeScenePayload} from '../sim/NativeScene';
import type {SceneCommand} from '../protocol/types';
import {createLocalLiberoEngine} from './engine';
import {createLocalRecorder,type LocalRecorder} from './recording';
import {startCanvasVideoCapture} from './videoCapture';
import './local-collection.css';

const SCENE_HASH='4a7eb0dbbe6e6b968ee9e5589df8b537a72659635525a18fc631b3d231723104';
const TASK_ID='libero_object__pick_up_the_alphabet_soup_and_place_it_in_the_basket';
type Engine=Awaited<ReturnType<typeof createLocalLiberoEngine>>;
type Phase='loading'|'ready'|'replay'|'replay_paused'|'running'|'paused'|'done'|'error';
type ReplayFrame={step:number;sim_time:number;qpos:number[];qvel:number[]};
const clamp=(v:number)=>Math.max(-1,Math.min(1,v));
const seed=()=>{const bits=new Uint32Array(1);crypto.getRandomValues(bits);return bits[0]||1;};
async function xmlHash(xml:string){const bytes=new TextEncoder().encode(xml);return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');}

export default function LocalCollectionView(){
 const [phase,setPhase]=useState<Phase>('loading');
 const [scene,setScene]=useState<NativeScenePayload|null>(null);
 const [frame,setFrame]=useState<NativeFrame|null>(null);
 const [steps,setSteps]=useState(0);
 const [completed,setCompleted]=useState(0);
 const [segment,setSegment]=useState<'recovery'|'task'>('recovery');
 const [error,setError]=useState('');
 const [lastFile,setLastFile]=useState('');
 const [startSeed,setStartSeed]=useState<number|null>(null);
 const [replayFrames,setReplayFrames]=useState<ReplayFrame[]>([]),[replayIndex,setReplayIndex]=useState(0);
 const engine=useRef<Engine|null>(null),recorder=useRef<LocalRecorder|null>(null);
 const frameRef=useRef<NativeFrame|null>(null),sceneHash=useRef(SCENE_HASH),modelHash=useRef(SCENE_HASH);
 const timer=useRef<ReturnType<typeof setInterval>|null>(null),policyTimer=useRef<ReturnType<typeof setInterval>|null>(null),policyIndex=useRef(0),capture=useRef<ReturnType<typeof startCanvasVideoCapture>|null>(null),keys=useRef(new Set<string>()),grip=useRef(-1);
 const pending=useRef<[number,number,number]>([0,0,0]),rotationPending=useRef<[number,number,number]>([0,0,0]),rightHeld=useRef(false),tilting=useRef(false),active=useRef(false);
 const clickStart=useRef<{x:number;y:number}|null>(null);
 const stageRef=useRef<HTMLDivElement|null>(null),lockTarget=useRef<Element|null>(null),lockAttempted=useRef(false);

 const stopClock=useCallback(()=>{active.current=false;if(timer.current){clearInterval(timer.current);timer.current=null;}},[]);
 const stopPolicy=useCallback(()=>{if(policyTimer.current){clearInterval(policyTimer.current);policyTimer.current=null;}},[]);
 const startCapture=useCallback(()=>{capture.current?.cancel();capture.current=null;const attach=(tries:number)=>{const canvas=stageRef.current?.querySelector('canvas');if(canvas){try{capture.current=startCanvasVideoCapture(canvas,`rg-policy-human-${Date.now()}`);}catch(caught){setError(String(caught));}}else if(tries<15)setTimeout(()=>attach(tries+1),100);};setTimeout(()=>attach(0),100);},[]);
 const finish=useCallback(async(success:boolean,reason:string)=>{
  stopClock();
  const record=recorder.current;if(!record)return;
  try{
   const current=frameRef.current;
   record.finish({success,reason,finalState:current?{qpos:current.qpos,qvel:current.qvel,time:current.sim_time}:undefined});
   const saved=await record.download();const video=await capture.current?.stop();capture.current=null;setLastFile(video?`${saved.filename} + ${video.filename}`:saved.filename);setCompleted(v=>v+1);setPhase('done');
  }catch(caught){setError(String(caught));setPhase('error');}
 },[stopClock]);

 const tick=useCallback(()=>{
  if(!active.current||!engine.current||!recorder.current)return;
  const held=keys.current,pointer=pending.current;
  const keyMove:[number,number,number]=[
   (held.has('s')?0.008:0)-(held.has('w')?0.008:0),
   (held.has('d')?0.008:0)-(held.has('a')?0.008:0),
   (held.has('r')?0.006:0)-(held.has('f')?0.006:0),
  ];
  const usingPointer=Math.hypot(...pointer)>1e-7;
  const requested=usingPointer?pointer:keyMove;
  const length=Math.hypot(...requested),fraction=length>.009?.009/length:1;
  const applied=requested.map(value=>value*fraction) as [number,number,number];
  pending.current=usingPointer?pointer.map((value,i)=>Math.abs(value-applied[i])<1e-7?0:value-applied[i]) as [number,number,number]:[0,0,0];
  const rotation=rotationPending.current.map(value=>clamp(value/.5)) as [number,number,number];rotationPending.current=[0,0,0];
  const action=[
   applied[0]/.05,applied[1]/.05,applied[2]/.05,
   rotation[0]+(((held.has('arrowdown')||held.has('x')) ? .09 : 0)-((held.has('arrowup')||held.has('z')) ? .09 : 0)),
   rotation[1]+(((held.has('arrowright')||held.has('v')) ? .09 : 0)-((held.has('arrowleft')||held.has('c')) ? .09 : 0)),
   rotation[2]+(rightHeld.current&&!tilting.current?-.09:0)+(held.has('q')?.09:0)-(held.has('e')?.09:0),grip.current,
  ];
  for(let i=3;i<6;i++)action[i]=clamp(action[i]);
  try{
   const out=engine.current.step(action);
   recorder.current.recordStep(out);
   frameRef.current=out.frame;setFrame(out.frame);setSteps(v=>v+1);
   if(out.successAfter)void finish(true,'local_mujoco_success');
  }catch(caught){stopClock();setError(String(caught));setPhase('error');}
 },[finish,stopClock]);

 const runClock=useCallback(()=>{if(active.current)return;active.current=true;setPhase('running');timer.current=setInterval(tick,50);},[tick]);

 useEffect(()=>{let cancelled=false;fetch('/replay/base-flail.json').then(response=>response.json()).then(data=>{if(!cancelled&&Array.isArray(data.frames))setReplayFrames(data.frames);}).catch(caught=>{if(!cancelled)setError(`Recorded policy unavailable: ${String(caught)}`);});return()=>{cancelled=true;};},[]);
 useEffect(()=>{
  let cancelled=false;
  (async()=>{
   try{
    const response=await fetch(`/local-scene?expected_scene_sha256=${SCENE_HASH}`);
    if(!response.ok)throw new Error(`Local scene unavailable (HTTP ${response.status})`);
    const payload=await response.json() as NativeScenePayload;
    if(cancelled)return;
    modelHash.current=await xmlHash(payload.xml);
    const local=await createLocalLiberoEngine(payload);
    if(cancelled){local.dispose();return;}
    engine.current=local;setScene(payload);setPhase('ready');
   }catch(caught){if(!cancelled){setError(String(caught));setPhase('error');}}
  })();
  return()=>{cancelled=true;stopClock();stopPolicy();capture.current?.cancel();engine.current?.dispose();engine.current=null;};
 },[stopClock,stopPolicy]);

 useEffect(()=>{
  const down=(event:KeyboardEvent)=>{
   if(event.target instanceof HTMLInputElement||event.target instanceof HTMLSelectElement)return;
   if(phase==='replay'&&event.key==='Escape'){stopPolicy();setPhase('replay_paused');return;}
   if(phase!=='running')return;
   const key=/^Key[A-Z]$/.test(event.code)?event.code.slice(3).toLowerCase():event.code.startsWith('Arrow')?event.code.toLowerCase():event.key.toLowerCase();
   if(['arrowup','arrowdown','arrowleft','arrowright','w','a','s','d','r','f','q','e','z','x','c','v',' '].includes(key))event.preventDefault();
   if(key===' '){if(!event.repeat)grip.current=grip.current>0?-1:1;return;}
   if(key==='escape'){keys.current.clear();stopClock();setPhase('paused');return;}
   if(key==='enter'&&segment==='recovery'&&steps>0){try{recorder.current?.markRecoveryComplete();setSegment('task');}catch(caught){setError(String(caught));}return;}
   keys.current.add(key);
  };
  const up=(event:KeyboardEvent)=>keys.current.delete(/^Key[A-Z]$/.test(event.code)?event.code.slice(3).toLowerCase():event.code.startsWith('Arrow')?event.code.toLowerCase():event.key.toLowerCase());
  window.addEventListener('keydown',down);window.addEventListener('keyup',up);
  return()=>{window.removeEventListener('keydown',down);window.removeEventListener('keyup',up);};
 },[phase,segment,steps,stopClock,stopPolicy]);

 useEffect(()=>{
  const released=()=>{if(phase==='running'&&lockAttempted.current&&!document.pointerLockElement){keys.current.clear();stopClock();setPhase('paused');}};
  document.addEventListener('pointerlockchange',released);
  return()=>document.removeEventListener('pointerlockchange',released);
 },[phase,stopClock]);

 useEffect(()=>{
  const move=(event:MouseEvent)=>{if(phase!=='running'||!rightHeld.current||!tilting.current)return;rotationPending.current[0]+=event.movementY*.006;rotationPending.current[1]+=event.movementX*.006;};
  const up=()=>{rightHeld.current=false;tilting.current=false;};
  document.addEventListener('mousemove',move);window.addEventListener('mouseup',up);window.addEventListener('blur',up);
  return()=>{document.removeEventListener('mousemove',move);window.removeEventListener('mouseup',up);window.removeEventListener('blur',up);};
 },[phase]);

 const start=async()=>{
  if(!engine.current||!scene)return;
  stopClock();setError('');setLastFile('');setSteps(0);setSegment('recovery');keys.current.clear();pending.current=[0,0,0];rotationPending.current=[0,0,0];rightHeld.current=false;tilting.current=false;grip.current=-1;lockAttempted.current=false;
  try{
   const value=seed();const reset=await engine.current.reset(value);
   frameRef.current=reset.frame;setFrame(reset.frame);setStartSeed(value);
   recorder.current=createLocalRecorder({taskId:TASK_ID,modelSha256:modelHash.current,sceneSha256:sceneHash.current,
    randomization:reset.randomization,initialRobotQpos:(reset.randomization as {joint_qpos?:number[]}).joint_qpos??reset.frame.joints??reset.frame.qpos.slice(0,7),controlHz:20});
   runClock();startCapture();
  }catch(caught){setError(String(caught));setPhase('error');}
 };
 const playReplay=()=>{
  if(!engine.current||replayFrames.length<2)return;
  stopClock();stopPolicy();setError('');setLastFile('');setSegment('recovery');keys.current.clear();pending.current=[0,0,0];
  if(policyIndex.current>=replayFrames.length-1)policyIndex.current=0;
  if(policyIndex.current===0){recorder.current=null;startCapture();}
  const advance=()=>{const index=policyIndex.current;if(index>=replayFrames.length){stopPolicy();setPhase('replay_paused');return;}
   try{const next=engine.current!.restoreRecordedFrame(replayFrames[index]);frameRef.current=next;setFrame(next);setReplayIndex(index);setSteps(index);policyIndex.current=index+1;
    if(policyIndex.current>=replayFrames.length){stopPolicy();setPhase('replay_paused');}}
   catch(caught){stopPolicy();setError(String(caught));setPhase('error');}};
  setPhase('replay');advance();policyTimer.current=setInterval(advance,50);
 };
 const pauseReplay=()=>{stopPolicy();setPhase('replay_paused');};
 const takeReplayControl=()=>{
  if(!engine.current||!frameRef.current||!['replay','replay_paused'].includes(phase))return;
  stopPolicy();setError('');const current=engine.current.beginHumanControl();frameRef.current=current;setFrame(current);setSegment('recovery');setSteps(0);grip.current=current.gripper?.[0]&&current.gripper[0]<.02?1:-1;
  recorder.current=createLocalRecorder({taskId:TASK_ID,sceneSha256:sceneHash.current,modelSha256:modelHash.current,
   randomization:{seed:0,source:'recorded_smolvla_base_policy_handoff',recorded_policy_step:replayIndex,recorded_model:'lerobot/smolvla_base'},
   initialRobotQpos:current.qpos.slice(0,7),controlHz:20});
  runClock();
 };
 const command=(value:Omit<SceneCommand,'id'>)=>{
  if(phase!=='running')return;
  if(value.type==='move'&&value.delta){const next=pending.current.map((v,i)=>v+value.delta![i]) as [number,number,number];const xy=Math.hypot(next[0],next[1]);if(xy>.03){next[0]*=.03/xy;next[1]*=.03/xy;}next[2]=Math.max(-.04,Math.min(.04,next[2]));pending.current=next;}
  if(value.type==='gripper'&&value.gripper!==undefined)grip.current=value.gripper>.5?-1:1;
 };
 const wheelAnywhere=(event:React.WheelEvent<HTMLDivElement>)=>{
  if(phase!=='running')return;
  event.preventDefault();
  const pixels=(event.shiftKey&&!event.deltaY?event.deltaX:event.deltaY)*(event.deltaMode===1?16:event.deltaMode===2?400:1);
  if(event.shiftKey)rotationPending.current[2]-=pixels*.0025;
  else command({type:'move',delta:[0,0,Math.max(-.025,Math.min(.025,-pixels*.00009))]});
 };
 const mark=()=>{try{recorder.current?.markRecoveryComplete();setSegment('task');}catch(caught){setError(String(caught));}};

 return <div className="rg-local">
  <header className="rg-local-header"><strong>RG Service Center</strong><span>RECORDED SMOLVLA → LOCAL HUMAN CONTROL</span><span>{completed} episodes saved</span></header>
  <main className="rg-local-main">
   <div className="rg-local-stage" ref={stageRef} onWheel={wheelAnywhere} onContextMenu={event=>event.preventDefault()}
    onPointerDown={event=>{if(phase!=='running')return;if(event.button===2){rightHeld.current=true;tilting.current=event.shiftKey;event.preventDefault();return;}if(event.button===0)clickStart.current={x:event.clientX,y:event.clientY};}}
    onClick={event=>{const start=clickStart.current;clickStart.current=null;if(phase!=='running'||!start||Math.hypot(event.clientX-start.x,event.clientY-start.y)>=6)return;
     if(!document.pointerLockElement){lockAttempted.current=true;lockTarget.current=event.target instanceof HTMLCanvasElement?event.target:stageRef.current;void (lockTarget.current as HTMLElement|null)?.requestPointerLock?.();return;}
     grip.current=grip.current>0?-1:1;}}>
    {scene&&frame?<NativeScene scene={scene} frame={frame} manual={phase==='running'} onOperatorCommand={command} view="orbit" ghost={false} renderProfile="performance" mouseStyle="wemine"/>:
     <div className="rg-local-loading">{phase==='loading'?'Loading local LIBERO workcell…':error||'Start a randomized episode'}</div>}
    <div className="rg-local-stage-label">Alphabet soup can → basket · {phase==='replay'?'Recorded SmolVLA playing':phase==='replay_paused'?'Recording paused · Take Control':phase==='running'?'Human control · recording':phase}</div>
   </div>
   <aside className="rg-local-controls">
    <h1>SmolVLA → human control</h1>
    <p>Recorded base policy · local LIBERO scene · 20 Hz</p>
    <div className="rg-local-progress"><span className={segment==='recovery'?'current':''}>3′ Recover</span><span>→</span><span className={segment==='task'?'current':''}>2 → 3 → 4 Finish</span></div>
    <button className="primary" onClick={playReplay} disabled={phase==='loading'||phase==='running'||phase==='replay'||replayFrames.length<2}>{phase==='replay_paused'?'Resume SmolVLA recording':'Play recorded SmolVLA'}</button>
    <button onClick={pauseReplay} disabled={phase!=='replay'}>Pause recording</button>
    <button onClick={takeReplayControl} disabled={!['replay','replay_paused'].includes(phase)}>Take Control · record human</button>
    <button onClick={()=>void start()} disabled={phase==='loading'||phase==='running'||phase==='replay'}>Start random joints instead</button>
    <button onClick={mark} disabled={!['running','paused'].includes(phase)||segment!=='recovery'||steps===0}>Recovery complete · keep control</button>
    <div className="rg-local-grip"><button onClick={()=>{grip.current=-1;}} disabled={phase!=='running'}>Open gripper</button><button onClick={()=>{grip.current=1;}} disabled={phase!=='running'}>Close gripper</button></div>
    <button onClick={()=>{stopClock();setPhase('paused');}} disabled={phase!=='running'}>Pause</button>
    <button onClick={runClock} disabled={phase!=='paused'}>Resume</button>
    <button onClick={()=>void finish(Boolean(frame?.success),'operator_finished')} disabled={!['running','paused'].includes(phase)||steps===0}>Finish & download</button>
    <button onClick={()=>{void recorder.current?.downloadAll().then(result=>setLastFile(`All episodes: ${result.filename}`)).catch(caught=>setError(String(caught)));}} disabled={completed===0}>Download all episodes</button>
    <div className="rg-local-status">{phase==='replay'||phase==='replay_paused'?`Recorded policy step ${replayIndex+1}/${replayFrames.length}`:`Human step ${steps}`} · {startSeed===null?'recorded policy start':`seed ${startSeed}`}</div>
    {lastFile&&<div className="rg-local-saved">Saved {lastFile}</div>}
    {error&&<div role="alert" className="rg-local-error">{error}</div>}
    <p className="rg-local-help">Mouse: XY · Wheel: height<br/>First click: capture mouse · Later click: grip<br/>WASD: XY · R/F: height<br/>Q/E: yaw · Arrows: pitch/roll<br/>Space: grip · Enter: recovery done<br/>Esc: pause/release mouse</p>
    <small>Recorded policy motion is replayed from saved LIBERO states. Human actions and this view are saved locally; native camera replay is required before VLA training.</small>
   </aside>
  </main>
 </div>;
}
