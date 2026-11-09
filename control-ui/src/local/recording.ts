/** Human-only local MuJoCo collection. The simulator owns every recorded action. */

export type LocalState = {time:number;qpos:number[];qvel:number[];[key:string]:unknown};
export type LocalStep = {
  preState:LocalState;
  action:number[];
  postFrame?:Record<string,unknown>;
  frame?:Record<string,unknown>;
  successAfter?:boolean;
};
export type LocalRecorderOptions = {
  taskId:string;
  modelSha256?:string|null;
  sceneSha256:string;
  randomization:Record<string,unknown>;
  initialRobotQpos?:number[];
  controlHz?:20;
};

const finite=(value:unknown):value is number=>typeof value==='number'&&Number.isFinite(value);
const finiteArray=(value:unknown):value is number[]=>Array.isArray(value)&&value.length>0&&value.every(finite);
const hash=(value:string)=>/^[a-f0-9]{64}$/i.test(value);
const copy=<T>(value:T):T=>JSON.parse(JSON.stringify(value)) as T;
const uuid=()=>globalThis.crypto?.randomUUID?.()??`${Date.now()}-${Math.random().toString(16).slice(2)}`;
const DATABASE='rg-local-mujoco-episodes-v1';
type SavedEpisode={id:string;createdAt:string;taskId:string;body:string};

function openEpisodeDatabase():Promise<IDBDatabase>{
  return new Promise((resolve,reject)=>{
    if(typeof indexedDB==='undefined'){reject(new Error('IndexedDB is unavailable'));return;}
    const request=indexedDB.open(DATABASE,1);
    request.onupgradeneeded=()=>{if(!request.result.objectStoreNames.contains('episodes'))request.result.createObjectStore('episodes',{keyPath:'id'});};
    request.onsuccess=()=>resolve(request.result);
    request.onerror=()=>reject(request.error??new Error('Could not open local episode storage'));
  });
}

async function storeEpisode(record:SavedEpisode){
  const db=await openEpisodeDatabase();
  try{await new Promise<void>((resolve,reject)=>{
    const transaction=db.transaction('episodes','readwrite');
    transaction.objectStore('episodes').put(record);
    transaction.oncomplete=()=>resolve();
    transaction.onerror=()=>reject(transaction.error??new Error('Could not save episode'));
    transaction.onabort=()=>reject(transaction.error??new Error('Episode save aborted'));
  });}finally{db.close();}
}

async function storedEpisodes():Promise<SavedEpisode[]>{
  const db=await openEpisodeDatabase();
  try{return await new Promise<SavedEpisode[]>((resolve,reject)=>{
    const request=db.transaction('episodes','readonly').objectStore('episodes').getAll();
    request.onsuccess=()=>resolve((request.result as SavedEpisode[]).sort((a,b)=>a.createdAt.localeCompare(b.createdAt)));
    request.onerror=()=>reject(request.error??new Error('Could not read saved episodes'));
  });}finally{db.close();}
}

async function downloadText(text:string,filename:string){
  let blob:Blob;
  if(typeof CompressionStream!=='undefined'){
    blob=await new Response(new Blob([text]).stream().pipeThrough(new CompressionStream('gzip'))).blob();
  }else blob=new Blob([text],{type:'application/x-ndjson'});
  const url=URL.createObjectURL(blob),anchor=document.createElement('a');
  anchor.href=url;anchor.download=filename+(typeof CompressionStream!=='undefined'?'.gz':'');
  document.body.append(anchor);anchor.click();anchor.remove();
  setTimeout(()=>URL.revokeObjectURL(url),5000);
  return {filename:anchor.download,bytes:blob.size};
}

export async function downloadAllLocalEpisodes(){
  const saved=await storedEpisodes();
  if(!saved.length)throw new Error('No finished local episodes are saved yet.');
  const header={type:'collection',schema:'rg.local_mujoco_collection.v1',episode_ids:saved.map(row=>row.id),
    count:saved.length,ready_for_lerobot_training:false,requires_native_replay:true};
  const result=await downloadText(JSON.stringify(header)+'\n'+saved.map(row=>row.body).join(''),
    `rg-local-collection-${new Date().toISOString().slice(0,10)}.jsonl`);
  return {...result,episodes:saved.length};
}

export function createLocalRecorder(options:LocalRecorderOptions){
  if(!options.taskId||!hash(options.sceneSha256)||(options.modelSha256!=null&&!hash(options.modelSha256)))throw new Error('Local recording requires an exact scene SHA256 and a valid model hash when available.');
  if(options.controlHz!==undefined&&options.controlHz!==20)throw new Error('Local recording requires 20 Hz actions.');
  const initialRobotQpos=options.initialRobotQpos??options.randomization?.robotJointQposAfter;
  if(!finiteArray(initialRobotQpos))throw new Error('Initial robot joint positions are required.');
  if(!options.randomization||typeof options.randomization!=='object'||!finite(options.randomization.seed))throw new Error('Robot joint randomization and its seed are required.');

  const episodeId=uuid(),createdAt=new Date().toISOString();
  const steps:Array<Record<string,unknown>>=[];
  const events:Array<Record<string,unknown>>=[];
  let firstSimTime:number|null=null;
  let recoveryCompleteStep:number|null=null;
  let terminal:Record<string,unknown>|null=null;
  let stateShape:{nq:number;nv:number}|null=null;
  let savePromise:Promise<void>|null=null;
  let saveError:string|null=null;

  function recordStep(step:LocalStep){
    if(terminal)throw new Error('Episode is already finished.');
    const state=step.preState;
    if(!state||!finite(state.time)||!finiteArray(state.qpos)||!finiteArray(state.qvel))throw new Error('Full finite pre-action qpos/qvel and simulator time are required.');
    if(!finiteArray(step.action)||step.action.length!==7||step.action.some(value=>value < -1.00001||value > 1.00001))throw new Error('Executed action must contain seven bounded values.');
    if(!stateShape)stateShape={nq:state.qpos.length,nv:state.qvel.length};
    if(state.qpos.length!==stateShape.nq||state.qvel.length!==stateShape.nv)throw new Error('Simulator state dimensions changed mid-episode.');
    if(firstSimTime===null)firstSimTime=state.time;
    const expected=firstSimTime+steps.length/20;
    if(Math.abs(state.time-expected)>.002)throw new Error(`Noncontiguous 20 Hz simulation time at step ${steps.length}.`);
    const row={type:'step',step:steps.length,sim_time_before:state.time,
      segment:recoveryCompleteStep===null?'recovery':'task',control_source:'human',
      pre_action_state:copy(state),executed_action:copy(step.action),
      ...((step.postFrame??step.frame)?{post_frame:copy(step.postFrame??step.frame)}:{}),
      success_after:step.successAfter===true};
    steps.push(row);
    return row;
  }

  function markRecoveryComplete(){
    if(terminal)throw new Error('Episode is already finished.');
    if(recoveryCompleteStep!==null)throw new Error('Recovery boundary was already marked.');
    if(steps.length===0)throw new Error('Record at least one recovery action before marking the boundary.');
    recoveryCompleteStep=steps.length;
    events.push({type:'recovery_complete',step:recoveryCompleteStep,
      meaning:'R_prime ends; human continues with task R from this state'});
    return recoveryCompleteStep;
  }

  function finish(result:{success:boolean;reason:string;finalState?:Record<string,unknown>}){
    if(terminal)throw new Error('Episode is already finished.');
    if(!result.reason||typeof result.success!=='boolean')throw new Error('Result needs an explicit success and reason.');
    terminal={type:'terminal',step:steps.length,success:result.success,reason:result.reason,
      ...(result.finalState?{final_state:copy(result.finalState)}:{})};
    events.push(terminal);
    savePromise=storeEpisode({id:episodeId,createdAt,taskId:options.taskId,body:jsonl()})
      .catch(error=>{saveError=error instanceof Error?error.message:String(error);});
    return snapshot();
  }

  function snapshot(){
    return {episodeId,steps:steps.length,recoverySteps:recoveryCompleteStep??steps.length,
      taskSteps:recoveryCompleteStep===null?0:steps.length-recoveryCompleteStep,
      recoveryCompleteStep,finished:terminal!==null,success:terminal?.success===true,
      replayRequired:true,readyForLerobotTraining:false,saveError};
  }

  function jsonl(){
    if(!terminal)throw new Error('Finish the episode before export.');
    const metadata={type:'metadata',schema:'rg.local_mujoco_human_episode.v1',episode_id:episodeId,
      created_at:createdAt,source:'human_browser_unverified',task_id:options.taskId,
      control_hz:20,model_sha256:options.modelSha256??null,scene_sha256:options.sceneSha256,
      initial_robot_qpos:copy(initialRobotQpos),robot_joint_randomization:copy(options.randomization),
      controller:'local_mujoco_reimplementation',action_schema:'relative_OSC_POSE_delta_xyz3_rotation3_gripper1',
      action_dim:7,pre_action_state_schema:'full_local_mujoco_qpos_qvel',
      trainability:{ready_for_lerobot:false,requires_native_replay:true,
        missing:['verified 8D LeRobot observation.state','two aligned native LIBERO policy camera images','native controller/action replay parity']},
      recovery_complete_step:recoveryCompleteStep,steps:steps.length,
      segments:{R_prime:{start_step:0,end_step_exclusive:recoveryCompleteStep??steps.length},
        R:recoveryCompleteStep===null?null:{start_step:recoveryCompleteStep,end_step_exclusive:steps.length}},
      result:copy(terminal)};
    return [metadata,...events,...steps].map(row=>JSON.stringify(row)).join('\n')+'\n';
  }

  async function download(){
    const result=await downloadText(jsonl(),`rg-local-${episodeId}.jsonl`);
    return {...result,...snapshot()};
  }

  async function downloadAll(){
    if(savePromise)await savePromise;
    if(saveError)throw new Error(`Latest episode was not saved: ${saveError}`);
    return downloadAllLocalEpisodes();
  }

  return {recordStep,markRecoveryComplete,finish,snapshot,jsonl,download,downloadAll};
}

export type LocalRecorder=ReturnType<typeof createLocalRecorder>;
