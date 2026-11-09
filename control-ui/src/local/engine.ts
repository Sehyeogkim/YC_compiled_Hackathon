import type {NativeFrame,NativeScenePayload} from '../sim/NativeScene';
import {nativeDisplayXml} from '../sim/nativeDisplayXml';
import initial from './initial-state.json';

type Mj=any;
export type LocalAction=[number,number,number,number,number,number,number];
export type LocalPreState={qpos:number[];qvel:number[];time:number};
export type LocalRandomization={seed:number;robotJointQposBefore:number[];robotJointQposAfter:number[];robotJointOffsets:number[];canDeltaXY:[number,number];actualEef:[number,number,number];attempts:number;source:string};
export type LocalFrame=NativeFrame&{step:number;success:boolean;done:boolean;control_source:'operator'};
export type LocalStep={preState:LocalPreState;action:LocalAction;frame:LocalFrame;successAfter:boolean};

const LIMIT=0.03;
const finite=(xs:ArrayLike<number>)=>Array.from(xs).every(Number.isFinite);
const clamp=(v:number,lo:number,hi:number)=>Math.max(lo,Math.min(hi,v));
const vec3=(xs:ArrayLike<number>,at:number):[number,number,number]=>[xs[at],xs[at+1],xs[at+2]];
function rng(seed:number){let value=seed>>>0||0x9e3779b9;return()=>{value^=value<<13;value^=value>>>17;value^=value<<5;return(value>>>0)/4294967296;};}
function solveLinear(a:number[][],b:number[]):number[]{
  const n=b.length;
  const m=a.map((row,i)=>[...row,b[i]]);
  for(let c=0;c<n;c++){
    let best=c;for(let r=c+1;r<n;r++)if(Math.abs(m[r][c])>Math.abs(m[best][c]))best=r;
    if(Math.abs(m[best][c])<1e-10)return Array(n).fill(0);
    [m[c],m[best]]=[m[best],m[c]];const divisor=m[c][c];for(let k=c;k<=n;k++)m[c][k]/=divisor;
    for(let r=0;r<n;r++)if(r!==c){const f=m[r][c];for(let k=c;k<=n;k++)m[r][k]-=f*m[c][k];}
  }
  return m.map(row=>row[n]);
}
const dot=(a:number[],b:number[])=>a.reduce((v,x,i)=>v+x*b[i],0);
function matmul(a:number[],b:number[]):number[]{return Array.from({length:9},(_,k)=>{const r=Math.floor(k/3),c=k%3;return a[r*3]*b[c]+a[r*3+1]*b[3+c]+a[r*3+2]*b[6+c];});}
function axisRotation(v:number[]):number[]{
  const angle=Math.hypot(...v);if(angle<1e-12)return [1,0,0,0,1,0,0,0,1];
  const [x,y,z]=v.map(a=>a/angle),c=Math.cos(angle),s=Math.sin(angle),t=1-c;
  return [t*x*x+c,t*x*y-s*z,t*x*z+s*y,t*x*y+s*z,t*y*y+c,t*y*z-s*x,t*x*z-s*y,t*y*z+s*x,t*z*z+c];
}
function rotationError(current:number[],target:number[]):number[]{
  const out=[0,0,0];
  for(let c=0;c<3;c++){
    const a=[current[c],current[3+c],current[6+c]],b=[target[c],target[3+c],target[6+c]];
    out[0]+=.5*(a[1]*b[2]-a[2]*b[1]);out[1]+=.5*(a[2]*b[0]-a[0]*b[2]);out[2]+=.5*(a[0]*b[1]-a[1]*b[0]);
  }
  return out;
}

/** Local physics only. The controller and success check are task-specific approximations,
 * so exported episodes must be replayed/validated before use as native LIBERO training data. */
export class LocalLiberoEngine{
  readonly mj:Mj;readonly model:Mj;readonly data:Mj;readonly sceneSha256:string;
  readonly controllerKind='local_6dof_damped_jacobian_pd_approx.v2';
  readonly successKind='local_can_in_basket_heuristic.v1';
  readonly names:string[];private dt:number;private goal:[number,number,number]=[0,0,0];private goalRotation:number[]=[1,0,0,0,1,0,0,0,1];private grip=0;
  private stepCount=0;private settle=0;private won=false;private jacp:Mj;private jacr:Mj;
  private constructor(mj:Mj,model:Mj,data:Mj,payload:NativeScenePayload&{scene_sha256?:string}){
    this.mj=mj;this.model=model;this.data=data;this.sceneSha256=payload.scene_sha256??'';this.names=payload.body_names??[];
    const opt=model.opt;this.dt=Number(opt.timestep);opt.delete?.();
    if(!Number.isFinite(this.dt)||this.dt<=0)throw new Error('Invalid local MuJoCo physics timestep');
    this.jacp=new mj.DoubleBuffer(3*model.nv);this.jacr=new mj.DoubleBuffer(3*model.nv);
  }
  static async create(payload:NativeScenePayload&{scene_sha256?:string}){
    const {default:load}=await import('mujoco-js');const mj=await load();
    const root='/rg_local';try{mj.FS.mkdir(root);}catch{}
    for(const asset of payload.assets){
      const relative=asset.path.replace(/^\/+/, '');if(relative.split('/').includes('..'))throw new Error('Unsafe scene asset path');
      const path=root+'/'+relative;let directory='';for(const segment of path.split('/').slice(0,-1)){if(!segment)continue;directory+='/'+segment;try{mj.FS.mkdir(directory);}catch{}}
      const raw=atob(asset.data);const bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);mj.FS.writeFile(path,bytes);
    }
    mj.FS.writeFile(root+'/scene.xml',nativeDisplayXml(payload.xml));
    const model=mj.MjModel.loadFromXML(root+'/scene.xml');const data=new mj.MjData(model);
    if(model.nq!==58||model.nv!==51||payload.nq!==58||payload.nv!==51){data.delete();model.delete();throw new Error('Unexpected LIBERO Object0 topology');}
    return new LocalLiberoEngine(mj,model,data,payload);
  }
  get frame():LocalFrame{
    const d=this.data;const names=this.names;
    return {qpos:Array.from(d.qpos),qvel:Array.from(d.qvel),sim_time:Number(d.time),
      eef:vec3(d.site_xpos,8*3),eef_orientation:Array.from(d.site_xmat.slice(8*9,8*9+9)),joints:Array.from(d.qpos.slice(0,7)),gripper:[Number(d.qpos[7]),Number(d.qpos[8])],
      body_poses:{names,xpos:names.map((_,i)=>vec3(d.xpos,i*3)),xquat:names.map((_,i)=>Array.from(d.xquat.slice(i*4,i*4+4)))},
      step:this.stepCount,success:this.won,done:this.won,control_source:'operator'};
  }
  private preState():LocalPreState{return {qpos:Array.from(this.data.qpos),qvel:Array.from(this.data.qvel),time:Number(this.data.time)};}
  private robotWorldContacts(){
    const d=this.data,m=this.model;
    const pairs=new Set<string>();
    for(let i=0;i<d.ncon;i++){
      const c=d.contact.get(i);try{
        const a=m.geom_bodyid[c.geom1],b=m.geom_bodyid[c.geom2];
        if((a>0&&a<19&&b>=19)||(b>0&&b<19&&a>=19))pairs.add([Math.min(c.geom1,c.geom2),Math.max(c.geom1,c.geom2)].join(':'));
      }finally{c.delete?.();}
    }
    return pairs;
  }
  reset(seed:number):{frame:LocalFrame;randomization:LocalRandomization}{
    if(!Number.isSafeInteger(seed)||seed<0)throw new Error('Seed must be a nonnegative integer');
    const mj=this.mj,m=this.model,d=this.data,random=rng(seed);
    mj.mj_resetData(m,d);
    if(initial.qpos.length!==58||initial.qvel.length!==51)throw new Error('Pinned native initial state dimension changed');
    d.qpos.set(initial.qpos);d.qvel.set(initial.qvel);d.time=0;mj.mj_forward(m,d);
    const before=Array.from(d.qpos.slice(0,7)) as number[];
    const canXY:[number,number]=[d.qpos[9],d.qpos[10]];
    const baselineContacts=this.robotWorldContacts();
    let offsets:number[]=[],canDelta:[number,number]=[0,0],attempts=0,accepted=false;
    for(let trial=0;trial<80;trial++){
      attempts=trial+1;d.qpos.set(initial.qpos);d.qvel.set(initial.qvel);
      offsets=Array.from({length:7},()=>((random()*2)-1)*LIMIT);
      for(let i=0;i<7;i++){
        const low=m.jnt_range[i*2],high=m.jnt_range[i*2+1];
        d.qpos[i]=clamp(before[i]+offsets[i],low+0.025,high-0.025);
        offsets[i]=d.qpos[i]-before[i];
      }
      canDelta=[(random()*2-1)*.04,(random()*2-1)*.04];
      d.qpos[9]=canXY[0]+canDelta[0];d.qpos[10]=canXY[1]+canDelta[1];
      if(d.qpos[9]<-.25||d.qpos[9]>.15||d.qpos[10]<-.30||d.qpos[10]>.30)continue;
      mj.mj_forward(m,d);
      const can=vec3(d.xpos,19*3),basket=vec3(d.xpos,20*3);
      const eef=vec3(d.site_xpos,8*3);
      const novelContacts=[...this.robotWorldContacts()].some(pair=>!baselineContacts.has(pair));
      if(Math.hypot(can[0]-basket[0],can[1]-basket[1])<.15||eef[2]<.08||novelContacts||!finite(d.qpos)||!finite(d.qvel))continue;
      accepted=true;break;
    }
    if(!accepted)throw new Error('Could not sample a collision-free local start');
    this.goal=vec3(d.site_xpos,8*3);this.goalRotation=Array.from(d.site_xmat.slice(8*9,8*9+9));this.grip=-1;this.stepCount=0;this.settle=0;this.won=false;
    for(let i=0;i<7;i++)d.ctrl[i]=0;
    d.ctrl[7]=.04;d.ctrl[8]=-.04;
    const frame=this.frame;
    return {frame,randomization:{seed,robotJointQposBefore:before,robotJointQposAfter:Array.from(d.qpos.slice(0,7)),robotJointOffsets:offsets,canDeltaXY:canDelta,actualEef:frame.eef!,attempts,source:initial.source}};
  }
  step(action:number[]):LocalStep{
    if(this.won)throw new Error('Local episode already succeeded');
    if(action.length!==7||!finite(action)||action.some(v=>Math.abs(v)>1))throw new Error('Expected bounded 7D action');
    const preState=this.preState(),m=this.model,d=this.data,mj=this.mj;
    const currentPos=vec3(d.site_xpos,8*3);
    for(let i=0;i<3;i++)this.goal[i]=clamp(this.goal[i]+clamp(action[i]*.05,-.009,.009),currentPos[i]-(i===2?.04:.03),currentPos[i]+(i===2?.04:.03));
    this.goalRotation=matmul(axisRotation(action.slice(3,6).map(v=>v*.5)),this.goalRotation);
    this.grip=action[6]===0?this.grip:Math.sign(action[6]);
    const substeps=Math.max(1,Math.round(.05/this.dt));
    for(let k=0;k<substeps;k++){
      mj.mj_forward(m,d);mj.mj_jacSite(m,d,this.jacp,this.jacr,8);
      const jp=this.jacp.GetView() as Float64Array,jr=this.jacr.GetView() as Float64Array;
      const jac=[0,1,2,3,4,5].map(r=>Array.from({length:7},(_,j)=>(r<3?jp[(r)*m.nv+j]:jr[(r-3)*m.nv+j])));
      const now=vec3(d.site_xpos,8*3),currentR=Array.from(d.site_xmat.slice(8*9,8*9+9)) as number[];
      const orient=rotationError(currentR,this.goalRotation);
      const error=[...this.goal.map((v,i)=>clamp(v-now[i],-.03,.03)),...orient.map(v=>clamp(v,-.15,.15))];
      const gram=jac.map((row,r)=>jac.map((other,c)=>dot(row,other)+(r===c?.0025:0)));
      const effort=solveLinear(gram,error);
      for(let j=0;j<7;j++){
        let dq=0;for(let r=0;r<6;r++)dq+=jac[r][j]*effort[r];
        const torque=d.qfrc_bias[j]+130*dq-16*d.qvel[j];
        d.ctrl[j]=clamp(torque,m.actuator_ctrlrange[j*2],m.actuator_ctrlrange[j*2+1]);
      }
      // Native XML has opposing position actuators: open at (+.04, -.04), closed at (0,0).
      d.ctrl[7]=this.grip>0?0:.04;d.ctrl[8]=this.grip>0?0:-.04;
      mj.mj_step(m,d);
    }
    this.stepCount++;
    const can=vec3(d.xpos,19*3),basket=vec3(d.xpos,20*3);
    const inside=Math.hypot(can[0]-basket[0],can[1]-basket[1])<.065&&can[2]<basket[2]+.12&&this.grip<0;
    this.settle=inside?this.settle+1:0;this.won=this.settle>=10;
    return {preState,action:[...action] as LocalAction,frame:this.frame,successAfter:this.won};
  }
  dispose(){this.jacp.delete();this.jacr.delete();this.data.delete();this.model.delete();}
}
export const createLocalLiberoEngine=(scene:NativeScenePayload&{scene_sha256?:string})=>LocalLiberoEngine.create(scene);
