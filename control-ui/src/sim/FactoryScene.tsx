import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { SceneCommand, SimSnapshot } from '../protocol/types';
import { NativeMaterials, nativeGeometry } from './NativeMaterials';
import { calculateAnalyticalIK } from './FrankaAnalyticalIK';
import { FactoryBackdrop, WORKBENCH_HEIGHT, type BackgroundState, type SplatQuality } from './FactoryBackdrop';
import { ScaleReference } from './ScaleReference';

type Props = { command: SceneCommand | null; onUpdate: (snapshot: SimSnapshot) => void; view: 'orbit'|'top'|'robot'; ghost: boolean; cameraRevision?: number; onOperatorCommand?: (command: Omit<SceneCommand,'id'>) => void };
type Status = SimSnapshot['status'];
// MuJoCo's generated embind API is intentionally contained in this adapter.
type Mj = any;
const HOME = [0, -.6, 0, -2.1, 0, 1.65, .78];
const v3 = (a: ArrayLike<number>, i=0): [number,number,number] => [a[i],a[i+1],a[i+2]];
const DOWN = new THREE.Quaternion(1,0,0,0);

function nameOf(model: Mj, address: number) {
  let s=''; for(let i=address; i<model.names.length && model.names[i];i++) s+=String.fromCharCode(model.names[i]); return s;
}
function findId(model: Mj, kind: string, name: string) {
  const adr=model[`${kind}_nameadr`] ?? model[`name_${kind}adr`];
  if (!adr) return -1;
  for(let i=0;i<adr.length;i++) if(nameOf(model,adr[i])===name)return i;
  return -1;
}

class FactorySim {
  mj: Mj; model: Mj; data: Mj;
  scene = new THREE.Scene(); camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: OrbitControls;
  meshes: {id:number;mesh:THREE.Mesh}[]=[]; frame=0; alive=true; status:Status='loading'; stage='Standby'; message='Loading physical simulation';
  feed=true; feedTime=0; phase=0; phaseTime=0; target=new THREE.Vector3(.48,.13,.33); grip=255; retries=0; success=false; manualMoves=0; failure=false;
  canId=-1; tcpId=-1; canQ=9; lastEmit=0; prev=0; lastAction=[0,0,0]; handoff=new THREE.Vector3(.48,.13,.33);
  overlayScene=new THREE.Scene();operatorGizmo=new THREE.Group(); operatorHit:THREE.Mesh; operatorLeash:THREE.Line; raycaster=new THREE.Raycaster();
  onOperatorCommand?:Props['onOperatorCommand']; pointerDrag:null|{id:number;start:THREE.Vector3;anchor:THREE.Vector3;plane:THREE.Plane}=null;
  canSkinMaterials:NativeMaterials|null=null;manualStart=new THREE.Vector3(); ghostMesh:THREE.Group; stripes:THREE.Mesh[]=[]; onUpdate:(s:SimSnapshot)=>void;
  backdrop:FactoryBackdrop; fallbackFloor=new THREE.Group();
  scaleReference=new ScaleReference();
  constructor(el:HTMLElement,onUpdate:(s:SimSnapshot)=>void,onBackground:(s:BackgroundState)=>void) {
    this.onUpdate=onUpdate;
    this.scene.background=new THREE.Color('#cad1d3');
    this.camera=new THREE.PerspectiveCamera(49,1,.02,100);this.camera.up.set(0,0,1);this.camera.position.set(1.45,-1.9,1.25);
    this.renderer=new THREE.WebGLRenderer({antialias:false,alpha:false});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=THREE.PCFSoftShadowMap;this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.0;
    el.appendChild(this.renderer.domElement);this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.target.set(.3,0,.27);this.controls.enableDamping=true;this.controls.minDistance=.45;this.controls.maxDistance=4.5;this.controls.maxPolarAngle=Math.PI*.48;
    const ambient=new THREE.HemisphereLight('#eef5fa','#67716b',.8);ambient.position.set(0,0,5);this.scene.add(ambient);
    const key=new THREE.DirectionalLight('#fff4e5',2.1);key.position.set(1,-1,4);key.castShadow=true;key.shadow.mapSize.set(2048,2048);key.shadow.camera.left=-2;key.shadow.camera.right=2;key.shadow.camera.top=2;key.shadow.camera.bottom=-2;key.shadow.bias=-.0003;this.scene.add(key);
    const fill=new THREE.DirectionalLight('#c9e1ec',.7);fill.position.set(-2,1,3);this.scene.add(fill);
    this.ghostMesh=new THREE.Group();const mat=new THREE.MeshBasicMaterial({color:'#72e8cb',wireframe:true,transparent:true,opacity:.7});
    this.ghostMesh.add(new THREE.Mesh(new THREE.SphereGeometry(.045,16,12),mat));
    const ring=new THREE.Mesh(new THREE.TorusGeometry(.065,.002,6,64),new THREE.MeshBasicMaterial({color:'#72e8cb'}));this.ghostMesh.add(ring);this.ghostMesh.visible=false;this.scene.add(this.ghostMesh);
    // Newly implemented operator handle. The cyan ring is independent of the green recovery guide.
    const cyan=new THREE.MeshBasicMaterial({color:'#82e7ff',transparent:true,opacity:1,depthTest:false,side:THREE.DoubleSide,toneMapped:false});
    const circle=new THREE.Mesh(new THREE.RingGeometry(.052,.060,64),cyan);circle.renderOrder=20;this.operatorGizmo.add(circle);const outline=new THREE.Mesh(new THREE.RingGeometry(.047,.066,64),new THREE.MeshBasicMaterial({color:'#07131b',transparent:true,opacity:.9,depthTest:false,side:THREE.DoubleSide,toneMapped:false}));outline.renderOrder=17;this.operatorGizmo.add(outline);
    const dot=new THREE.Mesh(new THREE.CircleGeometry(.008,20),cyan);dot.renderOrder=21;this.operatorGizmo.add(dot);
    this.operatorHit=new THREE.Mesh(new THREE.SphereGeometry(.069,12,8),new THREE.MeshBasicMaterial({color:'#75dfff',transparent:true,opacity:.035,depthTest:false}));this.operatorHit.renderOrder=19;this.operatorGizmo.add(this.operatorHit);
    const tick=new THREE.Mesh(new THREE.BoxGeometry(.018,.003,.002),cyan);tick.position.x=.061;tick.renderOrder=21;this.operatorGizmo.add(tick);this.overlayScene.add(this.operatorGizmo);
    this.operatorLeash=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]),new THREE.LineBasicMaterial({color:'#75dfff',transparent:true,opacity:.6,depthTest:false}));this.operatorLeash.renderOrder=18;this.overlayScene.add(this.operatorLeash);
    const canvas=this.renderer.domElement;canvas.addEventListener('pointerdown',this.beginPointer,true);canvas.addEventListener('pointermove',this.movePointer,true);canvas.addEventListener('pointerup',this.endPointer,true);canvas.addEventListener('pointercancel',this.endPointer,true);canvas.addEventListener('lostpointercapture',this.endPointer,true);canvas.addEventListener('wheel',this.wheelPointer,{capture:true,passive:false});
    this.addFactoryDecor();this.resize(el);
    this.scene.add(this.scaleReference);
    this.backdrop=new FactoryBackdrop(this.scene,this.renderer,state=>{if(state.status==='ready')this.fallbackFloor.visible=false;onBackground(state);});
    void this.backdrop.load();
  }
  pointerRay(event:PointerEvent|WheelEvent){const rect=this.renderer.domElement.getBoundingClientRect();this.raycaster.setFromCamera(new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1),this.camera);return this.raycaster;}
  handleHit(event:PointerEvent|WheelEvent){this.operatorGizmo.updateMatrixWorld(true);return this.pointerRay(event).intersectObject(this.operatorHit,false).length>0;}
  sendPointerTarget(desired:THREE.Vector3){
    if(this.status!=='manual'||!this.onOperatorCommand)return;
    const actual=new THREE.Vector3(...this.eef());
    const lead=new THREE.Vector2(desired.x-actual.x,desired.y-actual.y);if(lead.length()>.11)lead.setLength(.11);
    desired.set(THREE.MathUtils.clamp(actual.x+lead.x,.15,.7),THREE.MathUtils.clamp(actual.y+lead.y,-.55,.6),THREE.MathUtils.clamp(desired.z,Math.max(.15,actual.z-.10),Math.min(.65,actual.z+.10)));
    const delta=desired.sub(this.target);if(delta.length()>.0002)this.onOperatorCommand({type:'move',delta:delta.toArray() as [number,number,number]});
  }
  beginPointer=(event:PointerEvent)=>{
    if(event.button!==0||this.status!=='manual'||!this.handleHit(event))return;
    const plane=new THREE.Plane(new THREE.Vector3(0,0,1),-this.target.z);const anchor=this.pointerRay(event).ray.intersectPlane(plane,new THREE.Vector3());if(!anchor)return;
    event.preventDefault();event.stopImmediatePropagation();this.controls.enabled=false;this.pointerDrag={id:event.pointerId,start:this.target.clone(),anchor,plane};this.renderer.domElement.setPointerCapture(event.pointerId);this.renderer.domElement.style.cursor='grabbing';
  };
  movePointer=(event:PointerEvent)=>{
    const drag=this.pointerDrag;if(!drag){this.renderer.domElement.style.cursor=this.status==='manual'&&this.handleHit(event)?'grab':'';return;}
    if(event.pointerId!==drag.id)return;event.preventDefault();event.stopImmediatePropagation();
    const point=this.pointerRay(event).ray.intersectPlane(drag.plane,new THREE.Vector3());if(point)this.sendPointerTarget(drag.start.clone().add(point.sub(drag.anchor)));
  };
  endPointer=(event:PointerEvent)=>{
    if(!this.pointerDrag||event.pointerId!==this.pointerDrag.id)return;this.pointerDrag=null;this.controls.enabled=true;this.renderer.domElement.style.cursor='';if(this.renderer.domElement.hasPointerCapture(event.pointerId))this.renderer.domElement.releasePointerCapture(event.pointerId);
  };
  wheelPointer=(event:WheelEvent)=>{
    if(this.status!=='manual'||event.shiftKey||(!this.pointerDrag&&!this.handleHit(event)))return;
    event.preventDefault();event.stopImmediatePropagation();const goal=this.target.clone();goal.z-=THREE.MathUtils.clamp(event.deltaY*.00015,-.025,.025);this.sendPointerTarget(goal);
  };
  resize(el:HTMLElement){this.renderer.setSize(el.clientWidth,el.clientHeight);this.camera.aspect=el.clientWidth/el.clientHeight;this.camera.updateProjectionMatrix();}
  addFactoryDecor(){
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(30,30),new THREE.MeshStandardMaterial({color:'#929b9d',roughness:.85}));floor.receiveShadow=true;floor.position.z=-WORKBENCH_HEIGHT-.006;this.fallbackFloor.add(floor);this.scene.add(this.fallbackFloor);
    const contactShadow=new THREE.Mesh(new THREE.PlaneGeometry(6,6),new THREE.ShadowMaterial({opacity:.24,depthWrite:false}));contactShadow.position.z=-WORKBENCH_HEIGHT+.008;contactShadow.receiveShadow=true;this.scene.add(contactShadow);
    const box=(s:number[],p:number[],c:string)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(...s as [number,number,number]),new THREE.MeshStandardMaterial({color:c,metalness:.45,roughness:.4}));m.position.set(...p as [number,number,number]);m.castShadow=true;m.receiveShadow=true;this.scene.add(m);return m;};
    // 1.65 x 1.95 m workbench; visual factory floor is z=-.78, physics/tabletop stays z=0.
    box([1.65,1.95,.06],[.24,.06,-.032],'#657680');
    for(const x of [-.48,.96])for(const y of [-.79,.91]){
      box([.065,.065,WORKBENCH_HEIGHT-.08],[x,y,-WORKBENCH_HEIGHT/2-.02],'#33444b');
      box([.11,.11,.035],[x,y,-WORKBENCH_HEIGHT+.018],'#26363b');
    }
    box([1.48,.045,.05],[.24,-.79,-.53],'#44565e');
    box([.5,.008,.003],[.25,-.89,.002],'#f5d367');
    for(let i=0;i<=5;i++)box([.003,i===0||i===5?.045:.023,.003],[i*.1,-.89,.003],'#f5d367');
    box([.24,.24,.055],[0,0,.018],'#273b47');
    for(const x of [.25,.71])for(const y of [-.5,.75])box([.035,.035,.20],[x,y,-.08],'#293d49');
    for(let i=0;i<30;i++){const s=box([.405,.006,.002],[.48,-.55+i*.046,.092],'#263a45');this.stripes.push(s);}
    for(const y of [-.61,.85]){const r=new THREE.Mesh(new THREE.CylinderGeometry(.045,.045,.46,24),new THREE.MeshStandardMaterial({color:'#8697a0',metalness:.8,roughness:.3}));r.rotateZ(Math.PI/2);r.position.set(.48,y,.045);this.scene.add(r);}
    box([.018,1.5,.008],[.74,.12,.045],'#ef873d');box([.018,1.5,.008],[.22,.12,.045],'#ef873d');
    for(const x of [-.18,.88])box([.025,.025,1.2],[x,.87,.6],'#334955');box([1.1,.035,.035],[.35,.87,1.18],'#334955');
    const lamp=box([.8,.025,.012],[.35,.85,1.155],'#caebf7');(lamp.material as THREE.MeshStandardMaterial).emissive.set('#c3e5ff');(lamp.material as THREE.MeshStandardMaterial).emissiveIntensity=2;
    // Floor demarcation and a service cabinet give the workcell its industrial context.
    for(const x of [-.28,.95])box([.012,1.8,.002],[x,.05,.004],'#b88835');
    box([.22,.22,.55],[-.4,.45,.275],'#253a48');box([.17,.008,.15],[-.4,.335,.4],'#111f2b');
  }
  async init(){
    const {default:load}=await import('mujoco-js');this.mj=await load();if(!this.alive)return;
    const base='/models/panda/';const xml=await (await fetch(base+'factory.xml')).text();const manifest:string[]=await(await fetch(base+'manifest.json')).json();
    if(!this.alive)return;try{this.mj.FS.mkdir('/working');}catch{}try{this.mj.FS.mkdir('/working/assets');}catch{}
    await Promise.all(manifest.map(async f=>this.mj.FS.writeFile('/working/assets/'+f,new Uint8Array(await(await fetch(base+'assets/'+f)).arrayBuffer()))));
    if(!this.alive)return;this.mj.FS.writeFile('/working/factory.xml',xml);
    this.model=this.mj.MjModel.loadFromXML('/working/factory.xml');this.data=new this.mj.MjData(this.model);
    this.canId=findId(this.model,'body','can');this.tcpId=findId(this.model,'site','tcp');
    if(this.canId<0||this.tcpId<0)throw Error('Physical robot landmarks unavailable');
    const canJoint=this.model.body_jntadr[this.canId];this.canQ=this.model.jnt_qposadr[canJoint];
    this.buildGeoms();this.reset();this.loop(0);void this.loadCanSkin().catch(e=>console.warn("LIBERO can visual asset unavailable",e));
  }
  buildGeoms(){
    const m=this.model;
    for(let i=0;i<m.ngeom;i++){
      if(m.geom_group[i]===3 || nameOf(m,m.name_geomadr[i])==='floor')continue;
      const type=Number(m.geom_type[i]), sz=v3(m.geom_size,i*3);let g:THREE.BufferGeometry;
      if(type===7){const id=m.geom_dataid[i];const va=m.mesh_vertadr[id],vn=m.mesh_vertnum[id],fa=m.mesh_faceadr[id],fn=m.mesh_facenum[id];
        g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(Array.from(m.mesh_vert.slice(va*3,(va+vn)*3)),3));
        g.setIndex(Array.from(m.mesh_face.slice(fa*3,(fa+fn)*3)));g.computeVertexNormals();
      }else if(type===6)g=new THREE.BoxGeometry(sz[0]*2,sz[1]*2,sz[2]*2);
      else if(type===5){g=new THREE.CylinderGeometry(sz[0],sz[0],sz[1]*2,32);g.rotateX(Math.PI/2);}
      else if(type===3){g=new THREE.CapsuleGeometry(sz[0],sz[1]*2,8,16);g.rotateX(Math.PI/2);}
      else if(type===2)g=new THREE.SphereGeometry(sz[0],20,16);else continue;
      const mid=m.geom_matid[i];const rgba=mid>=0?v3(m.mat_rgba,mid*4):v3(m.geom_rgba,i*4);
      const mesh=new THREE.Mesh(g,new THREE.MeshStandardMaterial({color:new THREE.Color(...rgba),metalness:type===7?.26:.35,roughness:type===7?.32:.44}));
      mesh.castShadow=true;mesh.receiveShadow=true;this.scene.add(mesh);this.meshes.push({id:i,mesh});
    }
  }
  async loadCanSkin(){
    const files=['alphabet_soup.xml','texture_map.png','visual/textured_vis.msh'];const root='/libero_can_visual';
    try{this.mj.FS.mkdir(root);}catch{}try{this.mj.FS.mkdir(root+'/visual');}catch{}
    await Promise.all(files.map(async file=>{const response=await fetch('/models/libero-alphabet-soup/'+file);if(!response.ok)throw Error('Missing LIBERO soup asset');const bytes=new Uint8Array(await response.arrayBuffer());if(this.alive)this.mj.FS.writeFile(root+'/'+file,bytes);}));
    if(!this.alive)return;const model=this.mj.MjModel.loadFromXML(root+'/alphabet_soup.xml');const geometry=nativeGeometry(model,0);if(!geometry){model.delete();return;}
    this.canSkinMaterials=new NativeMaterials(model);const material=this.canSkinMaterials.material(0);geometry.computeBoundingBox();const size=geometry.boundingBox!.getSize(new THREE.Vector3()),center=geometry.boundingBox!.getCenter(new THREE.Vector3());geometry.translate(-center.x,-center.y,-center.z);geometry.scale(.054/size.x,.054/size.y,.11/size.z);
    // Offline skin only: collision dimensions and all task dynamics remain unchanged.
    for(const {id,mesh} of this.meshes){const name=nameOf(this.model,this.model.name_geomadr[id]);if(name==='can_body'){mesh.geometry.dispose();(mesh.material as THREE.Material).dispose();mesh.geometry=geometry;mesh.material=material;}else if(name==='can_top'||name==='can_bottom')mesh.visible=false;}
    model.delete();
  }
  reset(){
    this.mj.mj_resetData(this.model,this.data);for(let i=0;i<7;i++){this.data.qpos[i]=HOME[i];this.data.ctrl[i]=HOME[i];}this.data.qpos[7]=.04;this.data.qpos[8]=.04;this.data.ctrl[7]=255;
    this.data.qpos[this.canQ]=.48;this.data.qpos[this.canQ+1]=.38;this.data.qpos[this.canQ+2]=.15;this.data.qpos[this.canQ+3]=1;
    this.mj.mj_forward(this.model,this.data);this.status='ready';this.stage='Awaiting run';this.message='Franka Panda ready · scripted demo controller';this.feed=true;this.feedTime=0;this.phase=0;this.phaseTime=0;this.retries=0;this.success=false;this.failure=false;this.grip=255;this.manualMoves=0;this.target.set(.48,.13,.33);this.handoff.copy(this.target);
  }
  eef(){return v3(this.data.site_xpos,this.tcpId*3)}
  can(){return v3(this.data.xpos,this.canId*3)}
  command(c:SceneCommand){
    if(!this.data)return;
    switch(c.type){
      case 'reset':this.reset();break;
      case 'start':if(this.status!=='ready'&&this.status!=='paused'){this.message='Reset the episode or complete the operator handoff before starting.';break;}this.status='running';this.stage='Approach target';this.message='Autonomous run · scripted IK demo controller';break;
      case 'pause':this.status='paused';this.message='Simulation paused';break;
      case 'stop':this.status='stopped';this.message='Emergency stop · actuator targets held';break;
      case 'inject_failure':{
        if(this.status!=='running'&&this.status!=='recovering')break;this.data.qpos[this.canQ+1]+=.13;this.mj.mj_forward(this.model,this.data);this.feed=false;this.failure=true;this.status='attention';this.stage='Grasp alignment lost';this.retries++;this.message='Can displaced · approach target no longer valid. Human assistance required.';const p=this.can();this.handoff.set(p[0],p[1],.33);this.target.set(...this.eef());for(let i=0;i<7;i++)this.data.ctrl[i]=this.data.qpos[i];break;}
      case 'take_control':if(['stopped','completed','error','loading'].includes(this.status))break;this.status='manual';this.stage='Human remote recovery';this.target.set(...this.eef());this.manualStart.set(...this.eef());this.manualMoves=0;this.message='Drag the cyan ring to move X/Y, scroll over it for height, then hand off near the green guide.';break;
      case 'move':if(this.status==='manual'&&c.delta){this.target.add(new THREE.Vector3(...c.delta));this.target.x=THREE.MathUtils.clamp(this.target.x,.15,.7);this.target.y=THREE.MathUtils.clamp(this.target.y,-.55,.6);this.target.z=THREE.MathUtils.clamp(this.target.z,.15,.65);this.lastAction=c.delta;this.manualMoves++;}break;
      case 'gripper':if(this.status==='manual'){this.grip=THREE.MathUtils.clamp(c.gripper??1,0,1)*255;this.manualMoves++;}break;
      case 'resume_policy':
        if(this.status==='paused'){this.status='running';break;}
        if(this.status==='manual'&&this.manualMoves>0){const d=new THREE.Vector3(...this.eef()).distanceTo(this.handoff);if(d>.08||new THREE.Vector3(...this.eef()).distanceTo(this.manualStart)<.008){this.message='Continue jogging closer to the green target before handoff.';break;}this.feed=false;this.phase=0;this.phaseTime=0;this.status='recovering';this.failure=false;this.message='Human handoff accepted · scripted task controller resumes';}else this.message='Take control and perform a recovery action before handoff.';
        break;
    }
    this.emit();
  }
  solve(){
    const t=new THREE.Matrix4().compose(this.target,DOWN,new THREE.Vector3(1,1,1));const current=Array.from(this.data.qpos.slice(0,7)) as number[];
    let best:number[]|null=null,cost=Infinity;const candidates=[current[6],.78,0,-.78,1.5,-1.5,2.3,-2.3];
    for(const q7 of candidates){for(const q of calculateAnalyticalIK(t,q7)){const n=q.reduce((s,x,i)=>s+(x-current[i])**2,0);if(n<cost&&q.every(Number.isFinite)){cost=n;best=q;}}}
    if(best)for(let i=0;i<7;i++)this.data.ctrl[i]+=THREE.MathUtils.clamp(best[i]-this.data.ctrl[i],-.012,.012);this.data.ctrl[7]=this.grip;
  }
  updateController(dt:number){
    const p=this.can();
    if(this.feed){this.feedTime+=dt;this.stage='Conveyor transport';this.message='Conveyor feeding can · velocity-controlled transport approximation';this.target.set(.48,.13,.34);this.grip=255;if(p[1]<=.134||this.feedTime>5){this.feed=false;this.phaseTime=0;this.message='Belt stopped · scripted grasp controller active';}return;}
    this.phaseTime+=dt;
    const stages=[{name:'Approach can',pos:[p[0],p[1],.34],grip:255,t:4},{name:'Align gripper',pos:[p[0],p[1],.15],grip:255,t:3},{name:'Close gripper',pos:[p[0],p[1],.15],grip:0,t:1.5},{name:'Lift can',pos:[p[0],p[1],.4],grip:0,t:3},{name:'Transfer to metal bin',pos:[.32,-.4,.4],grip:0,t:4},{name:'Lower over bin',pos:[.32,-.4,.26],grip:0,t:2},{name:'Release can',pos:[.32,-.4,.26],grip:255,t:1.5},{name:'Verify placement',pos:[.32,-.4,.42],grip:255,t:2.2}];
    const s=stages[Math.min(this.phase,stages.length-1)];this.stage=s.name;this.grip=s.grip;
    // Hold the grasp XY after closing rather than chasing a moving carried object.
    if(this.phase<=2||this.phase>=4)this.target.set(...s.pos as [number,number,number]);else this.target.z=.4;
    if(this.phaseTime>s.t){this.phase++;this.phaseTime=0;if(this.phase>=stages.length){const c=this.can();this.success=Math.abs(c[0]-.32)<.12&&Math.abs(c[1]+.4)<.1&&c[2]>.07&&c[2]<.22&&Math.hypot(...Array.from(this.data.qvel.slice(this.model.jnt_dofadr[this.model.body_jntadr[this.canId]],this.model.jnt_dofadr[this.model.body_jntadr[this.canId]]+3)) as number[])<.12;
      this.status=this.success?'completed':'attention';this.stage=this.success?'Task completed':'Placement needs assistance';this.message=this.success?'Verified: can settled inside metal bin.':'Physical placement not complete. Take control to recover.';this.handoff.set(c[0],c[1],.33);}}
  }
  // Approximate an actuated conveyor with a free-body velocity servo.
  // Rendering always reads the resulting MuJoCo body pose, never a visual-only offset.
  driveConveyor(){const dof=this.model.jnt_dofadr[this.model.body_jntadr[this.canId]];const y=this.can()[1];this.data.qvel[dof+1]=Math.max(-.09,Math.min(0,(.13-y)*5));const sign=this.data.qpos[this.canQ+3]>=0?1:-1;for(let j=0;j<3;j++)this.data.qvel[dof+3+j]=-50*sign*this.data.qpos[this.canQ+4+j];}
  emit(){if(!this.data)return;const can=this.can(),eef=this.eef();this.onUpdate({status:this.status,simTime:this.data.time,joints:Array.from(this.data.qpos.slice(0,7)),eef,gripper:this.data.qpos[7]/.04,canPosition:can,targetPosition:this.handoff.toArray() as [number,number,number],retries:this.retries,taskProgress:this.success?1:Math.min(this.phase/8,.95),beltRunning:this.status==='running'&&this.feed,message:this.message,physicsReady:true,lastAction:Array.from(this.data.ctrl),success:this.success,stage:this.stage,contactCount:this.data.ncon});}
  loop=(time:number)=>{if(!this.alive)return;const dt=Math.min((time-this.prev)/1000||.016,.04);this.prev=time;
    if(['running','recovering','manual','attention'].includes(this.status)){
      if(this.status==='running'||this.status==='recovering')this.updateController(dt);
      if(this.status!=='attention')this.solve();const end=this.data.time+dt;while(this.data.time<end){if(this.feed&&this.status==='running')this.driveConveyor();this.mj.mj_step(this.model,this.data);}
    }
    for(const {id,mesh}of this.meshes){mesh.position.set(...v3(this.data.geom_xpos,id*3));const a=this.data.geom_xmat,o=id*9;mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(a[o],a[o+1],a[o+2],0,a[o+3],a[o+4],a[o+5],0,a[o+6],a[o+7],a[o+8],0,0,0,0,1));}
    if(this.status==='running'&&this.feed)for(const stripe of this.stripes){stripe.position.y-=dt*.09;if(stripe.position.y<-.55)stripe.position.y=.83;}
    this.operatorGizmo.position.set(...(this.status==='manual'?this.target.toArray() as [number,number,number]:this.eef()));
    this.operatorGizmo.visible=this.status!=='loading';this.operatorGizmo.scale.setScalar(this.status==='manual'?1.15:1);
    const leash=this.operatorLeash.geometry.getAttribute('position') as THREE.BufferAttribute;const tcp=this.eef();leash.setXYZ(0,...tcp);leash.setXYZ(1,this.target.x,this.target.y,this.target.z);leash.needsUpdate=true;this.operatorLeash.visible=this.status==='manual';
    if(this.pointerDrag&&this.status!=='manual'){const id=this.pointerDrag.id;this.pointerDrag=null;this.controls.enabled=true;if(this.renderer.domElement.hasPointerCapture(id))this.renderer.domElement.releasePointerCapture(id);}
    this.ghostMesh.position.copy(this.handoff);this.ghostMesh.rotation.z=time*.0005;this.controls.update();this.renderer.render(this.scene,this.camera);const autoClear=this.renderer.autoClear;this.renderer.autoClear=false;this.renderer.clearDepth();this.renderer.render(this.overlayScene,this.camera);this.renderer.autoClear=autoClear;if(time-this.lastEmit>110){this.emit();this.lastEmit=time;}this.frame=requestAnimationFrame(this.loop);
  }
  setView(view:Props['view']){if(view==='top'){this.camera.position.set(.38,.05,2.8);this.controls.target.set(.38,.05,0);}else if(view==='robot'){this.camera.position.set(.08,-.12,.7);this.controls.target.set(.48,.15,.15);}else{this.camera.position.set(1.45,-1.9,1.25);this.controls.target.set(.3,0,.27);}this.controls.update();}
  dispose(){const canvas=this.renderer.domElement;canvas.removeEventListener('pointerdown',this.beginPointer,true);canvas.removeEventListener('pointermove',this.movePointer,true);canvas.removeEventListener('pointerup',this.endPointer,true);canvas.removeEventListener('pointercancel',this.endPointer,true);canvas.removeEventListener('lostpointercapture',this.endPointer,true);canvas.removeEventListener('wheel',this.wheelPointer,true);this.canSkinMaterials?.dispose();this.operatorLeash.geometry.dispose();(this.operatorLeash.material as THREE.Material).dispose();this.alive=false;cancelAnimationFrame(this.frame);this.backdrop.dispose();this.scaleReference.disposeLabel();this.controls.dispose();this.scene.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();if(Array.isArray(o.material))o.material.forEach(m=>m.dispose());else o.material.dispose();}});this.overlayScene.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();if(Array.isArray(o.material))o.material.forEach(m=>m.dispose());else o.material.dispose();}});this.renderer.dispose();this.renderer.domElement.remove();try{this.data?.delete();this.model?.delete();}catch{}}
}

export default function FactoryScene({command,onUpdate,view,ghost,onOperatorCommand,cameraRevision}:Props){
  const operatorRef=useRef(onOperatorCommand);operatorRef.current=onOperatorCommand;const lastCommand=useRef(-1);const host=useRef<HTMLDivElement>(null);const sim=useRef<FactorySim|null>(null);const onUpdateRef=useRef(onUpdate);onUpdateRef.current=onUpdate;const[error,setError]=useState('');
  const[background,setBackground]=useState<BackgroundState>({status:'loading',message:'Loading factory background',quality:'500k'});
  const[roomScale,setRoomScale]=useState(1);
  const[showScale,setShowScale]=useState(false);
  useEffect(()=>{if(!host.current)return;const s=new FactorySim(host.current,v=>onUpdateRef.current(v),setBackground);sim.current=s;s.onOperatorCommand=c=>operatorRef.current?.(c);const observer=new ResizeObserver(()=>host.current&&s.resize(host.current));observer.observe(host.current);void s.init().catch(e=>{if(s.alive){setError(String(e));onUpdateRef.current({status:'error',simTime:0,joints:HOME,eef:[0,0,0],gripper:1,canPosition:[.48,.13,.15],targetPosition:[.48,.13,.33],retries:0,taskProgress:0,beltRunning:false,message:String(e),physicsReady:false,lastAction:[],success:false});}});return()=>{observer.disconnect();s.dispose();sim.current=null;};},[]);
  useEffect(()=>{if(command&&command.id!==lastCommand.current){lastCommand.current=command.id;sim.current?.command(command);}},[command]);useEffect(()=>sim.current?.setView(view),[view,cameraRevision]);useEffect(()=>{if(sim.current)sim.current.ghostMesh.visible=ghost;},[ghost]);
  return <div ref={host} style={{width:'100%',height:'100%',position:'relative',minHeight:300}}>{error&&<div style={{position:'absolute',inset:20,color:'#ffaf8a',fontSize:12,zIndex:3}}>Simulation error: {error}</div>}<details className={`factory-background-badge ${background.status}`}><summary>{background.message}</summary>{background.status==='ready'&&<div className="factory-background-settings"><label>Background detail<select aria-label="Factory background quality" value={background.quality} onChange={e=>{void sim.current?.backdrop.selectQuality(e.target.value as SplatQuality).catch(()=>setBackground({...background,message:'Could not switch background quality'}));}}><option value="full">Original · full resolution</option><option value="500k">Balanced</option><option value="100k">Performance</option></select></label><label>Room scale <span>×{roomScale.toFixed(2)}</span><input aria-label="Factory room scale" type="range" min="0.5" max="1.5" step="0.05" value={roomScale} onChange={e=>{const scale=Number(e.target.value);setRoomScale(scale);sim.current?.backdrop.setRoomScale(scale);}}/></label><p>Workbench 0.78 m · belt 1.40 m<br/>Yellow table gauge 0.50 m</p><p>{background.scaleDescription}</p><label>1 m reference cube<input aria-label="Show one meter scale reference" type="checkbox" checked={showScale} onChange={e=>{setShowScale(e.target.checked);if(sim.current)sim.current.scaleReference.visible=e.target.checked;}}/></label><button type="button" onClick={()=>{setRoomScale(1);sim.current?.backdrop.setRoomScale(1);}}>Reset scale</button></div>}</details></div>;
}
