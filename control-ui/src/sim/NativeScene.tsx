import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { nativeDisplayXml } from './nativeDisplayXml';
import { NativeMaterials, nativeGeometry, visibleNativeGeoms } from './NativeMaterials';
import type { SceneCommand } from '../protocol/types';
import { FactoryBackdrop, WORKBENCH_HEIGHT, type BackgroundState, type SplatQuality } from './FactoryBackdrop';
import { ScaleReference } from './ScaleReference';
import { getRenderPerformance, subscribeRenderPerformance, publishRenderStats, publishRenderBenchmark, type RenderBenchmarkRequest, type RenderProfile, type RenderStats } from './renderPerformance';
export type { RenderProfile, RenderStats } from './renderPerformance';
import { isNativeRoomSkin, nativeWorkcellLayout, createNativeWorkcellSkin } from './NativeWorkcellSkin';

export type NativeScenePayload = {
  xml: string;
  assets: { path: string; data: string; mime?: string }[];
  nq?: number; nv?: number;
  body_names?: string[]; geom_names?: string[]; site_names?: string[];
  cameras?: string[]; camera_observations?: string[];
  coordinate_frame?: string;
};
export type NativeFrame = {
  qpos: number[]; qvel: number[]; sim_time: number;
  eef?: [number,number,number]; eef_orientation?: number[];
  joints?: number[]; gripper?: number[]; state?: number[];
  body_poses?: { names: string[]; xpos: number[][]; xquat: number[][] };
  [key: string]: unknown;
};
const PROFILE = { performance:{quality:'100k',dpr:1,shadow:0}, balanced:{quality:'500k',dpr:1.25,shadow:512}, full:{quality:'full',dpr:2,shadow:2048} } as const;
type Props = {
  scene: NativeScenePayload | null;
  frame: NativeFrame | null;
  manual: boolean;
  renderProfile?: RenderProfile; onRenderStats?: (stats:RenderStats)=>void;
  onOperatorCommand?: (command: Omit<SceneCommand,'id'>) => void;
  mouseStyle?: 'wemine';
  onReady?: (ready: boolean, error?: string) => void;
  view: 'orbit'|'top'|'robot'; ghost: boolean; cameraRevision?: number;
  referenceFrame?: NativeFrame | null; trajectory?: [number,number,number][];
};
type Mj = any;
const xyz = (a:ArrayLike<number>, offset=0):[number,number,number] => [a[offset],a[offset+1],a[offset+2]];
function nameAt(model:Mj, address:number){let s='';for(let i=address;i<model.names.length&&model.names[i];i++)s+=String.fromCharCode(model.names[i]);return s;}
// Ignore sub-pixel floating-point drift from OrbitControls in a settled camera.
function sameViewMatrix(a:THREE.Matrix4,b:THREE.Matrix4){return a.elements.every((v,i)=>Math.abs(v-b.elements[i])<=1e-10*Math.max(1,Math.abs(v),Math.abs(b.elements[i])));}

/** Native server remains the sole owner of physics, action execution and success.
 * This viewer only copies received state and calls mj_forward for mesh transforms.
 * It never calls mj_step, executes a task controller or creates replacement state.
 */
class NativeRenderer {
  compileDiagnostics:string[]=[];mj:Mj;model:Mj;data:Mj;referenceData:Mj;referenceFrame:NativeFrame|null=null;showGhost=false;materials:NativeMaterials|null=null;alive=true;raf=0;hasFrame=false;manual=false;view:Props['view']='orbit';pending:NativeFrame|null=null;
  renderer:THREE.WebGLRenderer;scene=new THREE.Scene();camera:THREE.PerspectiveCamera;controls:OrbitControls;
  ghostMeshes:{id:number,mesh:THREE.Mesh}[]=[];historicalGhost=new THREE.Group();trail=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:'#75dfff',transparent:true,opacity:.6,depthWrite:false}));
  stationBounds:THREE.Box3|null=null;displayOffset=WORKBENCH_HEIGHT;nativeWorkcell=new THREE.Group();overlayWorkcell=new THREE.Group();meshes:{id:number,mesh:THREE.Mesh}[]=[];context=new THREE.Group();overlayScene=new THREE.Scene();handle=new THREE.Group();hit:THREE.Mesh;leash:THREE.Line;guide:THREE.Mesh;
  eef=new THREE.Vector3();target=new THREE.Vector3();guidePosition:THREE.Vector3|null=null;center=new THREE.Vector3(0,0,.9);framed=false;ray=new THREE.Raycaster();
  drag:null|{id:number,anchor:THREE.Vector3,start:THREE.Vector3,plane:THREE.Plane}=null;
  onOperatorCommand?:Props['onOperatorCommand'];onReady?:Props['onReady'];onRenderStats?:Props['onRenderStats'];mouseStyle?:Props['mouseStyle'];lastHover:THREE.Vector3|null=null;
  profile:RenderProfile='full';keyLight:THREE.DirectionalLight;dirty=true;settleUntil=0;
  statsSince=performance.now();rafCount=0;drawCount=0;drawMs=0;nativeCount=0;lastCalls=0;lastTriangles=0;
  backgroundDirty=true;backgroundDrawCount=0;backgroundCacheHits=0;backgroundCache:THREE.FramebufferTexture|null=null;
  backgroundCopyScene=new THREE.Scene();backgroundCopyCamera=new THREE.Camera();backgroundCameraWorld=new THREE.Matrix4();backgroundCameraProjection=new THREE.Matrix4();
  backgroundCopyMaterial=new THREE.ShaderMaterial({uniforms:{image:{value:null}},vertexShader:'varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',fragmentShader:'uniform sampler2D image;varying vec2 vUv;void main(){gl_FragColor=texture2D(image,vUv);}',depthTest:false,depthWrite:false,toneMapped:false});
  gpuExt:any=null;gpuQueries:{query:WebGLQuery;benchmarkId?:number}[]=[];gpuLastQuery=0;gpuTotalMs=0;gpuSamples=0;
  benchmark:(RenderBenchmarkRequest&{started:number;ends:number;draws:number;cpuTotal:number;gpuTotal:number;gpuSamples:number;backgroundDraws:number;cacheHits:number;nativeFrames:number})|null=null;
  invalidate=(settleMs=0)=>{this.dirty=true;this.settleUntil=Math.max(this.settleUntil,performance.now()+settleMs);};
  invalidateBackground=()=>{this.backgroundDirty=true;this.invalidate();};
  cameraChanged=()=>this.invalidate();
  backdrop:FactoryBackdrop;backgroundLayer=new THREE.Scene();backgroundVisible=true;backgroundReady=false;fallbackFloor=new THREE.Group();scaleReference=new ScaleReference(0);backgroundOptions={floorZ:0,onDirty:()=>this.invalidateBackground()};roomScale=1;
  constructor(public host:HTMLElement,onBackground:(state:BackgroundState)=>void,profile:RenderProfile='full'){
    this.scene.background=new THREE.Color('#cad1d3');
    // One display-only translation. Native qpos, poses, policy cameras and meter commands stay unchanged.
    this.nativeWorkcell.name='Native state · display Tz +0.78 m';this.nativeWorkcell.position.z=this.displayOffset;this.scene.add(this.nativeWorkcell);this.overlayWorkcell.position.z=this.displayOffset;this.overlayScene.add(this.overlayWorkcell);
    this.camera=new THREE.PerspectiveCamera(40,1,.02,100);this.camera.up.set(0,0,1);this.camera.position.set(1.6,-1.9,2.1);
    this.renderer=new THREE.WebGLRenderer({antialias:false});this.renderer.info.autoReset=false;this.renderer.setPixelRatio(Math.min(devicePixelRatio,1.25));this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=THREE.PCFShadowMap;this.renderer.shadowMap.autoUpdate=false;this.renderer.outputColorSpace=THREE.SRGBColorSpace;this.renderer.toneMapping=THREE.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.0;host.appendChild(this.renderer.domElement);
    this.gpuExt=this.renderer.getContext().getExtension('EXT_disjoint_timer_query_webgl2');
    this.backgroundCopyScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),this.backgroundCopyMaterial));
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.target.copy(this.center);this.controls.minDistance=.45;this.controls.maxDistance=4.5;this.controls.maxPolarAngle=Math.PI*.49;this.controls.addEventListener('change',this.cameraChanged);
    this.scene.add(new THREE.HemisphereLight('#eef5fa','#67716b',.8));const light=this.keyLight=new THREE.DirectionalLight('#fff4e5',2.1);light.position.set(1,-2,4);light.castShadow=true;light.shadow.mapSize.set(512,512);Object.assign(light.shadow.camera,{left:-3,right:3,top:3,bottom:-3});light.shadow.bias=-.0003;this.scene.add(light);const rim=new THREE.DirectionalLight('#c9e1ec',.7);rim.position.set(-2,2,3);this.scene.add(rim);
    const cyan=new THREE.MeshBasicMaterial({color:'#82e7ff',depthTest:false,side:THREE.DoubleSide,transparent:true,opacity:1,toneMapped:false});const ring=new THREE.Mesh(new THREE.RingGeometry(.052,.060,64),cyan);ring.renderOrder=21;this.handle.add(ring);const outline=new THREE.Mesh(new THREE.RingGeometry(.047,.066,64),new THREE.MeshBasicMaterial({color:'#07131b',transparent:true,opacity:.9,depthTest:false,side:THREE.DoubleSide,toneMapped:false}));outline.renderOrder=17;this.handle.add(outline);const dot=new THREE.Mesh(new THREE.CircleGeometry(.008,20),cyan);dot.renderOrder=22;this.handle.add(dot);
    this.hit=new THREE.Mesh(new THREE.SphereGeometry(.069,12,8),new THREE.MeshBasicMaterial({color:'#75dfff',transparent:true,opacity:.025,depthTest:false}));this.hit.renderOrder=20;this.handle.add(this.hit);this.handle.visible=false;this.overlayWorkcell.add(this.handle);
    this.leash=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]),new THREE.LineBasicMaterial({color:'#75dfff',depthTest:false,transparent:true,opacity:.6}));this.leash.renderOrder=19;this.overlayWorkcell.add(this.leash);
    this.guide=new THREE.Mesh(new THREE.TorusGeometry(.07,.002,6,64),new THREE.MeshBasicMaterial({color:'#72e8cb',transparent:true,opacity:.6}));this.guide.visible=false;this.nativeWorkcell.add(this.guide);this.scene.add(this.context);this.historicalGhost.visible=false;this.trail.visible=false;this.nativeWorkcell.add(this.historicalGhost,this.trail);
    this.context.add(this.fallbackFloor);this.scene.add(this.scaleReference);
    // Start at native floor z=0, then read the actual plane pose on the first frame.
    // The raw model and camera data stay native. Display groups and the 3D camera receive the shared +0.78 m translation only.
    this.backgroundLayer.background=this.scene.background;this.backdrop=new FactoryBackdrop(this.backgroundLayer,this.renderer,state=>{
      if(!this.alive)return;
      if(state.status==='ready')this.backgroundReady=true;this.fallbackFloor.visible=!this.backgroundVisible||!this.backgroundReady;this.invalidateBackground();
      onBackground(state);
    },this.backgroundOptions);
    this.setProfile(profile);void this.backdrop.load();
    const canvas=this.renderer.domElement;canvas.addEventListener('pointerdown',this.pointerDown,true);canvas.addEventListener('pointermove',this.pointerMove,true);canvas.addEventListener('pointerleave',this.pointerLeave,true);canvas.addEventListener('pointerup',this.pointerUp,true);canvas.addEventListener('pointercancel',this.pointerUp,true);canvas.addEventListener('lostpointercapture',this.pointerUp,true);canvas.addEventListener('wheel',this.wheel,{capture:true,passive:false});document.addEventListener('mousemove',this.lockedMouseMove,true);this.resize();this.loop();
  }
  resize(){const w=this.host.clientWidth,h=this.host.clientHeight;this.renderer.setSize(w,h);this.camera.aspect=w/Math.max(1,h);this.camera.updateProjectionMatrix();this.invalidateBackground();}
  setProfile(profile:RenderProfile){
    this.profile=profile;const config=PROFILE[profile];this.renderer.setPixelRatio(Math.min(devicePixelRatio,config.dpr));
    this.renderer.shadowMap.enabled=config.shadow>0;this.keyLight.castShadow=config.shadow>0;
    if(config.shadow&&this.keyLight.shadow.mapSize.x!==config.shadow){this.keyLight.shadow.map?.dispose();this.keyLight.shadow.map=null;this.keyLight.shadow.mapSize.set(config.shadow,config.shadow);}
    this.renderer.shadowMap.needsUpdate=true;this.invalidateBackground();
    void this.backdrop.selectQuality(config.quality).catch(error=>console.warn('Selected factory detail unavailable',error));
  }
  async load(payload:NativeScenePayload){
    const {default:load}=await import('mujoco-js');this.mj=await load({printErr:(message:string)=>this.compileDiagnostics.push(message)});if(!this.alive)return;
    const base='/native_scene';try{this.mj.FS.mkdir(base);}catch{}
    for(const asset of payload.assets){
      if(!this.alive)return;const relative=asset.path.replace(/^\/+/, '');if(relative.split('/').includes('..'))throw Error('Invalid native asset path');
      const file=base+'/'+relative;let dir='';for(const segment of file.split('/').slice(0,-1)){if(!segment)continue;dir+='/'+segment;try{this.mj.FS.mkdir(dir);}catch{}}
      const raw=atob(asset.data);const bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);this.mj.FS.writeFile(file,bytes);
    }
    this.mj.FS.writeFile(base+'/scene.xml',nativeDisplayXml(payload.xml));
    try{this.model=this.mj.MjModel.loadFromXML(base+'/scene.xml');}catch(error){let detail=String(error);try{const pointer=(error as {excPtr?:number}).excPtr;if(pointer)detail=this.mj.getExceptionMessage(pointer).join(': ');}catch{}const diagnostics=this.compileDiagnostics.slice(-3).join(' ');throw new Error('Native model could not compile: '+detail+(diagnostics?' · '+diagnostics:''));}this.data=new this.mj.MjData(this.model);
    if(payload.nq!==undefined&&payload.nq!==this.model.nq)throw Error('Native scene qpos size differs from server');
    this.buildMeshes();if(this.pending)this.apply(this.pending);this.setReference(this.referenceFrame);
  }
  buildMeshes(){
    const m=this.model;this.materials=new NativeMaterials(m);
    for(const id of visibleNativeGeoms(m)){
      if(isNativeRoomSkin(m,id))continue;
      const geometry=nativeGeometry(m,id);if(!geometry)continue;const material=this.materials.material(id);if(material.opacity<.01){geometry.dispose();material.dispose();continue;}
      const mesh=new THREE.Mesh(geometry,material);mesh.visible=false;mesh.castShadow=true;mesh.receiveShadow=true;this.meshes.push({id,mesh});this.nativeWorkcell.add(mesh);
    }
  }
  isRobotGeometry(id:number){
    // Classify the robot subtree, never task objects, furniture or room geometry.
    let body=this.model.geom_bodyid[id];
    for(let depth=0;body>0&&depth<this.model.nbody;depth++){
      const name=nameAt(this.model,this.model.name_bodyadr[body]);
      if(/(?:^|_)(?:robot[0-9]*|gripper[0-9]*|panda|link[0-9]+|hand|left_finger|right_finger)(?:_|$)/i.test(name))return true;
      body=this.model.body_parentid[body];
    }
    return false;
  }
  setReference(frame:NativeFrame|null){
    this.referenceFrame=frame;this.guidePosition=null;this.invalidate();
    if(!this.referenceData||!frame){this.updateGhostVisibility();return;}
    if(frame.qpos.length!==this.model.nq||frame.qvel.length!==this.model.nv||!frame.qpos.every(Number.isFinite)||!frame.qvel.every(Number.isFinite)){this.referenceFrame=null;this.updateGhostVisibility();return;}
    // A separate MjData reconstructs an observed historical pose without modifying current physics.
    this.referenceData.qpos.set(frame.qpos);this.referenceData.qvel.set(frame.qvel);if(Number.isFinite(frame.sim_time))this.referenceData.time=frame.sim_time;
    this.mj.mj_forward(this.model,this.referenceData);
    for(const {id,mesh} of this.ghostMeshes){mesh.position.set(...xyz(this.referenceData.geom_xpos,id*3));const a=this.referenceData.geom_xmat,o=id*9;mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(a[o],a[o+1],a[o+2],0,a[o+3],a[o+4],a[o+5],0,a[o+6],a[o+7],a[o+8],0,0,0,0,1));}
    if(frame.eef?.every(Number.isFinite))this.guidePosition=new THREE.Vector3(...frame.eef);this.updateGhostVisibility();
  }
  setTrajectory(points:[number,number,number][]=[]){
    this.invalidate();
    const geometry=new THREE.BufferGeometry().setFromPoints(points.filter(p=>p.length===3&&p.every(Number.isFinite)).slice(-4000).map(p=>new THREE.Vector3(...p)));this.trail.geometry.dispose();this.trail.geometry=geometry;this.updateGhostVisibility();
  }
  updateGhostVisibility(){this.historicalGhost.visible=false;this.guide.visible=false;this.trail.visible=(this.trail.geometry.getAttribute('position')?.count??0)>1;}
  apply(frame:NativeFrame){
    this.pending=frame;if(!this.model||!this.alive)return;
    if(frame.qpos.length!==this.model.nq||frame.qvel.length!==this.model.nv||!Number.isFinite(frame.sim_time)||!frame.qpos.every(Number.isFinite)||!frame.qvel.every(Number.isFinite))throw Error('Rejected invalid native state dimensions');
    this.data.qpos.set(frame.qpos);this.data.qvel.set(frame.qvel);this.data.time=frame.sim_time;
    this.mj.mj_forward(this.model,this.data);
    for(const {id,mesh} of this.meshes){mesh.visible=true;mesh.position.set(...xyz(this.data.geom_xpos,id*3));const a=this.data.geom_xmat,o=id*9;mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(a[o],a[o+1],a[o+2],0,a[o+3],a[o+4],a[o+5],0,a[o+6],a[o+7],a[o+8],0,0,0,0,1));}
    if(frame.eef)this.eef.set(...frame.eef);else{let site=-1;for(let i=0;i<this.model.nsite;i++)if(/grip_site|eef_site|tcp/.test(nameAt(this.model,this.model.name_siteadr[i]))){site=i;break;}if(site>=0)this.eef.set(...xyz(this.data.site_xpos,site*3));}
    if(!this.drag&&this.mouseStyle!=='wemine')this.target.copy(this.eef);else if(!this.hasFrame)this.target.copy(this.eef);this.handle.visible=true;this.hasFrame=true;
    if(!this.framed){this.framed=true;this.center.copy(this.eef).add(new THREE.Vector3(-.15,0,this.displayOffset-.12));this.backgroundOptions.floorZ=this.nativeFloorHeight();this.scaleReference.position.z=this.backgroundOptions.floorZ+.5;this.backdrop.setRoomScale(this.roomScale);this.addFactoryContext();this.setView(this.view);}
    if(this.benchmark)this.benchmark.nativeFrames++;this.nativeCount++;this.renderer.shadowMap.needsUpdate=true;this.invalidate();this.onReady?.(true);
  }
  nativeFloorHeight(){
    // Read the authoritative horizontal plane pose; use 0 only when no floor plane exists.
    const floors:{z:number;named:boolean}[]=[];
    for(let id=0;id<this.model.ngeom;id++){
      if(Number(this.model.geom_type[id])!==0||Math.abs(this.data.geom_xmat[id*9+8])<.95)continue;
      const z=Number(this.data.geom_xpos[id*3+2]);
      if(!Number.isFinite(z)||z>this.eef.z+.05)continue;
      const name=nameAt(this.model,this.model.name_geomadr[id]);floors.push({z,named:/floor|ground|terrain/i.test(name)});
    }
    floors.sort((a,b)=>Number(b.named)-Number(a.named)||a.z-b.z);
    return floors[0]?.z??0;
  }
  setFactoryBackground(visible:boolean){this.backgroundVisible=visible;this.fallbackFloor.visible=!visible||!this.backgroundReady;this.invalidate(500);}
  setRoomScale(scale:number){this.roomScale=scale;this.backdrop.setRoomScale(scale);this.invalidate(500);}
  addFactoryContext(){
    // Original task geometry stays native within nativeWorkcell. Only static room/support visuals are replaced.
    const layout=nativeWorkcellLayout(this.model,this.data,this.displayOffset,this.backgroundOptions.floorZ);
    this.stationBounds=layout.stationBounds.clone();this.center.copy(this.stationBounds.getCenter(new THREE.Vector3()));this.context.add(createNativeWorkcellSkin(layout));
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(10,10),new THREE.MeshStandardMaterial({color:'#929b9d',roughness:.9}));floor.position.set(this.center.x,this.center.y,this.backgroundOptions.floorZ);floor.receiveShadow=true;this.fallbackFloor.add(floor);
  }
  setView(view:Props['view']){
    this.invalidate();this.view=view;this.controls.enabled=!this.drag&&view!=='robot'&&!(this.mouseStyle==='wemine'&&this.manual);this.camera.up.set(0,0,1);
    const halfFov=THREE.MathUtils.degToRad(this.camera.fov/2),aspect=Math.max(.5,this.camera.aspect);
    const radius=this.stationBounds?.getBoundingSphere(new THREE.Sphere()).radius??.95;
    const horizontalHalf=Math.atan(Math.tan(halfFov)*aspect),distance=1.06*radius/Math.sin(Math.min(halfFov,horizontalHalf));
    this.controls.maxDistance=Math.max(4.5,distance*1.4);
    if(view==='top'){
      this.camera.up.set(0,1,0);const size=this.stationBounds?.getSize(new THREE.Vector3())??new THREE.Vector3(1.4,1.6,1);
      const topDistance=Math.max(2.6,1.1*Math.max(size.y,size.x/aspect)/(2*Math.tan(halfFov)));this.camera.position.copy(this.center).add(new THREE.Vector3(0,0,topDistance));
    }else this.camera.position.copy(this.center).add(new THREE.Vector3(1.6,-1.9,1.3).normalize().multiplyScalar(distance));
    this.controls.target.copy(this.center);this.camera.lookAt(this.center);this.controls.update();
  }
  pointerRay(event:PointerEvent|WheelEvent){const r=this.renderer.domElement.getBoundingClientRect();this.ray.setFromCamera(new THREE.Vector2((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1),this.camera);return this.ray;}
  floorPoint(x:number,y:number){const ray=this.pointerRay({clientX:x,clientY:y} as PointerEvent);return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,0,1),-this.displayOffset),new THREE.Vector3());}
  sendWemineDelta(delta:THREE.Vector3){if(!this.manual||!this.hasFrame||!this.onOperatorCommand)return;const desired=this.target.clone().add(delta);const lead=desired.clone().sub(this.eef);lead.z=0;lead.clampLength(0,.03);desired.x=this.eef.x+lead.x;desired.y=this.eef.y+lead.y;const applied=desired.clone().sub(this.target);applied.z=0;if(applied.length()<1e-6)return;this.target.copy(desired);this.invalidate();this.onOperatorCommand({type:'move',delta:applied.toArray() as [number,number,number]});}
  pointerLeave=()=>{this.lastHover=null;};
  lockedMouseMove=(event:MouseEvent)=>{if(this.mouseStyle!=='wemine'||!this.manual||!this.hasFrame||(event.buttons&2)||!document.pointerLockElement?.contains(this.renderer.domElement))return;const bounds=this.renderer.domElement.getBoundingClientRect(),x=bounds.left+bounds.width/2,y=bounds.top+bounds.height/2;const a=this.floorPoint(x,y),b=this.floorPoint(x+event.movementX,y+event.movementY);if(a&&b)this.sendWemineDelta(b.sub(a));};
  picked(event:PointerEvent|WheelEvent){this.overlayWorkcell.updateMatrixWorld(true);return this.pointerRay(event).intersectObject(this.hit,false).length>0;}
  sendTarget(desired:THREE.Vector3){if(!this.manual||!this.hasFrame||!this.onOperatorCommand)return;const lead=desired.clone().sub(this.eef);lead.clampLength(0,.1);desired.copy(this.eef).add(lead);const delta=desired.clone().sub(this.target);delta.clampLength(0,.03);if(delta.length()<.0002)return;this.target.add(delta);this.invalidate();this.onOperatorCommand({type:'move',delta:delta.toArray() as [number,number,number]});}
  pointerDown=(e:PointerEvent)=>{if(this.mouseStyle==='wemine')return;if(!this.manual||!this.hasFrame||e.button!==0||!this.picked(e))return;const plane=new THREE.Plane(new THREE.Vector3(0,0,1),-(this.target.z+this.displayOffset)),anchor=this.pointerRay(e).ray.intersectPlane(plane,new THREE.Vector3());if(!anchor)return;e.preventDefault();e.stopImmediatePropagation();this.drag={id:e.pointerId,anchor,start:this.target.clone(),plane};this.controls.enabled=false;this.renderer.domElement.setPointerCapture(e.pointerId);this.renderer.domElement.style.cursor='grabbing';};
  pointerMove=(e:PointerEvent)=>{if(!this.drag){if(this.mouseStyle==='wemine'&&this.manual&&this.hasFrame&&!document.pointerLockElement&&!(e.buttons&2)){const point=this.floorPoint(e.clientX,e.clientY);if(point&&this.lastHover)this.sendWemineDelta(point.clone().sub(this.lastHover));this.lastHover=point;}this.renderer.domElement.style.cursor=this.mouseStyle==='wemine'?'crosshair':this.manual&&this.hasFrame&&this.picked(e)?'grab':'';return;}if(e.pointerId!==this.drag.id)return;e.preventDefault();e.stopImmediatePropagation();const point=this.pointerRay(e).ray.intersectPlane(this.drag.plane,new THREE.Vector3());if(point)this.sendTarget(this.drag.start.clone().add(point.sub(this.drag.anchor)));};
  pointerUp=(e:PointerEvent)=>{if(!this.drag||e.pointerId!==this.drag.id)return;this.drag=null;this.controls.enabled=this.view!=='robot';this.renderer.domElement.style.cursor='';if(this.renderer.domElement.hasPointerCapture(e.pointerId))this.renderer.domElement.releasePointerCapture(e.pointerId);};
  wheel=(e:WheelEvent)=>{if(this.mouseStyle==='wemine'||!this.manual||e.shiftKey||(!this.drag&&!this.picked(e)))return;e.preventDefault();e.stopImmediatePropagation();const target=this.target.clone();target.z-=THREE.MathUtils.clamp(e.deltaY*.00015,-.025,.025);this.sendTarget(target);};
  drawFactoryBackground(){
    // Cache the *final original-resolution pixels*, not a reduced splat asset or
    // lower-resolution render target. Camera/scene changes and Spark async dirty
    // events invalidate this cache. Native robot-only changes reuse it exactly.
    this.camera.updateMatrixWorld();const size=this.renderer.getDrawingBufferSize(new THREE.Vector2());
    if(!this.backgroundCache||this.backgroundCache.image.width!==size.x||this.backgroundCache.image.height!==size.y){
      this.backgroundCache?.dispose();this.backgroundCache=new THREE.FramebufferTexture(size.x,size.y);
      this.backgroundCopyMaterial.uniforms.image.value=this.backgroundCache;this.backgroundDirty=true;
    }
    if(!sameViewMatrix(this.backgroundCameraWorld,this.camera.matrixWorld)||!sameViewMatrix(this.backgroundCameraProjection,this.camera.projectionMatrix))this.backgroundDirty=true;
    if(this.backgroundDirty){
      this.backgroundDirty=false;this.renderer.render(this.backgroundLayer,this.camera);
      // Raw RGBA framebuffer copy + raw shader sampling preserve Spark's exact
      // displayed RGB, without a second tone-map or color-space conversion.
      this.renderer.copyFramebufferToTexture(this.backgroundCache);
      if(this.benchmark)this.benchmark.backgroundDraws++;this.backgroundCameraWorld.copy(this.camera.matrixWorld);this.backgroundCameraProjection.copy(this.camera.projectionMatrix);this.backgroundDrawCount++;
    }else{this.renderer.render(this.backgroundCopyScene,this.backgroundCopyCamera);this.backgroundCacheHits++;if(this.benchmark)this.benchmark.cacheHits++;}
  }
  startBenchmark(request:RenderBenchmarkRequest){
    if(this.benchmark)return;
    if(!this.hasFrame||!this.backgroundReady){publishRenderBenchmark({...request,durationMs:0,draws:0,cpuMs:0,gpuMs:null,gpuSamples:0,backgroundDraws:0,cacheHits:0,nativeFrames:0,quality:this.backdrop.currentQuality,pixelRatio:this.renderer.getPixelRatio(),shadowPolicy:'reuse_existing',error:'Wait for the original scene and background to finish loading.'});return;}
    const now=performance.now();this.benchmark={...request,started:now,ends:now+3000,draws:0,cpuTotal:0,gpuTotal:0,gpuSamples:0,backgroundDraws:0,cacheHits:0,nativeFrames:0};this.invalidate();
  }
  finishBenchmark(now:number){
    const b=this.benchmark;if(!b)return;this.benchmark=null;
    publishRenderBenchmark({id:b.id,mode:b.mode,durationMs:now-b.started,draws:b.draws,cpuMs:b.draws?b.cpuTotal/b.draws:0,gpuMs:b.gpuSamples?b.gpuTotal/b.gpuSamples:null,gpuSamples:b.gpuSamples,backgroundDraws:b.backgroundDraws,cacheHits:b.cacheHits,nativeFrames:b.nativeFrames,quality:this.backdrop.currentQuality,pixelRatio:this.renderer.getPixelRatio(),shadowPolicy:'reuse_existing'});
  }
  pollGpuTimers(){
    if(!this.gpuExt||!this.gpuQueries.length)return;const gl=this.renderer.getContext() as WebGL2RenderingContext;
    if(gl.getParameter(this.gpuExt.GPU_DISJOINT_EXT)){for(const {query} of this.gpuQueries)gl.deleteQuery(query);this.gpuQueries=[];this.gpuTotalMs=0;this.gpuSamples=0;if(this.benchmark){this.benchmark.gpuTotal=0;this.benchmark.gpuSamples=0;}return;}
    this.gpuQueries=this.gpuQueries.filter(({query,benchmarkId})=>{if(!gl.getQueryParameter(query,gl.QUERY_RESULT_AVAILABLE))return true;const ns=gl.getQueryParameter(query,gl.QUERY_RESULT);if(Number.isFinite(ns)){this.gpuTotalMs+=ns/1e6;this.gpuSamples++;if(this.benchmark&&this.benchmark.id===benchmarkId){this.benchmark.gpuTotal+=ns/1e6;this.benchmark.gpuSamples++;}}gl.deleteQuery(query);return false;});
  }
  beginGpuTimer(now:number){
    if(!this.gpuExt||this.gpuQueries.length>=8||now-this.gpuLastQuery<200)return null;
    const gl=this.renderer.getContext() as WebGL2RenderingContext;if(gl.getQuery(this.gpuExt.TIME_ELAPSED_EXT,gl.CURRENT_QUERY))return null;
    const query=gl.createQuery();if(!query)return null;gl.beginQuery(this.gpuExt.TIME_ELAPSED_EXT,query);this.gpuLastQuery=now;return query;
  }
  loop=()=>{
    if(!this.alive)return;
    const now=performance.now();this.rafCount++;this.pollGpuTimers();
    if(this.view!=='robot')this.controls.update();
    if(this.benchmark&&now>=this.benchmark.ends)this.finishBenchmark(now);
    // Camera damping and incoming native frames invalidate the image. Keep rAF cheap
    // while paused: no repeated splat sort/render or shadow pass for an unchanged view.
    if(this.dirty||now<this.settleUntil||this.benchmark){this.dirty=false;const drawStart=performance.now();const gpuQuery=this.beginGpuTimer(now);
    if(this.drag&&!this.manual){const id=this.drag.id;this.drag=null;this.controls.enabled=this.view!=='robot';if(this.renderer.domElement.hasPointerCapture(id))this.renderer.domElement.releasePointerCapture(id);}
    this.updateGhostVisibility();
    if(this.hasFrame){this.handle.position.copy(this.manual?this.target:this.eef);this.handle.scale.setScalar(this.manual?1.15:1);this.leash.visible=this.manual;const p=this.leash.geometry.getAttribute('position') as THREE.BufferAttribute;p.setXYZ(0,...this.eef.toArray());p.setXYZ(1,...this.target.toArray());p.needsUpdate=true;if(this.guidePosition)this.guide.position.copy(this.guidePosition);
      if(this.view==='robot'){let camera=-1;for(let i=0;i<this.model.ncam;i++)if(/eye_in_hand|wrist/.test(nameAt(this.model,this.model.name_camadr[i]))){camera=i;break;}if(camera>=0){const a=this.data.cam_xmat,o=camera*9;this.camera.position.set(...xyz(this.data.cam_xpos,camera*3));this.camera.position.z+=this.displayOffset;this.camera.up.set(a[o+1],a[o+4],a[o+7]);this.camera.lookAt(this.camera.position.clone().add(new THREE.Vector3(-a[o+2],-a[o+5],-a[o+8])));}}
    }else this.leash.visible=false;
    this.renderer.info.reset();const autoClear=this.renderer.autoClear;const background=this.scene.background;if(this.backgroundVisible&&this.backgroundReady){if(this.benchmark?.mode==='direct')this.backgroundDirty=true;this.drawFactoryBackground();this.renderer.autoClear=false;this.renderer.clearDepth();this.scene.background=null;}this.renderer.render(this.scene,this.camera);this.scene.background=background;this.renderer.autoClear=false;this.renderer.clearDepth();this.renderer.render(this.overlayScene,this.camera);this.renderer.autoClear=autoClear;
    if(gpuQuery){(this.renderer.getContext() as WebGL2RenderingContext).endQuery(this.gpuExt.TIME_ELAPSED_EXT);this.gpuQueries.push({query:gpuQuery,benchmarkId:this.benchmark?.id});}
    const cpuMs=performance.now()-drawStart;if(this.benchmark){this.benchmark.draws++;this.benchmark.cpuTotal+=cpuMs;}this.drawCount++;this.drawMs+=cpuMs;this.lastCalls=this.renderer.info.render.calls;this.lastTriangles=this.renderer.info.render.triangles;
    }
    if(now-this.statsSince>=1000){const seconds=(now-this.statsSince)/1000;this.onRenderStats?.({fps:this.rafCount/seconds,drawFps:this.drawCount/seconds,frameMs:this.drawCount?this.drawMs/this.drawCount:0,benchmarkMode:this.benchmark?.mode??null,gpuMs:this.gpuSamples?this.gpuTotalMs/this.gpuSamples:null,backgroundDrawFps:this.backgroundDrawCount/seconds,backgroundCacheHits:this.backgroundCacheHits,nativeFrameHz:this.nativeCount/seconds,profile:this.profile,quality:this.backdrop.currentQuality,loadedSplatCount:this.backdrop.loadedSplatCount,pixelRatio:this.renderer.getPixelRatio(),drawCalls:this.lastCalls,triangles:this.lastTriangles});this.statsSince=now;this.rafCount=0;this.drawCount=0;this.drawMs=0;this.nativeCount=0;this.backgroundDrawCount=0;this.backgroundCacheHits=0;this.gpuTotalMs=0;this.gpuSamples=0;}
    this.raf=requestAnimationFrame(this.loop);
  };
  dispose(){this.alive=false;cancelAnimationFrame(this.raf);const c=this.renderer.domElement;c.removeEventListener('pointerdown',this.pointerDown,true);c.removeEventListener('pointermove',this.pointerMove,true);c.removeEventListener('pointerleave',this.pointerLeave,true);c.removeEventListener('pointerup',this.pointerUp,true);c.removeEventListener('pointercancel',this.pointerUp,true);c.removeEventListener('lostpointercapture',this.pointerUp,true);c.removeEventListener('wheel',this.wheel,true);document.removeEventListener('mousemove',this.lockedMouseMove,true);this.backdrop.dispose();this.scaleReference.disposeLabel();this.controls.removeEventListener('change',this.cameraChanged);this.controls.dispose();this.materials?.dispose();this.scene.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();const m=o.material;if(Array.isArray(m))m.forEach(v=>v.dispose());else m.dispose();}});this.overlayScene.traverse(o=>{if(o instanceof THREE.Mesh||o instanceof THREE.Line){o.geometry.dispose();const m=o.material;if(Array.isArray(m))m.forEach(v=>v.dispose());else m.dispose();}});this.backgroundCache?.dispose();this.backgroundCopyMaterial.dispose();this.backgroundCopyScene.traverse(o=>{if(o instanceof THREE.Mesh)o.geometry.dispose();});const gl=this.renderer.getContext() as WebGL2RenderingContext;for(const {query} of this.gpuQueries)gl.deleteQuery(query);this.renderer.dispose();this.renderer.domElement.remove();try{this.referenceData?.delete();this.data?.delete();this.model?.delete();}catch{}}
}

export default function NativeScene({scene,frame,manual,onOperatorCommand,onReady,view,ghost,cameraRevision,referenceFrame=null,trajectory=[],renderProfile,onRenderStats,mouseStyle}:Props){
  const host=useRef<HTMLDivElement>(null),renderer=useRef<NativeRenderer|null>(null);const callbacks=useRef({onOperatorCommand,onReady,onRenderStats});callbacks.current={onOperatorCommand,onReady,onRenderStats};const latest=useRef({frame,manual,view,ghost,referenceFrame,trajectory,renderProfile,mouseStyle});latest.current={frame,manual,view,ghost,referenceFrame,trajectory,renderProfile,mouseStyle};const[error,setError]=useState('');
  const[background,setBackground]=useState<BackgroundState>({status:'loading',message:'Loading factory background',quality:'full'});
  const[roomScale,setRoomScale]=useState(1),[showScale,setShowScale]=useState(false),[showFactory,setShowFactory]=useState(true);
  useEffect(()=>{
    if(!host.current||!scene)return;setError('');setRoomScale(1);setShowScale(false);setShowFactory(true);callbacks.current.onReady?.(false);const r=new NativeRenderer(host.current,setBackground,latest.current.renderProfile??getRenderPerformance().profile);renderer.current=r;r.onRenderStats=stats=>{publishRenderStats(stats);callbacks.current.onRenderStats?.(stats);};r.onOperatorCommand=c=>callbacks.current.onOperatorCommand?.(c);r.onReady=(v,e)=>callbacks.current.onReady?.(v,e);r.manual=latest.current.manual;r.mouseStyle=latest.current.mouseStyle;r.controls.enabled=!(r.mouseStyle==='wemine'&&r.manual);r.view=latest.current.view;r.showGhost=latest.current.ghost;r.referenceFrame=latest.current.referenceFrame;r.setTrajectory(latest.current.trajectory);r.pending=latest.current.frame;
    let seenBenchmark=getRenderPerformance().benchmarkRequest?.id??0;
    const unsubscribe=subscribeRenderPerformance(()=>{const request=getRenderPerformance().benchmarkRequest;if(request&&request.id!==seenBenchmark){seenBenchmark=request.id;r.startBenchmark(request);}if(latest.current.renderProfile===undefined){const profile=getRenderPerformance().profile;if(r.profile!==profile)r.setProfile(profile);}});
    const resize=new ResizeObserver(()=>r.resize());resize.observe(host.current);void r.load(scene).catch(e=>{if(r.alive){setError(String(e));callbacks.current.onReady?.(false,String(e));}});return()=>{unsubscribe();resize.disconnect();r.dispose();renderer.current=null;};
  },[scene]);
  useEffect(()=>{if(frame)try{renderer.current?.apply(frame);}catch(e){setError(String(e));callbacks.current.onReady?.(false,String(e));}},[frame]);
  useEffect(()=>{if(renderer.current){renderer.current.manual=manual;renderer.current.controls.enabled=!(renderer.current.mouseStyle==='wemine'&&manual);renderer.current.invalidate();}},[manual]);useEffect(()=>{if(renderer.current){renderer.current.mouseStyle=mouseStyle;renderer.current.lastHover=null;renderer.current.controls.enabled=!(mouseStyle==='wemine'&&renderer.current.manual);}},[mouseStyle]);useEffect(()=>renderer.current?.setView(view),[view,cameraRevision]);useEffect(()=>{if(renderer.current){renderer.current.showGhost=ghost;renderer.current.updateGhostVisibility();renderer.current.invalidate();}},[ghost]);
  useEffect(()=>renderer.current?.setProfile(renderProfile??getRenderPerformance().profile),[renderProfile]);
  useEffect(()=>renderer.current?.setReference(referenceFrame),[referenceFrame]);useEffect(()=>renderer.current?.setTrajectory(trajectory),[trajectory]);
  return <div ref={host} style={{width:'100%',height:'100%',position:'relative',minHeight:300}}>{(!scene||!frame)&&<div style={{position:'absolute',inset:24,color:'#8ca5b4',fontSize:12,zIndex:2}}>Waiting for authoritative native simulation state…</div>}{error&&<div style={{position:'absolute',inset:24,color:'#ffaf8a',fontSize:12,zIndex:3}}>Native renderer: {error}</div>}{scene&&<details className={`factory-background-badge ${background.status}`}><summary>{showFactory?background.message:"Native workcell · background hidden"}</summary>{background.status==='ready'&&<div className="factory-background-settings"><label>Factory background<input aria-label="Show native factory background" type="checkbox" checked={showFactory} onChange={e=>{setShowFactory(e.target.checked);renderer.current?.setFactoryBackground(e.target.checked);}}/></label><p>Visual context only. Native robot and objects draw in front.</p><label>Background detail<select aria-label="Factory background quality" value={background.quality} onChange={e=>{const r=renderer.current;if(r)void r.backdrop.selectQuality(e.target.value as SplatQuality).catch(()=>{if(r.alive)setBackground(current=>({...current,message:'Could not switch background quality'}));});}}><option value="full">Original · full resolution</option><option value="500k">Balanced</option><option value="100k">Performance</option></select></label><label>Room scale <span>×{roomScale.toFixed(2)}</span><input aria-label="Factory room scale" type="range" min="0.5" max="1.5" step="0.05" value={roomScale} onChange={e=>{const scale=Number(e.target.value);setRoomScale(scale);renderer.current?.setRoomScale(scale);}}/></label><p>Native scene · meters · fitted factory floor<br/>Robot and table retain server dimensions.</p><p>{background.scaleDescription}</p><label>1 m reference cube<input aria-label="Show one meter scale reference" type="checkbox" checked={showScale} onChange={e=>{setShowScale(e.target.checked);if(renderer.current)renderer.current.scaleReference.visible=e.target.checked;renderer.current?.invalidate();}}/></label><button type="button" onClick={()=>{setRoomScale(1);renderer.current?.setRoomScale(1);}}>Reset scale</button></div>}</details>}</div>;
}
