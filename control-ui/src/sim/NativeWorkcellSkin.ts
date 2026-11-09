import * as THREE from 'three';
type Mj=any;
const readName=(m:Mj,address:number)=>{let s='';for(let i=address;m.names[i];i++)s+=String.fromCharCode(m.names[i]);return s;};

/** Only named static room/support geometry is replaced visually. Task props are retained. */
export function isNativeRoomSkin(m:Mj,id:number):boolean {
  const geom=readName(m,m.name_geomadr[id]);
  if(/^wall_(?:leftcorner|rightcorner|left|right|rear|front)_visual$/.test(geom))return true;
  let body=m.geom_bodyid[id];
  for(let depth=0;body>0&&depth<m.nbody;depth++){
    if(m.body_jntnum[body]>0)return false;
    const name=readName(m,m.name_bodyadr[body]);
    if(/^(?:floor|ground|table|table0|living_room_table|kitchen_table)$/.test(name))return true;
    body=m.body_parentid[body];
  }
  return /^(?:floor|ground|floor_visual|table_visual|table_top|table_collision|table_leg_[0-9]+)$/.test(geom);
}
function freeBodyRoot(m:Mj,id:number){let body=m.geom_bodyid[id];for(let depth=0;body>0&&depth<m.nbody;depth++) {if(m.body_jntnum[body]>0&&Number(m.jnt_type[m.body_jntadr[body]])===0)return body;body=m.body_parentid[body];}return -1;}
function geomBounds(m:Mj,d:Mj,id:number){
  const offset=id*6,a=m.geom_aabb,r=id*9,p=id*3,R=d.geom_xmat;
  const center=a?new THREE.Vector3(a[offset],a[offset+1],a[offset+2]):new THREE.Vector3();
  const half=a?new THREE.Vector3(a[offset+3],a[offset+4],a[offset+5]):new THREE.Vector3(m.geom_size[p],m.geom_size[p+1],m.geom_size[p+2]);
  const world=new THREE.Vector3(d.geom_xpos[p]+R[r]*center.x+R[r+1]*center.y+R[r+2]*center.z,d.geom_xpos[p+1]+R[r+3]*center.x+R[r+4]*center.y+R[r+5]*center.z,d.geom_xpos[p+2]+R[r+6]*center.x+R[r+7]*center.y+R[r+8]*center.z);
  const extent=new THREE.Vector3(Math.abs(R[r])*half.x+Math.abs(R[r+1])*half.y+Math.abs(R[r+2])*half.z,Math.abs(R[r+3])*half.x+Math.abs(R[r+4])*half.y+Math.abs(R[r+5])*half.z,Math.abs(R[r+6])*half.x+Math.abs(R[r+7])*half.y+Math.abs(R[r+8])*half.z);
  return new THREE.Box3(world.clone().sub(extent),world.add(extent));
}
export type WorkcellLayout={surfaceZ:number;floorZ:number;displayOffset:number;beltLength:number;beltWidth:number;beltCenter:[number,number];taskBounds:THREE.Box3;supplyBounds:THREE.Box3;basketBounds:THREE.Box3;stationBounds:THREE.Box3;robotBase:THREE.Vector3};
export function nativeWorkcellLayout(m:Mj,d:Mj,displayOffset:number,floorZ:number):WorkcellLayout {
  const taskBounds=new THREE.Box3(),supplyBounds=new THREE.Box3(),basketBounds=new THREE.Box3();let support=0;let robotBase=new THREE.Vector3(-.6,0,0);
  for(let id=0;id<m.ngeom;id++){
    const root=freeBodyRoot(m,id);
    if(root>=0){const bounds=geomBounds(m,d,id);if(!Number.isFinite(bounds.min.x)||bounds.max.x-bounds.min.x>2)continue;taskBounds.union(bounds);if(/^basket(?:_|$)/.test(readName(m,m.name_bodyadr[root])))basketBounds.union(bounds);else supplyBounds.union(bounds);}
    const geom=readName(m,m.name_geomadr[id]);
    if(/^(?:table_top|table_collision|table_visual)$/.test(geom)&&Number(m.geom_type[id])===6)support=Math.max(support,geomBounds(m,d,id).max.z);
  }
  for(let i=0;i<m.nbody;i++)if(/^robot[0-9]*_base$/.test(readName(m,m.name_bodyadr[i]))){robotBase.set(d.xpos[i*3],d.xpos[i*3+1],d.xpos[i*3+2]);break;}
  if(taskBounds.isEmpty())taskBounds.set(new THREE.Vector3(-.25,-.35,0),new THREE.Vector3(.25,.35,.2));
  if(supplyBounds.isEmpty())supplyBounds.copy(taskBounds);
  const center=supplyBounds.getCenter(new THREE.Vector3()),length=1.4,width=Math.max(.5,supplyBounds.max.x-supplyBounds.min.x+.16);
  // Supply belt stops before the fixed receiving basket. Both retain their native XY frame.
  const endY=basketBounds.isEmpty()?supplyBounds.max.y+.035:Math.min(supplyBounds.max.y+.035,basketBounds.min.y-.055);
  const centerY=endY-length/2,top=support+displayOffset;
  const stationBounds=new THREE.Box3(new THREE.Vector3(Math.min(center.x-width/2-.1,robotBase.x-.2),Math.min(centerY-length/2-.03,robotBase.y-.2),floorZ),new THREE.Vector3(Math.max(center.x+width/2+.22,robotBase.x+.2),Math.max(endY+.03,robotBase.y+.2),Math.max(top+.95,taskBounds.max.z+displayOffset+.1)));
  if(!basketBounds.isEmpty())stationBounds.expandByPoint(new THREE.Vector3(basketBounds.min.x-.035,basketBounds.min.y-.035,top)).expandByPoint(new THREE.Vector3(basketBounds.max.x+.035,basketBounds.max.y+.035,top+.2));
  return {surfaceZ:support,floorZ,displayOffset,beltLength:length,beltWidth:width,beltCenter:[center.x,centerY],taskBounds,supplyBounds,basketBounds,stationBounds,robotBase};
}

/** Static industrial skin in display meters. It owns no physics and moves no native object. */
export function createNativeWorkcellSkin(layout:WorkcellLayout):THREE.Group {
  const root=new THREE.Group();root.name='Factory workcell · static visual skin · 1.40 m conveyor';
  root.userData.layout={beltLength:layout.beltLength,beltWidth:layout.beltWidth,supportNativeZ:layout.surfaceZ,displayOffset:layout.displayOffset,floorZ:layout.floorZ};
  const top=layout.surfaceZ+layout.displayOffset,[cx,cy]=layout.beltCenter,width=layout.beltWidth,length=layout.beltLength;
  const metal=new THREE.MeshStandardMaterial({color:'#a9b5bc',metalness:.78,roughness:.32});
  const dark=new THREE.MeshStandardMaterial({color:'#202b31',metalness:.48,roughness:.45});
  const blue=new THREE.MeshStandardMaterial({color:'#287ca1',metalness:.46,roughness:.4});
  const black=new THREE.MeshStandardMaterial({color:'#1e2427',metalness:.05,roughness:.87});
  const yellow=new THREE.MeshStandardMaterial({color:'#edbc39',metalness:.18,roughness:.55});
  const box=(size:[number,number,number],pos:[number,number,number],material:THREE.Material)=>{const o=new THREE.Mesh(new THREE.BoxGeometry(...size),material);o.position.set(...pos);o.castShadow=true;o.receiveShadow=true;root.add(o);return o;};
  const frameDepth=.085,frameTop=top-.009;
  // Keep the metal bed below the rubber belt to avoid coplanar depth artifacts.
  box([width+.09,length+.04,frameDepth],[cx,cy,frameTop-frameDepth/2],metal).name='conveyor-frame';
  box([width,length,.008],[cx,cy,top-.004],black).name='supply-belt';
  // Shallow transverse belt seams remain static, matching the paused native support plane.
  for(let i=0;i<35;i++)box([width-.015,.001,.0007],[cx,cy-length/2+.02+i*(length-.04)/34,top+.00035],dark);
  for(const x of [cx-width/2-.027,cx+width/2+.027]){
    box([.018,length+.06,.035],[x,cy,top+.004],metal);
    box([.006,length-.02,.007],[x,cy,top+.024],yellow);
  }
  const footZ=layout.floorZ+.012,legTop=frameTop-frameDepth,legHeight=Math.max(.025,legTop-footZ);
  for(const x of [cx-width/2+.035,cx+width/2-.035])for(const y of [cy-length/2+.15,cy+length/2-.15]){
    box([.04,.04,legHeight],[x,y,footZ+legHeight/2],metal);
    box([.09,.09,.024],[x,y,footZ],dark);
  }
  box([width-.08,.025,.025],[cx,cy-length*.32,layout.floorZ+legHeight*.35],metal);
  box([width-.08,.025,.025],[cx,cy+length*.32,layout.floorZ+legHeight*.35],metal);
  const motor=new THREE.Mesh(new THREE.CylinderGeometry(.052,.052,.14,20),blue);motor.rotation.z=Math.PI/2;motor.position.set(cx+width/2+.12,cy-length*.33,top-.09);motor.castShadow=true;root.add(motor);
  // A dedicated fixture stand meets the unchanged native robot base exactly.
  const rp=layout.robotBase.clone();rp.z+=layout.displayOffset;
  box([.29,.29,.035],[rp.x,rp.y,rp.z-.0175],metal).name='robot-mount-plate';
  const mountHeight=Math.max(.03,rp.z-.035-layout.floorZ);
  box([.19,.19,mountHeight],[rp.x,rp.y,layout.floorZ+mountHeight/2],dark);
  box([.34,.34,.018],[rp.x,rp.y,layout.floorZ+.009],metal).name='robot-mount-foot';
  for(const dx of [-.115,.115])for(const dy of [-.115,.115])box([.013,.013,.006],[rp.x+dx,rp.y+dy,rp.z+.003],dark);
  // Independent, stationary receiving stand outside the belt footprint.
  // The basket mesh and its native pose are not moved or parented to the conveyor.
  if(!layout.basketBounds.isEmpty()){
    const center=layout.basketBounds.getCenter(new THREE.Vector3()),size=layout.basketBounds.getSize(new THREE.Vector3());
    const platformX=size.x+.045,platformY=size.y+.045,basketTop=layout.basketBounds.min.z+layout.displayOffset-.0005;
    box([platformX,platformY,.012],[center.x,center.y,basketTop-.006],metal).name='basket-platform';
    const receivingLegHeight=Math.max(.03,basketTop-.012-(layout.floorZ+.012));
    for(const dx of [-1,1])for(const dy of [-1,1]){
      const x=center.x+dx*(platformX/2-.025),y=center.y+dy*(platformY/2-.025);
      box([.022,.022,receivingLegHeight],[x,y,layout.floorZ+.012+receivingLegHeight/2],blue).name='basket-table-leg';
      box([.055,.055,.024],[x,y,layout.floorZ+.012],dark).name='basket-table-foot';
    }
    for(const dx of [-1,1])box([.014,platformY,.04],[center.x+dx*(platformX/2-.007),center.y,basketTop-.032],blue);
  }
  // Measured 1.40 m belt gauge, independent of the background's room-scale setting.
  const gaugeX=cx-width/2-.075;
  box([.006,length,.004],[gaugeX,cy,top+.014],yellow);
  for(let i=0;i<=14;i++)box([i%5===0?.035:.02,.004,.005],[gaugeX,cy-length/2+i*.1,top+.015],yellow);
  return root;
}
