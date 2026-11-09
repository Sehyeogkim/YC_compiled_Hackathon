import * as THREE from 'three';
type Mj = any;

/** Geometry, indexed UVs and texture pixels come directly from the compiled MJCF. */
export function nativeGeometry(model:Mj, id:number):THREE.BufferGeometry|null {
  const type=Number(model.geom_type[id]),s=Array.from(model.geom_size.slice(id*3,id*3+3)) as [number,number,number];
  if(type===7){
    const mesh=model.geom_dataid[id],va=model.mesh_vertadr[mesh],vn=model.mesh_vertnum[mesh],fa=model.mesh_faceadr[mesh],fn=model.mesh_facenum[mesh],ta=model.mesh_texcoordadr?.[mesh]??-1;
    const g=new THREE.BufferGeometry();
    // MuJoCo indexes UVs per face corner, independently of position indices.
    if(ta>=0&&model.mesh_facetexcoord){
      const p=new Float32Array(fn*9),uv=new Float32Array(fn*6),normal=new Float32Array(fn*9);let hasNormals=true;
      const na=model.mesh_normaladr?.[mesh]??-1;
      for(let corner=0;corner<fn*3;corner++){
        const vi=va+model.mesh_face[fa*3+corner],ti=ta+model.mesh_facetexcoord[fa*3+corner];
        for(let axis=0;axis<3;axis++)p[corner*3+axis]=model.mesh_vert[vi*3+axis];
        uv[corner*2]=model.mesh_texcoord[ti*2];uv[corner*2+1]=model.mesh_texcoord[ti*2+1];
        const ni=na>=0?na+model.mesh_facenormal[fa*3+corner]:-1;
        if(ni>=0)for(let axis=0;axis<3;axis++)normal[corner*3+axis]=model.mesh_normal[ni*3+axis];else hasNormals=false;
      }
      g.setAttribute('position',new THREE.BufferAttribute(p,3));g.setAttribute('uv',new THREE.BufferAttribute(uv,2));
      if(hasNormals)g.setAttribute('normal',new THREE.BufferAttribute(normal,3));else g.computeVertexNormals();
    }else{g.setAttribute('position',new THREE.Float32BufferAttribute(Array.from(model.mesh_vert.slice(va*3,(va+vn)*3)),3));g.setIndex(Array.from(model.mesh_face.slice(fa*3,(fa+fn)*3)));g.computeVertexNormals();}
    return g;
  }
  if(type===6)return new THREE.BoxGeometry(s[0]*2,s[1]*2,s[2]*2);
  if(type===5){const g=new THREE.CylinderGeometry(s[0],s[0],s[1]*2,40);g.rotateX(Math.PI/2);return g;}
  if(type===3){const g=new THREE.CapsuleGeometry(s[0],s[1]*2,8,20);g.rotateX(Math.PI/2);return g;}
  if(type===2)return new THREE.SphereGeometry(s[0],24,18);
  if(type===4){const g=new THREE.SphereGeometry(1,24,18);g.scale(...s);return g;}
  return null;
}

export class NativeMaterials {
  textures=new Map<string,THREE.DataTexture>();
  model:Mj;
  constructor(model:Mj){this.model=model;}
  material(id:number){
    const m=this.model,mid=m.geom_matid[id],rgba=mid>=0?m.mat_rgba.slice(mid*4,mid*4+4):m.geom_rgba.slice(id*4,id*4+4);
    let map:THREE.DataTexture|undefined;
    if(mid>=0&&m.ntex){
      const stride=Math.max(1,Math.floor(m.mat_texid.length/m.nmat));let tex=m.mat_texid[mid*stride+(stride>=10?1:0)];if(tex<0&&stride>=10)tex=m.mat_texid[mid*stride+8];
      if(tex>=0&&Number(m.tex_type[tex])===0){
        const rx=m.mat_texrepeat?.[mid*2]??1,ry=m.mat_texrepeat?.[mid*2+1]??1,key=`${tex}:${rx}:${ry}`;map=this.textures.get(key);
        if(!map){const w=m.tex_width[tex],h=m.tex_height[tex],channels=m.tex_nchannel?.[tex]??3,offset=m.tex_adr[tex],source=m.tex_data??m.tex_rgb,pixels=new Uint8Array(w*h*4);
          for(let p=0;p<w*h;p++){const i=offset+p*channels;pixels[p*4]=source[i];pixels[p*4+1]=source[i+(channels>1?1:0)];pixels[p*4+2]=source[i+(channels>2?2:0)];pixels[p*4+3]=channels===4?source[i+3]:255;}
          map=new THREE.DataTexture(pixels,w,h,THREE.RGBAFormat);map.colorSpace=THREE.SRGBColorSpace;map.wrapS=map.wrapT=THREE.RepeatWrapping;map.repeat.set(rx,ry);map.flipY=false;map.generateMipmaps=true;map.minFilter=THREE.LinearMipmapLinearFilter;map.magFilter=THREE.LinearFilter;map.anisotropy=4;map.needsUpdate=true;this.textures.set(key,map);
        }
      }
    }
    return new THREE.MeshStandardMaterial({color:new THREE.Color(rgba[0],rgba[1],rgba[2]),map:map??null,roughness:mid>=0?THREE.MathUtils.clamp(1-(m.mat_shininess?.[mid]??.2),.25,.9):.55,metalness:mid>=0?Math.min(.35,m.mat_reflectance?.[mid]??.1):.1,transparent:rgba[3]<1,opacity:rgba[3]});
  }
  dispose(){for(const t of this.textures.values())t.dispose();this.textures.clear();}
}

export function visibleNativeGeoms(model:Mj):Set<number>{
  const bodyWithVisual=new Set<number>();
  for(let i=0;i<model.ngeom;i++)if(model.geom_type[i]!==0&&model.geom_group[i]!==3&&model.geom_contype[i]===0&&model.geom_conaffinity[i]===0)bodyWithVisual.add(model.geom_bodyid[i]);
  const visible=new Set<number>();
  for(let i=0;i<model.ngeom;i++){
    if(model.geom_type[i]===0||model.geom_group[i]===3)continue;
    if(bodyWithVisual.has(model.geom_bodyid[i])&&(model.geom_contype[i]!==0||model.geom_conaffinity[i]!==0))continue;
    visible.add(i);
  }
  return visible;
}
