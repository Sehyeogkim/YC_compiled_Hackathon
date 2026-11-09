import {createReadStream, statSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
import type {IncomingMessage, ServerResponse} from 'node:http';
import type {Plugin} from 'vite';

/** Local copies of an already verified scene, keyed by the authoritative reset hash.
 * Missing/different scenes fall through to the live GPU. No simulator state is cached.
 */
export function nativeSceneCache():Plugin {
  const directory=fileURLToPath(new URL('../.cache/native-scenes/',import.meta.url));
  function serve(req:IncomingMessage,res:ServerResponse,next:()=>void){
    const url=new URL(req.url??'/', 'http://localhost');
    const hash=url.searchParams.get('expected_scene_sha256');
    if(req.method!=='GET'||!hash||!/^[a-f0-9]{64}$/.test(hash))return next();
    const file=join(directory,`${hash}.json`);
    let size:number;
    try{size=statSync(file).size;}catch{return next();}
    res.setHeader('Content-Type','application/json');
    res.setHeader('Content-Length',size);
    res.setHeader('Cache-Control','no-store');
    res.setHeader('X-RG-Scene-Cache','HIT');
    const stream=createReadStream(file);
    res.once('close',()=>stream.destroy());
    stream.on('error',()=>{if(!res.headersSent)res.statusCode=500;res.end();});
    stream.pipe(res);
  }
  return {name:'rg-native-scene-cache',
    configureServer(server){server.middlewares.use('/robot-api/scene',serve);server.middlewares.use('/robot-api-base/scene',serve);server.middlewares.use('/local-scene',serve);},
    configurePreviewServer(server){server.middlewares.use('/robot-api/scene',serve);server.middlewares.use('/robot-api-base/scene',serve);server.middlewares.use('/local-scene',serve);},
  };
}
