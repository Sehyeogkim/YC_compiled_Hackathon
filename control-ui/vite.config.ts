import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import {nativeSceneCache} from './dev/nativeSceneCache';
import {Agent} from 'node:http';
const nativeAgent=new Agent({keepAlive:true,keepAliveMsecs:1000,maxSockets:2});
export default defineConfig({plugins:[nativeSceneCache(),react()],server:{host:'0.0.0.0',proxy:{'/robot-api-base':{target:'http://127.0.0.1:8011',agent:nativeAgent,changeOrigin:true,rewrite:path=>path.replace(/^\/robot-api-base/,'')},'/robot-api':{target:'http://127.0.0.1:8000',agent:nativeAgent,changeOrigin:true,rewrite:path=>path.replace(/^\/robot-api/,'')}}},preview:{proxy:{'/robot-api-base':{target:'http://127.0.0.1:8011',agent:nativeAgent,changeOrigin:true,rewrite:path=>path.replace(/^\/robot-api-base/,'')},'/robot-api':{target:'http://127.0.0.1:8000',agent:nativeAgent,changeOrigin:true,rewrite:path=>path.replace(/^\/robot-api/,'')}}},build:{chunkSizeWarningLimit:1600}});
