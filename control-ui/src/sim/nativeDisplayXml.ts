/** Only the browser's unused solver workspace allocation changes.
 * Joint/body/geom/asset/camera definitions and authoritative server XML stay intact.
 * The viewer performs mj_forward only and never executes dynamics.
 */
export function nativeDisplayXml(xml:string):string {
  const workspace=(attributes:string)=>attributes.replace(/\s+(?:njmax|nconmax|nstack|memory)\s*=\s*(["']).*?\1/g,'');
  if(/<size\b[^>]*\/>/.test(xml))return xml.replace(/<size\b([^>]*)\/>/,(_match,attributes:string)=>`<size${workspace(attributes)} memory="32M"/>`);
  return xml.replace(/<mujoco\b[^>]*>/,match=>match+'\n<size memory="32M"/>');
}
