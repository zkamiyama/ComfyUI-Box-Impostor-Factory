// A rectangular through-hole lies in the XY plane. The narrow blue/white tab stands
// proud of the front face of the upper crossbar and extends into the opening.
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { makeFixtureGeometry } from './make_fixture.mjs';

export function makeOpenFrame() {
  const rectangle=(w,h)=>[[-w,-h],[w,-h],[w,h],[-w,h]];
  const outer=rectangle(.65,.48), inner=rectangle(.554,.384);
  const lift=(p,z)=>[...p,z], front=.06, back=-.06;
  const quads=[];
  for(let i=0;i<4;i++){
    const j=(i+1)%4, a=outer[i], b=outer[j], c=inner[j], d=inner[i];
    const rim=[a,b,c,d];
    quads.push([[0,0,1],rim.map(p=>lift(p,front))]);
    quads.push([[0,0,-1],rim.toReversed().map(p=>lift(p,back))]);
    const n=[Math.sign(b[1]-a[1]),Math.sign(a[0]-b[0]),0];
    quads.push([n,[lift(a,back),lift(b,back),lift(b,front),lift(a,front)]]);
    quads.push([n.map(v=>-v),[lift(c,back),lift(d,back),lift(d,front),lift(c,front)]]);
  }
  return makeFixtureGeometry([
    [[ .31,.426,.106], [.07,.074,.094], [.12,.19,.83]],
    [[ .31,.344,.153], [.07,.226,.04], [.92,.9,.8]],
  ],quads);
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const output=process.argv[2] ?? '.test-output/open-frame.glb';
  await mkdir(dirname(output),{recursive:true});
  await writeFile(output,makeOpenFrame());
  console.log(output);
}
