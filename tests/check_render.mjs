// Decode RGBA PNGs without an image dependency to verify the real renderer's output.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { inflateSync } from 'node:zlib';

function paeth(a,b,c) {
  const p=a+b-c, da=Math.abs(p-a), db=Math.abs(p-b), dc=Math.abs(p-c);
  return da<=db && da<=dc ? a : db<=dc ? b : c;
}
function decodePNG(file) {
  const magic=Buffer.from([137,80,78,71,13,10,26,10]);
  assert.ok(file.subarray(0,8).equals(magic),'Not a PNG');
  let offset=8, width, height, bpp, depth, kind;
  const chunks=[];
  while(offset<file.length) {
    const length=file.readUInt32BE(offset);
    const type=file.toString('ascii',offset+4,offset+8);
    const data=file.subarray(offset+8,offset+8+length);
    if(type==='IHDR') {
      width=data.readUInt32BE(0); height=data.readUInt32BE(4);
      depth=data[8]; kind=data[9];
      bpp=kind===6?4:kind===2?3:0;
      assert.equal(depth,8); assert.ok(bpp,'Unsupported PNG pixel type');
    }
    if(type==='IDAT') chunks.push(data);
    offset+=12+length;
    if(type==='IEND') break;
  }
  assert.ok(width && height && chunks.length,'PNG missing header or pixels');
  const raw=inflateSync(Buffer.concat(chunks)), stride=width*bpp;
  assert.equal(raw.length,(stride+1)*height,'Unexpected inflated PNG size');
  const out=Buffer.alloc(stride*height);
  for(let y=0;y<height;y++) {
    const f=raw[y*(stride+1)];
    assert.ok(f>=0 && f<=4,'Unsupported PNG filter');
    for(let x=0;x<stride;x++){
      const i=y*stride+x;
      const left=x>=bpp?out[i-bpp]:0;
      const up=y?out[i-stride]:0;
      const upperLeft=y&&x>=bpp?out[i-stride-bpp]:0;
      const input=raw[y*(stride+1)+1+x];
      const predictor=f===0?0:f===1?left:f===2?up:f===3?Math.floor((left+up)/2):paeth(left,up,upperLeft);
      out[i]=(input+predictor)&255;
    }
  }
  return {width,height,kind,pixels:out};
}

const [basePath,improvedPath,widthArg,heightArg]=process.argv.slice(2);
if(!basePath || !improvedPath) throw Error('Usage: check_render.mjs BASE.png IMPROVED.png WIDTH HEIGHT');
const [baseline,reduced]=await Promise.all([basePath,improvedPath].map(async path=>decodePNG(await readFile(path))));
const width=Number(widthArg),height=Number(heightArg);
for(const image of [baseline,reduced]){
  assert.equal(image.width,width);
  assert.equal(image.height,height);
  assert.equal(image.kind,6,'Expected transparent RGBA output');
  let opaque=0, transparent=0;
  for(let i=3;i<image.pixels.length;i+=4){
    if(image.pixels[i]>8) opaque++;
    else transparent++;
  }
  assert.ok(opaque > width*height*0.02,'Missing rendered Box foreground');
  assert.ok(transparent > width*height*0.01,'Background alpha was not preserved');
}
assert.ok(!baseline.pixels.equals(reduced.pixels),'Sampling option did not change the rendered frame');
console.log(JSON.stringify({width,height,format:'RGBA8',pixelsDifferent:true}));
