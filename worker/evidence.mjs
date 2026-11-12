import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { deflateSync } from 'node:zlib';

function pngChunk(tag, payload) {
  const data = Buffer.concat([Buffer.from(tag), payload]);
  let crc = 0xffffffff;
  for (const byte of data) {
    crc ^= byte;
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
  }
  const chunk = Buffer.alloc(12 + payload.length);
  chunk.writeUInt32BE(payload.length);
  data.copy(chunk,4);
  chunk.writeUInt32BE((crc ^ 0xffffffff) >>> 0,chunk.length - 4);
  return chunk;
}

// A lossless RGBA8 contact sheet. Row 1 is the live Baker; row 2 is its reload.
export function contactSheet(before, after, backend) {
  const {width:w,height:h} = before[0];
  const W = w * before.length, stride = 1 + W*4;
  const data = Buffer.alloc(stride * h * 2);
  [before,after].forEach((set,group) => set.forEach((frame,index) => {
    const source = Buffer.from(frame.color,'base64');
    for(let y=0; y<h; y++){
      const srcY = backend === 'webgl2' ? h-1-y : y;
      const at = (group*h+y)*stride+1+index*w*4;
      source.copy(data,at,srcY*w*4,(srcY+1)*w*4);
    }
  }));
  const header = Buffer.alloc(13);
  header.writeUInt32BE(W);header.writeUInt32BE(2*h,4);header[8]=8;header[9]=6;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),pngChunk('IHDR',header),
    pngChunk('IDAT',deflateSync(data)),pngChunk('IEND',Buffer.alloc(0))]);
}

export async function writeEvidence(dir, before, after, report, backend) {
  await mkdir(dir,{recursive:true});
  await writeFile(join(dir,'roundtrip.json'),JSON.stringify({backend,...report},null,2)+'\n');
  await writeFile(join(dir,'live-and-reloaded.png'),contactSheet(before,after,backend));
}
