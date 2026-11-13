// A chat/report-friendly loader for the original three-impostor viewer.
// Fetch the portable Box ZIP, validate that every member stays inside the asset
// namespace, then serve it from memory without installing a service worker.
const revision='9785a8d';
const fallback=window.BI_BOX_FALLBACK_IMAGE || 'https://raw.githubusercontent.com/zkamiyama/ComfyUI-Box-Impostor-Factory/main/docs/verification/chat-walking-dog-preview.webp';
const defaultArchive='https://raw.githubusercontent.com/zkamiyama/ComfyUI-Box-Impostor-Factory/'+revision+'/examples/cc0-walking-dog-qwen-six-20261113.box.zip';
const status=document.getElementById('status');
const overlay=document.getElementById('chat-overlay');
const params=new URLSearchParams(location.search);
const zipURL=window.BI_BOX_ZIP || params.get('zip') || defaultArchive;
if(window.BI_BOX_LABEL) document.querySelector('header h1').textContent=String(window.BI_BOX_LABEL);
const MAX_ARCHIVE_BYTES=160*1024*1024;

function say(value){
  status.textContent=value;
  if(overlay) overlay.querySelector('span').textContent=value;
}
function rejectMember(name){
  if(!name || name.startsWith('/') || name.includes('\\') || name.includes('..') ||
      name.includes('%') || name.split('/').some(x=>!x||x==='.'||x==='..')) {
    throw Error('Invalid archive member: '+name);
  }
}
function allowSource(value){
  const url=new URL(value);
  if(url.protocol!=='https:' || !['raw.githubusercontent.com','cdn.jsdelivr.net'].includes(url.hostname))
    throw Error('Use a GitHub-hosted, published .box.zip archive');
  if(!url.pathname.endsWith('.box.zip')) throw Error('The source must end with .box.zip');
  return url;
}
async function fetchArchive(url){
  const response=await fetch(url,{mode:'cors'});
  if(!response.ok) throw Error('ZIP HTTP '+response.status);
  const size=Number(response.headers.get('content-length'))||0;
  if(size>MAX_ARCHIVE_BYTES) throw Error('Archive exceeds the safe download limit');
  const chunks=[];let bytes=0;
  const reader=response.body?.getReader();
  if(!reader) return new Uint8Array(await response.arrayBuffer());
  while(true){
    const {done,value}=await reader.read();
    if(done)break;
    chunks.push(value);bytes+=value.byteLength;
    if(bytes>MAX_ARCHIVE_BYTES){await reader.cancel();throw Error('Archive exceeds the safe download limit');}
    say('Boxを取得中 '+(size ? Math.round(bytes/size*100)+'%' : (bytes/1048576).toFixed(1)+' MiB'));
  }
  const merged=new Uint8Array(bytes);
  let off=0;
  for(const chunk of chunks){merged.set(chunk,off);off+=chunk.length;}
  return merged;
}
function showFailure(err){
  const reason=String(err?.message || err);
  say('3D表示に失敗: '+reason);
  overlay?.classList.add('failed');
  console.error('Box Chat Viewer:',err);
}
try {
  if(!window.JSZip) throw Error('ZIP展開ライブラリを読み込めません');
  const safeURL=allowSource(zipURL);
  say('Boxを取得中…');
  const bytes=await fetchArchive(safeURL);
  say('Boxを展開中…');
  const zip=await window.JSZip.loadAsync(bytes, {checkCRC32:true});
  if(!zip.file('manifest.json')) throw Error('manifest.json がありません');
  const paths=Object.keys(zip.files);
  for(const file of paths){rejectMember(file.endsWith('/') ? file.slice(0,-1) : file);}
  if(paths.length>512) throw Error('ZIPのファイル数が多すぎます');
  const manifest=JSON.parse(await zip.file('manifest.json').async('string'));
  if(manifest.schemaVersion!==1 || manifest.layout!=='box') throw Error('Box形式が正しくありません');
  if(manifest.faceOrder?.length!==6) throw Error('6面のデータがありません');
  // Pass the decoded archive directly to the existing loader. This also
  // works in opaque-origin chat frames (about:srcdoc/data:) without history
  // rewriting, a service worker, or virtual HTTP request interception.
  window.__BI_CHAT_BOX__={zip,manifest};
  say('Three.jsでBoxを構築中…');
  await import('./viewer.mjs');
  // The original viewer initializes asynchronously and writes the selected
  // backend to #status only when the impostor is actually renderable.
  await new Promise((resolve,reject)=>{
    let attempt=0;
    function check(){
      const current=status.textContent;
      if(current.startsWith('webgl2') || current.startsWith('webgpu')) return resolve();
      if(current.startsWith('表示エラー')) return reject(new Error(current));
      if(attempt++>900) return reject(new Error('3D renderer initialization timed out'));
      setTimeout(check,100);
    }
    check();
  });
  // The upstream viewer fits the complete bounds very conservatively. For a
  // report card start a little closer; the user's wheel can still zoom out.
  const canvas=document.querySelector('#canvas');
  if(canvas)canvas.dispatchEvent(new WheelEvent('wheel',{deltaY:-1200,deltaMode:0,bubbles:true,cancelable:true}));
  overlay?.remove();
} catch(err) {
  showFailure(err);
  const preview=document.getElementById('chat-fallback-image');
  if(preview){preview.src=fallback;preview.hidden=false;}
}
