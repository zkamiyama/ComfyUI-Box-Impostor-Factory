// A Vulkan compositor can present a WebGPU SwiftShader texture in Linux headless Chrome.
// Offscreen WebGPU passes can succeed even when the canvas swapchain cannot present.
export function chromiumOptions(backend='auto') {
  const software=process.env.BI_SOFTWARE_GPU==='1';
  const args=software ? ['--use-angle=swiftshader','--enable-unsafe-swiftshader'] : [];
  if (backend==='webgpu' || (software && backend==='auto')) args.push('--enable-unsafe-webgpu');
  if (software && process.platform==='linux' && backend!=='webgl2') {
    args.push('--enable-features=Vulkan','--use-vulkan=swiftshader','--disable-vulkan-surface');
  }
  return {headless:true,args,executablePath:process.env.BI_CHROMIUM || undefined};
}

// Executed inside the page after a frame. The canvas is transparent outside the fitted asset.
export function presentedForeground(panes=1) {
  const sampled=document.createElement('canvas');
  sampled.width=128*panes;sampled.height=128;
  const ctx=sampled.getContext('2d');
  ctx.drawImage(document.querySelector('#canvas'),0,0,sampled.width,sampled.height);
  const rgba=ctx.getImageData(0,0,sampled.width,sampled.height).data;
  const covered=new Array(panes).fill(0);
  for (let y=0;y<128;y++) for(let x=0;x<sampled.width;x++) {
    if (rgba[4*(y*sampled.width+x)+3]>0) covered[Math.floor(x/128)]++;
  }
  return covered.every(count=>count>0 && count<128*128);
}
