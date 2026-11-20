import {readFile,writeFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {dirname,join} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';

const root=dirname(fileURLToPath(import.meta.url)),output=join(root,'final');
const require=createRequire(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES
 ?join(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES,'idol-qwen-report.cjs'):import.meta.url);
const {marked}=await import(pathToFileURL(require.resolve('marked')).href);
const markdown=await readFile(join(root,'README.md'),'utf8');
const base='https://github.com/zkamiyama/ComfyUI-Box-Impostor-Factory/blob/experiment/idol-qwen-box-20261120/scripts/idol_qwen_experiment/';
let body=marked.parse(markdown,{gfm:true,async:false});
body=body.replace(/href="([^":#]+)"/g,(_,path)=>'href="'+base+path+'"');
const data=async(name,mime)=>'data:'+mime+';base64,'+(await readFile(join(output,name))).toString('base64');
const poster=await data('Idol_Qwen_Box_000.png','image/png');
const movie=await data('Idol_Qwen_Box_Comparison.mp4','video/mp4');
const side=await data('Idol_Qwen_Box_090.png','image/png');
const raw=await data('Idol_Qwen_FullBody.png','image/png');
const normals=await data('Idol_Qwen_Normal_Ablation.png','image/png');
const media='<section class="comparison"><h2>Boxの全身比較</h2><figure><video controls playsinline preload="metadata" poster="'+poster+'"><source src="'+movie+'" type="video/mp4"></video><figcaption>左：元のIdol Box。右：Qwenの正面RGB補正版。全身240フレーム、24fps、10秒。ほかの5面は元のRGBです。</figcaption></figure><details><summary>90°方向の全身比較</summary><img src="'+side+'" alt="90度方向の全身比較。左が元のIdol Box、右がQwenの正面RGB補正版"></details></section>'+
'<section class="comparison"><h2>Qwenへ渡した全身画像と生成画像</h2><figure><img src="'+raw+'" alt="全身RGBの比較。左がIdolの描画画像、右が選択したQwenの出力"><figcaption>左：Idolを描画した入力RGB。右：Qwenの出力。各1024×1024pxの全身画像を切り出さず並べています。この画像上の改善と、Box表示での改善は分けて評価しています。</figcaption></figure></section>';
const marker='<h2>生成条件と試行</h2>';
if(!body.includes(marker))throw new Error('Report heading changed');
body=body.replace(marker,media+marker);
const diagnostic='<details class="diagnostic"><summary>採用しなかった深度由来法線の比較</summary><figure><img src="'+normals+'" alt="元のBox、Qwen RGBのみ、深度由来法線のみ、Qwenと深度由来法線を適用した全身比較"><figcaption>法線の診断用候補です。保存したQwen試作には法線変更を含めていません。</figcaption></figure></details>';
body=body.replace('<h2>再実行に使うコード</h2>',diagnostic+'<h2>再実行に使うコード</h2>');
body=body.replaceAll('<table>','<div class="table-wrap"><table>').replaceAll('</table>','</table></div>');
let font='';
try{font=(await readFile(process.env.BOX_REPORT_FONT||join(root,'../zipsplat/report-regular.woff'))).toString('base64');}
catch(error){if(process.env.BOX_REPORT_FONT)throw error;}
const hash=createHash('sha256').update(markdown).digest('hex');
const html='<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="source-markdown-sha256" content="'+hash+'"><title>IdolにQwen Imageを適用したBoxテクスチャ実験</title><style>'+
(font?'@font-face{font-family:ReportJP;src:url(data:font/woff;base64,'+font+') format("woff");font-display:block}':'')+
':root{color-scheme:light;--ink:#173043;--muted:#506675;--line:#d9e4eb;--accent:#166c76}*{box-sizing:border-box}body{margin:0;background:#eef3f6;color:var(--ink);font:16px/1.9 ReportJP,"Noto Sans CJK JP",system-ui,sans-serif}.shell{max-width:1460px;margin:24px auto;border:1px solid var(--line);border-radius:14px;background:#fff;overflow:hidden}header{padding:22px 42px;background:#143243;color:#dcecf1;letter-spacing:.06em;font-size:13px}header strong{display:block;color:#fff;font-size:18px;letter-spacing:.01em;margin-top:3px}main{padding:32px 42px 40px}h1{font-size:32px;line-height:1.45;margin:0 0 20px}h2{font-size:24px;line-height:1.5;margin:38px 0 18px;padding-top:22px;border-top:1px solid var(--line)}p,li{max-width:1080px}p{margin:0 0 18px}a{color:#086d89;text-underline-offset:3px;overflow-wrap:anywhere}code{font-family:ui-monospace,monospace;font-size:.86em;background:#eef3f6;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}pre{padding:18px 20px;background:#f1f5f8;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;line-height:1.7}pre code{padding:0}ol,ul{padding-left:26px}li{margin:10px 0}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin:20px 0 24px}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.7}th,td{padding:11px 13px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}th{background:#e9f2f5}tbody tr:nth-child(even){background:#f8fafb}tr:last-child td{border-bottom:0}.comparison{margin:32px -12px 42px}.comparison h2{border:0;margin:0 12px 15px;padding:0}figure{margin:0;border:1px solid var(--line);border-radius:10px;overflow:hidden}video,img{display:block;width:100%;height:auto;background:#808080}figcaption{padding:12px 16px;font-size:13px;color:var(--muted);background:#f6f9fb}details{margin-top:14px}summary{cursor:pointer;padding:10px 14px;background:#edf5f7;border-radius:7px;font-weight:bold}details img{margin-top:10px;border-radius:8px}.diagnostic figure{margin-top:10px}.diagnostic img{margin-top:0;border-radius:0}footer{padding:20px 42px;background:#f5f8fa;border-top:1px solid var(--line);color:var(--muted);font-size:12px;overflow-wrap:anywhere}@media(max-width:700px){.shell{margin:0;border:0;border-radius:0}main,header,footer{padding:22px}h1{font-size:26px}h2{font-size:21px}.comparison{margin:28px -12px}th,td{padding:9px;font-size:13px}}@media print{body{background:white}.shell{border:0;margin:0}video{display:none}.table-wrap,figure{break-inside:avoid}h2{break-after:avoid}}'+
'</style></head><body><div class="shell"><header>COMFYUI BOX IMPOSTOR FACTORY<strong>Idol × Qwen Image：正面RGB補正の記録</strong></header><main>'+body+'</main><footer>動画と比較画像を埋め込んだレポート。モデルを回転して確認できるHTMLは、公式3D-Inspectorの出力を変更せず別ファイルとして保存しています。本文SHA256: '+hash+'</footer></div></body></html>';
const path=join(output,'Idol_Qwen_Report.html');await writeFile(path,html);
console.log(JSON.stringify({path,bytes:Buffer.byteLength(html),markdownSHA256:hash,sha256:createHash('sha256').update(html).digest('hex')}));
