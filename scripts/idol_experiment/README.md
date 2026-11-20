# Idol / TripoSplat 比較実験 20261120

同じ参照画像から生成したTripoSplat 20 steps、Idol 20 steps、Idol 50 stepsを、Box変換前の3D Gaussian Splatting（3DGS）とBox512の両方で比較した。3モデルともnative rendererの240フレーム、6面の512px Box、公式3D-Inspector HTML、Boxの240フレーム撮影まで完了している。実験識別子は `20261120` とする。[生成記録](evidence/render-comparison.json)・[native比較記録](evidence/media/native/comparison.json)・[Box比較記録](evidence/media/box/comparison.json)

## 見た目の結果

比較動画の目視では、Idolは髪と服の細部がTripoSplatより改善した。一方、参照画像の笑顔と脚の姿勢はTripoSplatの方が近い。

Idolを20から50 stepsへ増やすと、背面の髪と靴が小幅に改善した。今回の1枚では、50 stepsに増やしたことによる全面的な品質向上までは確認していない。

Box512では肌や顔のざらつきが目立ち、元の3DGSで見られた細部の改善がそのまま残るわけではなかった。ただし、native 3DGSとBoxではrenderer、照明、画角が異なる。この観察だけからBox変換を原因と断定せず、両動画をまたいだ画素差も品質指標として使わない。

動画の列順はどちらも **TripoSplat 20 / Idol 20 / Idol 50**。各列512×512px、全体1536×584px、240フレーム、24fps、10秒で、比較画像のcropやresizeは行っていない。[native比較記録](evidence/media/native/comparison.json)・[Box比較記録](evidence/media/box/comparison.json)

## 入力と生成条件

TripoSplatは以前生成したNPZを再利用した。Idolは旧 `minimal-prompt.json` のdenoiserとdecoderを差し替え、20 stepsと50 stepsを順に実行した。Qwenによる画像補正は今回の比較に加えていない。

| 項目 | 共通条件 |
|---|---|
| 入力 | `triposplat-person-20261120/inputs/source-original.png`、1024×1024、RGB |
| 背景除去 | `birefnet.safetensors` → `BIPrepareReference`、`mask_mode=foreground` |
| 前処理 | `TripoSplatPreprocessImage`、`erode_radius=1`、`size=1024` |
| Conditioning | `dino_v3_vit_h.safetensors` と `flux2-vae.safetensors` |
| Sampler | `dpmpp_2m`、scheduler `simple`、CFG 3、denoise 1 |
| Sampling seed | `46` |
| Decoder seed | `790219963981395` |
| Gaussian数 | 各モデル `262144` |

入力PNGのSHA256は `9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b`。前処理後のRGBとcoverageはIdolの両runで旧Tripoとfloat32の全要素が一致し、alphaの差は全画素で0だった。prepared referenceのcache identityも3モデルで `bb85341d2c3caf034417b3d16fed1dc6b78cf4fee9a93ea031f00bc7ec07efa3` と一致した。このidentityはPNGファイルのSHA256とは別の値で、RGBA配列の形状と内容に基づく。[Idol 20入力比較](evidence/idol-20steps-seed46/reference-comparison.json)・[Idol 50入力比較](evidence/idol-50steps-seed46/reference-comparison.json)

今回と旧Tripoの `nodes_triposplat.py` も同じSHA256だった。したがって、保持を確認した範囲は入力RGBA、前処理コード、グラフの条件であり、Idolのdenoiserとdecoder内部まで同じという意味ではない。[Idol 20事前記録](evidence/idol-20steps-seed46/preflight.json)・[Idol 50事前記録](evidence/idol-50steps-seed46/preflight.json)

## 重みの固定

Idolの2ファイルは [Comfy-Org/Idol](https://huggingface.co/Comfy-Org/Idol/tree/e907d8dca6c465735eaf4ce9b3a3c2c6401d5705) のrevision `e907d8dca6c465735eaf4ce9b3a3c2c6401d5705` から取得し、サイズとSHA256を検証した。DINOとFLUX VAEは旧Tripoで使ったファイルを共有した。DINOやVAEを同名の別ファイルへ置き換えない。[重みの実測記録](evidence/model-files.json)

| ファイル | 用途 | SHA256 |
|---|---|---|
| `diffusion_models/idol_fp16.safetensors` | Idol denoiser | `9abf6a99fc6bdad701901b12de8fe1b256f02cae4948f7489d25d86b11b52a93` |
| `vae/idol_decoder_fp16.safetensors` | Idol decoder | `6a146a1e6b2653f74422f7ea835eb8715c547c7de8fc4545051f15c247ae54a9` |
| `clip_vision/dino_v3_vit_h.safetensors` | 共通vision encoder | `a29ef35101a16966972a0d50732a6f3a608ff7cfffb2afa9bbe9007cb842cc53` |
| `vae/flux2-vae.safetensors` | 共通conditioning VAE | `d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5` |
| `diffusion_models/triposplat_fp16.safetensors` | 旧Tripo denoiser | `c870b97ac1d6bc9177608a5ec625e19ef9f3c5019aa68f64b0fb7803abcd6d20` |
| `vae/triposplat_vae_decoder_fp16.safetensors` | 旧Tripo decoder | `ed0d0c3d43b599e326845d0ec70f3cf77be9a55e2d97627ac3b34d2830763cc8` |

## 向きと共通範囲

native SPLATのY軸回転はTripoSplatが+90°、Idol 20 / 50が0°。参照に対する表示方向を合わせるための回転であり、姿勢や形状を参照画像へ変形して合わせたものではない。比較用PLYにはこの回転を適用済みなので、表示時に重ねて回転しない。推論直後のNPZは各runに残している。[共通範囲の記録](evidence/common-frame/bounds.json)

共通範囲は、回転後の全モデル・全Gaussian中心のunionから計算した。unionの高さの3%を各軸へ加え、原点を中心とした対称なboundsを設定している。既存templateの範囲も下回らない。Gaussianの除去、モデルごとの拡大縮小、平行移動は行っていない。

```json
{
  "min": [-0.2524936217069626, -0.5301595348119735, -0.2242100256681442],
  "max": [ 0.2524936217069626,  0.5301595348119735,  0.2242100256681442]
}
```

native 3DGSの比較では3モデルのカメラと描画設定が一致した。`splatScale=1`、`sharpen=1`、`headlightShading=0`、`opacityThreshold=0`、背景 `#808080` を使い、全モデルで240フレーム中の画像端への接触は0だった。[native比較記録](evidence/media/native/comparison.json)・[native描画処理](render_native_splat.py)

## Box512と表示検証

共通範囲から6方向を描画し、nativeのRGB、alpha、normalとmetric depthを既存のBox builderへ渡した。depthはカメラ軸に沿った前方重み付き期待値で、alpha > 0.5 の画素を対象に共通boundsの各軸長で正規化している。保持したproxyはカメラ・metadata用で、proxyの形状をBoxの生成元には使っていない。[描画処理](render_native_splat.py)・[生成コマンド](evidence/render-comparison.json)

| 検査 | Tripo 20 | Idol 20 | Idol 50 |
|---|---:|---:|---:|
| 6面の `outOfBoundsAlphaPixels` | 全面0 | 全面0 | 全面0 |
| 6面の `faceEdgePixels` | 全面0 | 全面0 | 全面0 |
| Inspector撮影フレーム数 | 240 | 240 | 240 |
| 撮影画像のhashと順序 | 全240件検証 | 全240件検証 | 全240件検証 |

`outOfBoundsAlphaPixels` はalpha > 0.5なのに正規化depthが範囲外または非有限となった画素数、`faceEdgePixels` はalpha > 0.5のatlas画素が画像端に接した数である。3モデル×6面でどちらも0だった。この検査範囲で切れは検出されなかったが、Gaussianの薄い裾まで含めた無欠損を保証する値ではない。[生成記録](evidence/render-comparison.json)

3D表示には公式 `3D-Inspector-skills/skills/3d-inspector/view.py` の出力HTMLをそのまま使用した。HTMLを編集せず、既存UIとSave PNGでBoxを撮影した。3本とも共通reference、512×512px、正射影、240フレームで、対応フレームの実測yaw差は0°だった。[Box比較記録](evidence/media/box/comparison.json)

初回のBox撮影では、残っていた `DISPLAY` / `WAYLAND_DISPLAY` によりANGLE SwiftShaderが存在しないXCB displayを参照して失敗した。`capture_boxes.py` で子プロセスに渡す環境からこの2変数だけを除外すると、3本とも完了した。大域環境、公式HTML、撮影スクリプトは変更していない。[撮影環境記録](evidence/capture-environment.json)・[撮影実行処理](capture_boxes.py)

## 生成時間とVRAM

| 設定 | Comfy server execution | GPU全体の観測最大使用VRAM | 観測数 |
|---|---:|---:|---:|
| Idol 20 steps | 29.765秒 | 6,818 MiB | 144 |
| Idol 50 steps | 40.082秒 | 7,266 MiB | 198 |

時間はComfy historyの `execution_start` と `execution_success` から算出した。VRAMは200ms間隔のnvidia-smi観測最大値で、driverや他プロセスの使用量を含む。PyTorch allocatorのpeakではなく、観測間の短いpeakを見落とす可能性がある。[測定値](evidence/measurements.json)

Idol 50ではComfyがnode `1,4,5,6,7,8,9,11` を再利用しており、conditioningもcacheに含まれる。上表は各1回の実行コストで、stepsだけを変えた厳密な速度比較ではない。Idol 20の `clientWaitSeconds` は後から結果を回収した待ち時間なので、推論時間には使わない。[Idol 20結果](evidence/idol-20steps-seed46/result.json)・[Idol 50結果](evidence/idol-50steps-seed46/result.json)

旧Tripoには全prompt処理99.62秒というログが残っているが、BiRefNet、DINO、VAE、denoiser、decoderの読み込みを含む。今回と同条件の速度測定としては未測定と扱い、99.62秒からIdolの速度向上率を算出しない。旧Tripoのpeak VRAMも未測定。[旧Tripo時間の根拠](evidence/triposplat-historical-timing.json)・[ログ抜粋](evidence/triposplat-successful-inference-log-excerpt.txt)

## 実行手順

実行環境は270-PC。実験outputは `/mnt/f/_dev/box-impostor-lab/experiments/idol-triposplat-20261120`、Factory worktreeは `/home/dev/Projects/box-impostor-lab/worktrees/idol-triposplat-20261120`、ComfyUIは `/home/dev/Projects/box-impostor-lab/ComfyUI` を使用した。実行Pythonは `/home/dev/Projects/box-impostor-lab/.venv/bin/python`。

以下はこの実験の処理順序。完成済みのrunを再実行するときは別の出力先を用意し、保存済み結果の回収には `--collect` を使う。

1. [bootstrap_remote.py](bootstrap_remote.py) で固定revisionのIdol重みを取得し、SHA256を検証する。既存の同名ファイルが想定と異なる場合は停止する。
2. [run_idol.py](run_idol.py) で20 steps、次に50 stepsを実行する。元グラフの変更をdenoiser、decoder、stepsだけに限定し、送信前にqueueが空であることを確認する。
3. `BICachedTripoSplat` のstage cacheからNPZを、`PreviewGaussianSplat` のhistoryと `/view` からPLYを回収する。実行グラフ、時間、VRAM、prepared RGBAを同時に記録する。
4. [prepare_common_frame.py](prepare_common_frame.py) に3つのNPZとY回転を渡し、共通範囲を作る。
5. [render_comparison.py](render_comparison.py) で各native動画、6面、Box archive、公式Inspectorを順に生成し、[encode_native_pair.py](encode_native_pair.py) で3列のnative比較動画を作る。
6. [capture_boxes.py](capture_boxes.py) で3つの公式Box viewerを順に撮影し、[encode_box_comparison.py](encode_box_comparison.py) で3列の比較動画を作る。全フレームのhash、順序、カメラ条件と、書き出したMP4の240フレーム・10秒を検証する。

推論のCLIは次のとおり。`--execute` がなければ送信せず、promptと事前記録の作成まで行う。

```bash
python run_idol.py --steps 20 --execute --url http://127.0.0.1:8198
python run_idol.py --steps 50 --execute --url http://127.0.0.1:8198
python run_idol.py --steps 20 --collect PROMPT_ID --url http://127.0.0.1:8198
```

Comfyは検証時にCFGの `3` を `3.0`、denoiseの `1` を `1.0` へ変換するため、cache照合にはhistoryの実行グラフを使う。初回のIdol 20は送信前グラフでのcache照合が失敗したが、生成自体は成功していた。修正後に同じprompt IDから回収し、再度推論することなくNPZとPLYを保存した。[回収処理](run_idol.py)

後半の処理ではoutputの `scripts/` にnative描画adapter、比較encoder、既存 `build_metric_box.py` を配置し、以前の `experimental_gaussian_depth.py` と `capture_inspector.mjs` を指定している。実際の絶対パス、引数、実行順序、各処理時間は [render-comparison.json](evidence/render-comparison.json) に残した。公式 `view.py` のSHA256は `ac5582e709b88c5d02aad2c999147237f077f3dc56850c8d8e139abaec947ada`。

## 成果物と記録

実験outputの `final/` に次のファイルを生成した。3Dモデルは各 `.inspector.html` で確認する。

| 種類 | ファイル名 |
|---|---|
| native比較動画 | `Idol_vs_TripoSplat_3DGS.mp4` |
| Box比較動画 | `Idol_vs_TripoSplat_Box512.mp4` |
| 4方向の比較画像 | 各動画名に `_000.png`、`_090.png`、`_180.png`、`_270.png` |
| 比較用Gaussian PLY | `TripoSplat_20.ply`、`Idol_20.ply`、`Idol_50.ply` |
| Box archive | 各モデル名に `_Box512.box.zip` |
| Gaussian viewer | 各モデル名に `_GS.inspector.html` |
| Box viewer | 各モデル名に `_Box512.inspector.html` |

入力・推論の原記録は `evidence/idol-20steps-seed46/` と `evidence/idol-50steps-seed46/`、生成・撮影の原記録は `evidence/render-comparison.json`、`evidence/media/native/`、`evidence/media/box/` に保存した。remoteから回収した各JSONはSHA256を照合しており、[測定値のsource一覧](evidence/measurements.json) と [生成記録のchecksum一覧](evidence/render-evidence-checksums.json) に対応を残している。
