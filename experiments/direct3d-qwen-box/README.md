# Direct3D-V3 1024からBoxを作りQwenでRGBを生成する実験

Direct3D-V3で作った形状をnative Boxへ変換し、各面のRGBをQwen Imageで生成する再現手順です。形状、depth、alphaを保持し、normalをRGB比較の前段で固定したうえで、元写真の見た目を6面の色へ反映します。

灰色のclayガイドへ直接着色する4候補は、塗り残しや内部の位置ずれがあり不採用にしました。[clayの評価記録](evidence/clay-evaluation.json)

その後、既存coarseモデルのUVテクスチャをDirect3Dの表面へ転写し、追加照明なしの色付きガイドをQwenで修正しました。全6面でdenoise 0.9の候補を選び、承認済みmeshとnormalの扱いをそろえたBoxを採用しました。色転写済みbaselineと採用版を同一cameraの240フレームで比較したところ、描画alphaは全画素で一致しました。[最終比較の記録](evidence/final-smooth-all6-090/summary.json)

評価画像は全身を含むフレーム全体を使います。6面を生成できたこと、面の境界で色が自然につながること、元のcoverageを保持したことは、それぞれ別に確認します。

## 採用した構成

採用したQwen入力は6面それぞれの `<slug>-prior-unlit-090` です。slugは `pz`、`px`、`nx`、`nz`、`py`、`ny` で、生成画像は1024×1024、貼り戻すBoxは各面512×512です。元のDirect3D proxyを保持し、比較に使うgray、色転写のみ、Qwen後の3つで同じnormal mapを使います。[6面の実行記録](evidence/qwen-selected-six-summary.json) [normalを固定した記録](evidence/smooth-normals/normal-revision-result.json)

| 項目 | 採用値 |
|---|---|
| asset ID | `095fb922d6ace93bd4ae2b8679d2be1d91c5ee79dd1a2ef70d5b1580272272ee` |
| manifest SHA-256 | `1ae379eb8b72a15c30d716c269c02b78352f0a1858b025c247efc5381c53eb13` |
| colorMap SHA-256 | `d8c45302c88ab5b4fda7de3abb7c2727c6c2b2dcfb17801eac4d5b236188cda3` |
| normalMap SHA-256 | `bfe952bb631c95666f8054a2940cf28a0f3c721680549ee4f0d53987e0ed7741` |
| 公式3D-Inspector HTML SHA-256 | `277075c56f8a2ec433b03607c140f2bdfeca19c2c63a2f38e5cdfd96e840a6e2` |

normalを変える前後で、最終6面のcolor atlasは同じbytesです。normalの変更による表示差は小さく、主な目的は承認済みmeshのnormal処理との整合です。[atlasの比較](evidence/smooth-normals/all6-smooth-native-comparison.json) [公式viewerの生成記録](evidence/smooth-normals/all6-smooth-viewer.json)

## 入力と保存先

形状入力はDirect3D-V3の1024出力から作った、Y-up、正面+Zのcanonical GLBです。GLBは形状のみを持ちます。この実験では追加の中心合わせ、拡大縮小、回転を加えません。

今回の入力記録は実行ディレクトリの `prepared/prepared.json` にあります。

| 入力 | 保存先 | SHA-256 |
|---|---|---|
| Direct3Dの形状 | `$RUN_DIR/inputs/direct3d-1024.glb` | `d7f5e2eb289a4565815782165486485cb9a161094695c1786141c81aea6064ee` |
| 元写真、1024×1024 | `$RUN_DIR/inputs/source-original.png` | `9d92ad97b34bb949b3aa555afed321b8dae427c0102c862b2afb96745b93390b` |
| 色の転写元となるcoarse GLB | `$COARSE_GLB` | `8aac16b2d7125bc41f76990eb414c46dbd2e5d664f89267791d0151a3113a0b3` |
| Qwenの元API graph | `$SCRIPTS/qwen_success_template.json` | `7c3d96f0ea0b7485f130ecd847cfe1bd78e4a09effa4864a10700ffec2c04bce` |

形状の元ファイルは次です。実行時は同じbytesを入力用ディレクトリへコピーして使います。

```text
/mnt/f/_dev/box-impostor-lab/experiments/direct3d-v3-20261121/rgb1024/canonical.glb
```

coarse GLBは既存のPixal3D出力で、49,522頂点、81,180三角形とUVテクスチャを持ちます。このファイルは読み取り元として参照し、再コピーしません。入力のSHAとscene変換は [color-priorの記録](evidence/color-prior/result.json) にあります。

```text
/mnt/f/_dev/box-impostor-lab/experiments/direct3d-v3-20261121/inputs/fullbody/coarse-original.glb
```

実験の保存先は次です。

```text
/mnt/f/_dev/box-impostor-lab/experiments/direct3d-qwen-box-20261121
```

再実行時は新しい保存先を指定します。以下のコマンドのパスは使用する環境の絶対パスに置き換えてください。

```bash
RUN_DIR=/absolute/path/to/new-run
FACTORY_ROOT=/absolute/path/to/box_impostor_factory
SCRIPTS=/absolute/path/to/this-experiment
CANONICAL_GLB=/absolute/path/to/canonical.glb
ORIGINAL_PHOTO=/absolute/path/to/source-original.png
COARSE_GLB=/absolute/path/to/coarse-original.glb
EXPERIMENT_PYTHON=/absolute/path/to/python
INSPECTOR_BUILD_SCRIPT=/absolute/path/to/3d-inspector/scripts/build_viewer.py
COMFY_INPUT=/absolute/path/to/ComfyUI/input
COMFY_SERVER=http://127.0.0.1:8198
COMFY_INPUT_PREFIX=direct3d-qwen-new-run
CHROMIUM=/absolute/path/to/chrome
```

`COMFY_INPUT_PREFIX` はComfyUIのinput内で使う、この実行専用の相対パスです。以前の候補と重ならない名前にしてください。

## 実行環境

CPU側のPythonスクリプトにはPython 3.11以降、NumPy、Pillow、SciPyを使います。coarseからの色転写にはtrimeshとrtreeも必要です。今回の色転写はPython 3.12.3、trimesh 5.2.1で実行し、rtree 1.4.1だけを実験用ディレクトリへ追加しました。共有Python環境の依存は変更していません。各ライブラリの実際のバージョンはcolor-priorのQA JSONに記録しています。[色転写のQA](evidence/color-prior/baseline512/qa/color-prior-13dcf86d0f78aab1.json)

QwenのクライアントはPillowと標準ライブラリを使い、推論は接続先のComfyUIが実行します。[select_guide_rgb.py](select_guide_rgb.py) は標準ライブラリだけで動作します。

native bakeにはFactoryの `worker/cli.mjs` と、その `package-lock.json` に対応するNode.js、Playwright、Chromiumが必要です。この実験ではフルChromium 1254を明示し、WebGL2とSwiftShaderでbakeしました。`FACTORY_ROOT` から既存の `node_modules` を参照できる状態にしてください。

ComfyUIには次のモデルとAPI graphのノードが必要です。クライアントはこれらをダウンロードしません。

| 用途 | モデルまたはノード |
|---|---|
| Qwen Image | `qwen_image_2.1_int8_convrot.safetensors` |
| text encoder | `qwen3vl_8b_int8_convrot.safetensors` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` |
| AnyAngle LoRA | `QI2.1_AnyAngle.safetensors`。既定のstrengthは0 |
| Qwen固有ノード | `QwenImage21Cache`、`TextEncodeQwenImage21` |
| 出力mask | `RemoveBackground` を使うBiRefNetの処理と `MaskToImage` |

API graphにはLoRA loaderを残しています。strengthが0でも、graphが参照するモデル名とノードをComfyUI側で扱える必要があります。モデル、ノード、入力順序の条件は [run_qwen_face.py](run_qwen_face.py) の `graph_from_success()` と `prepare()` を参照してください。

## 1. 入力を固定する

新しい実行ディレクトリを作り、形状と写真をコピーします。既存の実行ディレクトリに上書きしないよう、ここでは `mkdir -p` を使っていません。

```bash
mkdir "$RUN_DIR"
mkdir "$RUN_DIR/inputs"
cp "$CANONICAL_GLB" "$RUN_DIR/inputs/direct3d-1024.glb"
cp "$ORIGINAL_PHOTO" "$RUN_DIR/inputs/source-original.png"
sha256sum "$RUN_DIR/inputs/direct3d-1024.glb" "$RUN_DIR/inputs/source-original.png"
sha256sum "$COARSE_GLB"
```

SHAを入力表と照合します。今回のrunnerは元写真とAPI graphのSHAを検査するため、別の被写体や別のgraphをそのまま代入する汎用ツールにはしていません。

## 2. identityのままnative Boxへ変換する

Factoryのbakeは `--canonical-meta` を省略するとcanonical化を行います。すでにcanonicalのDirect3D GLBにはidentityのmetadataを渡し、元GLBのbytesと座標を保ちます。

`$RUN_DIR/identity-canonical.json` に次を保存します。

```json
{
  "glbSha256": "d7f5e2eb289a4565815782165486485cb9a161094695c1786141c81aea6064ee",
  "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  "inverseTransform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
}
```

各面512×512でbakeします。下の実行はソフトウェア描画を使います。

```bash
env -u DISPLAY -u WAYLAND_DISPLAY \
BI_SOFTWARE_GPU=1 BI_CHROMIUM="$CHROMIUM" \
node "$FACTORY_ROOT/worker/cli.mjs" bake \
  --input "$RUN_DIR/inputs/direct3d-1024.glb" \
  --output "$RUN_DIR/native512" \
  --resolution 512 \
  --backend webgl2 \
  --canonical-meta "$RUN_DIR/identity-canonical.json"
```

出力にはGLBのproxy、color、normal、interval、range、6面のguide PNGとdepth sidecarが含まれます。通常のbakerが作るboundsとcameraを使い、各面のfrustum、view matrix、projection matrixをmanifestに保存します。bake boxはmeshのboundsに最大辺の2%を各方向へ加えた範囲です。

今回の `native512/manifest.json` のSHAは `f7c86b4591cddea531560e03e749cceb9a1173808ddb49c6a12775b1bf0a4445` です。metadataの内容や実装の変更でmanifestのSHAは変わるため、別実行のmanifestはその実行の入力・設定とともに記録します。

## 3. グレーのBoxと6面ガイドを作る

[prepare_mesh_box.py](prepare_mesh_box.py) はnative bakeから色の基準となるBoxとQwenの入力画像を作ります。

```bash
"$EXPERIMENT_PYTHON" "$SCRIPTS/prepare_mesh_box.py" \
  --native-manifest "$RUN_DIR/native512/manifest.json" \
  --mesh "$RUN_DIR/inputs/direct3d-1024.glb" \
  --reference "$RUN_DIR/inputs/source-original.png" \
  --output "$RUN_DIR/prepared"
```

`prepared/baseline512` はcolor atlasのRGBをsRGB 128のグレーに変更したBoxです。RGBはlinear-sRGBへ変換し、元のalphaを掛けてRGBA16Fへ保存します。alphaはuint16のbit列で保持し、normal、interval、range、depth、proxy、cameraも保持します。元のnative guide PNGは変更せず、Qwen用の画像を別に保存します。

今回の `prepared/baseline512/manifest.json` のSHAは `0fcd3e3d57f75aa35f8ec07af8541dca0c0dfe56167b49fef65f373533fd88ba` です。入力・出力ファイルのSHAと構造の比較は `prepared/prepared.json` に記録しています。

### 面と投影

| face | ディレクトリ | カメラの位置する側 | camera up |
|---|---|---|---|
| +X | `px` | +Xから−Xを見る | +Y |
| −X | `nx` | −Xから+Xを見る | +Y |
| +Y | `py` | +Yから下を見る | −Z |
| −Y | `ny` | −Yから上を見る | +Z |
| +Z | `pz` | +Zから正面を見る | +Y |
| −Z | `nz` | −Zから背面を見る | +Y |

Boxの各tileは512×512ですが、物理的なfrustumは長方形です。ガイドでは縦横の実寸比を保って1024×1024へ配置し、長辺の両端に2%ずつ余白を加えます。上下の面もその面から実際に見える面を使います。隠れた手足や顔を表示するための加工はありません。

物理的な幅を `W`、高さを `H`、ガイドの辺を `S`、nativeの解像度を `N` とすると、正方形の物理的な辺 `L` とx方向の変換は次です。y方向も `W` を `H` に置き換えて計算します。

```text
L = max(W, H) * 1.04
sx = S * W / (L * N)
tx = (S - S * W / L) / 2 + sx / 2 - 0.5
x_square = sx * x_native + tx
```

座標はtop-downで、0始まりの画素中心です。正逆のaffineを `guide.json` の `nativeToSquarePixel` と `squareToNativePixel` に保存します。atlasの各tileは上下反転してtop-downへ読み出します。native depth sidecarはすでにtop-downです。

### RGBとdepth

各面の `prepared/guides/<slug>/` に次を保存します。

| ファイル | 内容 |
|---|---|
| `rgb.png` | 1024×1024のRGB。元のnormalから陰影を計算したclay画像 |
| `alpha.npy`、`alpha.png` | native coverageを同じaffineで配置した1024×1024のalpha |
| `depth.npy` | 1024×1024、float32。0が手前、1が奥、背景はNaN |
| `depth.png` | 1024×1024のdepth表示画像 |
| `depth-512.png` | Qwenの3枚目に渡す512×512のdepth表示画像 |
| `guide.json` | camera、座標変換、入力・出力SHA、normalとdepthの参照元 |

clayの陰影はguideだけに加えます。native world normalをdecodeし、Lambertの項にambient 0.35とkey intensity 0.8を使います。光の向きはカメラ座標の `[-0.35, 0.5, 0.85]` を正規化した値です。guideのalbedoはsRGB 168、背景は128です。Box本体のグレーのalbedoにはこの陰影を加えません。

depth表示画像は `(1 - depth) * coverage` を使い、近い面ほど明るく、背景を黒にします。alpha、RGB、depthは同じ投影で配置します。元の512画像を1024へ補間するため、guideの解像度を上げても形状の細部が新しく増えるわけではありません。

## 4. coarseの色を転写して照明なしガイドを作る

[prepare_color_prior.py](prepare_color_prior.py) は、Direct3Dのnative depthとcameraから復元した表面点に、coarse GLBの色を転写します。Direct3Dの位置、三角形、proxy、depth、normal、alpha、cameraは保持し、6面のRGBだけを更新した新しいbaselineを作ります。

### 実験用の依存と実行

既存のPython環境にNumPy、Pillow、SciPy、trimeshを用意したうえで、rtreeを今回の実行ディレクトリへ追加します。以下はこの実験で使ったバージョンです。

```bash
"$EXPERIMENT_PYTHON" -m pip install \
  --target "$RUN_DIR/python-deps-rtree" \
  "rtree==1.4.1"

PYTHONPATH="$RUN_DIR/python-deps-rtree" \
"$EXPERIMENT_PYTHON" "$SCRIPTS/prepare_color_prior.py" \
  --baseline-manifest "$RUN_DIR/prepared/baseline512/manifest.json" \
  --guides "$RUN_DIR/prepared/guides" \
  --source-glb "$COARSE_GLB" \
  --output "$RUN_DIR/color-prior"
```

`--chunk-size` の既定値は2048、元のDirect3D表面との照合に使う `--validation-samples` は各面256です。出力先は新しいディレクトリに限ります。モデル推論やmeshの再bakeは行いません。

主な出力は次のとおりです。

| 出力 | 内容 |
|---|---|
| `color-prior/baseline512/manifest.json` | 6面の色を転写したBox。albedoに追加照明を含めない |
| `color-prior/baseline512/qa/*.json` | 入力SHA、座標の照合、表面距離、UVの扱い、構造の比較 |
| `color-prior/guides/<slug>/albedo.png` | 転写した色を元と同じ1024×1024の投影へ配置した画像 |
| `color-prior/guides/<slug>/rgb.png` | 同じ色に、clayガイドと同じLambertの陰影を加えた画像 |
| `color-prior/guides/<slug>/guide.json` | 新baselineへの参照、元guideのSHA、色転写の記録 |
| `color-prior/result.json` | 全6面の出力と構造比較をまとめた記録 |

今回の色付きbaselineのmanifest SHAは `e6e4dc0039029141fddc6655c9670a20dd6955b79bfb21bc114d2db601c13e50` です。[出力記録](evidence/color-prior/result.json)

### 表面座標とUV

Direct3Dのnative Boxはすでにcanonical座標です。coarse GLBはscene内のworld変換を一度だけ適用すると同じ座標系になります。頂点やUVの継ぎ目をmergeせず、読み込み後の頂点順・三角形順を元GLBと照合します。復元したDirect3Dの表面点へcoarse側の変換をもう一度掛ける処理はありません。

nativeのdepthはbake boxの各面から軸方向に測った距離を、その軸の長さで正規化した値です。cameraのnear/farで正規化した値として扱うと、表面位置が変わります。各面のtop-down画素中心と保存済みview matrixから、次のcamera座標を復元します。view/projection matrixの16値はThree.jsのcolumn-major順で読みます。

```text
x_cam = left + (i + 0.5) / 512 * (right - left)
y_cam = top - (j + 0.5) / 512 * (top - bottom)
z_cam = -(camera_margin + native_depth * box_axis_span)
world = inverse(view_matrix) * [x_cam, y_cam, z_cam, 1]
```

復元点に最も近いcoarseの三角形を探し、その三角形上の最近点でbarycentric座標を求めます。三角形の3頂点のUVをその重みで補間し、baseColor textureから色を取得します。depthが有限な元alpha>0.5の画素を転写対象とし、depthを持たない境界画素は元のRGBとalphaを保持します。

今回のGLBでは、trimeshが読み込んだUVのvは元glTFの `1-v` です。全頂点のUVでこの対応を照合し、画像の画素中心を `x=u*W-0.5`、`y=(1-v)*H-0.5` としてsampleします。画像を別途上下反転しません。元のsamplerに合わせ、LINEARで補間してCLAMP_TO_EDGEで端を処理します。

テクスチャのRGBをsRGBからlinearへ変換してからbilinear補間し、元GLBのlinearな `baseColorFactor` を掛けます。今回のfactorは `[1,1,1,1]` です。metallic/roughness、AO、normal textureは色転写に使いません。RGBA16Fへ保存するときは、取得したlinear RGBに元のalphaを掛けます。[実装](prepare_color_prior.py)

表面座標は別のplane式による復元、元の画素中心への再投影、軸方向depthへの逆変換でも照合します。Direct3D表面との距離、coarse表面との距離分布、元normalと転写元三角形normalの内積もQAに保存します。距離やnormalによる色の除外は行っていないため、近接する別部位から色を取得する可能性は画像で確認します。[色転写のQA](evidence/color-prior/baseline512/qa/color-prior-13dcf86d0f78aab1.json)

色付きbaselineは全atlasのalphaをuint16のbit列で保持し、depth、normal、interval、range、proxy、cameraを保持します。manifestの変更は色のSHAとasset ID、parent、mode、QA参照に限ります。guideのalphaとdepthファイルも、元のgray guideと同じbytesを使います。

### Qwen用に照明なしのRGBを選ぶ

Qwenの出力をBoxのalbedoへ貼るため、追加したLambertの陰影まで色へ焼き込むことを避け、今回の推論には `albedo.png` を使います。元のguideを変更せず、[select_guide_rgb.py](select_guide_rgb.py) で別の6面guideを作ります。

```bash
"$EXPERIMENT_PYTHON" "$SCRIPTS/select_guide_rgb.py" \
  --guides "$RUN_DIR/color-prior/guides" \
  --output "$RUN_DIR/color-prior/guides-unlit" \
  --source-key albedo
```

新しい `guides-unlit/<slug>/rgb.png` は元の `albedo.png` と同じbytesです。baseline SHA、投影、camera、構造情報、alpha、depthはそのままにし、元guideと選んだ画像のpath・SHAを記録します。`shading.applied=false` は追加照明がないことを示し、元テクスチャに含まれる明暗を除去したことは示しません。[guideの選択記録](evidence/color-prior/guides-unlit/result.json)

## 5. Qwenで1面ずつ生成する

[run_qwen_face.py](run_qwen_face.py) は1面分の候補を作ります。今回の色付きguideの入力と、runnerの既定値を使う設定は次のとおりです。

| 項目 | 入力・設定 |
|---|---|
| profile | `anchored` |
| 画像1 | 対象面の `guides-unlit/<slug>/rgb.png`、1024×1024 |
| 画像2 | 元写真、1024×1024 |
| 画像3 | 対象面のdepth表示画像、512×512 |
| initial latent | 画像1をVAEEncodeしたlatent |
| denoise | 0.9 |
| AnyAngle LoRA strength | 0 |
| steps / CFG | 10 / 3 |
| sampler / scheduler | `euler` / `simple` |
| seed | `2026113031` |
| Qwen cache | CPU、int8 |
| text encoderのresolution指定 | 0。各入力画像の解像度を保つ |

depthは3枚目の画像としてQwenへ渡します。このgraphにdepth用ControlNetはありません。`alpha.npy` は位置合わせと評価に使い、Qwenの画像入力にもBox alphaの置換にも使いません。

正面の入力とgraphを準備する例です。今回の色付きguide向けpromptは、runnerの既定promptから変更しています。同梱した正面候補の `result.json` から実際に使ったpromptを取り出します。候補の記録がある別の面を再現するときも、その面のresultに保存されたpromptを使ってください。

```bash
FACE=pz
CASE=pz-prior-unlit-090
"$EXPERIMENT_PYTHON" - \
  "$SCRIPTS/evidence/qwen/$CASE/result.json" \
  "$RUN_DIR/$CASE.prompt.txt" <<'PY'
import json
import sys
from pathlib import Path
source = json.loads(Path(sys.argv[1]).read_text())
with Path(sys.argv[2]).open("x") as stream:
    stream.write(source["prompt"] + "\n")
PY
```

`--execute` を付けない次の実行は入力とgraphの準備だけを行います。

```bash
"$EXPERIMENT_PYTHON" "$SCRIPTS/run_qwen_face.py" \
  --face "$FACE" \
  --target-rgb "$RUN_DIR/color-prior/guides-unlit/$FACE/rgb.png" \
  --target-depth "$RUN_DIR/color-prior/guides-unlit/$FACE/depth-512.png" \
  --target-alpha "$RUN_DIR/color-prior/guides-unlit/$FACE/alpha.npy" \
  --box-manifest "$RUN_DIR/color-prior/baseline512/manifest.json" \
  --original "$RUN_DIR/inputs/source-original.png" \
  --template "$SCRIPTS/qwen_success_template.json" \
  --prompt-file "$RUN_DIR/$CASE.prompt.txt" \
  --profile anchored \
  --denoise 0.9 \
  --lora-strength 0 \
  --comfy-input "$COMFY_INPUT" \
  --input-prefix "$COMFY_INPUT_PREFIX/$FACE/$CASE" \
  --server "$COMFY_SERVER" \
  --output "$RUN_DIR/qwen/$CASE"
```

`prepared.json` と `prompt.json` を確認したあと、同じコマンドに `--execute` を付けて1件投稿します。ComfyUIのqueueが空の場合に実行します。投稿後の再収集は同じoutputとfaceに `--collect PROMPT_ID` を指定し、同じ候補を再投稿しません。

```bash
"$EXPERIMENT_PYTHON" "$SCRIPTS/run_qwen_face.py" \
  --face pz \
  --server "$COMFY_SERVER" \
  --output "$RUN_DIR/qwen/pz-prior-unlit-090" \
  --collect PROMPT_ID
```

正面を確認したあと、`FACE` を `px`、`nx`、`nz`、`py`、`ny` に変え、面に対応するpromptと新しい `CASE` を使って候補を順に作ります。設定を変えて試す場合も、outputと入力prefixを分けます。`--prompt-file` で変更したpromptは本文とSHAを保存します。

色付きguide向けpromptは、画像1のcamera、輪郭、姿勢、材質の領域、顔や靴の位置を維持し、画像2から見た目を参照する指示です。背面では顔を追加せず、上面・下面では通常の立ち姿へ描き直さないよう指定しています。この指示だけで投影や姿勢が保たれるとは限らないため、生成後の全身画像で確認します。

保存される主な記録は `submitted.json`、`history.json`、`execution-prompt.json`、`generated.png`、`mask.png`、`result.json` です。出力maskはBiRefNetが生成画像から求めたもので、native coverageとは別のデータです。

## 6. RGB比較に使うnormalを固定する

元のDirect3D GLBはNORMAL属性を持ちません。公式3D-Inspectorのmesh表示は頂点normalを求めますが、最初のnative bakeは面のnormalを使いました。採用版では別GLBに同じ計算方法のNORMAL属性を追加してbakeし、そのnormalMapをRGB比較の前に固定します。頂点位置や三角形を平滑化する処理ではありません。[NORMAL追加の記録](evidence/smooth-normals/prepare-verification.json)

### NORMAL属性を追加した別GLBをbakeする

[prepare_smooth_normals.py](prepare_smooth_normals.py) は、元のPOSITION、indices、sceneのJSON、元binaryの内容を保ったまま、角度で重みを付けた頂点normalを追加します。元GLBは変更せず、新しいファイルとidentity metadataを作ります。

```bash
"$EXPERIMENT_PYTHON" "$SCRIPTS/prepare_smooth_normals.py" prepare \
  --input "$RUN_DIR/inputs/direct3d-1024.glb" \
  --identity-meta "$RUN_DIR/identity-canonical.json" \
  --inspector-script "$INSPECTOR_BUILD_SCRIPT" \
  --output-root "$RUN_DIR/smooth-normals"

env -u DISPLAY -u WAYLAND_DISPLAY \
BI_SOFTWARE_GPU=1 BI_CHROMIUM="$CHROMIUM" \
node "$FACTORY_ROOT/worker/cli.mjs" bake \
  --input "$RUN_DIR/smooth-normals/canonical.smooth-normals.glb" \
  --output "$RUN_DIR/smooth-normals/native512" \
  --resolution 512 \
  --backend webgl2 \
  --canonical-meta "$RUN_DIR/smooth-normals/identity-canonical.json"

"$EXPERIMENT_PYTHON" "$SCRIPTS/prepare_smooth_normals.py" compare-bakes \
  --before "$RUN_DIR/native512" \
  --after "$RUN_DIR/smooth-normals/native512" \
  --output "$RUN_DIR/smooth-normals/native-comparison.json"
```

今回のbakeではcolor、color/normalのalpha、6面depth、interval、range、camera、boundsがすべて元と一致しました。normalMapのRGBだけが552,462画素で変わっています。この新しいGLBはnormalを取得するために使い、最終Boxのproxyは元のDirect3D GLBのbytesを保ちます。[native bakeの比較](evidence/smooth-normals/native-comparison.json)

### 比較用baselineとguideの参照をそろえる

[apply_normal_revision.py](apply_normal_revision.py) は、grayと色転写済みbaselineへ同じnormalMapを反映した新しいassetを作ります。proxy、color、depth、interval、rangeなどの既存データファイルとnormalMapのalphaを保持し、manifestはnormalのSHA、asset ID、parent、mode、QA参照だけを更新します。

```bash
"$EXPERIMENT_PYTHON" "$SCRIPTS/apply_normal_revision.py" \
  --gray "$RUN_DIR/prepared/baseline512" \
  --prior "$RUN_DIR/color-prior/baseline512" \
  --smooth-native "$RUN_DIR/smooth-normals/native512" \
  --guides "$RUN_DIR/color-prior/guides-unlit" \
  --output-root "$RUN_DIR/smooth-normals"
```

比較に使うgrayは `smooth-normals/baselines/gray`、Qwenを貼る前の色付きbaselineは `smooth-normals/baselines/prior` です。貼り戻し用guideは `smooth-normals/guides-unlit` を使います。guideのbaseline参照とnormalのSHAを更新し、参照変更の記録を追加します。

追加照明なしのguideはnormalに依存しないため、各面7ファイルの画像・NPYを再生成して元bytesとの一致を照合しています。Qwenへ渡したRGBとdepthのSHAも変わりません。したがって、このnormalだけの変更についてQwenを再実行する必要はなく、元のQwen resultやpromptも書き換えません。[normal変更とguideの記録](evidence/smooth-normals/normal-revision-result.json)

この前段のあと、色付きbaselineとQwen合成後のnormalを同じに保ちます。gray、色転写のみ、Qwen後の見た目を比較するときも、3つのnormalMapをそろえます。normal変更による差と、QwenのRGBによる差を同じ前後比較へ混ぜません。

## 7. 選んだ候補のRGBを貼り戻す

[compose_faces.py](compose_faces.py) は、明示的に選んだ1〜6面の候補を同じbaselineへ合成します。面ごとの `result.json` と `guide.json` を指定します。以下は全6面を選ぶときの入力形式です。

```json
{
  "faces": [
    {"face": "+X", "result": "qwen/px-prior-unlit-090/result.json", "guide": "smooth-normals/guides-unlit/px/guide.json"},
    {"face": "-X", "result": "qwen/nx-prior-unlit-090/result.json", "guide": "smooth-normals/guides-unlit/nx/guide.json"},
    {"face": "+Y", "result": "qwen/py-prior-unlit-090/result.json", "guide": "smooth-normals/guides-unlit/py/guide.json"},
    {"face": "-Y", "result": "qwen/ny-prior-unlit-090/result.json", "guide": "smooth-normals/guides-unlit/ny/guide.json"},
    {"face": "+Z", "result": "qwen/pz-prior-unlit-090/result.json", "guide": "smooth-normals/guides-unlit/pz/guide.json"},
    {"face": "-Z", "result": "qwen/nz-prior-unlit-090/result.json", "guide": "smooth-normals/guides-unlit/nz/guide.json"}
  ]
}
```

この6候補が今回選んだ入力です。`$RUN_DIR/selection.json` に保存した場合、相対パスはそのファイルのあるディレクトリから解決します。`reviewNote` と `visualReviewStatus` を各項目へ追加して確認内容を残せます。

```bash
"$EXPERIMENT_PYTHON" "$SCRIPTS/compose_faces.py" \
  --baseline-asset "$RUN_DIR/smooth-normals/baselines/prior/manifest.json" \
  --selection "$RUN_DIR/selection.json" \
  --output "$RUN_DIR/variants/all6-smooth-prior-unlit-090"
```

生成maskとguide alphaを使い、対象全体に1つのaffineを当てて位置を合わせます。XY scaleは0.96〜1.06、回転は±3度、平行移動は画像の辺の±1/64です。1024画像では移動幅は±16 pxです。identityと最適化後を比較し、IoUが高い方を使います。顔crop、局所warp、inpaintingはありません。IoUの既定参照値0.88は診断用で、これを下回るだけでは比較用のvariant作成を中断しません。

RGBはsRGBからlinearへ変換して生成maskを掛け、画素中心の変換とaffineの逆変換を合わせてsampleします。nativeの各画素で、guide alphaに対する生成maskの割合を重み `w` とします。

```text
RGB_out = (1 - w) * RGB_old + w * Qwen_RGB_linear * alpha_old
alpha_out = alpha_old のuint16 bit列
```

`RGB_old` はalphaを掛けた保存値です。生成maskの支援が弱い領域では、転写済みの色を大きな割合で残します。貼り戻しは選択した面のRGBだけを変更し、選択していない面、全atlasのalpha、depth、normal、interval、range、proxyとcameraを保持します。入力のSHAが変わっていないことも確認します。

出力は新しいassetディレクトリ、検査記録のJSON、`.box.zip` です。既存のbaselineや以前のvariantへ上書きしません。元のguide PNGはそのままなので、変更後の見た目はcolor atlasを描画する3D-Inspectorで確認します。

## 8. 同じcameraで比較する

baselineとvariantの `.box.zip` を公式3D-Inspectorの `view.py` でHTMLへ変換します。生成されたHTMLを変更せず、既定表示を使います。

[compare_boxes.py](compare_boxes.py) はその2つのviewerを同じreferenceで撮影し、構造の保存と描画alphaを比較します。`BASELINE_VIEWER` は `smooth-normals/baselines/prior`、`VARIANT_VIEWER` はそのbaselineから合成した採用版を表示するHTMLです。実行例の `VARIANT_RECEIPT` は合成時に出力されたasset ID付きのJSONです。

```bash
"$EXPERIMENT_PYTHON" "$SCRIPTS/compare_boxes.py" \
  --baseline-asset "$RUN_DIR/smooth-normals/baselines/prior/manifest.json" \
  --receipt "$VARIANT_RECEIPT" \
  --before-viewer "$BASELINE_VIEWER" \
  --after-viewer "$VARIANT_VIEWER" \
  --reference "$RUN_DIR/prepared/baseline512/manifest.json" \
  --capture-script "$SCRIPTS/capture_inspector.mjs" \
  --alpha-script "$SCRIPTS/compare_capture_alpha.py" \
  --playwright-root "$FACTORY_ROOT" \
  --chromium "$CHROMIUM" \
  --output "$RUN_DIR/comparison"
```

比較には同梱した `capture_inspector.mjs` と `compare_capture_alpha.py` を指定します。比較helperの既定パスはこの実験機の保存先を指すため、別環境では明示してください。撮影のreferenceは元のgray baselineにそろえ、gray、色転写、Qwenの比較で画角を変えません。既存の撮影を使う場合は `--before-capture` と `--after-capture` を指定できます。viewer、asset、reference、撮影設定、SHAの対応を検査します。

撮影は512×512、正投影、pitch 0度、yaw 1.5度刻みの240フレームです。比較helperは記録されたcameraが全フレームで完全一致することを要求します。PNGのalphaをuint8で比較し、全画素のMAE、最大差、差のある画素数を記録します。RGBの変化は別に集計します。

動画を作る場合は `--ffmpeg` と `--encode-script "$SCRIPTS/encode_comparison.py"` を指定します。24 fps、10秒は動画の再生設定で、描画性能のFPSではありません。比較画像は全身を含むフレーム全体を使います。

## 測定の意味

| 記録 | 読み方 |
|---|---|
| `prepared/prepared.json` | 入力、baseline、6面guideのSHAと、native構造を保持した比較記録 |
| `color-prior/result.json` とQA | 座標復元、coarseとの表面距離、UV/sampler、RGB以外の構造を保持した比較記録 |
| `color-prior/guides-unlit/result.json` | 追加照明なしのRGBを選んだ記録と、元guideのalpha/depth/cameraの保持 |
| Qwenの `serverExecutionSeconds` | ComfyUIの `execution_start` から `execution_success` まで。graph全体の時間 |
| Qwenの `clientWaitSeconds` | そのクライアント起動で待機した時間。`--collect` で再収集した場合は再収集時の待機時間 |
| `gpu-memory-summary.json` | 200 ms間隔のnvidia-smiによるGPU全体の使用VRAM。driverや他processを含み、短いpeakを見逃す場合がある |
| 合成のface別QA | 位置合わせ前後のIoU、変更RGB画素数、元の色を残した画素数、構造のSHA比較 |
| `alpha-240.json` | 同じcamera・同じPNG条件での描画alphaとRGBの差 |
| 撮影の `captureElapsedSeconds` | フレーム保存を含むoffline撮影時間 |
| 正面pilotの `supportedForegroundFraction` | 元alpha>0.5の画素のうち、実際のQwen合成重みが0.5以上の画素の割合 |
| 正面pilotの `fallbackForegroundPixels` | 同じ前景のうち、Qwen合成重みが0.5未満の画素数。RGBが全く変わらなかった画素数ではない |

Qwenの実行時間には、その回で実行されたモデル読み込み、生成、VAE処理、mask作成、保存などが含まれます。cacheの有無も記録から確認します。純粋なsampler時間として扱いません。VRAMはPyTorch allocatorの最大確保量とは異なる測定です。

## 6面の実行時間とVRAM

選んだ6候補の値は次のとおりです。共通設定は1024×1024、10 steps、CFG 3、denoise 0.9、AnyAngle strength 0です。各回に再利用されたnodeの記録がありますが、入力画像とpromptは面ごとに異なります。[集計と各resultのSHA](evidence/qwen-selected-six-summary.json)

| 面 | 候補 | ComfyUI graphの実行時間 | GPU全体の最大使用VRAMのsample |
|---|---|---:|---:|
| +Z | `pz-prior-unlit-090` | 63.465秒 | 13,732 MiB |
| +X | `px-prior-unlit-090` | 63.021秒 | 13,721 MiB |
| −X | `nx-prior-unlit-090` | 63.852秒 | 13,720 MiB |
| −Z | `nz-prior-unlit-090` | 63.081秒 | 13,720 MiB |
| +Y | `py-prior-unlit-090` | 62.775秒 | 13,720 MiB |
| −Y | `ny-prior-unlit-090` | 63.556秒 | 13,720 MiB |

6回のgraph実行時間の合計は379.750秒です。この合計には不採用試行、0.75の色付きguide試行、native bake、色転写、レビュー、RGB合成、撮影を含めていません。実験全体の所要時間ではありません。

初回の不採用clay試行はcoldなモデル・cache状態で実行し、`cachedNodes=[]` でした。選択した6回とは分けて記録します。入力やpromptも異なるため、同じ入力によるcold/warmの厳密な比較にはなりません。

| 初回の別試行 | ComfyUI graphの実行時間 | GPU全体の最大使用VRAMのsample | 再利用node |
|---|---:|---:|---|
| `pz-clay-090`、不採用 | 269.941秒 | 13,697 MiB | なし |

いずれもVRAMは200 ms間隔で観測したGPU全体の使用量です。driverや他processを含み、短いpeakを見逃す可能性があります。各面の値を残し、すべての方向が同じ負荷だったとは扱いません。

## 確認できた試行

### clayへ直接着色した4候補

灰色guideから試した次の4候補はすべて不採用です。`pz-materials-090` では全身に色が入りましたが、頭の向き、髪、靴の位置が変わりました。denoiseを下げた `pz-materials-075` は多くの部位が灰色のままで、色の完成度と内部の位置対応を両立できませんでした。[clayの評価記録](evidence/clay-evaluation.json)

| 候補 | 面 | denoise | 実画像での判断 | 採否 |
|---|---|---:|---|---|
| `pz-clay-090` | +Z | 0.9 | 顔と髪のみ着色。服、手、脚、靴がclayのまま | 不採用 |
| `pz-materials-090` | +Z | 0.9 | 全身の材質は着色。頭向き・髪位置がずれ、奥側のかかとが短くなる | 不採用 |
| `pz-materials-075` | +Z | 0.75 | 靴に色が入るが、身体の大部分が灰色のまま | 不採用 |
| `px-materials-090` | +X | 0.9 | 前腕、脚、靴は着色。髪と背中上部にclayが残る | 不採用 |

初回 `pz-clay-090` の `serverExecutionSeconds` は269.941秒、GPU全体の最大使用VRAMのsampleは13697 MiB、`cachedNodes=[]` でした。モデルの読み込みなどを含むため、この1回をwarm後の定常時間として扱いません。外観を確認した `generated.png` のSHAは `fd6bf9535b92d86316f357c257407f0d3de49540c07a1177ac5bf52f23a98935` です。[実行記録](evidence/qwen/pz-clay-090/result.json)

### 色付きguideを使った正面の試験

coarseの色を転写したguideには、目元の大きな黒い部分や靴の粗い色が残っていました。追加照明なしのguideで試した `pz-prior-unlit-075` は目元の黒い部分を残しました。denoiseを0.9にした `pz-prior-unlit-090` ではその欠陥が改善し、正面の全身画像とBoxの8視点表示を確認して仮採用としました。

正面だけを貼り戻したpilotの値は次のとおりです。支援率は元の+Zでalpha>0.5となる87,326画素を分母とし、実際のQwen合成重みが0.5以上の画素の割合を求めています。

| 正面候補 | 位置合わせ後のmask IoU | Qwenの支援率 | 支援が弱い画素数 |
|---|---:|---:|---:|
| clayの `pz-materials-090` | 0.94938555 | 96.3505% | 3,187 |
| 色付きguideの `pz-prior-unlit-090` | 0.9920252614 | 99.7114% | 252 |

出典は [clay正面pilot](evidence/pilots/front-clay-pilot.json) と [色付きguideの正面pilot](evidence/pilots/front-prior-pilot.json) です。支援が弱い画素数は `fallbackForegroundPixels` で、合成重みが0.5未満の画素を数えています。色付きbaselineでは、この領域に転写済みRGBが大きな割合で残ります。灰色guide、色付きguide、promptが異なるため、この比較を単一条件だけの効果とは扱いません。

このpilotでQwenのRGBを更新した面は+Zだけです。残る5面はclay試験ではgray、色付きguide試験では転写済みbaselineの色を保持しています。輪郭の重なりと支援率の改善は目・口・指などの内部位置や6面の継ぎ目の正しさを証明しません。画面下15%の支援統計も靴を意味的に切り出した値ではありません。

### 6面の外観とnormalの選択

6面のRGBを更新したflat-normal候補は、水平8方向の全身画像で色転写済みbaselineと比較しました。目元の黒い部分と髪の大きな色むらが減り、広い材質の混入や手足の大きな位置ずれは見当たりませんでした。胴体や背中の筋状の陰影は元のpriorにもあり、RGB更新後も残っています。靴は暗く、指や目の細かな位置も全身512画像では評価が限られます。[全身8方向の目視記録](evidence/pilots/all6-flat-visual-review.json)

このRGBを保ったままnormalだけを変更した採用版は、同一cameraの8方向でflat-normal版と比較しました。全フレームの描画alpha差は0で、背景を含む全画面のRGB MAEはuint8値で0.0128〜0.0400でした。表示差は小さく、承認済みmeshとnormal処理をそろえる目的で採用しています。陰影が大幅に改善したとは扱いません。[8方向のnormal比較](evidence/smooth-normals/all6-flat-smooth-sparse-comparison.json) [RGB MAEの記録](evidence/smooth-normals/normal-render-rgb-mae.json)

### 採用版の240フレーム比較

normalを固定した色転写済みbaselineとQwen後の採用版を、公式3D-Inspectorの既定表示で比較しました。カメラは240フレームすべてで完全一致し、PNGのalphaは62,914,560画素すべてで差が0でした。[比較の全記録](evidence/final-smooth-all6-090/summary.json)

| 検査 | 結果 |
|---|---|
| 記録されたcamera | 全240フレームで一致、最大差0 |
| 描画alpha | 全240フレームで一致、MAE 0、最大差0、差のある画素0 |
| 描画RGB | 全240フレームで変化 |
| color atlasのalpha | uint16のbit差0 |
| geometry、depth、normal、interval、range、proxy、camera | RGB合成の前後で保持 |

RGBが変わったことは、色が更新されたことを示します。差の大きさを画質向上の点数として使いません。保存alphaのbit一致と描画alphaの一致は別の検査で、後者は今回の240視点と8-bit出力でのcoverage保持を示します。

上面・下面に近い2視点でも、比較前後のcamera差と描画alpha差は0でした。OrbitControlsの制限により±Y軸から約0.0000573度ずれた視点で、厳密な極点ではありません。512×512の元フレームをcropやresizeせずに並べた比較シートも作成しています。[上下2視点の比較](evidence/final-smooth-all6-090/top-bottom-comparison.json)

同じnormalを使うgray、色転写のみ、Qwen後の3列動画は1536×584、240フレーム、24 fps、10秒のH.264です。3列の全組み合わせで240フレームのcamera差とalpha差は0で、入力720枚のPNGはSHA、順序、非空、画面端での切れがないことを確認しました。0、90、180、270度の全身posterと上下比較シートも目視で確認しています。動画の24 fpsは再生設定で、描画性能の測定ではありません。[動画の記録](evidence/final-smooth-all6-090/encoding-summary.json) [3列の入力比較](evidence/final-smooth-all6-090/three-column-capture-validation.json)

## 評価の限界

RGBの更新はDirect3Dの形状、depth、normalの精度を改善しません。生成maskとのIoUは輪郭の重なりを示しますが、目・手・服の模様などが形状上の正しい位置にあることまでは示しません。

色転写は既存coarseの見た目を引き継ぐ処理です。元テクスチャにない細部や隠れた面の色を復元する処理ではありません。近い表面から色を取得しても、手と服、髪と顔などの意味的な対応まで保証しないため、距離のQAと全身画像の確認を併用します。

6面の生成は独立しています。面ごとの色、明るさ、髪や衣服の模様が境界でつながるかを、回転表示で確認します。上面と下面は手足や顔が重なる投影なので、Qwenが隠れた部位を描き足したり、通常の立ち姿へ変えたりしていないかも確認します。

保存alphaのbit一致と240フレームの描画alpha一致は別の検査です。後者を確認できた場合も、その240視点と8-bit出力でのcoverage保持を示す範囲です。描画depth bufferの一致や、任意の視点での画質を証明するものではありません。

設定変更を比較するときは、入力画像、seed、steps、denoise、LoRA strength、解像度、promptを実行ごとに記録します。複数条件を同時に変えた比較から、解像度だけの効果を断定しません。
