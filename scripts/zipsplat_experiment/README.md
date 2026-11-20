# ZipSplatからBoxを作る実験

複数の入力画像からZipSplatの3DGSを推定し、TokenBAの解像度による違いと、512px/面のBoxへ変換したときの違いを調べる実験コードです。ComfyUI本体やZipSplat本体は変更しません。

比較では入力画像・順序・初期チェックポイント・圧縮率・最適化回数を固定します。入力画像への適合、未知方向の見た目、GSからBoxへの変換誤差をそれぞれ記録します。

実測結果は[REPORT.md](../../docs/verification/zipsplat-qwen-20261120/REPORT.md)を参照してください。このREADMEには再実行の手順と記録用テンプレートを載せています。

## 1. 実行環境

以下はリポジトリのルートから実行します。`RUN_DIR`は実験ごとに新しいディレクトリへ変更してください。

```bash
FACTORY_REPO="$(pwd)"
SCRIPTS="$FACTORY_REPO/scripts/zipsplat_experiment"
ZIP_SRC="$FACTORY_REPO/third_party/ZipSplat"
ZIP_ENV="$ZIP_SRC/.venv"
ZIP_PY="$ZIP_ENV/bin/python"
RUN_DIR="$FACTORY_REPO/output/zipsplat-run"
mkdir -p "$FACTORY_REPO/third_party" "$RUN_DIR"

git clone https://github.com/cvg/ZipSplat.git "$ZIP_SRC"
git -C "$ZIP_SRC" checkout 73ab63f048fcc15ee54ada4ef9cdd79b925611a8
uv venv --python 3.10.19 "$ZIP_ENV"
uv pip install --python "$ZIP_PY" \
  torch==2.9.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$ZIP_PY" -e "$ZIP_SRC" \
  -r "$SCRIPTS/requirements.txt" -c "$SCRIPTS/constraints-cu128.txt"
uv pip freeze --python "$ZIP_PY" > "$RUN_DIR/environment.freeze.txt"
nvcc --version > "$RUN_DIR/nvcc-version.txt"
nvidia-smi > "$RUN_DIR/gpu-environment.txt"
```

CUDA GPUとCUDA Toolkitの`nvcc`が必要です。この例はCUDA 12.8のPyTorchを使います。最初のレンダリングではgsplatのCUDAカーネルをコンパイルする場合があります。依存パッケージ全体のバージョンは`environment.freeze.txt`に保存します。

`requirements.txt`はCPU評価とBox変換の直接依存を定義します。推論用の依存は[固定したZipSplat](https://github.com/cvg/ZipSplat/tree/73ab63f048fcc15ee54ada4ef9cdd79b925611a8)からインストールします。

| 用途 | 実行環境 |
|---|---|
| ZipSplat推論、TokenBA、Box変換、入力評価 | 上記の`ZIP_PY` |
| `render_multiview_guides.py` | Gaussian Splatノードを利用できる既存ComfyUIのPython 3.11以降 |
| `generate_multiview.py`、`make_inspector_contact.py` | Pillowを利用できるPython 3.11以降 |
| `prepare_qwen_dataset.py` | NumPyとPillowを利用できるPython 3.11以降。既存ComfyUIのPython 3.12環境を推奨 |
| `encode_capture_comparison.py` | Pillowを利用できるPython 3.11以降とffmpeg。CPUで比較動画を生成 |
| 3D表示 | 3D-Inspectorの`view.py`と、その実行環境 |
| Inspectorの撮影 | Node.js、Playwright、Chromium |

`capture_inspector.mjs`は`--playwright-root`で指定したディレクトリの`package.json`からPlaywrightを解決します。既存のPlaywright環境を指定してください。Chromiumの場所は必要に応じて`--chromium`で渡します。

## 2. 入力を固定する

`views.json`にはZipSplatへ渡す順に画像を並べます。相対パスはこのJSONの場所を基準に解決します。最初の画像の推定カメラを、後の位置・向き合わせの基準に使います。

```json
{
  "views": [
    {"id": "front", "path": "images/front.png"},
    {"id": "yaw045", "path": "images/yaw045.png"},
    {"id": "yaw090", "path": "images/yaw090.png"},
    {"id": "yaw135", "path": "images/yaw135.png"}
  ]
}
```

全身の構図、ポーズ、衣装を確認し、採用する画像を先に決めます。画像のSHA256・寸法・順序はランナーが`inputs.json`へ保存します。同じ画像を重複して入力すると停止します。視点名や生成時の角度は記録用で、カメラの校正値としてZipSplatへ渡しません。

`masks.json`には同じ`id`で前景マスクを指定します。白が前景、黒が背景で、元画像と同じ寸法にします。生成画像を使う場合は、その生成画像のマスクを用意してください。

```json
{
  "views": [
    {"id": "front", "mask_path": "masks/front.png"},
    {"id": "yaw045", "mask_path": "masks/yaw045.png"},
    {"id": "yaw090", "mask_path": "masks/yaw090.png"},
    {"id": "yaw135", "mask_path": "masks/yaw135.png"}
  ]
}
```

新しい視点を作る場合は、`render_multiview_guides.py`で全身の透視投影ガイドを作り、`generate_multiview.py`へ渡します。後者はComfyUIのAPI形式JSONをテンプレートとして受け取り、ノードIDと型を検査します。必要な引数は各スクリプトの`--help`にあります。

`generate_multiview.py`の`--template "$SCRIPTS/qwen_anyangle_api_template.json"`で同梱のAPIテンプレートを指定できます。実行するComfyUI環境には、JSONが指定するQwen Image 2.1本体・CLIP・VAE・AnyAngle LoRA・BiRefNetの重みと、QwenImage21Cache・TextEncodeQwenImage21・背景除去などの使用ノードが必要です。入力画像名と保存先prefixはスクリプトが実行時に設定します。

AnyAngleの接続は元画像を`image_1`、対象視点のガイドを`image_2`とし、既定プロンプトは`Change the camera angle from <image2> to <image1>.`です。各視点を同じ元画像から生成し、全画像をpadで指定サイズに合わせます。顔などの部分クロップは行いません。生成の`manifest.json`はそのままZipSplatの入力に使えます。Box用マスクのJSONには各結果の`mask`を`mask_path`として転記します。

### 原図と8視点の黒背景入力

原図1枚と生成8枚を使う場合は、全視点の生成完了後に`prepare_qwen_dataset.py`で入力を固定します。この補助スクリプトは`hashlib.file_digest`を使うため、Python 3.11以降が必要です。上記のZipSplat用Python 3.10環境では実行できません。

```bash
COMFY_PY=/absolute/path/comfyui-env/bin/python
INPUT_DIR="$RUN_DIR/nine-inputs"
"$COMFY_PY" "$SCRIPTS/prepare_qwen_dataset.py" \
  --qwen "$RUN_DIR/qwen8" --original-mask "$RUN_DIR/original-mask.png" \
  --out "$INPUT_DIR" --background black
```

この処理は生成の完了状態・実行履歴・プロンプト・ハッシュを照合し、原図を先頭にした9枚の`views.json`と対応する`masks.json`を作ります。生成前の原図、生成画像、マスクはそのまま複製して残し、各画像のRGBをリニア光へ変換して対応マスクを1回乗算したあと、sRGBへ戻して`inputs_black/`に保存します。画像の寸法は変えず、クロップもしません。PNGの埋め込みalphaは重ねて乗算しません。

`qa.json`には元画像・黒背景画像・マスクのハッシュ、全身マスクの範囲と面積、原図と`yaw000`の重なりを記録します。原図と`yaw000`は両方残します。原図の`maxRGB > 4`によるマスクは近似であり、独立した正解マスクではありません。`contact.png`は左に元RGB、右に黒背景入力を並べ、`mask-contact.png`は対応マスクを並べます。各画像は全体を表示します。

このセットを使う場合、後続コマンドの`$RUN_DIR/views.json`と`$RUN_DIR/masks.json`を、それぞれ`$INPUT_DIR/views.json`と`$INPUT_DIR/masks.json`へ置き換えます。B504とC1024には同じ固定済み入力を渡します。

## 3. 少数視点で動作を確認する

まず近接する少数の視点で推論と標準TokenBAを実行します。以下は先頭4枚を使う例です。この結果だけで全周の品質を判断しません。

```bash
"$ZIP_PY" "$SCRIPTS/run_zipsplat_experiment.py" predict \
  --manifest "$RUN_DIR/views.json" --limit 4 \
  --compression 1.0 --out "$RUN_DIR/smoke/initial"
"$ZIP_PY" "$SCRIPTS/run_zipsplat_experiment.py" refine \
  --initial "$RUN_DIR/smoke/initial/prediction.pt" \
  --long-edge 504 --stock-ba --out "$RUN_DIR/smoke/ba504"
```

初回は重みのダウンロードが発生します。ローカルに重みがある場合は`predict --weights /absolute/path/checkpoint.tar`で指定できます。`--stock-ba`は公式の早期停止条件を使うため、実際の更新回数は`complete.json`で確認します。

ランナーは[公式デモと同じ範囲](https://github.com/cvg/ZipSplat/blob/73ab63f048fcc15ee54ada4ef9cdd79b925611a8/demo/app.py)をbf16化し、Gaussian headとcamera decoderをfp32で保持します。初期推論後はbackboneを解放し、追加調整では保存したGaussian headだけを読み込みます。

## 4. 同じ初期推論から解像度を比較する

採用した全画像で初期推論を一度だけ作ります。BとCは、この同じ`prediction.pt`から開始します。

```bash
"$ZIP_PY" "$SCRIPTS/run_zipsplat_experiment.py" predict \
  --manifest "$RUN_DIR/views.json" --compression 1.0 \
  --out "$RUN_DIR/full/initial"
"$ZIP_PY" "$SCRIPTS/run_zipsplat_experiment.py" refine \
  --initial "$RUN_DIR/full/initial/prediction.pt" \
  --long-edge 504 --rounds 4 --steps 96 --out "$RUN_DIR/full/B504"
"$ZIP_PY" "$SCRIPTS/run_zipsplat_experiment.py" refine \
  --initial "$RUN_DIR/full/initial/prediction.pt" \
  --long-edge 1024 --rounds 4 --steps 96 --out "$RUN_DIR/full/C1024"
```

この比較では`--stock-ba`を付けません。早期停止を無効にし、camera/token alternationを4回、joint refinementを96回実行します。画像のクロップ領域とencoderの解像度は共通です。Cでは同じ元画像のクロップを1024pxへ変換し、内部パラメータも対応する比率で拡大します。

2048pxを試す場合も、C1024の出力を続けて調整せず、同じ初期チェックポイントから同じ更新回数で別の出力先へ実行します。VRAM不足や品質の差はその条件と併せて記録します。

## 5. 同じ解像度で入力への適合を測る

BとCを同じ1024pxで再レンダリングします。

```bash
"$ZIP_PY" "$SCRIPTS/run_zipsplat_experiment.py" render \
  --checkpoint "$RUN_DIR/full/B504/prediction.pt" \
  --render-long-edge 1024 --out "$RUN_DIR/eval/B504"
"$ZIP_PY" "$SCRIPTS/run_zipsplat_experiment.py" render \
  --checkpoint "$RUN_DIR/full/C1024/prediction.pt" \
  --render-long-edge 1024 --out "$RUN_DIR/eval/C1024"
"$ZIP_PY" "$SCRIPTS/evaluate_zipsplat_inputs.py" \
  --left "$RUN_DIR/eval/B504/renders_1024" \
  --right "$RUN_DIR/eval/C1024/renders_1024" \
  --masks "$RUN_DIR/masks.json" --left-label B504 --right-label C1024 \
  --left-losses "$RUN_DIR/full/B504/losses.json" \
  --right-losses "$RUN_DIR/full/C1024/losses.json" \
  --out "$RUN_DIR/comparison1024"
```

`metrics.json`には全画像・全身の前景・前景を2px内側へ縮めた領域のL1とPSNRを保存します。Sobelの強度は細部やコントラストの量を示す統計で、入力への勾配誤差と併せて読みます。`contact.png`には全体画像を並べます。

これらは入力画像への適合値です。生成された入力にない方向の正しさや、実物の形状精度を測る値には使いません。PNGを使う評価は8bitに量子化した値なので、ランナーがfloat画像から求めた値とは微小な差が出る場合があります。

## 6. 共通の座標系で512px/面のBoxへ変換する

既存Boxの`manifest.json`と3D-Inspectorの実行パスを指定します。両分岐で同じ`--out`を使い、初期推論から作った共通の座標系とBox boundsを再利用します。

```bash
BOX_TEMPLATE=/absolute/path/existing-box/manifest.json
INSPECTOR_VIEWPY=/absolute/path/3D-Inspector/view.py
INSPECTOR_PY=/absolute/path/inspector-env/bin/python

"$ZIP_PY" "$SCRIPTS/process_zipsplat_box.py" \
  --initial-dir "$RUN_DIR/full/initial" --branch-dir "$RUN_DIR/full/B504" \
  --masks "$RUN_DIR/masks.json" --template "$BOX_TEMPLATE" \
  --out "$RUN_DIR/boxes" --label B504 \
  --inspector-viewpy "$INSPECTOR_VIEWPY" --inspector-python "$INSPECTOR_PY"
"$ZIP_PY" "$SCRIPTS/process_zipsplat_box.py" \
  --initial-dir "$RUN_DIR/full/initial" --branch-dir "$RUN_DIR/full/C1024" \
  --masks "$RUN_DIR/masks.json" --template "$BOX_TEMPLATE" \
  --out "$RUN_DIR/boxes" --label C1024 \
  --inspector-viewpy "$INSPECTOR_VIEWPY" --inspector-python "$INSPECTOR_PY"
```

この処理は前景マスクへのGaussian中心の投影、座標の正規化、6方向の正射影RGB・alpha・expected Z、Boxへの格納、公式Inspectorの生成を順番に行います。Gaussian中心のマスク判定だけでは、大きな楕円体や遮蔽を正確に分離できません。`selection.json`と全周画像を確認して採用を判断します。

TokenBAでは最初のカメラも動きます。`align_camera_gauge.py`は分岐ごとに最初のカメラを共通の初期カメラへ合わせ、変換行列とカメラ間距離の比を保存します。スケールは自動補正しません。

GSのalphaは累積不透明度、expected Zは寄与したGaussianの重み付き平均深度です。これらを前景マスクや最前面の表面深度と同一視しないでください。既存GLBはBox形式のメタデータとして残りますが、新しいGSの形状を表していません。

### v2の前景選別と共通範囲

大きな背景Gaussianと初期推論の範囲外へのはみ出しを調べる場合は、次のCPU処理で前景とBox範囲を準備します。3視点以上のマスクがGaussian中心を前景と判定し、最大標準偏差が共通の基準高さの5%以下のGaussianを残す設定です。これは今回試す選別条件であり、前景を正しく分離できる保証ではありません。

新しい入力セットでは初期推論から座標変換を一度だけ作り、B/Cの最初のカメラをその基準へ合わせます。最後に、選別後のB/Cそれぞれの0.1–99.9パーセンタイル範囲の和集合へ、高さの2%の余白を加えます。B/Cの座標変換とスケールは変えません。

```bash
PREP="$RUN_DIR/boxes-v2-prepared"
mkdir -p "$PREP"
"$ZIP_PY" "$SCRIPTS/zip_render_faces.py" select \
  --gaussians "$RUN_DIR/full/initial/gaussians.npy" \
  --cameras "$RUN_DIR/full/initial/cameras.json" --masks "$RUN_DIR/masks.json" \
  --min-views 3 --output "$PREP/initial-foreground"
"$ZIP_PY" "$SCRIPTS/zip_render_faces.py" align \
  --gaussians "$PREP/initial-foreground/gaussians.npy" \
  --cameras "$RUN_DIR/full/initial/cameras.json" --template "$BOX_TEMPLATE" \
  --output "$PREP/initial-canonical.json" --frame-dir "$PREP/reference-frame"
for BRANCH in B504 C1024; do
  "$ZIP_PY" "$SCRIPTS/align_camera_gauge.py" \
    --initial-cameras "$RUN_DIR/full/initial/cameras.json" \
    --branch-cameras "$RUN_DIR/full/$BRANCH/cameras.json" \
    --canonical "$PREP/initial-canonical.json" --output "$PREP/$BRANCH-canonical.json"
  "$ZIP_PY" "$SCRIPTS/zip_render_faces.py" select \
    --gaussians "$RUN_DIR/full/$BRANCH/gaussians.npy" \
    --cameras "$RUN_DIR/full/$BRANCH/cameras.json" --masks "$RUN_DIR/masks.json" \
    --min-views 3 --max-sigma-height-fraction 0.05 \
    --transform "$PREP/$BRANCH-canonical.json" --template "$PREP/reference-frame/manifest.json" \
    --output "$PREP/$BRANCH-foreground"
done
cat > "$PREP/branches.json" <<'JSON'
{"branches": [
  {"id": "B504", "gaussians": "B504-foreground/gaussians.npy", "transform": "B504-canonical.json"},
  {"id": "C1024", "gaussians": "C1024-foreground/gaussians.npy", "transform": "C1024-canonical.json"}
]}
JSON
"$ZIP_PY" "$SCRIPTS/build_common_frame.py" \
  --branches "$PREP/branches.json" --template "$PREP/reference-frame/manifest.json" \
  --quantile 0.001 --padding-height-fraction 0.02 --output "$PREP/common-frame"
```

`selection.json`の採用数・視点数・標準偏差と、`common-frame/bounds.json`の範囲外の中心数を確認します。以前の入力セットで作った変換は、新しい入力セットへ流用しません。同じ入力セットの既存B/Cをやり直す場合は、そのセットの初期変換と分岐の変換を再利用し、選別と共通範囲の生成を新しい出力先で行います。

準備後は3つの`--prepared-*`をまとめて指定します。ここからはGPUでの6面レンダリング、Box格納、`.box.zip`、表示用PLY、公式Inspectorの生成を行います。

```bash
for BRANCH in B504 C1024; do
  "$ZIP_PY" "$SCRIPTS/process_zipsplat_box.py" \
    --branch-dir "$RUN_DIR/full/$BRANCH" \
    --prepared-foreground "$PREP/$BRANCH-foreground" \
    --prepared-transform "$PREP/$BRANCH-canonical.json" \
    --prepared-frame "$PREP/common-frame/manifest.json" \
    --out "$RUN_DIR/boxes-v2" --label "$BRANCH" \
    --inspector-viewpy "$INSPECTOR_VIEWPY" --inspector-python "$INSPECTOR_PY"
done
```

この経路では前景選別・位置合わせ・共通範囲の生成を省略し、準備済みファイルを直接参照します。`--initial-dir`・`--masks`・`--template`は省略できます。`result.json`と`stages.json`に参照パスとSHA256を記録し、準備済み入力が変わった状態で完了済み出力を再利用しようとすると停止します。以前の結果を残すため、v2には新しい`--out`を使ってください。

## 7. Inspectorと同じカメラでGSを比較する

`view.py`が生成したHTMLをそのまま使います。以下はB504の240フレームを撮影し、そのカメラをGSで再現する例です。

```bash
PLAYWRIGHT_ROOT=/absolute/path/playwright-project
node "$SCRIPTS/capture_inspector.mjs" \
  --viewer "$RUN_DIR/boxes/branches/B504/B504-box.inspector.html" \
  --reference "$RUN_DIR/boxes/shared/frame/manifest.json" \
  --out "$RUN_DIR/capture/B504" --frames 240 --size 512 \
  --playwright-root "$PLAYWRIGHT_ROOT"
"$ZIP_PY" "$SCRIPTS/render_matched_gs.py" \
  --gaussians "$RUN_DIR/boxes/branches/B504/foreground/gaussians.npy" \
  --transform "$RUN_DIR/boxes/branches/B504/canonical.json" \
  --capture "$RUN_DIR/capture/B504/capture.json" \
  --output "$RUN_DIR/replay/B504"
```

GSとBoxの輪郭IoU、alphaの誤差、穴や二重像を確認します。ネイティブBoxとGSでは材質・照明・トーンマッピングが異なるため、RGBの見た目は条件を併記して比較します。形状を再構成する実験では既存Boxとのalpha完全一致を合否条件にしません。

v2では生成したv2のHTMLと共通範囲を指定し、撮影・再レンダリングの保存先も分けます。

```bash
node "$SCRIPTS/capture_inspector.mjs" \
  --viewer "$RUN_DIR/boxes-v2/branches/B504/B504-box.inspector.html" \
  --reference "$PREP/common-frame/manifest.json" \
  --out "$RUN_DIR/capture-v2/B504" --frames 240 --size 512 \
  --playwright-root "$PLAYWRIGHT_ROOT"
"$ZIP_PY" "$SCRIPTS/render_matched_gs.py" \
  --gaussians "$PREP/B504-foreground/gaussians.npy" \
  --transform "$PREP/B504-canonical.json" \
  --capture "$RUN_DIR/capture-v2/B504/capture.json" \
  --output "$RUN_DIR/replay-v2/B504"
```

### RGBだけを更新する対照

`retexture_box_rgb.py`は既存A512の6面のRGBをGSの色と混合し、全RGBA16F atlasのalphaビット列と、depth・normal・interval・range・proxyを保持します。入力する6面は`zip_render_faces.py render`の`--template`にA512の元manifestを指定して作ります。既定ではGSのalphaが0.5以下の画素にAの色を残し、0.5〜0.95で混合重みを滑らかに増やします。面ごとの支援alpha IoU・coverage・変更画素数はQAに保存します。

```bash
"$ZIP_PY" "$SCRIPTS/retexture_box_rgb.py" \
  --template /absolute/path/A512/manifest.json \
  --captures "$RUN_DIR/rgb-only-captures" --output "$RUN_DIR/rgb-only-box"
```

A512とRGB更新後のBoxを、公式Inspectorのdefault表示・同じ`--reference`でそれぞれ240フレーム撮影してから比較します。

```bash
"$ZIP_PY" "$SCRIPTS/compare_capture_alpha.py" \
  --left "$RUN_DIR/capture/A512/capture.json" \
  --right "$RUN_DIR/capture/D-RGB-only/capture.json" \
  --out "$RUN_DIR/alpha-A512-vs-D-RGB-only.json"
```

比較処理は完了状態・240枚の順序・PNGのSHA256・記録カメラを検査します。カメラの数値差が1e-9を超える場合は`comparisonValid=false`とし、alpha一致の結論を出しません。成立した比較では各フレームと全体のalphaのMAE・最大差・不一致画素数・完全一致フレーム数を保存し、RGB差は別集計にします。既存の結果JSONは上書きしません。

`encode_capture_comparison.py`は4条件の撮影結果を2×2に並べ、240フレームを24fps・10秒の比較動画にします。24fpsは保存動画の再生速度で、3D-InspectorやGPUの描画性能を示す値ではありません。

## 8. 保存物と再実行

| 保存物 | 用途 |
|---|---|
| `prediction.pt` | plain tensorとメタデータ。追加調整・再レンダリングの開始点 |
| `scene.ply`、`gaussians.npy` | 3DGS本体 |
| `inputs.json`、`cameras.json` | 入力のハッシュ・寸法・順序、元画像とencoderの内部パラメータ、c2w |
| `run.json`、`complete.json`、`losses.json` | 設定、所要時間、PyTorchの最大確保・予約VRAM、更新履歴 |
| `stages.json`、各段階のログ | Box処理の入力・出力ハッシュ、所要時間、失敗箇所 |
| `*.box.zip`、`*-canonical.ply`、`*.inspector.html` | Box、表示用GS、公式Inspector |

同じ設定で完了済みの処理はハッシュを確認して再利用します。設定やコードを変更した比較は、新しい出力先または分岐名で実行してください。推論後の出力処理だけが失敗した場合は、保存済み`prediction.pt`に対して`render`を実行できます。

## 実験記録用テンプレート

以下は再実験の記録用テンプレートです。実測ファイルを確認してから記入します。今回の実測結果は[REPORT.md](../../docs/verification/zipsplat-qwen-20261120/REPORT.md)を参照してください。

時間欄には`refinement.seconds`の追加調整の所要時間を、VRAM欄にはその段階の`peak_allocated_bytes`をGiBへ変換した値を記入します。モデルの読み込み、初期推論、Qwen生成、結果の保存は別工程です。PyTorchの最大確保量は`nvidia-smi`の使用量とは異なります。生成処理の`seconds`は投稿から結果収集までの時間で、処理を再開するまでの待ち時間も含み得ます。`captureElapsedSeconds`はInspectorのオフライン撮影時間です。

| 条件 | 入力セット | 更新回数 | 追加調整の所要時間 | 追加調整中のPyTorch最大確保VRAM | 共通解像度での前景PSNR | 240フレームの所見 |
|---|---|---|---|---|---|---|
| B504 | 未記入 | 未記入 | 未記入 | 未記入 | 未記入 | 未記入 |
| C1024 | 未記入 | 未記入 | 未記入 | 未記入 | 未記入 | 未記入 |

モデルと重みの利用条件は[ZipSplatの公開元](https://github.com/cvg/ZipSplat)を確認してください。重みは各実験の実行時にダウンロードし、このディレクトリには含めません。
