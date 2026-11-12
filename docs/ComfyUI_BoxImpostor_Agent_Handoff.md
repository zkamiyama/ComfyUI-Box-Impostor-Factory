# ComfyUI Box Impostor Factory 実装指示書

作成日：2026-11-12  
設計段階：v0.1（実装前）

## 1 目的と完成条件

参照画像1枚から、ComfyUI上で6方向の高精細な見た目を生成し、`three-impostor`の`box`レイアウトに互換なImpostorアセットを作成する。生成したアセットをthree.jsのインタラクティブなビューアで即座に360度確認し、保存・再読込できること。

画像生成バックエンドは**Qwen-Image 2.1**と**MiniMax H3-Image**を切り替えられること。カメラ方向はプロンプトだけに任せず、一度参照画像をメッシュまたは3D Gaussian Splatに変換して得られる幾何ガイドで制御すること。

初期版の完成条件は次のとおり。

- 参照画像1枚 → 形状プロキシGLB → 正投影6方向のガイド画像 → AIによる6方向RGB(A) → Box Impostorを一連のComfyUIワークフローとして実行できる。
- 6方向画像は個別に上書き・再生成でき、形状推定とベイクを不要に再実行しない。
- Box Impostorはテクスチャ、Bounding Box、メタデータを同一アセットとして保存し、ComfyUI再起動後にも読み込める。
- three.jsプレビューでカメラを回せる。プロキシGLBとのA/B比較、6枚の元画像、生成した各種マップを表示できる。
- Qwen-Image 2.1とH3-Imageの両方で同一の後処理・ベイク・プレビューに接続できる。
- 生成不能・画像不整合・GPUメモリ不足時に明確なエラーやフォールバックを返し、壊れたアセットを成功扱いしない。

現段階では、学習・LoRA作成、厳密なメッシュ復元、動画アニメーション、3DGSの直接Box化、商用向けの最適圧縮は対象外。

## 2 確認済みの既存実装と基本方針

1. 指定された混合順氏の記事と配布ワークフロー：Pixal3DでGLBを生成し、Qwen-Image 2.1 + `lilylilith/QI_2.1_AnyAngle`で別角度画像を生成する。GLB入力だけの後段ワークフローもある。
   - https://note.com/kongo_jun/n/ndbabf97a8f41?hl=en
2. `three-impostor`：`ImpostorBaker.bake(object, {layout:'box', resolution:256})`と`ImpostorNodeMaterial`を再利用する。MIT。GPUで複数マップを生成する実装は`Impostor.js`に存在する。
   - https://github.com/richardmonette/three-impostor
3. Pixal3D：ComfyUIネイティブの画像→PBRメッシュ生成。画素と3Dの位置合わせを意図した設計。**初期版の標準形状バックエンド**。
   - https://github.com/Comfy-Org/docs/blob/main/tutorials/3d/pixal3d.mdx
4. TripoSplat：ComfyUIネイティブの画像→3DGS。**代替ガイド生成器**として採用。3DGSのみでは既存の`ImpostorBaker`に直接入力できないため、Boxベイク時には別途GLBメッシュが必要。
   - https://www.comfyui.org/en/bringing-native-support-for-3d-gaussian
5. AnyAngle Studio T8：3Dカメラ・TripoSplat・角度ガイド・バッチ生成を既に持つ。ソースを参照し、必要なら任意連携するが、初期版で一から同等のエディタを再開発しない。
   - https://github.com/T8mars/Comfyui-Qwen-Image-2.1-MultiAngle-T8
6. Qwen-Image 2.1とH3-Imageは2026年11月時点でどちらもComfyUIネイティブ対応。H3-Imageは動画モデルH3とは別の画像モデル。
   - https://github.com/QwenLM/Qwen-Image-2.1
   - https://blog.comfy.org/p/minimax-h3-image-is-now-available

**重要な設計原則**：幾何形状の権威はPixal3D等のプロキシGLBに置く。AIで生成する6方向画像は主に表面色・微細な見た目を供給する。`intervalMap`・`rangeMap`・法線・深度をRGB画像だけから推測した結果で、無検証に上書きしない。

## 3 全体のデータフロー

```text
Load Reference IMAGE
       │
       ├── Mask/Matting (Qwen native RGBA or BiRefNet)
       │
       ▼
Proxy Geometry ─── Pixal3D [default] / existing GLB / optional TripoSplat
       │                 ※3DGS only ⇒ mesh proxy required for Box bake
       ▼
Canonicalize Y-up + bounds + source-view calibration
       │
       ├── 6 orthographic guide renders (+X,-X,+Y,-Y,+Z,-Z)
       │            │
       │            └── View Image Generation Adapter
       │                   ├── Qwen-Image 2.1 + AnyAngle LoRA
       │                   └── MiniMax H3-Image image editing (no AnyAngle LoRA)
       │                       │
       │                       ▼
       │               6 RGB(A) views → Alpha / registration / QA
       │                                    │
       ▼                                    │
three-impostor BOX bake (geometry truth)   │
       │                                    │
       ├── normalMap, intervalMap, rangeMap │
       └── baseline colorMap ───────────────┴── geometry-aligned color replacement
                                           │
                                     manifest + texture files
                                           │
                           three.js 360° preview / Save / Reload
```

**モードを分けること**

- `geometry_baseline`：AI色生成をスキップし、Pixal3D/GLBのテクスチャで正しいBoxアセットを作る。最優先で実装し、比較の正解基準とする。
- `qwen_enhanced`：AnyAngle LoRAを使用し、6方向のAI画像でカラーアトラスを高精細化する。
- `h3_enhanced`：H3-Imageの2画像参照編集で同じ6方向のガイドを利用する。Qwen専用LoRAを誤って適用しない。
- `manual_faces`：外部生成した6方向画像を入力できる。生成後の修正と比較に必須。

## 4 モデルごとの実装方針

### Qwen-Image 2.1

- 公式ネイティブComfyUI Image Editサブグラフを利用。`TextEncodeQwenImage21`に`image_1 = original reference`、`image_2 = geometric guide`を入力する。
- `lilylilith/QI_2.1_AnyAngle`のLoRAをモデル側に適用する。初期パラメータはLoRA強度1.0、20 steps以上、CFG 3を実験基準にする（標準QwenテンプレートのCFG 1と区別）。推奨プロンプト：`Change the camera angle from <image2> to <image1>.`。モデル仕様は運用時に再確認する。
- RGBAが得られる場合はalphaを使用する。ComfyUI IMAGE型の4チャンネルを扱えない既存ノードとの互換には、Split/Join alpha処理を使用。
- 高速版を使う場合は品質比較を経て別プリセットにする。

### MiniMax H3-Image

- ネイティブ`MiniMaxH3ImageConditioning`（モデル本体 + VAE + Qwen3-VL系エンコーダー）を利用。Qwen用AnyAngle LoRAは**非互換**。
- `image_1 = original reference`、`image_2 = target camera coarse guide`とし、`<Picture 1>`と`<Picture 2>`を明記するモデル専用プロンプトを組み立てる。
- 推奨実験設定は公式の6 steps、CFG 1。ネガティブプロンプトやLoRAを流用しない。元画像とガイドのうち、どちらに外観／幾何を従わせるか明確に記述する。
- ネイティブ透過出力は前提としない。BiRefNet等で背景を除去し、必ず透過輪郭を検査する。
- ローカルとAPI/Partner Nodesの双方を将来選べるようにする。API鍵はサーバー側環境変数で扱い、ワークフローJSONへ保存しない。APIでseedが指定できない場合はnullで記録し、出力画像をキャッシュする。

### 3D生成

- **Pixal3D GLBが既定**。実際の出力が参照画像のカメラに一致しているか確認してから正規化する。
- ユーザーが既存GLBを与えた場合は画像→3D生成を省略する。
- TripoSplatは新視点のガイド描画には有効だが、Boxは3Dメッシュの表面・裏面・ギャップをベイクするため、Pixal3D GLB、別メッシュ、または検証済みのスプラット→メッシュ抽出結果が必要。TripoSplatの描画だけは専用のsplatビューア／AnyAngle Studioから導出してよいが、初期版の`BI MakeSixGuides`はGLB入力を必須とし、SPZ/PLY入力を暗黙に受理しない。
- 将来のVisual Hull/深度融合は別バックエンドとし、初期版のパイプラインに混ぜない。

## 5 カメラと座標系の厳密な契約

- **Y-up、右手座標系、単位スケールを正規化**し、ローカルBounding Boxのmin/maxと中心を記録する。参照画像の撮影方向は必ず別フィールドに保持し、参照画像＝+Z正面と暗黙に決めない。
- Boxの6方向は**`+X, -X, +Y, -Y, +Z, -Z`**の順。three-impostorのコードはこの順で3列×2行の`colorMap`と`normalMap`に格納する。
- 元`ImpostorBaker`のカメラ向き、upベクトル、上下反転、正投影範囲、Bounding Boxの寸法をそのまま用いる。全方向で正投影／固定物体姿勢／固定BBox／中心を変更しない。
- 元参照が斜め視点でも、Pixal3Dの画像位置合わせ情報を用いて正面を確定する。自動推定が難しい場合は`source_view_rotation`の手動調整を許す。
- Top/Bottomではカメラupが退化しない向きにする。上下方向の画像回転はユニットテストで固定する。
- ガイド画像は各方向で影・背景・余計な物体を排除した正投影レンダーとし、必要に応じてnormal、depth、maskも別デバッグ出力する。

## 6 Box Impostorの正確なテクスチャ契約

`three-impostor/Impostor.js`の`_bakeBox()`が生成するレイアウトと同じ意味・精度を保つ。`resolution=256`が初期値。

| 名前 | 内容 | 256px時の寸法 | 用途 |
|---|---|---:|---|
| `colorMap` | RGBA、6方向の色とcoverage | 768×512 | 表面色 |
| `normalMap` | 6方向の法線とcoverageに関連する値 | 768×512 | シェーディング |
| `intervalMap` | XYZ軸ごとの前後面の区間と最初の内部ギャップ | 768×256 | Boxの幾何判定 |
| `rangeMap` | 8×8 texelセルごとの占有範囲 | 96×32 | Ray Marchingの空領域スキップ |
| `materialMap` | 任意。roughness/metalness/AO | 768×512 | PBR |
| `emissiveMap` | 任意 | 768×512 | 発光 |

`intervalMap`のRが正側の最前面位置、Gが`1 - 負側の最前面位置`、B/Aが最初の内部空隙に関する位置。`rangeMap`はセルごとのlow/high。元ソースにはcoverage境界、空領域、薄い形状、開放メッシュの補正があるため、この演算を初期版でPythonに書き直さない。

`ImpostorNodeMaterial`はBoxを実際にRay Marchingして着色し、シーンに正しい深度を出力する。カラーをAI画像で差し替える際も、**`normalMap`と`intervalMap`/`rangeMap`は必ず同じプロキシ形状から取得する**。AI側のマスクと形状のシルエットが合わない場合は、補正を試しても一致しなければその方向のAI色の適用を保留する。

### Color replacement

- 6方向のAI画像を「正投影の各画素→プロキシBoxの対応面」という同じキャプチャ座標に配置する。
- alphaは参照輪郭、ガイド輪郭と比較し、単なる背景の黒色を透明と誤認しない。
- 色は原則として未照明のAlbedoに近い状態で生成する。陰影の焼き込みが残る場合はPBRシェーディングで陰影が二重になる旨をUIで明示し、unlit比較プレビューを用意する。
- AI画像の見た目がプロキシと局所的にずれている場合は、正規化・2D位置合わせ・ガイド差分マスクで調整。大きな形状変形を無制限にワープしてごまかさない。
- いずれかの方向が破綻したときは、その方向だけプロキシベイクのベースカラーに戻せること。

## 7 実装技術スタック

| 層 | 依存・技術 | 方針 |
|---|---|---|
| ワークフロー | 最新安定ComfyUIのネイティブノード + カスタムノード | モデル本体をカスタムノードから直接実装しない |
| 画像モデル | Qwen-Image 2.1 / MiniMax H3-Image | 公式テンプレートの条件付けとサンプラーを流用 |
| 画像→3D | Pixal3D、代替TripoSplat、手動GLB | Geometry Adapterで統一 |
| マスク・幾何補助 | Qwen native alpha / BiRefNet / MoGe | 生成モデルの差をアダプターで吸収 |
| 画像整列とQA | Python 3.11+、NumPy、Pillow、OpenCV | alpha、bbox、IoU、エッジ誤差、色差 |
| Box Baker | `three-impostor` + three.js r186以上 | 既存BakerのJSを呼ぶ。独自の間隔マップ実装禁止 |
| 自動レンダー | Node.js + Playwright Chromium（WebGPU/WebGL2） | 専用プロセスで初期化し、GPU互換性を検証 |
| Preview | three.js + `ImpostorNodeMaterial` + OrbitControls | ComfyUI Web extensionとローカルWebページ |
| UI | ComfyUI Web extension (`WEB_DIRECTORY`等) | 独立したReact SPAを最初から作らない。必要なUIだけ追加 |
| 保存 | バージョン付きmanifest + half-floatテクスチャ + PNG | 保存・再ロードのroundtripを必ず試験 |
| 試験 | pytest、Vitest、Playwright | Pythonロジック、JSシェーダー、統合フローを分離 |

**自動ベイクの注意**：`three-impostor`のBakerはブラウザGPUレンダラー上で動く。PythonのComfyUIノードだけで`ImpostorBaker`を直接実行する設計にしない。Node.js/Playwrightによるheadless Chromiumワーカーを設ける。WSL2でWebGPUが不安定な場合はthree.jsの`WebGPURenderer({forceWebGL:true})`を試し、ベイクが不可能なら事前検証で明確に止める。必要ならBlender/EGL等を別フォールバックとして後から追加する。

## 8 推奨カスタムノード分割

**独自処理は小さなノードに分離する。** 最初から単一の巨大ノードに画像生成・3D・ベイクを埋め込まない。

- `BI PrepareReference`：IMAGE、任意MASK → 正規化参照画像、マスク、画像メタデータ。
- `BI CanonicalizeProxy`：FILE3DGLB、参照姿勢情報 → 正規化GLB、BBox、canonical transform。Pixal3D自体は標準ノードを使用。
- `BI MakeSixGuides`：正規化GLBとBBox → 6方向の`IMAGE`群と`BOX_FACE_META`。撮影は専用three.jsレンダーワーカーで固定カメラを再利用。
- `BI FaceInputs`：6方向の出力画像または手動差替え → `BOX_FACES`。画像を方向ID付きの構造として扱う。
- `BI AlignAndValidateFaces`：AI画像、ガイド、BBox、カメラ → 色マップ候補、個別スコア、警告、fallback状態。
- `BI BakeBox`：正規化GLB、BOX_FACES、resolution、PBRフラグ → `BOX_IMPOSTOR`、manifest path、デバッグマップ。専用JSワーカーへ依頼。
- `BI PreviewBox`：BOX_IMPOSTOR → プレビューURL、静止プレビューIMAGE、オービットUI起動用操作。
- `BI SaveBox` / `BI LoadBox`：manifest・マップ・bboxの書出しと再読込。

**モデルアダプターはComfyUIネイティブグラフで実装**。`GenerateView-Qwen21`と`GenerateView-H3Image`という別サブグラフを用意し、同じ`FACE_VIEW_RESULT`（方向ID、RGBA画像、生成条件）に接続する。選択していないモデルがメモリに読み込まれたり実行されたりしないこと（必要ならQwen用・H3用の上位ワークフローJSONを別々に保持）。6方向は同じモデルローダーを共有し、実行・再生成は方向別にキャッシュする。ComfyUI内部から`/prompt`を入れ子に投入して同一キューを待つデッドロック設計は避ける。

## 9 ファイルと再利用契約

`BOX_IMPOSTOR`はメモリに閉じたGPU RenderTargetではなく、下記manifestを持つ保存済みアセットへの参照とする。

```text
output/box_impostor/<asset_id>/
  manifest.json
  geometry/proxy.glb
  reference/reference.png
  guides/+X.png ... -Z.png
  views/+X.png ... -Z.png
  maps/color.rgba16f.bin
  maps/normal.rgba16f.bin
  maps/interval.rgba16f.bin
  maps/range.rgba16f.bin
  maps/material.rgba16f.bin      # optional
  maps/emissive.rgba16f.bin      # optional
  previews/hero.png
  qa/report.json
```

MVPではGPUから読み出したRGBA16Fデータをヘッダーなしの**little-endian half-float raw**で格納してよい。テクスチャごとの解像度、dtype、origin/rowOrder、色空間、alpha表現、使用モデル・seed・設定、BBox、座標変換、ソースコードのrevisionを必ずmanifestに入れる。ブラウザでは`DataTexture`に復元し、`ImpostorNodeMaterial`へ渡す。復元アダプターは`layout=box`、`box=Box3`、`resolution`、`maps`、`colorMap`、`normalMap`、`intervalMap`、`rangeMap`、任意の`materialMap`・`emissiveMap`、対応する`BoxGeometry`を正しく構築する。公開された`ImpostorData`のコンストラクタがGPUの`RenderTarget`を前提とするため、偽のRenderTargetを無理に挿入せず、`ImpostorNodeMaterial`が必要とする実データ契約をテストで確認する。EXR/KTX2最適化は後段階。RGBのプレビューPNGは本体のhalf-floatテクスチャの代用としない。

**最重要のroundtrip試験**：Baker直後のGPUテクスチャで描いた結果と、保存→ロード後に描いた結果を、全6正面＋斜め方向で比較する。色空間、UV反転、premultiplied-alpha、16F精度の差異をここで検出する。

## 10 ComfyUI内プレビュー仕様

- `BI PreviewBox`を選択すると、ComfyUIの同一オリジンに設置したthree.jsビューアを開く。Orbit・Pan・Zoom、正投影／透視、照明回転、ワイヤーフレーム表示を持つ。
- 表示切替は`Box Impostor / Proxy GLB / side-by-side`。背景をグレー、チェック、黒、白から選べる。正面6方向へワンクリックで切り替えられる。
- `steps`（Ray Marching最大ステップ数）、露出、AOの比較値を調整できる。法線／深度／interval／range／6方向カラーのタイルを表示するデバッグビューを設ける。
- スクリーンショットとmanifest配下への保存、再ロードを用意する。UIとワークフローは保存された`asset_id`/manifestで接続し、ブラウザだけに永続状態を置かない。
- 実際の描画には`WebGPURenderer`と`ImpostorNodeMaterial`を使用。WebGPUが使えなければthree.jsのWebGL2 fallbackを試す。
- カスタムサーバールートは専用プレフィックス（例`/box-impostor/*`）に限定し、ファイルパスの検証とディレクトリ逸脱防止を行う。

## 11 ソースコード構造

```text
ComfyUI-BoxImpostorFactory/
  __init__.py                  # Node registration / WEB_DIRECTORY
  pyproject.toml               # Comfy Registry向けメタデータ
  requirements.txt             # Python minimal deps
  nodes/
    reference.py
    canonicalize.py
    guides.py
    faces.py
    qa.py
    bake.py
    preview.py
    assets.py
  core/
    schema.py                  # BOX_FACES / BOX_IMPOSTOR / manifest
    geometry.py
    image_alignment.py
    quality_metrics.py
    file_store.py
    worker_client.py
  worker/
    package.json               # three.js / three-impostor / Playwright
    src/
      bake.ts                  # ImpostorBaker invocation
      guide_renderer.ts        # canonical six ortho cameras
      export_maps.ts           # readbacks / raw half-float serialization
      import_maps.ts           # runtime DataTexture / impostor-like object
      worker_server.ts
  web/
    extension.js               # ComfyUI node widget integration
    viewer/                    # minimal three.js / Vite viewer
  workflows/
    01_box_from_glb.json
    02_box_from_reference_qwen21.json
    03_box_from_reference_h3.json
    04_manual_six_faces.json
  examples/
    cube.glb
    hollow_shape.glb
    sample_reference.png
  tests/
    test_manifest.py
    test_alignment.py
    test_camera_axis.py
    test_bake_roundtrip.spec.ts
    test_pipeline_smoke.py
  docs/
    architecture.md
    install.md
    qa.md
    third_party_licenses.md
```

外部ライブラリを不用意にコピーしない。`three-impostor`・three.js・ComfyUIのバージョンをlockfile/commit hashで固定し、ライセンスを保持する。

## 12 開発順序

**Phase 0 — Bakerの正確な保存・復元**：まず外部GLBを入力し、AI生成を使わずに既存BakerでBoxをベイク。half-floatマップを保存・再ロードし、同一描画に見えることを確認する。プレビューのOrbit操作が必須。この段階が成立するまでモデルを統合しない。

**Phase 1 — ComfyUIノード化**：GLB選択、6方向ガイド、BoxBake、Preview、Save/Loadを個別のノードとして実装。`01_box_from_glb.json`がGUIから無人実行できる。

**Phase 2 — 参照画像→3D**：ComfyUI標準Pixal3DのGLB出力をPhase 1へ接続。source-view校正とcanonical camerasを検証する。

**Phase 3 — Qwen-Image 2.1**：指定記事のAnyAngle LoRAとプロンプトの接続を基準に6方向出力。AI色を形状由来のアトラスに合成し、視点間整合性を検査。方向ごとの再生成を可能にする。

**Phase 4 — MiniMax H3-Image**：モデル専用conditioningとプロンプトで同じ6方向を生成。GPUを食うため、Qwen/H3を同時常駐させない。APIとローカルの選択肢、モデルの可用性、ライセンス制約を明示する。

**Phase 5 — 任意拡張**：TripoSplatガイド、3DGS→mesh、Visual Hullによる幾何補正、KTX2圧縮、テクスチャ品質の自動再試行、Comfy Registry公開。

## 13 QA、障害対策、達成基準

### 再現性・品質

- 6つの軸方向のプレビューと、最低12個の斜め方向でベイク前GLBとBoxを比較する。
- テスト形状：単純立方体、非対称形状、細い脚のある形状、空洞／開放形状、透過輪郭のある毛・葉の形状。
- top/bottomの画像が90度回転・鏡像化していないことを自動テストする。
- AI色とプロキシ輪郭のIoU、境界ずれ、各視点での不一致、色バランスをログに残す。閾値は実験で決定可能にし、暫定値を公式仕様として固定しない。
- 不合格の方向はプロキシ色へfallbackし、**何方向が置換されたか**をmanifestへ記録する。
- geometry-onlyとAI-enhancedのプレビュー切替で細部の増減を見分けられること。

### 運用

- ComfyUIのキャンセル・実行失敗でプロセスが残留せず、tempを後処理で削除する。生成中間ファイルはjob IDで隔離し、成功時にatomic renameする。
- 同じ入力画像、3D backend、参照姿勢、model revision、prompt、seed、パラメータからキャッシュキーを作る。API側でseedを保証しない場合は結果ファイルのcontent hashを使う。
- 3D/画像モデルの各フェーズが独立して再試行でき、Queueが1つの失敗で止まり続けないこと。
- 2モデルを切り替えても、同じ形式の`BOX_IMPOSTOR`が得られる。選択されないモデルの重みは読み込まない。
- H3-Imageローカル推論は大容量モデルのため、24GB級GPUではオフロードや量子化でも動作保証しない。API/Partner Nodes利用も可能にする。
- モデルウェイトをcustom-nodeの配布物に同梱しない。MiniMax H3-Image Community Licenseには適用地域の制限（EU、UK、韓国、米国を除外）があり、利用前に原文確認。Qwenおよび各依存モデルの商用条件も別に検査する。

**完成判定**：参照画像をComfyUIに読み込み、`qwen_enhanced`または`h3_enhanced`を選ぶだけで、6方向の見た目と正しいBoxのマップが作られ、Orbit可能なBox Impostorプレビューと永続保存アセットを返す。失敗時には不一致の箇所と使用したfallbackを記録する。

## 14 根拠ソース

- 指定記事・配布JSON： https://note.com/kongo_jun/n/ndbabf97a8f41?hl=en
- AnyAngle LoRAモデルカード： https://huggingface.co/lilylilith/QI_2.1_AnyAngle
- AnyAngle Studio T8： https://github.com/T8mars/Comfyui-Qwen-Image-2.1-MultiAngle-T8
- `three-impostor`のBaker/Shader： https://github.com/richardmonette/three-impostor/blob/main/Impostor.js
- Pixal3D公式ComfyUI解説： https://github.com/Comfy-Org/docs/blob/main/tutorials/3d/pixal3d.mdx
- Qwen-Image 2.1公式ComfyUI： https://github.com/Comfy-Org/docs/blob/main/tutorials/image/qwen/qwen-image-2-1.mdx
- H3-Image公式ComfyUI： https://blog.comfy.org/p/minimax-h3-image-is-now-available
- H3-ImageモデルREADME： https://github.com/MiniMax-AI/MiniMax-H3-Image
- H3-Image利用条件： https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/LICENSE
- ComfyUI Web拡張： https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/docs/extensions/development.md
- three.js WebGPURenderer： https://threejs.org/manual/pages/webgpurenderer

## Agentへの最初の具体的な指示

1. 直ちにAIモデル統合から始めず、**Phase 0の単独GLB→BoxBake→保存→再読込→360度プレビュー**を完成させる。ここが既存Bakerへの互換性を証明する最小単位。
2. 必ず`three-impostor`の現行ソースとREADMEを確認して実装。`intervalMap`の仕様は本書の説明だけで再実装しない。
3. 次に指定記事のGLBから後半を生成するJSONを参照し、Qwen-Image 2.1 + AnyAngleの条件付けを独立サブグラフとして移植する。
4. 保存方式は試験可能なmanifest+raw half-floatから始める。不要な依存や独自UIフレームワークを増やさない。
5. 各Phaseで実際の出力画像とテスト結果を確認し、READMEに制約・未実装・再現手順を更新する。
