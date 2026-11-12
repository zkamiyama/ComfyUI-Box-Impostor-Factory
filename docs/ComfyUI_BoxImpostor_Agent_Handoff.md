# ComfyUI Box Impostor Factory 実装指示書

v0.3 ｜ 仕様、受け入れ基準、実装の検証状況

GLBのベイク・保存・再読込とComfyUIノードは実装済み。現在使えるワークフロー、導入手順、GPU推論の検証状況は [README](../README.md) に記載した。元メッシュとBoxの形状・色の差は [描画比較の記録](./verification/proxy-box-r128.json) を参照。

参照画像1枚から静止物体のメッシュを作り、6方向の見た目を個別に編集できるBox Impostorとして保存する。表示にはthree.jsとthree-impostorの既存Shaderを使う。最初の実装は、手元のGLBだけでベイク・保存・別プロセスでの再読込・360度プレビューまで通す。画像モデルの統合はその後に進める。[先行事例・比較実験・CPUでの確認結果](./research.md)

## 1　完成条件

利用者はComfyUI上でGLBまたは参照画像を入力し、元メッシュとBoxを同じカメラで比較できる。外観の入力は `geometry_baseline`（プロキシの色）、`manual_faces`（手動画像）、`qwen_enhanced`、`h3_enhanced` の4通りとする。どの入力からも同じ形式の保存アセットを返す。

1面だけを差し替えたときは、その面の整列・品質判定、影響する面間比較、カラーアトラスの合成、プレビューを更新する。GLB生成、幾何ベイク、ほかの5面の生成は再実行しない。再起動後も保存済み工程を再利用する。

元の色に戻した面は面ID・理由・枚数を表示する。ベイクや保存に失敗した場合はアセットを公開しない。モデル別の対応は、実際に生成した6面と斜視での結果を記録してから「検証済み」と表示する。経路ごとの検証状況を以下に示す。

| 経路 | 本改訂で確認したこと | 次に必要な証拠 |
|---|---|---|
| GLB → Box | WebGL2・WebGPUのソフトウェア描画で、保存前後およびZIP展開前後の色・深度・輪郭を18視点で比較した。 | 実機GPUでの互換性・速度 |
| Pixal3D → Qwen＋AnyAngle | 実際のComfyUIフロントエンドで、6面ぶんの生成画像と前景マスクの接続を確認した。 | 本プロジェクトの固定カメラを用いた6面と斜視での採用率 |
| H3-Image | ComfyUIに配布ワークフローを読み込み、元画像とガイドの順序、編集結果と前景マスクの接続を確認した。 | 画像の順序を変えた比較、正面・背面・上下面の一致、メモリ実測 |
| TripoSplat | 合成PLYをネイティブのSPLATとして読み込み、GLBの6方向のカメラで色付き目印の座標を照合した。 | 学習済みモデルで生成したスプラットと同一物体のGLBの位置合わせ・品質 |

## 2　幾何と外観の分担

標準の形状入力は正規化した三角形GLBとする。Boxの占有区間、占有範囲、法線、カラーの輪郭は、そのGLBを1回ベイクして決める。AIは各面の色と見た目を提案する。1枚の参照に写らなかった背面や底面には正解画像がないため、推定した外観として記録する。

AIを使った形状補正を試す場合はPhase 5に置く。元のベイク画像と生成画像を参照し、マスク・不一致領域や修正案を保存する。採用した修正から別バージョンのプロキシを作り、Bakerで全幾何マップを再生成する。学習器の出力を配布用の `normalMap`、`intervalMap`、`rangeMap` に直接書き込む経路は設けない。補正の採否は同じカメラで元のプロキシと比較できるようにする。

## 3　処理の順序と保存単位

| 工程 | 入力 | 保存する出力 |
|---|---|---|
| 参照の準備 | IMAGE、任意MASK、切抜き条件 | 元画像、参照RGB、前景coverage、切抜き・リサイズの変換 |
| 形状の確定 | Pixal3D出力または既存GLB、姿勢指定 | 正規化GLB、入力からの変換と逆変換、BBox、参照カメラ |
| ガイドの描画 | 正規化GLB、ベイク範囲、面解像度 | 6組のRGB・coverage・深度、カメラ行列。必要に応じて法線・表面ID |
| 幾何ベイク | 同じGLB、Bakerの版、解像度、材質設定 | 元のカラーアトラスと幾何・材質マップ。ガイドと共有するベイク範囲 |
| 面ごとの生成と整列 | 元画像、該当面のガイド、選択モデルまたは手動画像 | 生成原寸、整列後の色とcoverage、変換、モデル条件、合否 |
| 合成と公開 | 保存済みベイク、面別の採用結果 | 新しいカラーアトラス、manifest、品質レポート、プレビュー画像 |

ガイドはBakerと同じカメラ定義から作る。ガイドとベイクは同じ形状を参照するため、面の生成結果を待たずに実行できる。ベイク時の `data.box` がガイド用に計算したBBoxと違った場合は、面を生成する前に停止する。

## 4　形状・画像モデルの接続

### Pixal3DとTripoSplat

Pixal3Dの公式テンプレートでは、テクスチャを適用した枝の `ApplyTextureToMesh → MeshSmoothNormals → MeshToFile3D` から `FILE_3D_GLB` を受け取る。頂点色プレビューの枝との取り違えを防ぐため、既知の柄がある入力で材質とUVを確認する。既存GLB入力ではこの生成を省略する。Pixal3Dは入力視点の画素と3Dを対応づける。切抜き、余白、推定画角を保存し、参照が斜めならそのまま参照姿勢として扱う。意味上の「正面」は必要に応じて手動で合わせる。[公式テンプレート](https://github.com/Comfy-Org/workflow_templates/blob/a8bc07a76478dfada740fa8fa49ac87be1bff519/templates/3d_pixal3d_trellis2_image_to_model.json) · [Pixal3D論文](https://arxiv.org/abs/2605.10922)

TripoSplatのデコーダーは位置・スケール・回転・不透明度・球面調和係数を持つSplatを返す。Phase 5でガイドとして使う場合も、Boxベイクには別途GLBが必要になる。スプラットからメッシュを抽出する実験では、抽出法と生成物を別の形状として記録する。[デコーダーの出力定義](https://github.com/Comfy-Org/ComfyUI/blob/c5751c81abe131202eea9e4c66d000589013dd0b/comfy_extras/nodes_triposplat.py#L140-L181)

### 6面の生成

| 項目 | Qwen-Image 2.1＋AnyAngle | MiniMax H3-Image |
|---|---|---|
| ネイティブ分岐 | `TextEncodeQwenImage21` にVAEを接続し、モデルにAnyAngle LoRAを適用 | `MiniMaxH3ImageConditioning` に専用VAEとテンプレートのテキストエンコーダーを接続 |
| 画像の順序 | `image_1` が元画像、`image_2` が目的視点の粗いガイド | `image_1` が編集対象。候補Aはガイドを第1画像、元画像を第2画像とし、候補Bでは逆順にする。役割をプロンプトにも明記 |
| 比較の開始条件 | 作者の指定はLoRA強度1、CFG 3、20 steps以上。通常の編集テンプレートのCFG 1とは別のプリセットにする。編集指示は `Change the camera angle from <image2> to <image1>.` | 公式ローカル編集テンプレートの6 steps、CFG 1、KSamplerの `euler/simple` を保持。画像の参照名は `<Picture 1>`、`<Picture 2>`。A/Bでseed・キャンバス・ガイドを固定する |
| 解像度と透過 | 第1参照の縦横比が出力latentの寸法に影響する。両入力を記録済みの変換で同じキャンバスに合わせ、1方向ずつ渡す。RGBAが得られた場合に限りアルファを取り出す | 自動寸法は第1画像に依存する。比較時は縦横を明示する。出力をマッティングし、輪郭をガイドと比較する |
| 採否 | ガイド由来の形状誤り、文字・部品の入替え、背面・底面の不一致を面別に記録 | 2画像編集が通ることと、指定視点を保てることを別々に判定する。Qwen用LoRAは読み込まない |

参考実装は[Qwen公式テンプレート](https://github.com/Comfy-Org/workflow_templates/blob/a8bc07a76478dfada740fa8fa49ac87be1bff519/templates/image_qwen_image_2_1_image_edit.json)、[Qwenエンコーダー](https://github.com/Comfy-Org/ComfyUI/blob/c5751c81abe131202eea9e4c66d000589013dd0b/comfy_extras/nodes_qwen.py#L111-L181)、[AnyAngleモデルカード](https://huggingface.co/lilylilith/QI_2.1_AnyAngle)、[H3公式テンプレート](https://github.com/Comfy-Org/workflow_templates/blob/a8bc07a76478dfada740fa8fa49ac87be1bff519/templates/image_minimax_h3_image_edit.json)、[H3編集入力の説明](https://blog.comfy.org/p/minimax-h3-image-is-now-available)を参照する。

Qwen用・H3用のComfyUIサブグラフは同じ `FACE_VIEW_RESULT` を返す。入力は元画像、1面のガイド、`face_id`、seed、出力寸法、設定。出力は面ID、RGB(A)、元画像からの変換、モデル・LoRA・ワークフローのリビジョン、実際の生成設定とする。6面でモデルローダーを共有し、選択されなかったモデルの読み込みを防ぐ。最初は上位ワークフローを2本に分ける。ノードから同じComfyUIキューに `/prompt` を投入して待機する構成は避ける。

ローカル版とPartner Node/API版は別の検証結果を持つ。APIキーはサーバー側から読み、ワークフロー・キャッシュ・manifestに含めない。seed非対応なら `null` とし、実際の出力を保存する。24 GB級GPUでのH3推論、量子化版、Turboは実測してから対応環境に加える。

## 5　カメラと座標系

canonical空間は右手系、Y-up。ローカルGLBを一様にスケールし、採用した原点・単位・正面・変換行列を保存する。Bakerは入力ルート自身の平行移動・回転・スケールをリセットする。正規化済みの子を単位変換の親Groupに入れるか、変換を頂点へ適用してから渡す。入力メッシュのローカルBBox、正規化後の `tightBounds`、Bakerが返した `bakeBox` を区別する。空または非有限のBBoxはエラーとする。[Bakerの座標処理](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/Impostor.js#L395-L429)

Bakerは `tightBounds` の最大辺の2%を各側に加える。下表の `x,y,z` は、この余白付きBBox内で0から1に正規化した点。`face_id` は中心から見たカメラ位置の側を表し、カメラは中心を向く。タイル番号はShader側のUVで示す。PNGの先頭行とrawの先頭行は別途定義する。[6方向の定義](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/Impostor.js#L91-L130)

| face_id | camera up | 面内 (u,v)、vは上向き | アトラス (列,行) | 投影幅 × 高さ |
|---|---|---|---|---|
| +X | +Y | (1−z, y) | (0,0) | Z × Y |
| −X | +Y | (z, y) | (1,0) | Z × Y |
| +Y | −Z | (x, 1−z) | (2,0) | X × Z |
| −Y | +Z | (x, z) | (0,1) | X × Z |
| +Z | +Y | (x, y) | (1,1) | X × Y |
| −Z | +Y | (1−x, y) | (2,1) | X × Y |

ファイル・JSONの面IDはASCIIの `+X,-X,+Y,-Y,+Z,-Z` とする。余白付きBBoxの面に垂直な方向の半長をh、余白付き最大辺の0.1倍をmとすると、カメラ位置は中心＋side×(h+m)、nearはm/2、farは2h+1.5m。各面の幅と高さをそのまま正投影範囲とし、R×R画素に描く。横長の物体もBakerのタイルは正方形になる。[カメラ設定](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/Impostor.js#L447-L496)

生成モデルに自然な縦横比のガイドを渡す場合は、幅W・高さHの面を一様に縮放して正方形の中央に置く。L=max(W,H)として、横座標は `0.5+(u−0.5)W/L`、縦座標は `0.5+(v−0.5)H/L`。この余白・切抜き・上下反転を含む往復変換、view行列、projection行列、行列の格納順を面メタデータに記録する。画像を整列した後、Bakerのタイル座標に戻す。面ごとに輪郭の外接矩形を取り直して中心・スケールを変えない。

## 6　Boxテクスチャと色の合成

互換対象は `three-impostor 0.4.1` の `ImpostorBaker` と `ImpostorNodeMaterial`。Rは1面のベイク解像度、Cは `ceil(R/8)`。下表はGPU値をそのまま再現する配布形式とする。[Bakerのアトラス生成](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/Impostor.js#L584-L736)

| マップ | サイズ：一般式／R=256 | 格納型・フィルター | 値の意味 |
|---|---|---|---|
| colorMap | 3R×2R／768×512 | RGBA16F・linear | 線形の表面色とベイクしたcoverage |
| normalMap | 3R×2R／768×512 | RGBA16F・linear | 0..1に符号化したcanonical空間の法線。coverageを掛けた値を保持 |
| intervalMap | 3R×R／768×256 | RGBA16F・linear | X,Y,Z各軸に1タイル。Rは高側、Gは1−低側、B/Aは最初の内部空隙の上端・下端 |
| rangeMap | 3C×C／96×32 | RGBA16F・nearest | 8×8セルと周囲のサンプルから得る占有範囲。RGがlow/high、BAは0/1 |
| materialMap（任意） | 3R×2R／768×512 | RGBA8 UNORM・linear | roughness、metalness、AOを共有アトラスに格納 |
| emissiveMap（任意） | 3R×2R／768×512 | RGBA16F・linear | 発光する材質があり、emissiveを要求した場合の発光色 |

全マップのmipmap生成は無効。フィルター列はmin/magの両方に適用する。`intervalMap` には空隙が閉じた場合の符号化、輪郭近傍の補正、開口側の延長がある。`rangeMap` はこれと組にして使う。どちらも検査用の可視化PNGとは別に、Bakerが返した値を保存する。

**カラーの更新単位は、採用した面のRGBとする。** Shaderは色・法線・材質を `colorMap.a` で割り戻し、各面のcoverageと法線で重み付けする。生成画像のアルファはマッティング、位置合わせ、画素の採否に使う。出力のAにはベイク済みcoverageのビットを保持する。[Shaderのサンプリング](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/Impostor.js#L1426-L1462)

生成画像をstraight RGBとして解釈し、色プロファイルを記録して線形色へ変換する。ベイクしたcoverageをa、採用重みをwとすると、アトラスに書くRGBは `(1−w) × baselineRGB + w × a × generatedLinearRGB`。すでにpremultipliedの入力は先にstraightへ戻す。完全な空画素のRGBは0。w=0の面・画素にはベースのバイト列をそのまま使う。黒い模様は有効色とする。照明や影がすでに描かれた面は照明なしの比較表示でも確認し、PBR照明を掛けて二重の陰影が出れば採用を見直す。

生成原寸とアトラス内での解像度を別々に記録する。R=256に2Kの画像を戻した場合、最終アトラスに残るのは1面256画素である。Shaderは半texelのクランプにもRを使う。初版はマップ一式を同じRで生成する。Rを上げる実験ではVRAM、ベイク時間、レイマーチのsteps、細い形状の欠損を再計測する。[Shaderと解像度](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/Impostor.js#L1206-L1235)

## 7　実行環境

| 部分 | 採用技術とバージョン管理 |
|---|---|
| ノード・整列 | ComfyUIネイティブノード＋Python、NumPy、Pillow、OpenCV。ComfyUI本体・フロント・ワークフローの版を記録し、その組み合わせでテストする。 |
| ベイク | Node.jsからPlaywrightでChromiumを起動し、ブラウザ内の `WebGPURenderer` でBakerを呼ぶ。three-impostorの参照コミットは `4d525ca3aecc5ed2009d6bd9b8c6d6c289248768`。three.jsのCPU試験はr186。実機試験で使うthree.jsのパッチ版・Playwright・Chromiumも固定する。 |
| プレビュー | three.js、TSL、`ImpostorNodeMaterial`、OrbitControls。ComfyUI同一オリジンのWeb Extensionと、Phase 0用の単独ビューアで同じ復元関数を使う。 |
| 検査 | pytestでスキーマ・マスク・整列・キャッシュを、JSの単体試験でカメラとバイナリを、PlaywrightでGPU描画と再読込を検査する。 |

ワーカーは実際に選ばれたWebGPUまたはWebGL2 backendを記録する。開始時に小さいRGBA16FのMRT描画・読み戻し・DataTexture再アップロードを試す。WebGPUが使えない環境では `WebGPURenderer({forceWebGL:true})` を同じ試験に通す。どちらも失敗したら、backend、失敗段階、例外を返す。ソフトウェアGPUで通った結果は、実GPUでの速度とVRAMの測定から区別する。[three.jsのbackend選択](https://threejs.org/manual/pages/webgpurenderer) · [上流の依存条件](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/package.json)

## 8　ComfyUIのノード

次の `BOX_*` と `FACE_VIEW_RESULT` は本プロジェクトで定義する型である。3Dファイルと画像は接続先が受け付けるComfyUIの型に合わせる。各ノードは画面上の表示名のほかに、変更しない内部IDを持つ。

| ノード | 入力 | 出力と役割 |
|---|---|---|
| BI PrepareReference | IMAGE、任意MASK、その極性 | RGBと前景coverage（不透明=1）、参照メタデータ。元のアルファがなければ未指定とし、マッティングまたは明示された不透明条件を使う。 |
| BI CanonicalizeProxy | FILE_3D_GLBまたは既存GLB、姿勢 | 正規化GLB、tightBounds、変換・逆変換、参照姿勢 |
| BI MakeSixGuides | 正規化GLB、ベイク条件 | 6方向のIMAGEと `BOX_FACE_META`。RGB、深度、coverageを同じカメラで描く。 |
| BI FaceInputs | 各面の生成出力、任意の手動入力 | 面IDをキーとする `BOX_FACES`。重複・未知のIDを拒否し、未指定の面にはベース色を指定する。 |
| BI AlignAndValidateFaces | BOX_FACES、ガイド、カメラ | 方向別の位置合わせ、面間比較、採用重み、警告。色候補と `qa/report.json` を返す。 |
| BI BakeBox | 正規化GLB、R、材質設定、任意の整列済み面 | ベイクキャッシュから幾何マップを取得し、採用色を合成する。返す `BOX_IMPOSTOR` はmanifestを参照する。 |
| BI PreviewBox | BOX_IMPOSTOR | preview URL、静止IMAGE、元メッシュとの比較UI |
| BI SaveBox / BI LoadBox | BOX_IMPOSTORと保存先／保存済みmanifest | 持ち出せるアセット一式／検証済みのBOX_IMPOSTOR。Load時にモデルやBakerを起動しない。 |
| BI RenderBox | BOX_IMPOSTOR、カメラ、投影、画像サイズ、補間 | RGBのIMAGEと背景を1とするMASK。色にじみ低減はプレビューと最終描画で選ぶ。 |
| BI PackBox / BI LoadBox | BOX_IMPOSTORと書き出し名／保存済みの `.box.zip` | 可逆ZIP／展開・検証済みのBOX_IMPOSTOR。幾何マップのビット列を保持する。 |
| BI InspectFaceSilhouette | BOX_IMPOSTOR、面ID | 位置合わせ前の輪郭差を重ねたIMAGE、余分な前景・欠けた前景のMASK、計測JSON。 |
| BI RebakeProxyRevision | 元のBOX_IMPOSTOR、同じ元座標系の修正BOX_PROXY、描画backend | 修正版のBox、視線が一致する面の輪郭・深度を照合したJSON。元のアセットIDを保持する。 |

ComfyUIの `LoadImage.MASK` は透明側が1で、アルファのない入力は64×64のゼロマスクを返す場合がある。既知のLoadImage由来なら `1−mask` を前景coverageに変換する。外部マスクは極性と有効性を明示し、寸法だけで判定しない。QwenのRGBAは必要に応じてRGBとアルファに分け、最後に同じcoverage規約へそろえる。[LoadImageの定義](https://github.com/Comfy-Org/ComfyUI/blob/c5751c81abe131202eea9e4c66d000589013dd0b/nodes.py#L1764-L1805)

## 9　ファイル形式と再利用

`output/box_impostor/<asset_id>/` の下に次を保存する。各ファイルのパスはmanifestからの相対パスとする。

| パス | 内容 |
|---|---|
| `manifest.json` | 形式の版、生成物・親バージョン、モード、状態、面別採否、下記の読込情報 |
| `geometry/proxy.glb`、`reference/` | 正規化GLBと、入力に使った画像・coverage・参照姿勢。GLBだけのモードでは参照画像は任意。 |
| `guides/<face_id>/`、`views/<face_id>/` | ガイドRGB・深度・マスク・カメラ。生成原寸、整列後画像、変換、手動入力を上書きせず保存。 |
| `maps/` | `color_base.rgba16f.bin`、`color.rgba16f.bin`、`normal.rgba16f.bin`、`interval.rgba16f.bin`、`range.rgba16f.bin`。任意で `material.rgba8.bin` と `emissive.rgba16f.bin`。同内容の実体は共有可。 |
| `qa/report.json`、`previews/` | 面・面対の指標、失敗理由、比較画像、テスト条件 |

### manifestとrawの読み方

`schemaVersion`、`asset_id`、`mode`、`layout="box"`、`resolution`、`tightBounds`、`bakeBox` のmin/max、GLBの変換、6面の順序とカメラ、Bakerが実際に返した `maps` リストを必須とする。マップごとにパス、幅・高さ、channels、dtype、endian、行順、rowBytes、色空間、alpha規約、min/magFilter、wrap、mipmap、byteLength、SHA-256を保存する。参照入力、プロキシ、モデル・LoRA・テンプレート・ワーカーの版、seed、prompt、実GPU/backendも記録する。手動入力とGLBのみのモードでは該当しない項目を明示する。

ディスク上はヘッダーなし、RGBA順、行に余白なし、16FはIEEE 754 binary16のlittle-endian。行順は `increasingShaderV` と定義し、先頭行をShaderのv=0側の画素中心に対応させる。PNGの先頭行は画像の上端なので、面内vと混同しない。各backendからこの行順に変換する箇所を1か所に集め、非対称パターンで確定する。範囲外のパス、未知のmajor版、重複面、欠落した必須マップ、寸法・バイト数・ハッシュの不一致はロードエラーにする。

`renderer.readRenderTargetPixelsAsync` の戻り値から、行ごとの実データだけをコピーする。r186のWebGPUはrow strideを256バイトの倍数に切り上げ、最終行の末尾にはパディングを付けない。例えばR=128のrangeMapは48×16で、1行384バイト、GPU上の読み戻しstrideは512バイト、保存長は6,144バイト。各backendの型とstrideを検査する。16Fのビット列を8bit画像に通さない。[WebGPUの読み戻し](https://github.com/mrdoob/three.js/blob/r186/src/renderers/webgpu/utils/WebGPUTextureUtils.js#L803-L861) · [WebGL2の読み戻し](https://github.com/mrdoob/three.js/blob/r186/src/renderers/webgl-fallback/utils/WebGLTextureUtils.js)

復元時は16Fを `Uint16Array`＋`HalfFloatType`、RGBA8を `Uint8Array`＋`UnsignedByteType` のRGBA DataTextureにする。サンプラー設定と行順を復元し、`needsUpdate` を設定する。ベイク済みの線形RGBにはsRGBの再デコードを掛けない。空間データは色変換を無効にする。`Box3` から `BoxGeometry(size.x,size.y,size.z)` を作り、BBoxの中心へ平行移動する。マテリアルには `layout, box, resolution, maps, colorMap, normalMap, intervalMap, rangeMap` と実在する任意マップを渡す。復元用のアダプターがジオメトリとテクスチャの破棄も担当する。[ImpostorDataのgeometryと所有関係](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/Impostor.js#L292-L341)

### キャッシュと公開

| 変更 | 再計算するもの |
|---|---|
| +Xのseed、prompt、手動画像だけを変更 | +Xの生成または手動取込、+Xの整列、+Xが関わる面間比較、色の合成、manifest、プレビュー |
| 同じ画像でQA閾値を変更 | 保存済み面の指標を再判定し、採否が変わる場合は色を再合成 |
| GLBの内容・正規化姿勢・撮影範囲を変更 | 影響を受けるガイド、幾何ベイク、各面の生成・整列と後段。元の参照画像は再利用 |
| ベイク解像度Rを変更 | 幾何マップ一式とタイル座標への再サンプル。モデルに渡すガイド・キャンバスの内容が同一なら生成原寸を再利用 |
| カメラが見る向き、露出などプレビュー操作だけを変更 | プレビュー描画のみ |

キャッシュキーは工程名・実装版・入力内容のSHA-256・出力に影響するパラメータから作る。ファイル名だけをキーにしない。APIのseedが固定できない場合もリクエスト条件と返った画像のhashを関連付ける。明示的な「この面を再生成」で別の生成結果を採用できる。

再起動後に高価なノードを飛ばすには、画像生成やPixal3Dを評価する前にキャッシュを引く。軽量な入力情報からキーを確定し、ヒット時は保存パスから復元する。ミス時に限り、遅延入力として接続した生成サブグラフの出力を要求する。手動差し替え時も元の生成分岐は評価しない。再起動直後の再実行でモデルローダーの呼出し回数を検査する。[ComfyUIの遅延評価](https://docs.comfy.org/custom-nodes/backend/lazy_evaluation)

ジョブごとに一時領域を設け、同じキーの同時生成は1回にまとめる。全バイトの書込み・検証が成功してから同一ファイルシステム上でatomic renameし、manifestを最後に公開する。失敗・キャンセル・OOM時は一時領域とワーカーを片付ける。既存の公開版は参照できる状態を保つ。

## 10　プレビュー

同じカメラと照明で `Proxy GLB / Box（元の色）/ Box（採用色）` を即座に切り替え、並列表示も選べるようにする。正面6方向、斜視、Orbit/Pan/Zoom、正投影／透視、背景色／チェッカー、照明回転に対応する。色の比較には照明なしの表示も用意する。GPU検証用にはヒット深度、法線、区間、占有範囲、面別のガイドと生成結果を個別に表示する。

任意でsteps・露出・AO・ワイヤーフレームを切り替え、表示条件をスクリーンショットに添える。三角形が12枚であることだけを高速化の根拠にしない。実際のカメラ距離・表示サイズでGPU時間を測る。専用ルートは同一オリジンの `/box-impostor/` 配下に置き、サーバーはasset_idから公開済みファイルだけを解決する。再起動後の表示はmanifestから再現する。

## 11　実装ファイルの配置案

以下はこれから作る構成を示す。調査時点のリポジトリには設計資料とライセンスがある。

| 配置 | 役割 |
|---|---|
| `__init__.py`、`pyproject.toml`、`requirements.txt` | ノード登録、`WEB_DIRECTORY`、Pythonの最小依存 |
| `nodes/{reference,canonicalize,guides,faces,qa,bake,preview,assets}.py` | 各ノードの入出力、実行とエラーの通知 |
| `core/{schema,geometry,image_alignment,quality_metrics,file_store,worker_client}.py` | 共通データ型、座標、整列・指標、永続キャッシュ、ワーカー呼出し |
| `worker/src/{bake,guide_renderer,export_maps,import_maps,worker_server}.ts` | three.jsのBaker、固定カメラ、raw入出力、復元アダプター。依存はlockfileで固定 |
| `web/extension.js`、`web/viewer/` | ComfyUI内の操作と単独ビューア |
| `workflows/` | `01_box_from_glb.json`、`02_box_from_reference_qwen21.json`、`03_box_from_reference_h3.json`、`04_manual_six_faces.json` |
| `examples/`、`tests/`、`docs/` | 自作の非対称GLB、既知正解、CPU・GPUの試験、再現手順、環境・ライセンス・結果 |

## 12　開発順序と各段階の終了条件

| Phase | 最初に実装するもの | 段階を完了できる証拠 |
|---|---|---|
| 0 | 単独GLB → 正規化 → 既存Baker → raw＋manifest → 別プロセスでLoad → Orbit | 非対称GLB、全4必須マップ、任意マップの有無、6正面＋斜め12方向、深度、上下の向き、同一backendの保存前後差を記録 |
| 1 | ノードとWeb Extension、手動6面差し替え | GUIから01と04を実行。1面更新で幾何ベイク0回・他面生成0回、再起動後Load、元メッシュ比較。 |
| 2 | 公式Pixal3Dのテクスチャ付きGLBを既存入力へ接続 | 参照の切抜きと推定画角を記録。参照姿勢に戻した輪郭と手動補正後の向きを提示。既存GLB経路とも同じ形式で読める。 |
| 3 | Qwen＋AnyAngleの1面、次いで6面 | 手動の既知画像で合成を先に検証。同一GLBでベース色・手動・Qwenの斜視を比較し、採否・失敗条件・所要時間を記録。 |
| 4 | H3-Imageの独立分岐 | 入力順A/BとQwenを同じガイドで比較し、同じ後処理・保存・再読込を通す。未選択モデルのロード0回。 |
| 5 | TripoSplatガイド、形状補正、圧縮、任意の配布 | 各拡張を個別に試験。形状変更後は再ベイク。圧縮後の深度・輪郭・画質を未圧縮版と比較。 |

Phase 0の最初のfixtureには中心をずらした直方体を使い、面ごとに異なる色、上下面に回転を識別できる印、片側に突起を付ける。Bakerを直接呼ぶ基準描画と、保存ファイルだけを渡した別プロセスの描画を同時に用意する。検査に失敗したら生のタイル・カメラ・行順を比較できるよう、元のベイク結果も残す。

## 13　受け入れテストと失敗時の動作

| 試験 | 方法 | 合否・記録 |
|---|---|---|
| カメラと座標 | 非正方形、原点外の子、ルート・子の変換、上下の矢印、面別の既知パターンを描画 | 6面IDと投影先、左右と上方向、中心、変換の往復を自動照合。面の交換、鏡像、90度回転は不合格。 |
| 型とraw | 16Fの0、0.5、1、隣接する16F値を含む独立した既知パターンを使用。R=256とR=128を試験。PBR全有／全無、発光なしも確認 | ディスク上のbyteLength、SHA-256、ビット列を比較。行パディングが残る、8bitを経由する、optionalとmapsの内容が食い違う場合は不合格。 |
| 描画の往復 | ベイク直後と、プロセスを終了してLoadしたBoxを同一backend・カメラ・露出・照明・stepsで描画 | 全6正面＋少なくとも12斜視。各マップをサンプルする診断描画と最終色・深度を対比。同一環境での繰返し差を先に測り、色・深度の許容値を記録して固定。別backendは別の結果として記録。 |
| Boxの形状近似 | GLBとBoxの輪郭・ヒット深度、別メッシュを貫く前後関係を比較。脚、貫通穴、開底形状、alpha-cutoutの葉も試す | 輪郭IoU、深度差の分位点、面積、最悪視点を記録。奥の空隙が埋まる例と、保存時に生じた差を区別する。 |
| アルファと色 | 不透明な黒、透明、半透明の境界、既知の色票。無変更のカラーを取り出して戻す経路と1面だけ変更する経路を試す | 元のcolor.a、ほか5面のRGB、全幾何・材質マップのhashが一致。通常の半透明材質の再現は個別の対応項目とする。 |
| 個別の整列 | ガイドと候補の前景で平行移動・一様拡縮から始める。変換量に上限を設け、許容範囲を超える変形や左右反転が必要な場合はベース色に戻す | 面ごとに輪郭IoU、境界距離p95（pxとR比）、ランドマークのずれを変換の前後で記録。空マスクや裏返った文字は個別に検出／目視判定する。 |
| 面どうしの整合 | プロキシ上の同じ点を各カメラへ再投影し、両方の深度と一致する対だけで色・模様の位置を比較。輪郭、掠れ角、鏡面は別集計 | 色差、比較可能点数・面積率、最悪の面対、辺を横切る軌道での二重像を記録。重なりが少ない面対は「比較不能」。閾値は既知柄・手動整列の基準画像で調整し、評価セットの前に版を固定する。 |
| 障害と再実行 | キャッシュ済み状態から1面更新、同時ジョブ、破損raw、GPUメモリ不足、キャンセル、再起動を試す | 工程ごとの実行回数を記録。面不合格はベース色へ戻して理由を表示。GPU失敗・破損・欠落では成功manifestを公開しない。 |

評価セットには正解画像を持つ手続き生成アセットと、参照が1枚だけの実例を含める。成功面数・ベース色に戻した面数・比較不能な面対を併記する。全6面をベース色に戻した画像をモデルの生成成功として計上しない。レポートには実行環境、モデル版、最大VRAM、生成時間、再実行時間、表示サイズごとのGPU時間を残す。

## 14　依存関係と公開条件

three-impostorはMIT。Bakerの表現には閉じたメッシュが適し、複数段の深い空隙や開いた底は近似される。上流版では投影シャドウに制限がある。GPU実装の互換性は実際に通過した組み合わせごとに掲示する。[上流README](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/README.md#notes)

Qwen-Image 2.1本体はQwenのResearch Licenseで研究・評価用途を規定し、商用利用には別のライセンスを求める。AnyAngle LoRAのカードに記載されたApache-2.0はLoRA側の条件である。H3-Image Community LicenseはEU・英国・韓国・米国を適用地域から除き、第V条4項は出力の使用・配布・表示にも触れている。コード、モデル本体、LoRA、生成例、外部APIの契約を別々に記録する。公開リポジトリにはモデルウェイトを同梱せず、GPU検証には自作の手続き生成アセットを使う。[Qwen本体](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) · [AnyAngle](https://huggingface.co/lilylilith/QI_2.1_AnyAngle) · [MiniMax H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)

初回の開発報告には、Phase 0の固定バージョン、実行手順、使ったGLB、Baker直後と再読込後の同一視点画像、数値差、失敗した環境を添える。生のマップとプレビューを照合できてからPhase 1へ進む。
