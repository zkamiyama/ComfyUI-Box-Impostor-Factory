# ComfyUI Box Impostor Factory

GLBから6方向の正投影ガイドとthree-impostor互換のBoxアセットを作るComfyUIカスタムノード。元メッシュとBoxを同じカメラで比較できる。1面だけ色を変更しても、ベイク済みの法線・区間・占有範囲のマップを再利用する。

| 経路 | 検証状況 |
|---|---|
| GLB → Box → 保存 → 再読込 | RTX 4080上のWindows版EdgeとSwiftShaderでWebGL2・WebGPUを検証。街灯は1面2048 pxでも、ベイク直後と再読込後の18方向で色・深度・輪郭が一致した。 |
| CC0のPBRモデル | 木製街灯と金属ボトルで元メッシュとBoxを18方向から比較。街灯はRTX 4080でSD 1.5による6面編集も実行した。[比較画像・再現手順](docs/verification/cc0-pbr/README.md) · [生成済みの街灯Box](examples/cc0-lantern-sd15.box.zip) |
| ComfyUI、手動画像、プレビュー | CPU版ComfyUIの画面から01と04を読み込み、GLBのベイクと手動6面の採用を確認。キューで保存・再読込と1面更新も実行。ブラウザでメッシュとBoxの左右比較、視点変更、アトラス表示を操作。 |
| 軽量チェックポイントによる6面生成 | RTX 4080でSD 1.5を6面それぞれに実行。3種の幾何マップと輪郭を保持し、最終Boxの色に反映されることを確認。[実機の計測結果](docs/verification/rtx4080-sd15.json) |
| TripoSplatの追加ガイド | RTX 4080で街灯の参照画像から262,144個のガウスを生成。6方向に同じ位置・尺度補正を適用し、元メッシュとの平均輪郭IoUが0.545から0.812に改善した。[6方向の計測結果](docs/verification/cc0-pbr/lantern-triposplat.json) |
| Pixal3Dの単画像入力 | RTX 4080で同じ街灯の画像1枚を使用。軽量設定は75.71秒、標準設定は135.68秒。元メッシュと比較した未使用7視点の平均IoUはそれぞれ0.861、0.882だった。[軽量版のPBR付きBox](examples/cc0-lantern-pixal3d-compact.box.zip) · [標準版のBox](examples/cc0-lantern-pixal3d-default.box.zip) |
| Qwen-Image 2.1＋AnyAngle | 街灯の画像から生成した軽量版プロキシを使い、512 pxで6方向を編集した。真上だけseedを変えて6面とも採用。色を更新し、4枚の幾何・材質マップと最終描画の透明度を保持した。[編集済みPBR付きBox](examples/cc0-lantern-pixal3d-qwen-pbr.box.zip) · [比較画像・再現手順](docs/verification/cc0-pbr/README.md) |
| Qwenによる動物の6面編集 | CC0の歩行犬写真と保存済みPixal3Dメッシュから、Qwenだけで6方向の画像を生成。6面すべてを採用し、法線・区間・占有範囲・材質マップは維持した。ZIP展開後の18視点では色・深度の差が0。ただし面間の色差は残る。別の人物イラストは＋Z面の輪郭が合わず不採用。[6方向の比較](docs/verification/qwen-organic-dog-views.webp) · [計測結果](docs/verification/qwen-organic-20261113.json) · [完成Box](examples/cc0-walking-dog-qwen-six-20261113.box.zip) |
| MiniMax H3-Image | ワークフロー03をComfyUIの実行グラフに展開し、ガイドと元画像の入力順を切り替えられることを確認した。 |

## チャット向けThree.jsプレビュー

[チャット向けビューア](web/chat_viewer.html) は、公開済みの `.box.zip` をブラウザへ取得し、ZIP内のカラー・法線・区間・占有範囲のrawテクスチャとGLBを読み込む。描画には本リポジトリの `restoreBox` と `three-impostor` を使う。6面画像を箱に貼る代替表示ではない。ComfyUIサーバーやGPU推論は不要だが、ブラウザのWebGL2と公開ファイルへの通信は必要となる。

使用例は歩行犬のQwen 6面Box。初回表示では約35MBのZIPを取得する。回転・ズーム、元メッシュとBoxの左右比較、正面6方向、生成色とベース色、法線・深度・アトラスの表示を利用できる。読み込めない環境では静的な実機描画画像を表示する。

```sh
# 270-PCのWSLでWindows Edge／SwiftShaderを使用する
BI_SOFTWARE_GPU=1 BI_CHROMIUM='C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe' \
  scripts/wsl-windows-node.sh tests/chat_viewer_smoke.mjs .test-output/chat-preview.png
```

次回以降の最終アセット報告にも、同じチャット内プレビューに新しいZIPの公開URLを渡して使う。成功数や画質の根拠は[成果物の報告形式](docs/ARTIFACT_REPORTING.md)に従って添える。WebGL2が使えないチャット画面では静止画像とZIPへのリンクを併記し、実際の3D表示が動いたと断定しない。

## 汎用3D Inspector Skill

汎用3DモデルビューアとChatGPT向けSkillは [3D-Inspector-skills](https://github.com/zkamiyama/3D-Inspector-skills) に分離しました。完成したBox Impostorを [Three.jsとthree-impostorによるネイティブ描画](https://github.com/zkamiyama/3D-Inspector-skills/blob/main/docs/verification/native-box-20261113.md) で確認できます。このリポジトリのComfyUIノード、ベイカー、既存のブラウザプレビューは従来どおり残しています。

## ComfyUIでGLBを開く

Node.js 22以上、ComfyUIを実行するPython 3.11以上を使用する。以下の `python` はComfyUIと同じ環境の実行ファイルを指す。Windowsのポータブル版では同梱のPythonを指定する。

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/zkamiyama/ComfyUI-Box-Impostor-Factory.git
cd ComfyUI-Box-Impostor-Factory
python -m pip install -r requirements.txt
npm ci
npx playwright install chromium --only-shell
node tests/make_fixture.mjs ../../input/asymmetric.glb
```

LinuxでChromiumのシステムライブラリも必要な場合は `npx playwright install --with-deps chromium --only-shell` を使う。既存のChromiumを使う場合は環境変数 `BI_CHROMIUM` に実行ファイルの絶対パスを設定する。

ComfyUIを再起動し、[01_box_from_glb.json](workflows/01_box_from_glb.json) を画面にドロップして実行する。最初のノードのファイル名は `asymmetric.glb`。自分のGLBを使うときはComfyUIの `input/` に置き、この欄を変更する。正面を合わせるには `BI CanonicalizeProxy` の各軸の回転角（度）を指定する。

`BI PreviewBox` のボタンからビューアを開く。「左右で比較」を選ぶと、左にプロキシ、右にBoxが表示される。背景、照明、正投影／透視、6方向を切り替えられる。「色のみ」で照明の影響を外し、「法線」「ヒット深度」で同じ画素に見える表面を調べる。「アトラス・ガイド」では各方向の深度や、区間・占有範囲の各チャンネルを確認できる。

## ComfyUIを起動せずに試す

上記のリポジトリ直下で実行する。次の例は手続き生成した直方体で、行パディングの生じるR=128を検査する。`--verify` は保存後にChromiumを起動し直し、18方向の色・深度と輪郭を比較する。空の画像同士の一致は失敗として扱う。

```sh
node tests/make_fixture.mjs .test-output/asymmetric.glb
node worker/cli.mjs bake --input .test-output/asymmetric.glb --output .test-output/box --resolution 128 --backend webgl2 --maps roughness,metalness,ao --verify --evidence .test-output/evidence
node worker/cli.mjs inspect --asset .test-output/box
node worker/cli.mjs preview --asset .test-output/box
```

同じ非対称GLBを1面1024画素（カラーアトラス3072×2048）でも検証した。WebGL2・WebGPUとも、ベイク直後と新しいChromiumで再読込した後の18視点を各64×64で比較し、色・深度・輪郭の差は0だった。保存済みのBoxから1024×1024の透過PNGを書き出す試験も実施した。RTX 4080でCC0の街灯を1面2048画素・フルPBRでベイクし、保存・別プロセスでの再読込まで計測した。WebGPUは12.88秒、WebGL2は15.71秒だった。[R=1024の試験](docs/verification/high-resolution-roundtrip.json) · [実機の条件と計測値](docs/verification/cc0-pbr/rtx4080-render.json)

最後のコマンドが表示するローカルURLを開く。PNGだけを出力する場合は `node worker/cli.mjs thumbnail --asset .test-output/box --output .test-output/hero.png` を使う。自動選択には `--backend auto`、WebGPUを必須にするには `--backend webgpu` を指定する。CIと同じソフトウェア描画を試す環境変数は `BI_SOFTWARE_GPU=1`。入力や設定を変えたときは別の出力ディレクトリを選ぶ。

### WSLのComfyUIからWindowsのGPUでベイクする

Windows側のNode.js 22以降とEdgeまたはChromiumを使う。WSLのシェルで `node.exe --version` を確認し、このリポジトリのルートで次を設定する。ブラウザのパスは実際のインストール先に合わせる。

```sh
export BI_NODE="$PWD/scripts/wsl-windows-node.sh"
export BI_CHROMIUM='C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
export BI_SOFTWARE_GPU=0
```

同じシェルからComfyUIを起動する。通常の `BI BakeBox` と `BI RenderBox` がWindows側のワーカーを使う。ベイクしたアダプターはアセットのmanifestで確認できる。[RTX 4080での登録ノードの実測](docs/verification/cc0-pbr/rtx4080-comfy-waterbottle.json)

### ハードウェアGPUを使わずに統合テストを実行する

依存関係をインストールした状態で、リポジトリ直下から次を実行する。Python・JavaScriptの単体テストに続いて、ComfyUIを `--cpu` で起動し、GUIワークフローの読込み、Boxの作成・6面交換・保存・描画・形状更新を検査する。画像生成モデルの推論は実行しない。

```sh
bash scripts/verify_cpu_only.sh
```

このスクリプトはCUDAを無効化し、`BI_SOFTWARE_GPU=1` を設定する。WSLではWindows版Node.jsとEdgeを検出し、SwiftShaderで描画する。WSL側のChromiumがWebGL2を初期化できない環境でも試せる。Windows版Edgeが別の場所にある場合は、事前に `BI_CHROMIUM` を設定する。ComfyUIの場所を変更するには `BI_COMFY_DIR`、Pythonを指定するには `BI_PYTHON` を使う。実行証拠は `.test-output/cpu-only/` の実行別ディレクトリに残る。[270-PCのCPU-only検証記録（2026年11月13日）](docs/verification/cpu-only-270-20261113.json) に実行結果と既存配布ZIPの検査結果を記載した。

## ワークフローと入力

| ファイル | 入力と使い方 |
|---|---|
| [01　GLBからBox](workflows/01_box_from_glb.json) | 外部リソースを内包するGLB。モデルの重みは不要。 |
| [02　参照画像からQwen](workflows/02_box_from_reference_qwen21.json) | Primaryを1枚指定する。Pixal3Dはこの画像からGLBと6面ガイドを作る。Qwenの各面にはPrimaryとガイドに加えてSecondaryも渡せる。 |
| [03　参照画像からH3](workflows/03_box_from_reference_h3.json) | H3専用の分岐を使用する。共有の切り替えで、ガイドを編集する条件と元画像を編集する条件を比較できる。同じ面・seedで試す。 |
| [04　手動6面](workflows/04_manual_six_faces.json) | `input/` のGLBと `px.png, nx.png, py.png, ny.png, pz.png, nz.png` を指定する。前景アルファのあるPNGを使う。1面だけを適用するときは、ほかの入力枝を `BI FaceInputs` から外す。 |
| [05　Pixal3DからBox](workflows/05_pixal3d_baseline.json) | 参照画像から得たプロキシのベース色を使う。画像編集モデルを読み込まずに形状とカメラを確認できる。 |
| [06　保存済みBoxの1面を交換](workflows/06_replace_saved_face.json) | 現在のアセットIDまたは書き出し名と、交換したい面のPNGを指定する。初期接続は＋X。別の面に差すときは `BI FaceInputs` の接続先を変える。他の5面の色と幾何マップを保持する。 |
| [07　保存済みBoxを透過PNGにする](workflows/07_render_saved_box.json) | 先頭の `BI LoadBox` に自分のアセットIDまたは書き出し名を指定する。初期値は `fixture`。標準ノードで `IMAGE` と `MASK` を結合し、`output/BoxImpostor/` に透過PNGを保存する。 |
| [08　TripoSplatの追加ガイド](workflows/08_triposplat_guide.json) | 同じ物体の参照画像とGLBを指定し、位置を合わせたスプラットを選んだ面に投影する。位置合わせ後の3DGSを `Preview Splat` で回転して確認できる。デコードしたガウスのパラメータは別の面から見るときも再利用する。 |
| [09　修正GLBで形状を更新](workflows/09_rebake_proxy_revision.json) | 元のアセットIDまたは書き出し名と、同じ原点・単位・軸で編集したGLBを指定する。初期値は `fixture` と `corrected.glb`。新しいBoxと、修正前後の計測結果を出力する。 |
| [10　保存済みBoxの材質を推定](workflows/10_material_from_saved_box.json) | 生成した面を含むBoxのアセットIDまたは書き出し名を指定する。`PBR from generated faces` で推定方法と対象面を選ぶ。 |

02・03では `BI SecondaryReferences` に `ComfyUI/input/` からの相対パスを1行に1枚、最大8枚入力する。Secondaryは外観用で、未指定でも実行できる。02・03の初期値は、6面の画像生成、Qwen・H3の材質推定、最終Boxの各面ともに2048×2048。モデルパラメータを開くと、3つの解像度を個別に変えられる。Boxのサイズを変えると、位置合わせに使う6面ガイドも同じサイズになる。

02・03・05のモデルファイル名は各ローダーに設定済み。必要な重みと配置は公式テンプレートに従う。Qwenの第1画像は元画像、第2画像はガイド。LoRAはモデル側だけに適用する。H3では第1画像が編集対象になる。ガイドは余白を加えてモデルの正方形入力へ収める。生成後は同じ変換を逆にたどり、ガイドの画角に戻す。面別seedを変更すると、その面の生成からやり直せる。[Pixal3D](https://github.com/Comfy-Org/workflow_templates/blob/a8bc07a76478dfada740fa8fa49ac87be1bff519/templates/3d_pixal3d_trellis2_image_to_model.json) · [Qwen](https://github.com/Comfy-Org/workflow_templates/blob/a8bc07a76478dfada740fa8fa49ac87be1bff519/templates/image_qwen_image_2_1_image_edit.json) · [AnyAngle](https://huggingface.co/lilylilith/QI_2.1_AnyAngle) · [H3](https://github.com/Comfy-Org/workflow_templates/blob/a8bc07a76478dfada740fa8fa49ac87be1bff519/templates/image_minimax_h3_image_edit.json)

02・03を開くと、グラフの先頭にBox Impostorの操作ノードが表示される。Primary、Secondary、モデル、深度・材質の設定と6方向の途中結果をここで確認できる。直したい面のボタンでその面のseedを変え、ComfyUIの「Run」で実行する。入力や3D生成のseedを変えると、更新が必要な処理に `Dirty` と表示される。App Modeでは右の入力欄に同じ操作パネルが並ぶ。通常グラフではノードの幅を広げて6面を見比べられる。[6方向のガイドとBoxの操作画面](docs/verification/workflow-controls-graph.webp) · [App Modeの入力と出力](docs/verification/workflow-controls-app.webp)

`BI BakeBox` の `depth_source` は初期値が `geometry`。メッシュの座標を6方向から直接サンプリングする。`rgb_guided` を選ぶと、採用した面画像の色と元の深度を使い、色の境界や深度の段差をまたがない範囲で表面を平滑化する。補正した頂点から法線・区間・占有範囲をベイクし直し、6面を再度位置合わせする。この処理が変更できるのは既存メッシュの表面で、画像にだけ描かれた穴やパーツを追加する機能はない。

### 生成した面に合わせて材質を推定する

02・03・10の `PBR from generated faces` で `source` を選ぶ。`face` は位置合わせを通過した全方向（`all`）か、`+Z` などの1方向を指定する。

| source | ベースカラーと材質の扱い |
|---|---|
| `keep`（初期値） | 入力Boxの色と材質を使う。GLBのroughness・metalness・AOをベイクするには、上流のガイドとベイクの `material_maps` を `material` にそろえる。 |
| `qwen` / `h3` | 各モデルの画像編集を2回実行し、照明を除いた色と、roughness・metalness・AOの3チャンネル画像を生成する。出力の確認にはビューアの材質マップ表示を使う。 |
| `marigold` | Marigold IID Appearanceが色・roughness・metalnessを推定する。AOはGLBからベイクした値を使う。ベイクしていない場合は遮蔽を加えない。 |

元の面画像から輪郭と採用時の位置合わせを読み、推定マップをその解像度から直接Boxのタイルに写す。面画像を交換した場合は、その面の推定材質を破棄する。別の面の推定結果は次の実行で再利用できる。Qwen・H3の材質編集は初期値2048で、Marigoldの推定は長辺768画素。結果は指定したBox解像度で保存する。

Qwen・H3には、実行時に再照明するPBRレンダラーで使うことを明示する。ベースカラーは影・反射・色かぶりを除いたsRGB、Boxの材質マップはR＝roughness、G＝metalness、B＝AOの線形値を指定する。CLIでは `--size 2048 --material-source qwen --material-size 2048 --box-size 512` のように、推定を2Kで行い最終Boxだけを小さくできる。実行前のチェックに、選択した材質モデルの不足ファイルと追加Pythonパッケージが表示される。

旧プロンプトで歩行犬の＋Z面を256角に推定した実機比較では、Qwenは白い毛の一部をピンクに変え、H3では暗い帯が出た。Marigoldは白い毛の色を保ったものの、金属度を高めに推定した。保存したマップはビューアでチャンネル別に確認できる。[元画像と推定マップの比較](docs/verification/walking-dog-material-estimates.png)

同じ犬の＋Z面（元画像512 px）で、PBRの用途を明記したQwenとH3を各2048 pxで実行した。RTX 4080（16 GB）でベースカラーとRMAの2枚を作るのに、Qwenは28分32秒、H3は8分34秒かかった。

Qwenでは、白い毛の平均R−Gが旧256 pxで+0.328、旧2048 pxで+0.031、新256 pxで+0.008、新2048 pxで+0.009だった。尾先の細かい毛筋は新2048 pxで描き足された。H3では旧・新どちらの指示でも256 pxで横帯が出て、2048 pxで解消した。旧2048 pxは毛色を平坦な領域に分け、新2048 pxは毛の模様を残した。

平均金属度は不透明な28,490画素でQwenが0.502、H3が0.457となり、毛の一部に強い金属反射が出た。両方の結果を2048 pxの＋Zタイルに適用し、残り5面・透明度・幾何マップを保持したまま、Three.jsのビューアで光源角度を変えて描画した。[尾先の細部](docs/verification/qwen-pbr-tail-detail.webp) · [2K Boxの再照明](docs/verification/pbr-box-relighting.webp) · [プロンプト・解像度別の全マップ](docs/verification/pbr-prompt-resolution-dog.webp) · [機材・条件・前景の計測値](docs/verification/pbr-prompt-resolution.json)

Marigoldを使うにはComfyUIのPython環境に `diffusers`、`transformers`、`accelerate` を入れ、モデル一式を `ComfyUI/models/diffusers/marigold-iid-appearance-v1-1/` に置く。モデル直下に `model_index.json` が必要になる。[公式モデルカード](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1)

## BoxをIMAGEとしてレンダリング

`BI RenderBox` は保存済みのBoxを指定方向から描画し、透過PNG相当の `IMAGE` とComfyUI方式の `MASK`（背景=1）を出力する。画角は正投影・透視から、視点は6軸と斜め方向から選ぶ。幅・高さは64〜2048px。出力は条件別にキャッシュし、元の `colorMap`、`intervalMap`、`rangeMap` を変更しない。

`BI PreviewBox` の `IMAGE` サムネイルでも `sampling` を選べる。標準は `standard`。硬い境界の色にじみを抑えたい場合だけ `reduce_bleeding` を使う。この設定は描画時の `intervalMap` サンプリングを最近傍にするもので、穴・細部や深度が悪化する場合がある。ビューアでも「補間」から切り替えられる。

CLIから画像を得る場合:

```sh
node worker/cli.mjs render --asset .test-output/box --output .test-output/side.png --view +Z --projection orthographic --width 512 --height 512 --sampling reduce_bleeding
```

## 実モデルで1面から試す

ComfyUIの `input/` に参照画像を置き、ComfyUIを起動した状態で、リポジトリのルートから実行する。最初のコマンドで、起動中のComfyUIが認識しているモデル・ノード・参照画像を照合する。

```sh
node scripts/run_model_workflow.mjs --model qwen --input reference.png
node scripts/run_model_workflow.mjs --model qwen --input reference.png --face=+Z --run
```

全6面を生成するには `--face=+Z` を外す。H3は `--model h3`、Pixal3DによるGLBとBoxの確認は `--model pixal3d` を使う。ComfyUIの接続先は `--origin` で指定できる。実行記録は既定で `.test-output/models/` に残る。各面の採否と理由、実際に動いたサンプラー、ガイドと生成画像はその記録で確認できる。重みを同じファイル名で交換した場合はComfyUIを再起動する。

既存GLBで外観だけ試す場合は、参照画像と同じ物体のGLBを `input/` に置く。次の例ではQwenが＋Z面を512角で生成する。H3でも同じ指定を使える。H3は内部の参照画像を最低1024画素に拡大する。実行記録の `workflow.json` は、同じ設定でComfyUIの画面から開ける。

```sh
node scripts/run_model_workflow.mjs --model qwen --input reference.png --mesh matching.glb --face=+Z --size 512 --run
```

街灯を試したRTX 4080（16 GB）ではPixal3Dの標準設定と軽量設定の両方が完走した。短時間で試すには `--compact-proxy` を使う。4段のサンプリングを保ち、再メッシュを256、目標面数を10万、UVと法線マップを1024 pxにする。材質アトラスを保存するときは `--material-maps material+emissive` を足す。発光材質のないメッシュではラフネス・メタルネス・AOを保存する。Qwen・H3の6面のうち1面のseedを変えるには、たとえば `--face-seed=+Y:321` を指定する。`--face` は単独の面を試すときに使う。

Pixal3Dを通る02・03・05では、実際に使った切り抜きを `reference/cropped.png` に保存する。`manifest.reference.camera` にその寸法・推定水平画角（度）・焦点距離（ピクセル）を記録する。単画像用のPixal3Dの座標系とカメラを確認できるアセットでは、ビューアの「参照画像と元のメッシュ」「参照画像とBox」で、入力の切り抜きと同じカメラから形状を比較できる。

### 軽量モデルで6面の色と最終描画を検査する

SD 1.5のFP16チェックポイント（約2.13 GB）を `ComfyUI/models/checkpoints/` に置き、上の手順で作った `asymmetric.glb` を使う。参照メッシュの6枚のガイドをそれぞれimg2imgに渡す。輪郭にはガイドのアルファを使う。[チェックポイントの固定版](https://huggingface.co/Comfy-Org/stable-diffusion-1.5/blob/96a4eb63b0056942bd4c59222cdd5e6e2c9d0f89/v1-5-pruned-emaonly-fp16.safetensors)

```sh
python scripts/run_lightweight_smoke.py --mesh asymmetric.glb --backend webgl2 --resolution 256 --steps 6 --run
```

`.test-output/lightweight/run-*/` に6枚の生成画像、変更前後のBoxの透過PNG、manifest、検査結果が残る。各面の前景に色が反映された画素数、最終描画の色の変化、法線・区間・占有範囲のバイト一致を記録する。

RTX 4080の初回キューは256角・各6ステップで10.90秒。斜視の前景14,929画素すべてでRGBが変わり、透明度は全画素で一致した。推論はCUDA、Boxのベイクと描画はWSL2のChromium（WebGL2・SwiftShader）を使った。GPU全体の使用量は1秒間隔で計測し、開始時1,835 MiB、観測最大3,901 MiBだった。

| 元のBox | SD 1.5で6面を編集したBox |
|---|---|
| ![元の斜視](docs/verification/sd15-box-before.png) | ![編集後の同じ斜視](docs/verification/sd15-box-after.png) |

## TripoSplatをガイドに使う

`BI TripoSplatGuide` にComfyUIの `SPLAT` と、同じ物体のGLBから作った `BI MakeSixGuides` の `guides` をつなぐ。`face_id` で方向を選び、スプラットを元のGLBに位置合わせした場合は `source_glb`、正規化後のGLBに合わせた場合は `canonical_box` を指定する。位置・向き・スケールが違う場合は上流の `Transform Splat` で合わせる。

コマンドで試すには `node scripts/run_model_workflow.mjs --model triposplat --input reference.png --mesh matching.glb --face=+Z` を実行する。参照画像とGLBをComfyUIの `input/` に置く。重みとGLBの事前確認が通ったら `--run` を付ける。ガイド画像、同じ面のメッシュ画像、輪郭・深度の計測値を保存する。

出力の `guide` は画像生成に渡すRGB画像、`foreground` は物体を1とするマスク。`qa_json` でメッシュとの輪郭IoU、境界距離、重なる画素の深度順位相関を確認できる。Boxの法線・区間・占有範囲は、この場合もGLBからベイクする。

RTX 4080の実機試験では、同じ非対称GLBの＋Zガイドから参照画像を作り、公式重みで20ステップ推論した。＋Zの初回実行は60.28秒、保存済みスプラットを別方向の＋Xに投影した2回目は0.98秒。下表は手動位置合わせ前の測定値で、1に近いほど輪郭が重なる。[実行条件・重み・深度の計測値](docs/verification/rtx4080-triposplat.json)

| 方向 | GLBのガイド | TripoSplatのガイド | 輪郭IoU |
|---|---|---|---:|
| ＋Z | [元メッシュ](docs/verification/tripo-pz-mesh.png) | [スプラットの投影](docs/verification/tripo-pz-splat.png) | 0.298 |
| ＋X | [元メッシュ](docs/verification/tripo-px-mesh.png) | [スプラットの投影](docs/verification/tripo-px-splat.png) | 0.500 |

## 保存・再利用・品質の確認

ComfyUIでは `output/box_impostor/<asset_id>/manifest.json` がアセットの入口になる。GLB、RGBA16Fの幾何マップ、カラー、6方向のガイドと深度をまとめて参照する。`BI BakeBox` の `material_maps` でroughness・metalness・AOやemissiveを選べる。roughness・metalness・AOは上流と同じRGBA8アトラスに保存する。`BI SaveBox` で `output/box_impostor_exports/<name>/` に書き出し、`BI LoadBox` でアセットID・書き出し名・SaveBoxの出力パス・inputからの相対manifestパスを指定する。

Boxを1ファイルで渡すには `BI PackBox` にアセットと書き出し名をつなぐ。`output/box_impostor_exports/<name>.box.zip` に、GLB、manifest、各マップ、ガイド、QAを可逆圧縮する。`BI LoadBox` は書き出したZIPのパスや `input/` に置いたZIPの相対パスも開ける。展開時に各マップを検証し、同じZIPを再度開くときは取り込み済みのアセットを使う。

各面の深度ガイドは、物体上を0〜1の距離、背景を半精度の `NaN` として保存する。古い書き出しには、透明な画素の深度が0のものがある。外部で読み込むときはガイドPNGのアルファで背景を判定できる。UV付きGLBの試験には、PNGを内包し、色の目印と透明な穴を持つ [テスト用モデルの生成スクリプト](tests/make_textured_fixture.mjs) を使う。

手動・生成画像の整列では前景輪郭を比較する。採用した面でも元のアトラスのcoverageは保持し、線形RGBだけを合成する。輪郭が基準を満たさない指定面はプロキシのベース色に戻す。結果の `qa/report.json` に方向別の理由と、深度で照合できた共通表面の色差が残る。直方体のように隣接カメラから同じ面がほとんど見えない場合は比較不能として扱う。

ビューアの「6面の採否」を開くと、各面の表示色と判定理由が分かる。`BI InspectFaceSilhouette` は、生成画像を元のガイドの画角に戻し、**位置合わせ前**の前景を重ねて表示する。オレンジは生成・手動画像にだけある領域、水色はGLBのガイドにだけある領域。それぞれを個別のマスクとして取り出せる。面画像やカメラ変換が足りない場合は、計測できなかった理由を `qa_json` に記録する。

修正したGLBを採用するときは、09の `BI RebakeProxyRevision` を使う。元のBoxの解像度とマテリアルマップの選択を引き継ぎ、修正GLBから6方向のガイド、カラー、幾何マップを作り直す。元のアセットIDは引き続き使える。同じプロキシを渡した場合は既存のBoxを返す。`qa/geometry-revision.json` には元のGLBの座標系で比較した方向別の輪郭の重なりと深度差、相手の画角から外れた前景画素数が残る。視線方向が異なる面は比較不能として記録する。

保存前後の一致と、元メッシュに対する近似精度は別々に測る。下の比較では、斜めから見た輪郭は近いが、硬い辺を持つ物体のBox内部に隣の面の色が混じる。これは元のBakerで描いた段階から再現する。[18視点での元メッシュとの比較値](docs/verification/proxy-box-r128.json)

![同じ視点から見た手続き生成の非対称GLBと、上流Shaderで描画したBox](docs/verification/viewer.png)

元メッシュの見え方を基準に、区間マップの補間だけを変更する診断も行った。同じ硬い辺を持つ直方体をR128でベイクし、正面6方向と斜視12方向で重なる前景画素を測った。深度はカメラのnearからfarを0〜1とした値。どちらも誤差が小さいほど元メッシュに近い。

| 同じアセットの描画条件 | 色の平均絶対誤差（8bit、18方向の平均） | 深度誤差の95パーセンタイル（18方向の最大） |
|---|---:|---:|
| three-impostor 0.4.1の標準 | 22.29 / 255 | 0.001716 |
| intervalMapのみ最近傍補間（比較実験） | 2.10 / 255 | 0.002209 |

貫通穴を持つ薄い枠でも比較した。最近傍補間で色の平均誤差は22.48から7.84に下がり、最も輪郭が崩れた視点のシルエットIoUは0.825から0.934に上がった（1が完全一致）。穴を見通せた10方向の計6,799画素のうち、標準ではすべて背景のまま、最近傍補間では11画素が物体の色で埋まった。配布版は上流の標準設定を使用する。[色付き直方体の比較画像](docs/verification/color-fringe-comparison.png) · [直方体の18方向の計測値](docs/verification/color-fringe-comparison.json) · [開いた枠の18方向の計測値](docs/verification/open-frame-r128.json)

厚みのない板ポリゴンは、ガイドやカラーアトラスに正しく写っていても、最終Boxの描画から消えることがある。UVテクスチャ付きの直方体と、そこから離れた透過板を同じGLBに入れた試験では、正面から見た前景が元メッシュの1,162画素から644画素に減った（64×64で比較）。色にじみ低減を有効にしても、この向きの輪郭IoUは0.554のままだった。[透過板の18視点比較](docs/verification/textured-cutout-r128.json)

別のGLBでも、上記のCLIでベイクした後に `node tests/compare_proxy.mjs .test-output/box .test-output/comparison.json webgl2 --compare-interval` を実行すると、同じアセットを使って両方の補間を比較できる。枠のGLBは `node tests/make_open_frame.mjs .test-output/open-frame.glb` で作成できる。

## テストと設計資料

```sh
npm test
python -m unittest discover -s tests -p 'test_*.py' -v
```

GPU付きブラウザを使うビューア試験は `BI_TEST_ASSET` にベイク済みアセットを指定して実行する。GitHub Actionsでは描画結果とmanifestを バックエンド別の `box-roundtrip-*` アーティファクトに保存する。ComfyUI統合試験は固定コミットの本体を起動し、HTTPキュー、書き出し、再読込、1面更新を検査する。さらに配布用JSONをブラウザで読み込み、01・04・07・09を実行する。モデルを使うワークフローは実行用グラフに展開し、記録は `comfy-frontend-workflows` に保存する。[ブラウザ試験](https://github.com/zkamiyama/ComfyUI-Box-Impostor-Factory/actions/workflows/box-roundtrip.yml) · [ComfyUI試験](https://github.com/zkamiyama/ComfyUI-Box-Impostor-Factory/actions/workflows/comfy-native.yml) · [実装指示書](docs/ComfyUI_BoxImpostor_Agent_Handoff.md) · [先行研究](docs/research.md)

このリポジトリのコードはMIT。ワークフローの出典とライセンスは [Comfy-Orgの表記](workflows/LICENSE-Comfy-Org.txt) に収録した。モデル本体とLoRAは各配布元の条件に従う。Qwen-Image 2.1本体の商用利用は別ライセンスを要する。H3の公開ウェイトには利用地域および出力の取り扱いに条件がある。[Qwen本体のライセンス](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) · [H3のライセンス](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)
