# Box Impostorの実現性と先行事例

調査日 2026-11-12 ｜ 対象は、参照画像1枚から静止物体のBox Impostorを作るComfyUIワークフロー。実装手順とテスト仕様は[実装指示書](./ComfyUI_BoxImpostor_Agent_Handoff.md)にまとめた。

## どこまで実例があるか

既存GLBからのBoxベイクには、そのまま利用できるBakerと描画用マテリアルがある。参照画像と3Dガイドから別角度の画像を作る工程にも公開作例がある。両者の間に残る開発課題は、異なる視点の生成画像で同じ表面の模様をそろえ、Bakerが記録した輪郭・法線・奥行きを保ったまま色を差し替えることにある。[three-impostor 0.4.1](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/README.md) · [混合順のGLB入力ワークフローと作例](https://note.com/kongo_jun/n/ndbabf97a8f41?hl=en)

| 先例・原典 | 一次資料で確かめられること | このプロジェクトに引き継ぐ判断 |
|---|---|---|
| Max・Deussen・Keating（1999）、Oliveira・Bishop・McAllister（SIGGRAPH 2000） | 直交する観測画像から別視点を描く。後者の図13–14はBBoxの6面に変位つきの画像を配置し、斜視では別の面に投影がまたがる例を示す。 | 軸の6枚に加え、辺をまたぐカメラ移動で模様のずれと隠れ面を検査する。原典と現在のBakerは深度の表現・描画アルゴリズムが異なる。[1999年の著者ページ](https://graphics.uni-konstanz.de/publikationen/Max1999HierarchicalImageBased/index.html) · [Relief Texture Mapping](https://doi.org/10.1145/344779.344947) |
| Richard Monette、three-impostor 0.4.1 | Boxは6面から色・法線と軸ごとの占有区間をベイクし、12三角形の箱内をレイマーチしてヒット深度を書く。各軸が保持する内部の隙間は最初の1つ。裏面の向きが意味を持ち、開いたメッシュは開口側へ延長することがある。 | 椅子の脚と奥の胴、開底の器、二重の空洞を別々に測る。深い穴・ガラス・投影シャドウを初版の画質保証から外す。[実装と制限](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/README.md#notes) |
| 混合順、AnyAngle作者、AnyAngle Studio T8 | 元画像と、メッシュやスプラットを新しいカメラで描いた粗い画像をQwenに渡す。混合順はPixal3Dで作ったGLBから後半だけ再開するJSONも配布。T8はカメラと元画像・ガイド・結果の実例を公開している。作者はガイドの形状誤りが出力に残る例を示している。 | Qwenに最も近い先例。カメラ操作とLoRAの使い方を流用し、直交6面への整列・面ごとの再生成・Boxでの斜視品質は本プロジェクトで測る。[作例](https://note.com/kongo_jun/n/ndbabf97a8f41?hl=en) · [LoRAモデルカード](https://huggingface.co/lilylilith/QI_2.1_AnyAngle) · [T8](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-MultiAngle-T8/blob/f920d97fd9477a807a8d68e59bf3174a70fa1b30/README.md) |
| Hunyuan3D-Paint 2.1、Text2Tex（ICCV 2023）、MVPaint（CVPR 2025） | Hunyuanはメッシュと参照を受け取り、各視点の法線・位置から外観を生成してUVへ焼く。Text2Texは面と視線の角度を使って更新領域を選ぶ。MVPaintは複数視点を同期し、見えない領域の補完と継ぎ目の処理を行う。 | 深度で相互に見える同じ表面点を求め、模様と色を比較する。同じメッシュで「UVに外観を作って既存Bakerに通す」経路を任意の比較対象にできる。[Hunyuanの実装](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dpaint/textureGenPipeline.py#L119-L192) · [Text2Tex](https://daveredrum.github.io/Text2Tex/) · [MVPaint](https://mvpaint.github.io/) |
| Pixal3D、TripoSplat | Pixal3Dは入力視点で画素と3Dを対応させてメッシュを生成する。TripoSplatは単一画像から3D Gaussianを生成する。ComfyUIでは両者の出力型が異なる。 | Pixal3Dの入力視点と物体固有の正面は別々に保存する。スプラットは追加の視点ガイドに利用し、既存Bakerには同じ座標の三角形メッシュを渡す。[Pixal3D論文](https://arxiv.org/abs/2605.10922) · [Pixal3D公式テンプレート](https://github.com/Comfy-Org/workflow_templates/blob/a8bc07a76478dfada740fa8fa49ac87be1bff519/templates/3d_pixal3d_trellis2_image_to_model.json) · [TripoSplatの出力型](https://github.com/Comfy-Org/ComfyUI/blob/c5751c81abe131202eea9e4c66d000589013dd0b/comfy_extras/nodes_triposplat.py#L140-L181) |
| MiniMax H3-Image | ComfyUI公式のネイティブ編集とPartner Nodeがあり、複数画像を番号で指定できる。ローカル版では第1画像が編集対象。VAEを接続すると画像が参照latentとしても条件付けに入る。 | ガイドを第1画像に置く実験と、参照画像を第1画像に置く実験を比較する。確認した作例・公式資料からは、H3で指定の6面をBoxに整合させた結果までは判断できない。[ComfyUI公式](https://blog.comfy.org/p/minimax-h3-image-is-now-available) · [編集ノードの実装](https://github.com/Comfy-Org/ComfyUI/blob/c5751c81abe131202eea9e4c66d000589013dd0b/comfy_extras/nodes_minimax_h3.py) |

## 判断を分ける比較実験

| 実験 | 固定するものと変えるもの | 何を判定するか |
|---|---|---|
| A　GLBとBaker直後のBox | カメラ、物体姿勢、照明、解像度を共有する。既知の柄を持つ非対称な物体を使う。 | シルエット、輪郭の穴、斜視の奥行き差でBox表現自体の限界を測る。 |
| B　Baker直後と再読込後のBox | 同じベイク結果を別プロセスに読み込み、同じbackendで描く。 | マップごとの数値、深度、画素差で保存・復元を判定する。Aの近似誤差をBの許容差に流用しない。 |
| C　色を一度取り出し同じ色を戻す | カラーの取り出し、タイル配置、再合成だけを通す。次に、既知の1面だけを識別色へ置換する。 | 面ID、上下反転、線形色空間、輪郭のアルファ、1面だけを更新できることを調べる。 |
| D　Qwen / H3 / 手動6面 | 同じGLB・参照・ガイド・出力解像度で、1条件ずつ変更する。生成と合成にかかった時間を分ける。 | 面別の採用率、共通可視部の色差、柄の位置ずれ、斜めからの二重像、1面再実行の範囲。生成を飛ばした面を成功数に数えない。 |

Dでは、正解用の多視点画像を持つ物体と、参照が1枚しかない物体を分ける。前者は別視点の忠実度を測れる。後者で判断できるのは、見えている部分の再現、ガイドとの整合、生成した裏側が隣の視点と矛盾しないかまでである。無地の立方体では左右反転に気付きにくい。片側だけに把手を付け、上下面に向きのある記号を置く。

表示が軽くなるかは画面上の物体サイズにも左右される。同一カメラでGLBとBoxのGPUフレーム時間、画面占有率、同時表示個数、テクスチャ使用量を記録する。解像度を上げるとマップ容量は概ね面解像度の二乗で増える。高精細化は同じ表示サイズで見分けられる模様と、必要なGPUメモリの両方で判定する。

## 根拠をたどった経路と残る問い

- 幾何側は、元指示書のBakerからパッケージ定義、カメラ、アトラスの書き込み、Shaderのサンプリングへ進み、最後にthree.jsの画素読み戻しを照合した。この結果、任意の材質アトラスはRGBA8であること、WebGPUの行パディングを詰める必要があることを保存仕様に反映した。[Baker](https://github.com/richardmonette/three-impostor/blob/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768/Impostor.js#L584-L602) · [three.js r186の読み戻し](https://github.com/mrdoob/three.js/blob/r186/src/renderers/webgpu/utils/WebGPUTextureUtils.js#L803-L861)
- 生成側は、配布作例からLoRAの失敗例、ComfyUIのノード本体、公式ワークフローの入出力まで照合した。研究側は、BBoxの6面を使う原典から、ガイド条件つきの多視点生成とメッシュへの再投影を行う実装を調べた。各方式の出力形式が異なるため、画像・UVマップ・Boxアトラスをそのまま交換できるという根拠はない。
- 次に必要な証拠は、同じ非対称GLBを使った保存前後の画像対と、同じガイドで生成した面の比較結果である。調べた一次資料の範囲では、両画像モデルから得た結果に、このBakerの保存・再読込と斜視評価まで組み合わせた公開ベンチマークは確認できなかった。

## 今回実行したBakerの確認

Node.jsでthree.js r186の配布ビルドとthree-impostor 0.4.1のソースを読み込み、Bakerに描画呼び出しを記録するレンダラーを渡した。入力は親Groupの位置が (19,19,19)、子Meshの中心が (1,2,3)、寸法が (4,2,1) の発光直方体。実際の `bake(group, {layout: 'box', resolution: 41})` が設定した値を照合した。[three.js固定版](https://github.com/mrdoob/three.js/tree/9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8) · [Baker固定版](https://github.com/richardmonette/three-impostor/tree/4d525ca3aecc5ed2009d6bd9b8c6d6c289248768)

| 照合項目 | 観測値 |
|---|---|
| ベイク範囲 | min = (−1.08, 0.92, 2.42)、max = (3.08, 3.08, 3.58)。親の移動は範囲に含まれない。 |
| カメラ | 6台の位置・up・投影範囲・near/farが期待値と一致。例として+Xの位置は (3.496, 2, 3)、左右は±0.58、上下は±1.08、nearは0.208、farは4.784。 |
| RenderTarget | 色・法線・材質・発光は123×82、区間は123×41、占有範囲は18×6。材質はRGBA8、ほかはRGBA16F。占有範囲のみnearest、全マップのmipmap生成は無効。 |

この試験で照合したのは、描画前にBakerが設定する値である。Chromiumは起動時のソケット作成が実行環境に拒否され、GPUでの描画まで進まなかった。画素値、上下の向き、保存・再読込後の描画一致、Qwen/H3の推論品質には実測値がない。
