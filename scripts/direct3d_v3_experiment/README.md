# Direct3D-V3 全身メッシュ実験

実行日: 2026年11月21日（日本時間）

Direct3D-V3をRTX 4080のローカル環境で実行し、同じ人物の既存メッシュから512段階と1024段階の形状を生成した。全身表示では手指と靴に改善が見られた。一方、髪の塊感と服の平面的な帯は残り、初期形状の問題を全面的に解決したとは評価しない。

## 今回の比較

元の画像は1024×1024の全身写真である。入力メッシュは以前の実験で生成したPixal3D int8とTRELLIS.2のshape/texture VAEを組み合わせた結果を使用した。Pixal3DとTRELLIS.2を独立した2モデルとして比較した実験ではない。

処理順は「元メッシュ → Direct3D-V3 512 → Direct3D-V3 1024」。1024段階は512出力を粗形状として受け取り、512段階で背景除去した同一画像を再利用した。解像度だけを変えて同じ粗形状から生成した比較ではない。

各段階を別プロセスで実行し、steps 15、CFG 4、seed 46、float16、max_tokens 150000に統一した。各プロセスでseedをリセットしているため、公式の一括実行と乱数列まで一致するとは主張しない。顔の切り抜きは行っていない。

## 実測値

| 項目 | 元メッシュ | 512段階 | 1024段階 |
|---|---:|---:|---:|
| 頂点数 | 49,522 | 187,579 | 750,060 |
| 三角形数 | 81,180 | 375,162 | 1,500,124 |
| 形状生成時間 | — | 7.26秒 | 23.45秒 |
| 段階全体の時間 | — | 30.80秒 | 33.85秒 |
| GPU全体の観測最大使用量 | — | 10,362 MiB | 8,134 MiB |
| 当該プロセスのTorch最大割当量 | — | 5,928 MiB | 4,276 MiB |
| 使用した表面パッチ数 | — | 3,218 | 13,101 |

形状生成時間は生成処理の実測値であり、モデルの読み込みや背景除去を含まない。段階全体の時間は各実行プロセス内の処理と出力保存を含み、環境構築、重みのダウンロード、別プロセスでの事前ハッシュ確認、比較動画の作成を含まない。

1024段階では背景除去済み画像を再利用した。512段階とのGPUピークの差から、1024の方が一般に省メモリだとは判断できない。GPU全体の値は200ms間隔のnvidia-smi観測値で、他プロセスも含み、短いピークを取り逃す可能性がある。Torchの値は当該プロセスのみを対象とする。

実測の原記録は [512](evidence/rgb512.json)、[1024](evidence/rgb1024.json)、[入力来歴](evidence/prepared.json) に保存した。

## 見た目の評価

評価には全身を収めた灰色材質の表示を用いた。色や元のテクスチャに依存する細部の印象を除き、同じカメラと照明で形状を比較した。

- 元メッシュで塊に近かった手に、512段階で指の筋と分離が現れた。
- 512段階では靴の輪郭、甲、踵を読み取りやすくなった。ただし細い部分には不自然さが残る。
- 1024段階は512段階から局所的に変化した。顔の深い溝のような陰影は弱まったが、元写真への高い再現性を確認したわけではない。
- 胴体からスカートに大きな平面の帯が残る。元メッシュより角張って見える部分もあり、全身の細部が一様に改善したとは言えない。
- 髪は大きな一塊のままで、背面の髪、体形、脚の輪郭も512と1024で大きく変わらない。

これは今回の灰色表示の目視評価である。正解の3Dスキャンとの誤差評価、テクスチャ品質の評価、他の人物への一般化は行っていない。

## 表示の検証

最初のプレビューでは3D-Inspectorの既定の三角形数上限により面が間引かれ、512出力が粒状に見えた。元の375,162面に対し、初回HTMLに含まれた面は187,581面だった。このプレビューは品質評価から除外した。

比較スクリプトは公式view.pyに `--max-triangles` と各メッシュの全三角形数を渡す。出力HTMLのpayloadを復号し、描画される三角形数と元メッシュの三角形数が一致することを、撮影開始前の必須条件にした。公式HTMLは生成後に編集していない。

比較用の灰色GLBは頂点位置と面の接続を保持し、同一の材質と法線計算方法を使用する。動画撮影時は共通の境界範囲を公式viewerへ渡し、カメラの範囲と照明位置を揃える。配布する単体viewerは公式の表示動作をそのまま保つ。

初回表示の原因と検査値は [表示の診断記録](evidence/inspector-face-stride-diagnosis.json) に記録した。最終比較の完了状態、各フレームのカメラ差、画面端での欠け、動画の復号結果は [比較検証](evidence/comparison.json) を参照する。

最終動画は1536×584、24fps、240フレームの10秒である。全240フレームで各モデル間のカメラ値の差は0、全身の画面端での欠けは0だった。動画を最後まで復号し、240フレームを確認した。3つのviewerはそれぞれ81,180面、375,162面、1,500,124面をすべて保持している。

今回は形状そのものを変更する実験であるため、Boxのalpha一致は合格条件にしていない。

## 公開版の範囲と残る検証

確認した公開版は画像と粗いメッシュを入力として形状を生成する。粗形状の生成モデルとテクスチャ生成モデルは未公開である。今回の結果は既存の初期3D生成の出力にDirect3D-V3を適用したものである。

RGB入力は公式APIで受理される。作者は細かな形状を得る条件画像として法線画像または白いクレイ画像を推奨している。今回は元のRGB写真による試験であり、その推奨入力による品質は未検証である。今回生成したGLBは形状のみで、Direct3D-V3によるテクスチャ生成やBox用テクスチャの完成を意味しない。

公開範囲と入力条件の出典: [公式README（実行時の版）](https://github.com/DreamTechAI/Direct3D-V3/blob/b93ad9c0a451729bb76b60ce69cbf076e76e95af/README.md)。

## 実行環境

| 項目 | 使用した版・設定 |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 / 16GB |
| Python | 3.12.3 |
| Torch / Torchvision | 2.9.1+cu128 / 0.24.1+cu128 |
| Transformers | 4.57.6 |
| FlashAttention | 2.8.3 |
| CUDA拡張のビルド対象 | CUDA 12.8 / sm_89 |
| Direct3D-V3のcommit | b93ad9c0a451729bb76b60ce69cbf076e76e95af |
| 生成モデル | wushuang98/Direct3D-V3 |
| 生成モデルのrevision | 1f42e11da6efbc8cc0661c62132b0533c338e3d3 |

公式の `from_single_file` を使用し、必要な1段階だけを読み込む。DINOv3-H+はTripoSplat用に導入済みのローカル重みを再利用した。重みは変更せず、実際のキーと形状の一致、およびstrict state_dict loadを確認した。配布元とSHA256は入力来歴に記録している。

生成メッシュは公式の正規化座標から入力メッシュの座標へ戻して保存する。実装上のボクセル化の0.5倍を含め、逆変換の倍率は入力範囲を `0.95 × token_scale` で割った値になる。追加の軸回転は加えていない。各段階の実際の変換値は実行記録に保存した。

## 再実行

この実験の入力来歴に対応する準備、推論、比較スクリプトを同じディレクトリに保存した。任意の別画像向けの汎用ComfyUIノードではない。

```bash
python prepare_inputs.py --reference REFERENCE.png --coarse COARSE.glb --dino-weights DINO.safetensors --output INPUT_DIRECTORY
python run_direct3d.py --repo DIRECT3D_REPO --model MODEL_DIRECTORY --prepared INPUT_DIRECTORY/prepared.json --output RGB512_DIRECTORY --resolution 512 --steps 15 --cfg 4 --seed 46
python run_direct3d.py --repo DIRECT3D_REPO --model MODEL_DIRECTORY --prepared INPUT_DIRECTORY/prepared.json --output RGB1024_DIRECTORY --resolution 1024 --steps 15 --cfg 4 --seed 46 --coarse RGB512_DIRECTORY/canonical.glb --image RGB512_DIRECTORY/foreground.png --background-removed
python compare_geometry.py --before INPUT_DIRECTORY/coarse-original.glb --after RGB512_DIRECTORY/canonical.glb --after-label 'Direct3D-V3 512 / RGB' --extra 'Direct3D-V3 1024 / RGB' RGB1024_DIRECTORY/canonical.glb --out COMPARISON_DIRECTORY
```

準備と生成の出力先には新しいディレクトリを指定する。比較は公式3D-Inspector、既存captureスクリプト、Playwright、FFmpegの場所を引数で指定できる。

