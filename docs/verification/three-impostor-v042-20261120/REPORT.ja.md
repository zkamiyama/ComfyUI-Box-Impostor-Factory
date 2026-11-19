# three-impostor 0.4.2 の互換検証

## 版を切り替えた表示結果

既存アセットFとQ2の表示条件を `unlit / colorSampling=view` に揃えて0.4.1・0.4.2で描画したところ、45度の顔と頭部の切り出し画像は両アセットとも完全に一致した。この比較では、0.4.2への更新による顔の細部や二重像の改善は見られなかった。[比較結果](version-comparison.json)

各1024×1024 px画像で、RGBAの値が変わった画素数は次のとおりだった。全8組でalphaとカメラ行列は一致し、元の画像・深度マップのバイト列も一致している。RGBの最大差は1〜2階調だった。これは版による描画差の比較であり、形状精度の採点ではない。

| アセット | 0度 | 30度 | 45度 | 60度 |
| --- | ---: | ---: | ---: | ---: |
| F | 0 | 3 | 0 | 8 |
| Q2 | 0 | 3 | 4 | 6 |

この実写アセット比較では描画深度を取得していない。深度の保存前後の一致は、後述する独立fixtureで確認した。

## 変更内容

Factory と 3D-Inspector の Box 描画を 0.4.2 に更新した。保存済みの schemaVersion 1 アセットは、既存の `rangeMap` から新しい `distanceMap` を読み込み時に生成して使う。カラー・法線・深度の保存形式は維持し、メッシュの再ベイクや AI 画像の再生成を必要としない。[互換処理](../../../worker/distance_texture.mjs)

0.4.2 は空き領域をまとめて飛ばす距離グリッドを追加し、ヒット位置から遠い面のテクスチャ読み取りを省く。作者は約3分の1の描画高速化を報告している。これは作者の測定値であり、この検証の測定値ではない。ヒット点でのライティングと細い形状の欠落修正は、0.4.1 にすでに含まれている。[公式変更履歴](https://github.com/richardmonette/three-impostor/blob/df6caf4f459f58cf424df32af07c4aed0de4905c/CHANGELOG.md)

`distanceMap` は8 texel単位のセルについて、面を含み得るセルまでの距離を最大7で保存する R8 テクスチャである。CPU実装は半精度の `rangeMap` を読み、シェーダーの単精度演算と近傍更新を再現する。1024 pxでは1536×1408、2,162,688バイトとなる。このキャッシュは描画を省略する領域を表し、新しい観測方向や細部の形状は追加しない。[実装](../../../worker/distance_map.mjs)・[公式ソース](https://github.com/richardmonette/three-impostor/blob/df6caf4f459f58cf424df32af07c4aed0de4905c/Impostor.js)

## 保存前後の一致

非対称の色と形状を持つ小さな fixture を0.4.2でベイクし、schemaVersion 1 のバイト列から復元した。ネイティブの距離テクスチャとCPUで再構築したテクスチャは、同じシェーダーでRGBA8にコピーして比較した。GPUのR8直接読み取りには依存せず、使用されるセルだけを比較している。[検証スクリプト](../../../tests/check_box_distance_map.mjs)

WebGL2とWebGPUでそれぞれ全項目が一致した。描画は45度、192×192 pxで、元のベイクと復元後に同じステップ数を使った。RGB・alpha・24bitに符号化した深度の差分をそれぞれ比較した。バックエンドをまたぐ画素一致は判定していない。[WebGL2の実測JSON](gpu-distance-fixture/distance-map-report.json)・[WebGPUの実測JSON](gpu-distance-fixture-webgpu/distance-map-report.json)

| バックエンド | 面解像度 | 比較セル数 | 距離値の不一致 | RGBの不一致画素 | alphaの不一致画素 | 深度の不一致画素 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WebGL2 | 41 | 216 | 0 | 0 | 0 | 0 |
| WebGL2 | 64 | 512 | 0 | 0 | 0 | 0 |
| WebGL2 | 96 | 1,728 | 0 | 0 | 0 | 0 |
| WebGPU | 41 | 216 | 0 | 0 | 0 | 0 |
| WebGPU | 64 | 512 | 0 | 0 | 0 | 0 |
| WebGPU | 96 | 1,728 | 0 | 0 | 0 | 0 |

各画像は36,864画素で、前景と背景の両方を含む。前景の深度範囲はワールド座標で約0.808〜0.819だった。ブラウザーは Chromium 158.0.8127.18で、要求したWebGL2・WebGPUの使用を実測確認した。起動時に `BI_SOFTWARE_GPU=1` でSwiftShaderを指定した。ブラウザーエラーは両経路とも0件だった。

1024 pxのCPU検証では、有効な2,097,152セルを解析的に求めた距離と比較し、不一致は0だった。これはCPUのキャッシュ生成検証であり、描画速度の測定ではない。[1024 pxの記録](distance_map_1024_cpu_receipt.json)

既存機能とCPU距離生成の単体テストは15件通過、失敗0件だった。8の倍数でない解像度、未使用タイル、単精度の境界、半精度デコード、入力バイト列の保持も検査に含む。[単体テストの出力](unit-tests.log)

## 速度の測定

Q2について各版を3回ずつ、各回8回のウォームアップ後に30回の `api.draw() + gl.finish()` を記録した。180回中30回が0 msとなり、残りも約0.1 ms刻みだったため、この記録を描画速度の比較値として採用しなかった。今回の測定から高速化率は示せない。生の測定値と不採用理由は比較結果の `benchmark` に残している。[測定記録](version-comparison.json)

## 角度を増やす設計

現在のBox方式は、画素ごとに任意方向のレイを1本追跡し、そのヒット位置で六面の色と法線を混合する。六面は観測方向の数である。Boxのポリゴンを6面から24面に分割しても、六方向の画像や深度が増えなければ情報量は増えない。[公式ソースのVIEWS・axisUV・_setupBox](https://github.com/richardmonette/three-impostor/blob/df6caf4f459f58cf424df32af07c4aed0de4905c/Impostor.js)

斜めからの画像と深度を追加する設計は有効な検証候補である。水平の観測方向を90度刻みの4方向から45度刻みの8方向に増やし、上下を加えると、重複しない方向は10になる。既存形式を使う試作なら、同じメッシュを0度と45度の2つのBoxとしてベイクできる。保存上は6面×2組となり、マップ容量はおおむね2倍になる。

2組の描画を混合する場合は、各レイのヒット位置と可視性を共通座標で比較する必要がある。異なる深度の顔をそのまま色だけでフェードすると、二重像が残り得る。まず近い角度の1組を選ぶ方式を比較基準にし、ヒット位置が整合する領域だけを混合する方式を検証する。この増角度方式は設計案であり、今回の互換検証には含めていない。

## 再実行

プロジェクトの既存PlaywrightとChromium設定を使う。

```sh
BI_SOFTWARE_GPU=1 node tests/check_box_distance_map.mjs .test-output/box-distance-map webgl2 41,64,96
BI_SOFTWARE_GPU=1 node tests/check_box_distance_map.mjs .test-output/box-distance-map-webgpu webgpu 41,64,96
```

比較対象は公開npmパッケージの0.4.1と0.4.2である。Threeのpeer条件は両版とも `>=0.186.0` で変わっていない。公表tarballのファイル差分で変更内容を確認した。[npm 0.4.1](https://registry.npmjs.org/three-impostor/0.4.1)・[npm 0.4.2](https://registry.npmjs.org/three-impostor/0.4.2)

表示に使った0.4.2の3D-Inspector HTMLは、再現用コミットの[viewersディレクトリ](https://github.com/zkamiyama/ComfyUI-Box-Impostor-Factory/tree/43555805e07fd5b45844286bf595c29c549d12c4/docs/verification/three-impostor-v042-20261120/viewers)にgzipで保存している。
