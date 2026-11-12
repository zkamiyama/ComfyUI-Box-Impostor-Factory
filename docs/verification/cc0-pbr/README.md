# CC0の街灯と金属ボトルでの検証

Khronosが配布しているMicrosoft製のGLBを使った。Lanternは木の支柱と吊り下げ式の灯具を含む5,394三角形のモデル。Water Bottleは赤いキャップと金属製の胴を持つ4,510三角形のモデルで、ラフネス、メタルネス、AO、法線、発光の各テクスチャを検査できる。どちらもCC0 1.0で公開されている。[Lanternの配布元](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/Lantern/README.md#legal) · [Water Bottleの配布元](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/WaterBottle/README.md#legal)

## 元メッシュとBoxの比較

元メッシュと保存したBoxを同じカメラと照明で描いた。比較画像は256×256画素。6軸と斜め12方向で、輪郭の重なり（IoU、完全一致は1）と、両方に物体が写った画素のRGB平均絶対誤差を測った。描画にはChromiumのSwiftShaderを使用した。

| モデル | 1面の解像度 | 描画方式 | 6軸の平均IoU | 斜め12方向の平均IoU | 18方向の平均RGB誤差（0–255） |
|---|---:|---|---:|---:|---:|
| Lantern | 256 px | WebGL2 | 0.9614 | 0.9625 | 4.73 |
| Lantern | 256 px | WebGPU | 0.9623 | 0.9667 | 4.77 |
| Lantern | 512 px | WebGPU | 0.9775 | 0.9816 | 4.01 |
| Water Bottle | 256 px | WebGL2 | 0.9951 | 0.9958 | 2.02 |

街灯の支柱と灯具の発光はBoxにも写った。真上から見ると台座の角が削れ、正面では細い吊り鎖が途切れる。WebGPUで1面を256 pxから512 pxに上げると、真上のIoUは0.906から0.946に改善した。正面で鎖が写る13画素のうちBoxでも見えたのは8画素から10画素に増えた。[鎖と台座の拡大比較](lantern-details.png)

![街灯の元メッシュと1面256 pxのBoxを、同じ斜め視点で描いた結果](lantern.png)

金属ボトルでは標準補間で胴とキャップに縦筋が出た。1面を512 pxに上げても胴の縦筋は残る。1面256 pxで描画時の「色にじみ低減」を選ぶと、18方向の平均RGB誤差が2.02から1.34に下がった。肩とホルダーの縁には細い筋が残り、最も輪郭の差が大きい方向のIoUは0.992から0.982に下がった。

![ボトルの元メッシュ、標準補間の256 pxと512 px、色にじみ低減を使った256 pxの比較](waterbottle.png)

WebGL2とWebGPUの両方で、ベイク直後と新しいブラウザで再読込したBoxを64×64画素・18方向で比較した。両モデルとも色、深度、輪郭が全画素で一致した。ボトルの6面から選んだ14点では、元のGLBの三角形とUVから各PBR値を求め、アトラスの値と照合した。WebGL2での最大誤差は、ラフネス・メタルネス・AOが0.88/255、法線の向きが0.90度だった。[各方向の計測値とPBR検査結果](measurements.json)

## SD 1.5で街灯の6面を編集

RTX 4080で各面のカラーガイドをSD 1.5のimg2imgに渡し、元のガイドから輪郭を取った。1面256 px、各6ステップ、denoise 0.5で、6個のサンプラーがすべて実行された。同じGLBをベイクし、チェックポイントを一度読み込んだ環境で、このキューの実行時間は6.00秒だった。

6面とも編集色を採用し、最終の斜視では前景4,509画素のうち4,441画素の色が変わった。編集前後で透明度は全画素一致し、法線・区間・占有範囲・材質・発光の5マップはバイト単位で一致した。[実行条件とマップのSHA-256](sd15.json)

![街灯のBoxを同じ視点から描いた画像。左が編集前、右がSD 1.5で6面の色を編集した結果](lantern-sd15.png)

[検証済みの街灯Box](../../../examples/cc0-lantern-sd15.box.zip) を `ComfyUI/input/` に置き、`BI LoadBox` に `cc0-lantern-sd15.box.zip` を指定すると開ける。ZIPを書き出して読み直した後の斜視もRGBA全画素が一致した。

## 再現手順

リポジトリ直下で次を実行する。ボトルの元データを固定リビジョンから取得し、材質と発光を含めてベイクする。

```sh
mkdir -p .test-output
curl -fL https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/WaterBottle/glTF-Binary/WaterBottle.glb -o .test-output/WaterBottle.glb
node worker/cli.mjs bake --input .test-output/WaterBottle.glb --output .test-output/bottle --resolution 256 --backend webgl2 --maps roughness,metalness,ao,emissive --verify
python tests/check_pbr_bake.py .test-output/WaterBottle.glb .test-output/bottle
node tests/compare_proxy.mjs .test-output/bottle .test-output/bottle-comparison.json webgl2 --size 256 --compare-interval
```

ComfyUIで自分のGLBを試すときは、ワークフロー01の `BI BakeBox` で `material_maps` を `material+emissive` にする。ビューアの「アトラス・ガイド」から `materialMap` のR（ラフネス）、G（メタルネス）、B（AO）を個別に表示できる。元モデルにAOテクスチャのない街灯では、物体の表面のAOは255になる。

街灯の6面生成を再現するには、SD 1.5のFP16チェックポイントと `input/Lantern.glb` を用意し、ComfyUIの起動中に実行する。[検証に使った街灯GLB](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/Lantern/glTF-Binary/Lantern.glb)

```sh
python scripts/run_lightweight_smoke.py --mesh Lantern.glb --backend webgl2 --resolution 256 --steps 6 --seed 19590 --denoise 0.5 --material-maps material+emissive --positive-prompt "a weathered timber street-lamp post, crossbeam, hanging warm golden lantern, aged dark iron fittings, detailed wood grain, studio product photograph" --run
```
