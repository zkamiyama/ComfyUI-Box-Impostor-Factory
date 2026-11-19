# Qwen Image 2.1 による＋X画像の再生成

検証ID：20261119。実行：2026年11月19日夜〜20日未明（JST）。

Qwen Image 2.1にAnyAngleを加えたQ2では、実際のBox Impostorの＋45°と＋60°で、髪の中に重なっていた第二の顔を視認できなくなりました。Qwen単体のQ1では目・鼻・口が残りました。Q2でも頭頂の縦筋・まだら、顎下の不連続、胴体の縦境界、脚の灰色の帯は残っています。

## 生成条件

| 項目 | 以前のH3 | Q1 | Q2 |
|---|---|---|---|
| 生成モデル | MiniMax H3 | Qwen Image 2.1 | Qwen Image 2.1＋AnyAngle |
| 生成する面 | ＋X | ＋X | ＋X |
| 入力キャンバス | 1024×1024 | 1024×1024 | 1024×1024 |
| steps / CFG | 6 / 1 | 25 / 1 | 25 / 3 |
| seed | 101 | 101 | 101 |
| 参照順序 | ガイド、原画像、Depth、法線、mask | H3と同じ5枚 | 原画像、ガイド |
| LoRA | なし | なし | AnyAngle、強度1 |

Q1・Q2はeuler / simple、denoise 1、resolution 1024、CacheはCPU / defaultです。`TextEncodeQwenImage21` の空のlatentから生成しました。参照画像はencoderとVAEに渡し、ControlNetは使用していません。[Q1の実行グラフ](Q1/px/prepared-prompt.json)と[Q2の実行グラフ](Q2/px/prepared-prompt.json)のSHA256は、それぞれの推論結果に記録された値と一致します。

共通weightsは `qwen_image_2.1_int8_convrot.safetensors`、`qwen3vl_8b_int8_convrot.safetensors`、`qwen_image_2.1_vae_bf16.safetensors` です。Q2は `QI2.1_AnyAngle.safetensors` を追加しました。各ファイルのSHA256と画像の入力順序は[Q1の準備記録](Q1/prepared.json)と[Q2の準備記録](Q2/prepared.json)にあります。

Q1はH3のプロンプトを引き継ぎ、画像参照タグだけを `<Picture n>` から `<imagen>` に変更しました。Q2は作者の接続図に従い、image_1に原画像、image_2に＋Xガイドを渡し、`Change the camera angle from <image2> to <image1>.` を使いました。

## 生成画像と合成

[Q1の生成PNG](Q1/px/generated.png)には、ガイドで髪に隠れている位置に目・鼻・口が残っています。[Q2の生成PNG](Q2/px/generated.png)では顔が髪に隠れ、ガイドに近い見え方になりました。固定座標による比較方法と原画像のSHA256は[Q1](Q1-px-raw-provenance.json)・[Q2](Q2-px-raw-provenance.json)の記録にあります。

| 項目 | Q1 | Q2 |
|---|---:|---:|
| 新規推論の時間 | 109.68秒 | 126.4秒 |
| 合成前のシルエットIoU | 0.9854688 | 0.8999510 |
| 位置合わせ後のIoU | 0.9854688 | 0.9300731 |
| 合成前の輪郭距離P95 | 3.6056 px | 22.4722 px |
| 位置合わせ後の輪郭距離P95 | 3.6056 px | 18.0278 px |
| 適用scale | 1 | 1.0139678 |
| 適用shift YX | [0, 0] | [2.7400692, −10.2310462] |

数値の原記録は[Q1の推論](Q1/px/result.json)・[合成](Q1/composed.json)、[Q2の推論](Q2/px/result.json)・[合成](Q2/composed.json)です。Q2は顔の見える範囲がガイドに近づいた一方、外形の対応はQ1より低く、合成時に位置合わせも適用しています。

Fの＋XのRGBを差し替え、Q1は `90bc93b6d83833f93eef20492d011bc60376ddd5bb2a3c0deeda95e2dd8059ba`、Q2は `87b95907db2b663184783f1eb7d9c5b594542ce15aa1d1d777a85b1900ab213e` になりました。Depth等の保持確認ではproxy、geometry metadata、non-color maps、coverage alphaが一致し、他の5面のRGBA・参照元・metadataも保持されています。[Q1の対応表](Q1-direct-evidence.json)と[Q2の対応表](Q2-direct-evidence.json)に根拠をまとめました。

Q2の原画像の黒背景はRGBの値です。alphaは251〜255で、alpha 0の画素はありません。比較画像はRGB成分を表示し、別に生成したforeground maskは使っていません。[背景とalphaの記録](raw-alpha-display-convention.json)

## Box Impostorでの観察

同じ現行3D-InspectorでF-current・Q1・Q2を生成し、view / unlit、正投影、1024×1024で0°・＋30°・＋45°・＋60°を比較しました。3つのviewerの描画コードは同一です。古いKのHTMLとはコードが異なるため、比較基準には新たに生成したF-currentを使いました。[viewerの生成記録](viewer-builds.json)

全角度でカメラ行列の差は0、描画alphaはバイト単位で一致しました。＋45°の固定領域 `[435, 155, 486, 231]` ではF-currentとQ1に第二の顔が見え、Q2では髪になっています。＋60°でも同じ改善を確認しました。観察範囲は今回の4角度です。[表示比較の記録](Q1-Q2-box-provenance.json)

| 実際の描画PNG | F-current | Q1 | Q2 |
|---|---|---|---|
| ＋45° | [原画像](raw/F-current/yaw-p45_000.png) | [原画像](raw/Q1-qwen-five/yaw-p45_000.png) | [原画像](raw/Q2-qwen-anyangle/yaw-p45_000.png) |
| ＋60° | [原画像](raw/F-current/yaw-p60_000.png) | [原画像](raw/Q1-qwen-five/yaw-p60_000.png) | [原画像](raw/Q2-qwen-anyangle/yaw-p60_000.png) |

正面のRGBはQ1・Q2ともF-currentから204画素変化しました。固定10,936画素で測った参照画像とのMAEは、0〜255の値でF-current 0.607961、Q1 0.603511、Q2 0.602201です。別の顔領域4,287画素では、平均RGB誤差が16を超える画素はすべて24画素でした。これらは画像の保持状況を示す値で、3D精度や全体的な画質の評価値ではありません。

＋60°の輪郭内側3px帯も確認しました。alpha>250の8,296画素でmaxRGB<16だった画素はF-current 14、Q1・Q2各7で、新しく暗くなった画素は0でした。この角度としきい値での確認であり、他の角度の輪郭品質を保証するものではありません。[輪郭の確認記録](edge-review-provenance.json)

## 比較の限界

H3は6 steps、Q1・Q2は25 stepsで、生成モデル・encoder・VAEも異なります。Q2はLoRAに加えて参照数・順序・prompt・CFGを変えているため、LoRA単独の効果を測った比較ではありません。

今回変更したのは＋XのRGBで、Depthと3D形状は保持しています。この保持確認は元のDepthや3D形状が正しいことを示しません。全体の色と形の対応には調整が残るため、Q2は顔の向きの改善候補として扱います。縦筋や不連続が残り、6面全体の対応と3D形状の精度は未検証です。

## 参照資料とviewer

- [ComfyUIのQwen Image 2.1説明](https://github.com/Comfy-Org/docs/blob/main/tutorials/image/qwen/qwen-image-2-1.mdx)と[公式edit workflow](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/image_qwen_image_2_1_image_edit.json)。取得したworkflowのSHA256は `c1e6551759de131f083a040da072ac6f032c3c8401ceca202412cd912897d0fc`、commit IDは未確定です。
- [AnyAngle作者のモデルカード](https://huggingface.co/lilylilith/QI_2.1_AnyAngle)。
- [実際に比較したviewerの保存先](https://github.com/zkamiyama/ComfyUI-Box-Impostor-Factory/tree/9be1efb5bee8cef187e1efa22cafcc64da68f228/docs/verification/qwen-image-20261119)。公式view.pyが生成したHTMLを保存しています。
- [QAファイル一覧](qa-manifest.json)。比較ボードはチャットで提示し、repoには生成PNG、描画PNG、再現用スクリプト、JSON記録を保存しています。
