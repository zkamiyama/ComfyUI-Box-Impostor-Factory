# 3D-Inspector-skills

3Dモデルを指定すると、インタラクティブな **3D-Inspector** HTMLを生成するAgent向けSkillです。配布するHTMLにはモデルデータと必要な描画ライブラリを埋め込みます。

## 使い方

```sh
python skills/3d-inspector/view.py "/path/to/model.glb"
```

標準出力にHTMLの絶対パスが1行で出力されます。Agentはそのファイルをユーザーに渡すだけです。必要なPythonパッケージは専用キャッシュへ自動導入されます。出力場所を指定する場合は `-o /path/to/preview.html`、ライブラリだけをCDNから読み込む場合は `--runtime-source cdn` を追加します。モデルやテクスチャはどちらの形式でもHTMLに含まれます。

メッシュはGLB・glTF・OBJ・STL・PLY・OFF・3MF・DAE、アニメーション付きglTF、点群、Gaussian PLY・SPLAT、Box Impostorの `.box.zip` に対応します。FBX・BLEND、USD/USDZ、STEP/IGES/BREPは外部変換ツールが利用できる環境で変換します。3D表示はWebGPUを優先し、WebGL2またはCanvas 2Dへ切り替えます。Canvas 2Dでは描画表現が簡略化されます。

## 金属の照明プレビュー

金属や光沢のある材質は、**Materials → Lighting → Studio stripes** で映り込みを確認できます。明るいスタジオの縦帯を128×64の半精度環境マップとしてブラウザ内で生成し、Three.jsの標準PBR／粗さに応じたPMREM反射を利用します。HDRIの画像を同梱する必要はありません。必要なモデルでだけ生成し、**Neutral** に切り替えて元の照明と比較できます。Box ImpostorやGaussianの専用描画は変更しません。

## Box ImpostorのRGBと視差

画像に描かれた陰影を使ってBoxを表示する場合は、`--box-lighting unlit` を追加します。保存済みの深度で交差位置を求め、視点の移動に応じた視差を出します。表示時の照明とトーンマッピングは無効になります。

```sh
python skills/3d-inspector/view.py "/path/to/asset.box.zip" \
  --box-lighting unlit --box-color-sampling view \
  -o "/path/to/asset.inspector.html"
```

`--box-color-sampling` で、交差位置に投影する画像の選び方を指定できます。

| 値 | 色の選び方 |
|---|---|
| `native`（既定） | 元のBox描画処理による、表面の向きと深度に応じた6面の重み |
| `view` | 見ている方向に近い面を優先する連続的な重み |
| `dominant` | 可視候補のうち、見ている方向に最も近い1面 |

`view` と `dominant` は投影先の画素と深度を検査し、候補を得られない場所では元の色の計算へ戻ります。色の選択を替えても形状・深度・輪郭は変わりません。元画像同士の顔や髪の位置が異なると、`view` では中間の角度で像が重なり、`dominant` では面が切り替わるときに像が跳ぶ場合があります。

これらの設定は、native描画の `.box.zip` で使えます。照明の既定値は `lit`、色の選択の既定値は `native` です。

## 配布内容

Agentが使うものだけを `skills/3d-inspector/` に収録しています。

- [SKILL.md](skills/3d-inspector/SKILL.md)：Agent向けの実行手順
- `view.py`・`requirements.txt`・`scripts/`：モデル変換と依存関係
- `assets/`：形式別の描画ランタイムとHTMLテンプレート
- `agents/openai.yaml`：Agent向けの表示設定

サンプル、テスト、開発用ソース、ビルドツール、重複するZIPアーカイブはリポジトリから除いています。

## ライセンス

Skillの[MITライセンス](skills/3d-inspector/LICENSE)と、描画ランタイムに含まれる[第三者ライセンス](skills/3d-inspector/assets/THIRD_PARTY_LICENSES.txt)を参照してください。
