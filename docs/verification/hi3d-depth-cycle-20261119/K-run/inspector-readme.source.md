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

## 配布内容

Agentが使うものだけを `skills/3d-inspector/` に収録しています。

- [SKILL.md](skills/3d-inspector/SKILL.md)：Agent向けの実行手順
- `view.py`・`requirements.txt`・`scripts/`：モデル変換と依存関係
- `assets/`：形式別の描画ランタイムとHTMLテンプレート
- `agents/openai.yaml`：Agent向けの表示設定

サンプル、テスト、開発用ソース、ビルドツール、重複するZIPアーカイブはリポジトリから除いています。

## ライセンス

Skillの[MITライセンス](skills/3d-inspector/LICENSE)と、描画ランタイムに含まれる[第三者ライセンス](skills/3d-inspector/assets/THIRD_PARTY_LICENSES.txt)を参照してください。
