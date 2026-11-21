# Direct3D-V3 調査マップ

## 問いと調査範囲

Direct3D-V3を既存の人物画像とcoarse meshで試すために、公開機能、入力条件、単段実行経路を確認する。

実験を進めるための確認として、公開範囲、入力条件、単段実行、トークン制限の処理を対象にした。今回の出力の評価と計測値は実験報告で扱う。

## 論点と原文の場所

| 論点 | 確認結果 | 原文の場所 |
|---|---|---|
| 公開範囲 | coarse mesh生成とtexture生成は未公開 | README 34–35・42行、pipeline 196–197行 |
| 入力と出力 | 参照画像と向きをそろえたメッシュを入力し、正規化座標系のメッシュを返す | README 44・48・116行 |
| 単段実行 | 指定した段だけをロードできる | pipeline 84–114行 |
| トークン制限 | 占有パッチ数が上限に収まるまで形状を縮小する | voxelization 28–44行 |

## 用語・人物・媒体

coarse meshは生成に渡す初期形状。stageは解像度ごとの生成段。max_tokensは占有ボクセルパッチ数の上限を指す。これらの意味は上表のコードとREADMEで確認した。

著者表記はShuang Wu、Jiachen Qian、Youtian Lin、Feihu Zhang、Philip Torr、Yao Yao、Xun Cao。媒体表記はACM Transactions on Graphics（SIGGRAPH Asia 2026）。いずれもREADME 5–9行と171–184行による。公式コードはDreamTechAI/Direct3D-V3、公開モデルの案内先はwushuang98/Direct3D-V3。

## たどった経路

ユーザー指定のGitHub READMEから、[プロジェクト](https://dreamtechai.github.io/Direct3D-V3)、[モデル](https://huggingface.co/wushuang98/Direct3D-V3)、[ACM DOI](https://dl.acm.org/doi/10.1145/3842524)へ進んだ。DOIを手掛かりにした検索結果から、[公式arXiv](https://arxiv.org/abs/2611.26511)へ到達した。読み取りの失敗と代替経路はlog.mdに記録した。

## 原文（sources/）

| ファイル | 元の URL | この資料束での状態 |
|---|---|---|
| Direct3D-V3-README.md | [固定commitの原文](https://github.com/DreamTechAI/Direct3D-V3/blob/b93ad9c0a451729bb76b60ce69cbf076e76e95af/README.md) | 保存済み・本文確認済み |
| Direct3D-V3-pipeline.py | [固定commitの原文](https://github.com/DreamTechAI/Direct3D-V3/blob/b93ad9c0a451729bb76b60ce69cbf076e76e95af/direct3d/pipeline.py) | 保存済み・本文確認済み |
| Direct3D-V3-image_encoder.py | [固定commitの原文](https://github.com/DreamTechAI/Direct3D-V3/blob/b93ad9c0a451729bb76b60ce69cbf076e76e95af/direct3d/models/image_encoder.py) | 保存済み・本文確認済み |
| Direct3D-V3-voxelization.py | [固定commitの原文](https://github.com/DreamTechAI/Direct3D-V3/blob/b93ad9c0a451729bb76b60ce69cbf076e76e95af/direct3d/utils/voxelization.py) | 保存済み・本文確認済み |
| Direct3D-V3-requirements.txt | [固定commitの原文](https://github.com/DreamTechAI/Direct3D-V3/blob/b93ad9c0a451729bb76b60ce69cbf076e76e95af/requirements.txt) | 保存済み・本文確認済み |

取得元commitは`b93ad9c0a451729bb76b60ce69cbf076e76e95af`。親エージェントが保存した原文を本作業で読んだ。執筆時に再取得はしていない。SHA-256と取得経路はfetch-log.jsonlに記録した。

## 未完了・次の手掛かり

未保存の論文本文、モデル設定、重みヘッダーの数値は、この抜き書き資料では再検証できない。RGBと法線・クレイ入力の品質差、異なる初期形状の効果は比較実験が必要。16 GB環境での第三者の実行事例や量子化モデルは調べていない。

## 区切った理由

公開範囲・入力条件・単段ロードの根拠が固定した公式READMEとコードにそろったため、調査を区切って実行に進んだ。検索の網羅性や飽和は判定していない。
