# 作成時の動作確認

このファイルは実装の動作確認記録です。論文の再現精度を報告するものではありません。

- 環境: Linux x86_64 / Python 3.11.17 / torch 2.6.0+cpu / numpy 1.26.4 / h5py 3.16.0。
- `uv sync --extra cpu --group dev`: 成功。依存関係をuv.lockへ固定。
- `pytest -q`: **21 passed**。
- `ruff check src tests`: 成功。
- `bash -n setup.sh run_all.sh`: 成功。
- `uv lock --check`: 成功。
- `uv build`: sdist / wheel の生成成功。wheel内のSDKレシピ・分割JSONの存在を確認。
- 既定モデル（hidden=40、4 heads、Attention 2層、論文式GRU 2層、CE7）で、合成データによる全7構成×1seed×1epochの学習・検証・テストを実行し、全7構成の完了を確認。
- 標準PyTorch GRUとMAE回帰を組み合わせたtav構成でも、合成データによる1epochの完了を確認。
- テスト内でcheckpoint再読込のAcc2/Acc7一致と、完了済み全7構成のresumeを確認。

実データのアクセス確認ではMOSIの配布URLがこの実行環境からHTTP/HTTPSとも403となりました。実データのダウンロード、GPU学習、論文精度の再現は未検証です。

論文記載の著者コードURLはGitHub APIで404でした。実装上の解釈・特徴抽出の違いはREADME.mdの表に明記しています。
