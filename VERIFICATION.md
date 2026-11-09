# 作成時の動作確認

このファイルは実装の動作確認記録です。論文の再現精度を報告するものではありません。

## Google Drive対応の検証

- 環境: Linux x86_64 / Python 3.11.17 / torch 2.6.0+cpu / numpy 1.26.4 / h5py 3.16.0 / gdown 5.2.2。
- 指定フォルダ `1uEK737LXB9jAlf9kyqRs6B9N6cDncodq` へHTTP 200でアクセスし、4ファイルの一覧を確認。
- `mosi_data.pkl` (154,041,300 bytes)、`mosi_raw.pkl` (357,186,706 bytes)、`mosi.hdf5` (166,906,710 bytes) の実ファイル取得を確認。`mosi_unalign.hdf5` は一覧のみ確認。
- 既定の `mosi_data.pkl` を全量インポート: train=1,284 / valid=229 / test=686、特徴次元t=300 / a=5 / v=20。サンプルIDの公式SDK分割への一致を検証。2,199ラベルは以前取得したMMSAのCSVとも数値が一致。
- 任意の `mosi_raw.pkl` も全量インポート: train=1,283 / valid=214 / test=686、特徴次元t=300 / a=74 / v=35。サンプルIDの公式SDK分割への一致を検証。
- 両実ファイルのインポート結果を既定MulGモデルへ入力し、全7構成で有限な7クラス出力を確認。
- `uv sync --frozen --extra cpu --extra drive --group dev` による既定環境（raw用の依存関係なし）で **39 tests passed**。
- テストには、ファイルハッシュ、限定された数値配列読込、2種のID形式、公式分割、パディング・内部ゼロ保持、train-only標準化、全7構成のモデル入力、ダウンロード済みキャッシュ検証を追加。
- `ruff check src tests`、`bash -n setup.sh run_all.sh`、`uv lock --check`: 成功。

Drive実データによる100epochの全構成・全seed学習や論文の報告精度の再現は未実施です。配布特徴の抽出器・次元選択の詳細はファイル内に記されていません。実験条件はREADME.mdを参照してください。

## Kaggle対応時の検証記録

- 環境: Linux x86_64 / Python 3.11.17 / torch 2.6.0+cpu / torchvision 0.21.0+cpu / numpy 1.26.4 / h5py 3.16.0。
- 生データ用: opensmile 2.6.0 / opencv-python-headless 4.11.0.86 / pandas 2.3.3 / pyarrow 19.0.1。NumPy 1.xに対応する依存関係へ固定。
- `uv sync --extra cpu --extra raw --group dev`: 成功。uv.lockに固定。
- Kaggle対応時の `pytest -q`: **32 passed**。従来のAttention/GRU/指標/全7構成の学習・checkpoint再評価に加え、Kaggle取得、キャッシュ・ハッシュ、ZIP応答、正解ラベルの検証、Kaggle生データからのHDF5変換とtrain-only標準化を確認。
- `ruff check src tests`、`bash -n setup.sh run_all.sh`、`uv lock --check`: 成功。
- Kaggle公開APIで全5,049ファイル（展開後4,275,432,942 bytes）の一覧を確認。
- 実際のCLIで `.annotprocessed`、COVAREP MAT、分割WAV、分割MP4の計4ファイルを取得。直接ファイル応答と、Kaggleが自動ZIP化したMAT応答の両方に対応することを確認。
- `download-labels` でMMSAのラベルCSVを取得し、SHA256を検証。2,199区間すべてがKaggleの分割MP4名と一致。SDK公式分割との不一致0件（train=1,284 / valid=229 / test=686）。
- 実Kaggle区間 `2iD-tVS8NPw_3` でopenSMILE IS09特徴 `(228, 16)` と学習済みR3D-18特徴 `(2, 512)` の抽出に成功。R3D重みの取得も確認。
- 既定モデルで合成データによる全7構成×1seed×1epochの学習・評価を初版で確認済み。全7構成の学習・再読込は今回もテストで確認。

Kaggle全量ZIPの取得、全2,199区間の特徴抽出、実データ全量での学習、GPU学習、論文精度の再現は未実施です。Stanford GloVe ZIPはこの環境の接続確認で503だったため、ベクトル本体の取得は未検証です。ユーザーが取得したテキスト形式の事前学習埋め込みを指定できます。

SDK旧配布先は初版の確認で403、論文記載の著者コードURLは404でした。Kaggle経路の特徴量は従来のSDK経路と異なるため、同一の実験条件として混同しないでください。詳細はREADME.mdに記載しています。
