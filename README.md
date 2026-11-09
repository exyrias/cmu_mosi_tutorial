# MulG 再現実験用プログラム

添付論文「Multimodal GRU with directed pairwise cross-modal attention for sentiment analysis」(Qin et al., Scientific Reports 15, 10112, 2025) の記述から再構成した実装です。

MOSIの既定の取得先は、今回指定された [Google Driveのmosiフォルダ](https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq) です。**手動で取得した `mosi_data.pkl` 1ファイルから、追加のGloVe・ラベル取得なしで学習できます。** KaggleとSDKの取得経路も選択できます。MOSEIは従来のSDK取得を維持します。

CMU-MOSI / CMU-MOSEI を対象に、**t / a / v / ta / tv / av / tav の7構成を、それぞれ最初から独立に学習**し、同じテスト集合で Acc2・Acc7 を出します。t は text、a は audio、v は visual です。全モダリティで学習したモデルの入力を後からゼロにする実験ではありません。

## 再現の範囲と、先に確認する点

**著者実装と同一の厳密再現とは主張できません。** 論文末尾のコードURL `https://github.com/bitbitlemon/Multimodal-GRU` は作成時の確認で GitHub API が404を返しました。以下の不明点・相違を残さず記録し、再実験できるようにしています。

| 項目 | 論文の記述 | この実装 |
|---|---|---|
| 中核モデル | Conv1D、位置埋め込み、方向付きペアAttention、GRU、結合・FC・残差 | 図3と式(1)–(9)を基に構成。3モダリティでは6方向、2つでは2方向、単独ではAttentionを通さずGRUへ入力 |
| Attention | 式(5)は双線形スコア、図3はCT、表7は4 heads | 4分割のheadごとに双線形スコアを計算。ソース側時間軸でsoftmax。スケーリングなし。各方向2ブロック。CTのFFN幅4dとpost-LNは補完した設計 |
| GRU | 2層、隠れ40。式(3)はresetを行列積の前に適用 | 既定 `paper` は式(1)–(4)を直接実装（式にないbiasは付加しない）。`pytorch` は高速な標準GRUで、候補状態の式が異なる |
| 特徴抽出 | word2vec/CNN、openSMILE、3D-CNNなどを記述 | Driveは配布済み数値特徴（既定t=300/a=5/v=20）。Kaggleは固定GloVe等 / openSMILE IS09 LLD / 固定R3D-18。SDKはGloVe / COVAREP / OpenFace2。各経路の特徴量は異なる |
| 損失・出力 | Cross entropyの記述、連続値MAE/Corr、表7のoutput_dim=10が混在 | 既定 `ce7` は7クラスCE。比較用に `mae` / `mse` の連続値回帰も選択可能。10クラスにはしない |
| 残差・正規化 | post-LN / pre-LN / ResiDualを説明するが、最終構成は不明瞭 | CTと融合部に通常のpost-LN残差を使用。ResiDualやBatchNormの挿入位置は推測しない |
| データ分割 | 分割・前処理の詳細に不足あり | Driveはファイル内の分割を使い、全動画IDを公式SDK分割と照合。KaggleもCSVを公式IDと照合。SDKはラベル時刻で切り出す |
| 入力長 | 詳細不明 | 上限 t=128, a=512, v=512。DriveのPKLは既に長さ50の配列へ前処理済み。必要時は区間全体を等分平均。0で上限解除 |
| その他 | 学習詳細に不足あり | a/vの訓練集合だけで特徴別標準化、Adam、勾配clip=1、検証損失最小のcheckpointを選択。これらの補完設定も保存 |

論文の数値にも整合しない箇所があります。MOSIは表2で Acc2=82.2 / Acc7=40.5、表6のFull Modelで85.0 / 55.0です。MOSEIは表3・5で82.1 / 51.4です。また概要の82.2 / 82.1は7クラス正解率と書かれていますが、表ではAcc2です。**これらを一つの目標値として混ぜないでください。** 本プログラムは論文の値を結果として埋め込みません。

IEMOCAPは論文でも4感情の評価であり、今回のAcc7の対象には含めていません。

## 環境構築（bash / uv）

このリポジトリをcloneして実行します。Linux/macOS、Python 3.11を想定しています。検証はLinux CPUで実施しました。

```bash
# uvが未導入の場合
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone https://github.com/exyrias/cmu_mosi_tutorial.git
cd cmu_mosi_tutorial
uv python install 3.11

# CPU。macOSではPyPIのmacOS用PyTorchが使われます。
uv sync --frozen --extra cpu --extra drive --group dev
uv run --no-sync pytest -q
```

NVIDIA GPUを使うLinux/Windows環境では、CPU用のsyncの代わりに次を実行します。

```bash
uv sync --frozen --extra cu124 --extra drive --group dev
uv run --no-sync python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
uv run --no-sync pytest -q
```

CUDA 12.4に対応したNVIDIAドライバが必要です。Python依存関係は `uv.lock` で固定します。GPU用wheelでの実学習は今回検証していません。以降は **`uv run --no-sync`** を使い、syncで選択したCPU/GPUバックエンドを維持します。GPUへの切替時はもう一度対応する `uv sync` を実行してください。

同じ処理をまとめた `bash setup.sh cpu` / `bash setup.sh cu124` も利用できます。これらは既定でDrive用の環境を準備します。Kaggle生データの特徴抽出用には `bash setup.sh cpu raw` / `bash setup.sh cu124 raw` を使います。

## 指定Google DriveからMOSIを取得・前処理（既定）

指定フォルダへのアクセスとPKL本体のダウンロードに成功しました。まず **`mosi_data.pkl` 1ファイル**を使用してください。数値化済みの3モダリティ・連続値sentimentラベル・サンプルID・train/valid/testが含まれています。

| ファイル | サイズ | 件数 train / valid / test | 特徴次元 t / a / v |
|---|---:|---:|---:|
| `mosi_data.pkl`（推奨・対応） | 154,041,300 bytes | 1,284 / 229 / 686 | 300 / 5 / 20 |
| `mosi_raw.pkl`（任意・対応） | 357,186,706 bytes | 1,283 / 214 / 686 | 300 / 74 / 35 |

フォルダには `mosi.hdf5` と `mosi_unalign.hdf5` もあります。確認した `mosi.hdf5` は生単語の `words` 等を含む別構造です。今回の直接インポートには上表のPKLを選択します。

### 手動ダウンロードする場合

1. [指定フォルダ](https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq)をブラウザで開きます。
2. `mosi_data.pkl` をダウンロードします。
3. ファイルを任意の場所に置き、次を実行します。手動取得の場合はファイルをリポジトリ内へ移す必要はありません。

```bash
uv run --no-sync mulg prepare-drive \
  --input /path/to/mosi_data.pkl \
  --output data/mosi_drive_data.h5

uv run --no-sync mulg run \
  --data data/mosi_drive_data.h5 --output runs/mosi_drive_data \
  --modalities all --seeds 13 37 73
```

または、手動取得したファイルを指定して一括実行できます。

```bash
bash run_all.sh mosi drive /path/to/mosi_data.pkl
```

### 自動ダウンロードする場合

```bash
# --source drive がMOSIの既定。Google Driveの大きなファイル確認画面にはgdownで対応します。
uv run --no-sync mulg download --dataset mosi --output data/drive/mosi

uv run --no-sync mulg prepare-drive \
  --input data/drive/mosi/mosi_data.pkl --output data/mosi_drive_data.h5

uv run --no-sync mulg run \
  --data data/mosi_drive_data.h5 --output runs/mosi_drive_data \
  --modalities all --seeds 13 37 73
```

取得から全7構成の評価までをまとめて実行する場合:

```bash
bash run_all.sh mosi
```

自動取得には `--extra drive` のgdownが必要です。Drive側の混雑・アクセス制限等で失敗した場合は、同じPKLを手動取得して前述の `prepare-drive --input` に渡せます。`prepare-drive` 自体はネット接続を使いません。

`mosi_raw.pkl` を使う場合:

```bash
uv run --no-sync mulg download --dataset mosi --drive-file mosi_raw.pkl --output data/drive/mosi
bash run_all.sh mosi drive data/drive/mosi/mosi_raw.pkl
```

こちらの保存先は `data/mosi_drive_raw.h5` と `runs/mosi_drive_raw/` になります。2つのPKLは件数と特徴次元が異なるため、結果を別々に記録してください。

### この経路の前処理・再現条件

- 公式SDKの動画ID分割と全サンプルを照合します。ランダムな区間分割へ変更しません。
- 両PKLとも各モダリティの配列長は50です。`mosi_raw.pkl` という名前から未整列の生時系列とは推定せず、配布済みの整列・切詰めを前提に使用します。
- 先頭・末尾で全モダリティが同時にゼロのパディングを取り除きます。途中のゼロや、一部モダリティだけがゼロのフレームは保持します。全ゼロのモダリティ数を記録します。
- NaN/Infをゼロ置換し、a/vの標準化は訓練集合だけで計算します。textは再標準化しません。
- 全体のSHA256を照合した後、NumPy配列を許可する限定ローダーで読み込みます。ラベルの有限性・範囲[-3,3]・形状・ID重複・分割・次元を検証します。
- **特徴抽出器や選択した5/20次元の詳細はファイル内に記されていません。** この配布特徴を使うMulGの再構成実験であり、論文の抽出条件やKaggle版と同一の条件とは主張しません。
- `data/mosi_drive_data.manifest.json` 等に入力SHA256・出典・特徴次元・件数・パディング処理・標準化条件を残します。チェックポイントにも記録します。

確認したSHA256:

```text
mosi_data.pkl  1a113ad5edc8b9b625a7e29e6b94a97ecade70494c4e022d9f7ba5affca3bbdc
mosi_raw.pkl   5b8a29b2a603ce37611117d088570369c35246f843dade8865ff59cc5ee6f338
```

## KaggleからMOSIを取得・前処理（任意）

この経路を使う場合は `bash setup.sh cpu raw` 等で特徴抽出用の環境を準備します。


指定されたKaggle版は **5,049ファイル、展開後4,275,432,942 bytes（約4.28 GB）** です。動画・WAV・書き起こし・COVAREP MATを含みます。SDKのCSD形式の特徴量とsentiment正解ラベルは含まれません。追加するラベルと事前学習モデルも含め、空き容量は余裕を持たせてください。

### 1. Kaggleの生データを取得

```bash
uv run --no-sync mulg download --dataset mosi --source kaggle --output data/kaggle/mosi
```

Kaggleを使う場合は `--source kaggle` を明示します。Kaggleのversion 1を固定し、ZIPをストリーミング取得・展開します。公開APIによる取得を作成環境で確認しました。SHA256付きの `download_manifest.json` を保存し、再実行時は取得済みファイルの整合性を検証します。ソースや選択ファイルが異なる出力先への混在は拒否します。`--overwrite` はこのダウンローダーが作成した同じデータ構成だけを再取得します。

Kaggleが認証を要求する環境では、ローカルで `KAGGLE_API_TOKEN`、または `KAGGLE_USERNAME` と `KAGGLE_KEY` を設定してください。認証情報はコード・Git・manifestへ保存しません。ブラウザで取得したZIPを使う場合は次のように指定します。

```bash
uv run --no-sync mulg download --dataset mosi --source kaggle \
  --kaggle-archive /path/to/cmu-mosi.zip --output data/kaggle/mosi
```

取得の疎通確認だけなら、全体を取得せずファイルを指定できます。これは完全な学習データではありません。

```bash
uv run --no-sync mulg download --dataset mosi --source kaggle --output data/kaggle/check \
  --kaggle-file Raw/Transcript/Segmented/2iD-tVS8NPw.annotprocessed
```

### 2. 未所持の正解ラベルを取得

ラベルはKaggleとは別に、[MMSAが案内する公開配布先](https://github.com/thuiar/MMSA#2-datasets)から取得します。同じMOSIの `label.csv` で、作成時に次を検証しました。

- 2,199区間すべてがKaggleの `Raw/Video/Segmented/<video_id>_<clip_id>.mp4` と一致。
- 公式SDKの動画ID分割と全件一致。train=1,284 / valid=229 / test=686。
- 連続値の `label` 列を使用。`annotation` の文字ラベルや `label_T/A/V` は使用しない。

```bash
uv run --no-sync mulg download-labels --output data/annotations/mosi.csv
```

CSVのSHA256は `dec8b0affc5c7c923688040f091dddb61966afc729f54630ef91d3f51624b821` に固定しています。内容が変わったりサインインページが返った場合は検証で停止します。実データの代わりに推定ラベルや合成ラベルを使うことはありません。出典はCSV横のmanifestにも保存します。

### 3. テキスト用の事前学習ベクトルを用意

手元のテキスト形式GloVe/word2vecを指定します。例として [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) の `glove.6B.50d.txt` を使えます。バイナリword2vec形式には対応していません。

```bash
mkdir -p data/embeddings
curl -fL https://nlp.stanford.edu/data/glove.6B.zip -o data/embeddings/glove.6B.zip
unzip data/embeddings/glove.6B.zip glove.6B.50d.txt -d data/embeddings
```

このZIPは約0.86 GBです。上のStanford配布先は作成環境の接続確認で503だったため、ここではファイル本体の取得を検証していません。ローカルでも取得できない場合は、公式ページから入手したプレーンテキストの事前学習ベクトルを `--embedding-file` に指定してください。埋め込みをMOSIのラベルから学習し直すことはありません。

### 4. 生データから特徴量を作成

```bash
uv run --no-sync mulg prepare-kaggle \
  --raw-dir data/kaggle/mosi \
  --labels-csv data/annotations/mosi.csv \
  --embedding-file data/embeddings/glove.6B.50d.txt \
  --output data/mosi_kaggle.h5 --device auto
```

`--extra raw` の依存関係が必要です。`bash setup.sh cpu raw` / `cu124 raw` はこれらも導入します。

| 入力 | Kaggle経路の特徴量 |
|---|---|
| t | `.annotprocessed` の区間番号で対応させた書き起こし。小文字の単語に分割し、固定GloVe等へ変換。未知語はゼロ。未知語数・埋め込みSHA256を記録 |
| a | 分割済み16 kHz WAVからopenSMILE IS09のLow-Level Descriptorsを抽出。配布済みの動画全体のCOVAREP MATはこの経路では使わない |
| v | 分割済みMP4から約8 fpsでサンプリング。16フレームの非重複窓ごとに、Kinetics-400学習済みR3D-18のFC手前の512次元特徴を抽出。末尾の短い窓は最後のフレームで補う |

3D-CNNは固定して特徴抽出にだけ使用し、MulG学習時には更新しません。初回はtorchvisionの学習済み重み（約127 MiB）を取得します。`TORCH_HOME` が未指定なら出力HDF5の親ディレクトリの `pretrained/torch/hub` にキャッシュします。

これらは論文の抽出器の詳細が不明な部分を補完した構成で、以前のGloVe/COVAREP/OpenFace2のCSD経路とは特徴量が異なります。**従来のCSD実験と同一条件とは扱わず、`mosi_kaggle` として結果を保存してください。** モデルのMulG構造・全7構成・指標の定義は共通です。

欠損したメディア・書き起こし・不正ラベルがあれば処理を止めます。正常なら全2,199区間を保存します。a/vの標準化は訓練集合だけで計算します。前処理は時間を要し、途中で失敗した場合は再実行が必要です。

### 5. 全7構成を学習・評価

```bash
uv run --no-sync mulg run \
  --data data/mosi_kaggle.h5 --output runs/mosi_kaggle \
  --modalities all --seeds 13 37 73 --device auto
```

一括実行する場合は、事前学習ベクトルを用意した後に次を実行します。

```bash
bash run_all.sh mosi kaggle data/embeddings/glove.6B.50d.txt
```

## 従来のSDK CSD経路を使う場合

SDK取得は `--source sdk` を明示します。MOSEIは既定でもSDKです。Kaggleの指定URLはMOSIのみであり、MOSEIをここから取得することはできません。

```bash
uv run --no-sync mulg download --dataset mosi --source sdk --output data/raw/mosi
uv run --no-sync mulg prepare --dataset mosi --raw-dir data/raw/mosi --output data/mosi.h5

uv run --no-sync mulg download --dataset mosei --source sdk --output data/raw/mosei
uv run --no-sync mulg prepare --dataset mosei --raw-dir data/raw/mosei --output data/mosei.h5
```

SDK commit `2eadcd7e7b9e20e83b868cdad385b41830285c` のURLレシピと公式分割を `src/mulg/resources/sdk_manifest.json` に同梱しています。CSDをh5pyで読み、旧SDKパッケージ自体のインストールには依存しません。SDK旧配布URLは作成環境から403で取得できなかったため、利用可能な既存CSDを指定する経路も残しています。

### 既存のSDK CSDファイルを使う

4ファイルを任意の場所に置き、次のように指定できます。ファイル名がSDK配布時の名前のままでも構いません。CSDは `<root>/data/<video_id>/{features,intervals}` の形式を想定し、ラベルも同じ時刻基準のものを使います。

```bash
uv run --no-sync mulg prepare --dataset mosi \
  --text-csd /path/to/CMU_MOSI_TimestampedWordVectors.csd \
  --audio-csd /path/to/CMU_MOSI_COVAREP.csd \
  --visual-csd /path/to/CMU_MOSI_Visual_OpenFace_2.csd \
  --labels-csd /path/to/CMU_MOSI_Opinion_Labels.csd \
  --output data/mosi.h5
```

複数rootのCSDは `--text-key` / `--audio-key` / `--visual-key` / `--labels-key` でroot名を指定します。MOSEIのラベルでは先頭列のsentimentを使用し、後続の感情ラベルは使用しません。

SDK経路のMOSIでopenSMILE CSDを使う場合は `download --source sdk --audio-feature OpenSMILE` を指定できます。MOSEIのopenSMILE URLは指定SDKの有効なレシピに存在しないため、既存CSDを `--audio-csd` に渡してください。配布特徴を変更した結果は、既定のCOVAREPの結果と別実験として記録してください。

### word vector CSDを取得できない場合

SDKの生単語CSDと、手元にある**テキスト形式**GloVe/word2vecの固定済み事前学習ベクトルを組み合わせることもできます（バイナリword2vec形式は未対応）。大きなベクトルファイルから必要な単語だけを読み、未知語はゼロベクトルにし、その数を記録します。データセットを使った埋め込みの再学習は行いません。

```bash
uv run --no-sync mulg download --dataset mosi --source sdk \
  --text-feature words --output data/raw/mosi_words
uv run --no-sync mulg prepare --dataset mosi \
  --raw-dir data/raw/mosi_words \
  --embedding-file /path/to/glove.840B.300d.txt \
  --output data/mosi_words.h5
```

元のword vector CSDとはアラインメント・特徴が異なる可能性があります。特徴変更として報告してください。GloVeファイルの自動取得はしません。

### 前処理と比較条件

- 動画ID単位でSDKのtrain / valid / testへ分割し、話者・動画をまたぐランダムな区間分割は行いません。
- 各ラベル区間と時間的に重なる特徴を切り出します。区間の境界に触れるだけのフレームは含めません。3モダリティはそれぞれの長さを保持します。
- 欠落・空区間があるサンプルを全7構成で共通に除外し、入力構成によるテスト対象の違いを避けます。
- NaN/Infはゼロ置換し、標準化の平均・標準偏差は訓練集合のa/vだけで計算します。textには標準化を加えません。
- `data/mosi.manifest.json` 等に入力SHA256、入力次元、区間数、除外理由・ID、未知動画を記録します。公開済みの分割件数と違う場合は、この記録で原因を確認してください。
- `prepare` は既存出力を上書きしません。特徴・長さ・入力を変更するときは別の出力名を指定します。

## 全7構成を学習・評価（共通CLI）

Drive経路は前述の `data/mosi_drive_data.h5` / `runs/mosi_drive_data`、Kaggle経路は `data/mosi_kaggle.h5` / `runs/mosi_kaggle` を指定します。以下は従来のSDK CSD経路の例です。

```bash
uv run --no-sync mulg run \
  --data data/mosi.h5 --output runs/mosi \
  --modalities all --seeds 13 37 73 --device auto

uv run --no-sync mulg run \
  --data data/mosei.h5 --output runs/mosei \
  --modalities all --seeds 13 37 73 --device auto
```

各データセットにつき7構成×3seed=21モデルを学習します。各構成にはその入力に対応するパラメータのみを作成します。検証損失が最小のcheckpointを選んでから、テスト集合を一度評価します。テスト正解率でepochやハイパーパラメータを選びません。

論文表1に合わせた既定値:

| 設定 | MOSI | MOSEI |
|---|---:|---:|
| epochs | 100 | 50 |
| 実効batch size | 128 | 16 |
| 学習率 | 0.001 | 0.001 |
| optimizer | Adam | Adam |
| hidden size / GRU layers | 40 / 2 | 40 / 2 |
| text dropout | 0.2 | 0.3 |
| attention / output dropout | 0.1 / 0.1 | 0.1 / 0.1 |

メモリ使用量を抑えるため、既定でmicrobatch=4の勾配蓄積を使います。最後の不完全なbatchもサンプル数で重み付けします。batch sizeの数字は実効値です。通常はGPUを推奨します。特に式に忠実な `--gru-variant paper` は逐次計算のため遅くなります。長い配列でメモリ不足になった場合は `--micro-batch-size 1`、またはprepare時の最大長を調整してください。長さの変更は実験条件として明記します。

1seedだけ、または特定の構成だけの実行:

```bash
uv run --no-sync mulg run --data data/mosi.h5 \
  --output runs/mosi_one_seed --seeds 13 --modalities t a v ta tv av tav
```

高速GRU・回帰損失の比較実験（既定の再構成条件からの変更です）:

```bash
uv run --no-sync mulg run --data data/mosi.h5 \
  --output runs/mosi_torch_mae --modalities all --seeds 13 \
  --gru-variant pytorch --objective mae
```

同一設定で完了済み構成をスキップするには `--resume` を付けます。中断された構成はepoch 1から再学習します。epoch途中のoptimizer状態からの再開ではありません。データ・モデル・学習設定・実装ハッシュが異なる場合、同じ出力への再開を拒否します。

SDK経路を取得からまとめて実行する場合は `bash run_all.sh mosi sdk` / `bash run_all.sh mosei sdk` も使えます。Kaggle経路の一括コマンドは前述のとおりです。

## Acc2 / Acc7 の定義と保存先

すべてのaccuracyは **0–100の百分率** です。論文では境界処理の指定が不足しているため、次を本実装の定義として固定しています。

| 列 | 定義 |
|---|---|
| Acc2 | 正解の連続ラベル `y != 0` のみ。`score > 0` と `y > 0` の一致率。既定の主指標 |
| Acc2_with_zero | 全サンプルで `score >= 0` と `y >= 0` を比較。0を非負側へ含める別慣例 |
| Acc7 | 正解値を[-3,3]にclipし、最近傍整数へ丸めた7クラスとの一致率 |

`ce7` の予測クラスは7ロジットのargmaxです。Acc2用の連続スコアは `sum(p(k) * k), k=-3..3` とします。回帰の `mae` / `mse` では1出力をAcc2へ使い、Acc7はclipして丸めます。丸めはNumPy/PyTorchのties-to-even（例: 0.5→0、1.5→2）です。正解の連続値が0ではなく、丸めると0になるだけのサンプルはAcc2から除外しません。

`Acc2` 対象がゼロ件の場合はJSONでnull、CSVでは空欄とします。MAEとCorrは補助値として保存し、CE7の期待値スコアによるMAE/Corrと回帰出力によるMAE/Corrを混同しないよう設定を併記します。

主な出力:

```text
runs/mosi_drive_data/
  summary.csv                 # t/a/v/ta/tv/av/tavの7行、seed間のmeanと標本std
  results.json                # 各構成・seedの結果と件数
  tav/seed_13/
    config.json               # 全設定、データmanifest、入力・実装ハッシュ、依存バージョン
    history.csv               # epochごとのtrain/valid loss
    best.pt                   # 検証損失最小モデル
    metrics.json              # Acc2/Acc2_with_zero/Acc7/MAE/Corrと実測件数
    test_predictions.csv      # ID・正解・スコア・予測クラス
```

`summary.csv` の `Acc2_mean`, `Acc2_std`, `Acc7_mean`, `Acc7_std` が比較に使う列です。stdはseed間の標本標準偏差で、論文表のconfidence intervalとは異なります。1seedのstd=0は不確実性ゼロを意味しません。

保存済みモデルの再評価:

```bash
uv run --no-sync mulg evaluate \
  --checkpoint runs/mosi_drive_data/tav/seed_13/best.pt \
  --data data/mosi_drive_data.h5 --output runs/reevaluated_tav
```

## データなしで動作確認する

```bash
uv run --no-sync mulg smoke-data --output data/synthetic.h5
uv run --no-sync mulg run --data data/synthetic.h5 \
  --output runs/smoke --modalities all --seeds 13 --epochs 1 --device cpu
```

**これは合成データでの疎通確認です。論文・CMUデータセットの再現精度ではありません。** 合成データであることはHDF5、metrics.json、summary.csvおよびコンソール表示に残ります。実データの代用にはしないでください。

テストでは、Attentionの式、GRUの式、パディングの不変性、全7構成の勾配、時間境界、公式分割、訓練集合だけの標準化、2種のAcc2、7クラス化、全7構成の合成データ学習、checkpoint再評価と完了済み再開を確認します。Kaggleの一部実ファイル取得と音声・映像特徴抽出、MMSAの実ラベルCSV取得・全ID/公式分割照合も別途確認しました。全量の特徴抽出・実学習・論文の報告精度は検証していません。

## 参照

- 論文: https://doi.org/10.1038/s41598-025-93023-3
- 指定Driveフォルダ: https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq
- Kaggle MOSI: https://www.kaggle.com/datasets/mathurinache/cmu-mosi
- MMSAのラベル配布案内: https://github.com/thuiar/MMSA#2-datasets
- Stanford GloVe: https://nlp.stanford.edu/projects/glove/
- 指定SDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
- 使用commit: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/tree/2eadcd7e7b9e20e83b868cdad385b41830285c
- 論文記載の著者コードURL（作成時404）: https://github.com/bitbitlemon/Multimodal-GRU
- uv: https://docs.astral.sh/uv/

SDKレシピ・分割のライセンスは `src/mulg/resources/SDK_LICENSE.txt` を参照してください。データセット・特徴量・事前学習埋め込みを使用・発表する際は、それぞれの利用条件と引用要件を確認してください。
