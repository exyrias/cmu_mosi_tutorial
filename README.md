# MulG 再現実験用プログラム

添付論文「Multimodal GRU with directed pairwise cross-modal attention for sentiment analysis」(Qin et al., Scientific Reports 15, 10112, 2025) の記述から再構成した実装です。

CMU-MOSI / CMU-MOSEI を対象に、**t / a / v / ta / tv / av / tav の7構成を、それぞれ最初から独立に学習**し、同じテスト集合で Acc2・Acc7 を出します。t は text、a は audio、v は visual です。全モダリティで学習したモデルの入力を後からゼロにする実験ではありません。

## 再現の範囲と、先に確認する点

**著者実装と同一の厳密再現とは主張できません。** 論文末尾のコードURL `https://github.com/bitbitlemon/Multimodal-GRU` は作成時の確認で GitHub API が404を返しました。以下の不明点・相違を残さず記録し、再実験できるようにしています。

| 項目 | 論文の記述 | この実装 |
|---|---|---|
| 中核モデル | Conv1D、位置埋め込み、方向付きペアAttention、GRU、結合・FC・残差 | 図3と式(1)–(9)を基に構成。3モダリティでは6方向、2つでは2方向、単独ではAttentionを通さずGRUへ入力 |
| Attention | 式(5)は双線形スコア、図3はCT、表7は4 heads | 4分割のheadごとに双線形スコアを計算。ソース側時間軸でsoftmax。スケーリングなし。各方向2ブロック。CTのFFN幅4dとpost-LNは補完した設計 |
| GRU | 2層、隠れ40。式(3)はresetを行列積の前に適用 | 既定 `paper` は式(1)–(4)を直接実装（式にないbiasは付加しない）。`pytorch` は高速な標準GRUで、候補状態の式が異なる |
| 特徴抽出 | word2vec/CNN、openSMILE、3D-CNNなどを記述 | 既定はSDK公開済みGloVe / COVAREP / OpenFace2。論文記載の特徴抽出器をそのまま再現したものではない |
| 損失・出力 | Cross entropyの記述、連続値MAE/Corr、表7のoutput_dim=10が混在 | 既定 `ce7` は7クラスCE。比較用に `mae` / `mse` の連続値回帰も選択可能。10クラスにはしない |
| 残差・正規化 | post-LN / pre-LN / ResiDualを説明するが、最終構成は不明瞭 | CTと融合部に通常のpost-LN残差を使用。ResiDualやBatchNormの挿入位置は推測しない |
| データ分割 | 分割・前処理の詳細に不足あり | SDKの公式動画ID分割。ラベル区間に切り出し、モダリティ間の時間軸は揃えない |
| 入力長 | 詳細不明 | 最大長 t=128, a=512, v=512。超過分は区間全体を等分して平均。0で上限解除。設定と実データ数を記録 |
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
uv sync --frozen --extra cpu --group dev
uv run --no-sync pytest -q
```

NVIDIA GPUを使うLinux/Windows環境では、CPU用のsyncの代わりに次を実行します。

```bash
uv sync --frozen --extra cu124 --group dev
uv run --no-sync python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
uv run --no-sync pytest -q
```

CUDA 12.4に対応したNVIDIAドライバが必要です。Python依存関係は `uv.lock` で固定します。GPU用wheelでの実学習は今回検証していません。以降は **`uv run --no-sync`** を使い、syncで選択したCPU/GPUバックエンドを維持します。GPUへの切替時はもう一度対応する `uv sync` を実行してください。

同じ処理をまとめた `bash setup.sh cpu` / `bash setup.sh cu124` も利用できます。

## 実データの取得・前処理

指定SDKのcommit `2eadcd7e7b9e20e83b868cdad385b41830285c` に含まれるURLレシピと公式分割を `src/mulg/resources/sdk_manifest.json` に同梱しています。データセット自体は同梱しません。SDKのCSD形式をh5pyで直接読み、旧SDKパッケージのインストールに依存しない構成です。SDK由来データ・分割を使用しており、別データセットへの置換は行いません。

```bash
# MOSI
uv run --no-sync mulg download --dataset mosi --output data/raw/mosi
uv run --no-sync mulg prepare \
  --dataset mosi --raw-dir data/raw/mosi --output data/mosi.h5

# MOSEI
uv run --no-sync mulg download --dataset mosei --output data/raw/mosei
uv run --no-sync mulg prepare \
  --dataset mosei --raw-dir data/raw/mosei --output data/mosei.h5
```

**取得先の制約:** SDKのレシピは古い `immortal.multicomp.cs.cmu.edu` のURLを含み、SDK READMEにもGloVe/BERT特徴の公開停止・再アラインメントに関する注意があります。この作業環境からのMOSI URL確認ではHTTP/HTTPSとも403となったため、実データの取得・実学習・実測Acc2/Acc7は完了していません。ローカル環境でも取得できない場合、以下の既存CSD指定で進めます。エラー時に合成データへ自動置換することはありません。

HTTPSを既定にしています。配布元がHTTPのみを提供している場合は `download --scheme http` を明示してください。失敗したURLは `download_manifest.json` に残ります。取得済みファイルは検証して再利用し、`--overwrite` を指定した場合だけ置き換えます。

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

MOSIでopenSMILEを使う場合は `download --audio-feature OpenSMILE` を指定できます。MOSEIのopenSMILE URLは指定SDKの有効なレシピに存在しないため、既存CSDを `--audio-csd` に渡してください。配布特徴を変更した結果は、既定のCOVAREPの結果と別実験として記録してください。

### word vector CSDを取得できない場合

SDKの生単語CSDと、手元にある**テキスト形式**GloVe/word2vecの固定済み事前学習ベクトルを組み合わせることもできます（バイナリword2vec形式は未対応）。大きなベクトルファイルから必要な単語だけを読み、未知語はゼロベクトルにし、その数を記録します。データセットを使った埋め込みの再学習は行いません。

```bash
uv run --no-sync mulg download --dataset mosi \
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

## 全7構成を学習・評価

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

取得からまとめて実行する場合は `bash run_all.sh mosi` / `bash run_all.sh mosei` も使えます。

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
runs/mosi/
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
  --checkpoint runs/mosi/tav/seed_13/best.pt \
  --data data/mosi.h5 --output runs/reevaluated_tav
```

## データなしで動作確認する

```bash
uv run --no-sync mulg smoke-data --output data/synthetic.h5
uv run --no-sync mulg run --data data/synthetic.h5 \
  --output runs/smoke --modalities all --seeds 13 --epochs 1 --device cpu
```

**これは合成データでの疎通確認です。論文・CMUデータセットの再現精度ではありません。** 合成データであることはHDF5、metrics.json、summary.csvおよびコンソール表示に残ります。実データの代用にはしないでください。

テストでは、Attentionの式、GRUの式、パディングの不変性、全7構成の勾配、時間境界、公式分割、訓練集合だけの標準化、2種のAcc2、7クラス化、全7構成の合成データ学習、checkpoint再評価と完了済み再開を確認します。実データの特徴整合性や論文の報告精度は検証していません。

## 参照

- 論文: https://doi.org/10.1038/s41598-025-93023-3
- 指定SDK: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK
- 使用commit: https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/tree/2eadcd7e7b9e20e83b868cdad385b41830285c
- 論文記載の著者コードURL（作成時404）: https://github.com/bitbitlemon/Multimodal-GRU
- uv: https://docs.astral.sh/uv/

SDKレシピ・分割のライセンスは `src/mulg/resources/SDK_LICENSE.txt` を参照してください。データセット・特徴量・事前学習埋め込みを使用・発表する際は、それぞれの利用条件と引用要件を確認してください。
