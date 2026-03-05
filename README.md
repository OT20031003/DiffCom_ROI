![Python >=3.8](https://img.shields.io/badge/Python->=3.8-yellow.svg)
![PyTorch >=1.9](https://img.shields.io/badge/PyTorch->=1.9-blue.svg)

# DiffCom / ADJSCC 実装ガイド（日本語）

このリポジトリは、無線画像伝送向けの DiffCom 系コードと、ADJSCC の単体学習・評価コードを含みます。

- DiffCom 推論: `main_diffcom.py`（`configs/diffcom.yaml` を使用）
- ADJSCC 学習: `train_djscc.py`
- ADJSCC 評価: `test_djscc.py`
- 合成重要度マップ生成: `generate_ffhq_importance_maps.py`

## 1. ADJSCC 学習で使う損失関数

現行の `train_djscc.py` では、前景/背景の分離重み（`alpha`, `beta`）は使わず、
再構成誤差と相関正則化の和を最小化します。

再構成誤差（`--loss-type l1` または `mse`）:

$$
\mathcal{L}_{\mathrm{recon}} =
\begin{cases}
\frac{1}{N} \sum_{i=1}^{N} |x_i - \hat{x}_i| & (\text{L1}) \\
\frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{x}_i)^2 & (\text{MSE})
\end{cases}
$$

重要度マップ $I$ と誤差マップ $E$（実装では RGB 平均の絶対誤差）とのピアソン相関:

$$
\mathrm{corr}(I, E) =
\frac{\sum (I - \bar{I})(E - \bar{E})}
{\sqrt{\sum (I - \bar{I})^2 \sum (E - \bar{E})^2 + \varepsilon}}
$$

最終損失:

$$
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{recon}} + \lambda_{\mathrm{corr}}\,\mathrm{corr}(I, E)
$$

ここで $\lambda_{\mathrm{corr}}$ は `--lambda-corr` で指定します。

## 2. データ構成

`train_djscc.py` / `test_djscc.py` は、次の構成を想定しています。

```text
<split_dir>/
  images/
    xxx.png
    yyy.jpg
  importance/
    xxx.png
    yyy.png
```

- `importance/` がない、または対応ファイルが見つからない場合は all-ones マップにフォールバックします。
- `--train-images-dir` / `--train-importance-dir` のような override 引数も利用できます。

## 3. 環境構築（例）

依存関係は固定の `requirements.txt` がないため、最低限として以下を推奨します。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision tqdm pillow opencv-python pyyaml lpips
```

- `lpips` は評価時の LPIPS 計算で使います。未導入でも `test_djscc.py` は継続可能です。

## 4. 重みファイル

### ADJSCC 事前学習重み

- `configs/diffcom.yaml` のデフォルトは `_djscc/ckpt/ADJSCC_C=2.pth.tar` を参照します。
- 必要に応じて重みを `_djscc/ckpt/` に配置してください。
- 公開済み ADJSCC 重み: https://drive.google.com/drive/folders/1N0EzzxCv1wh6JeFr0g8vkmB0Qj23ozZJ?usp=sharing

### 拡散モデル重み（DiffCom 用）

- `ffhq_10m.pt` などを `model_zoo/` に配置してください。
- 256x256_diffusion_uncond.pt: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
- ffhq_10m.pt: https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing

## 5. 実行コマンド

### 5.1 重要度マップ生成（合成ランダム構造）

```bash
python3 generate_ffhq_importance_maps.py \
  --input-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --output-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --seed 42 \
  --min-blobs 3 \
  --max-blobs 8 \
  --min-blob-radius-frac 0.04 \
  --max-blob-radius-frac 0.22 \
  --center-prob 0.65 \
  --patch-prob 0.50 \
  --stripe-prob 0.35 \
  --blur-kernel 31
```

### 5.2 ADJSCC 学習（通常実行）

```bash
python3 train_djscc.py \
  --train-images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --train-importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --val-images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --val-importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --save-dir results/djscc_train_random_imp \
  --channel-num 2 \
  --image-size 256 \
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --snr-range -10 10 \
  --lambda-corr 0.1 \
  --loss-type l1 \
  --device cuda
```

### 5.3 ADJSCC 学習（`nohup` 実行）

```bash
mkdir -p results/djscc_train_random_imp

nohup python3 train_djscc.py \
  --train-images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --train-importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --save-dir results/djscc_train_random_imp \
  --channel-num 2 \
  --image-size 256 \
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --snr-range -10 10 \
  --lambda-corr 0.1 \
  --device cuda \
  > results/djscc_train_random_imp/train.log 2>&1 &

# ログ確認
tail -f results/djscc_train_random_imp/train.log
```

### 5.4 ADJSCC 学習再開（`nohup`）

```bash
nohup python3 train_djscc.py \
  --resume results/djscc_train_standard/best.pth \
  --epochs 4 \
  --train-images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --train-importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --save-dir results/djscc_train_standard \
  --channel-num 2 \
  --image-size 256 \
  --batch-size 8 \
  --lr 1e-4 \
  --snr-range -10 10 \
  --lambda-corr 0.1 \
  --device cuda \
  > results/djscc_train_random_imp/resume.log 2>&1 &
```

### 5.5 ADJSCC 評価（平均値出力）

```bash
python3 test_djscc.py \
  --checkpoint results/djscc_train_random_imp/best.pth \
  --images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --output-dir results/djscc_test_best \
  --channel-num 2 \
  --snr 0 \
  --loss-type l1 \
  --report-correlation \
  --device cuda
```

### 5.6 ADJSCC 評価（画像ごとの指標も表示）

```
rm -rf results/djscc_test_best
```

```bash
python3 test_djscc.py \
  --checkpoint results/djscc_train_random_imp/best.pth \
  --images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --output-dir results/djscc_test_best_proposed \
  --channel-num 2 \
  --snr -10 \
  --print-per-image-psnr \
  --print-per-image-lpips \
  --report-correlation \
  --print-per-image-correlation \
  --device cuda \
  --num-test-images 20 \
  --importance-threshold 0.5 \
  --print-per-image-split-mse
```
## 通常のADJSCC (比較用)
### 学習
```bash
nohup python3 train_djscc.py \
  --train-images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --train-importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --save-dir results/djscc_train_standard \
  --channel-num 2 \
  --image-size 256 \
  --batch-size 8 \
  --epochs 6 \
  --lr 1e-4 \
  --snr-range -10 10 \
  --lambda-corr 0 \
  --disable-importance-gating \
  --device cuda \
  > results/djscc_train_standard/train.log 2>&1 &
```
### 再開
```bash
nohup python3 train_djscc.py \
  --resume results/djscc_train_standard/best.pth \
  --epochs 6 \
  --train-images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --train-importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --save-dir results/djscc_train_standard \
  --channel-num 2 \
  --image-size 256 \
  --batch-size 8 \
  --lr 1e-4 \
  --snr-range -10 10 \
  --lambda-corr 0 \
  --disable-importance-gating \
  --device cuda \
  > results/djscc_train_standard/resume.log 2>&1 &
```

### テスト
```bash
python3 test_djscc.py \
   --checkpoint results/djscc_train_standard/best.pth \
   --images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
   --importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
   --output-dir results/djscc_test_standard \
   --channel-num 2 \
   --snr 10 \
   --report-correlation \
   --print-per-image-psnr \
   --print-per-image-lpips \
   --device cuda \
   --num-test-images 20 \
   --print-per-image-split-mse \
   --disable-importance-gating
```


| SNR | Method | PSNR (dB) ↑ | LPIPS ↓ | CORR (強い負が優) | MSE ($\geq 0.5$) ↓ | MSE ($< 0.5$) |
| --- | --- | --- | --- | --- | --- | --- |
| **10 dB** | Standard | **27.28** | **0.192** | -0.014 | 0.00198 | **0.00201** |
|  | **Proposed** | 21.85 | 0.261 | **-0.737** | **0.00159** | 0.00892 |
| **5 dB** | Standard | **26.15** | **0.228** | -0.008 | 0.00254 | **0.00259** |
|  | **Proposed** | 20.96 | 0.308 | **-0.714** | **0.00208** | 0.01087 |
| **0 dB** | Standard | **24.03** | **0.310** | 0.002 | 0.00414 | **0.00415** |
|  | **Proposed** | 19.57 | 0.399 | **-0.653** | **0.00325** | 0.01480 |
| **-5 dB** | Standard | **21.44** | **0.453** | 0.002 | 0.00762 | **0.00748** |
|  | **Proposed** | 18.10 | 0.535 | **-0.527** | **0.00589** | 0.02015 |
| **-10 dB** | Standard | **18.70** | **0.606** | -0.010 | 0.01395 | **0.01406** |
|  | **Proposed** | 16.62 | 0.643 | **-0.391** | **0.01094** | 0.02722 |

### 5.7 DiffCom 推論（`configs/diffcom.yaml` 使用）

```bash
python3 main_diffcom.py --opt ./configs/diffcom.yaml
```

`nohup` で実行する場合:

```bash
mkdir -p results/diffcom_run
nohup python3 main_diffcom.py --opt ./configs/diffcom.yaml \
  > results/diffcom_run/infer.log 2>&1 &

tail -f results/diffcom_run/infer.log
```

## 6. 主な出力

- ADJSCC 学習: `--save-dir` 配下に `latest.pth`, `best.pth`, `final.pth`
- ADJSCC 評価: `--output-dir` 配下に再構成画像を保存
- DiffCom 推論: `configs/diffcom.yaml` の設定に従ってログと画像を出力

## 7. 参考

- 論文: [DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling](https://arxiv.org/abs/2406.07390)
- プロジェクトページ: [https://semcomm.github.io/DiffCom/](https://semcomm.github.io/DiffCom/)

## 8. Citation

```bibtex
@article{wang2024diffcom,
  title={DiffCom: Channel Received Signal is a Natural Condition to Guide Diffusion Posterior Sampling},
  author={Wang, Sixian and Dai, Jincheng and Tan, Kailin and Qin, Xiaoqi and Niu, Kai and Zhang, Ping},
  journal={arXiv preprint arXiv:2406.07390},
  year={2024}
}
```
