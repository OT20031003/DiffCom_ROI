# DiffCom + DeepJSCC 実行メモ（日本語）

## 1. 重要度マップを生成（顔モデルなし）
このワークフローの重要度マップは、顔認識やランドマークではなく、ランダム構造マップです。

```bash
python3 /mnt/d/WSL_Work/diffcom_roi/diffcom/generate_ffhq_importance_maps.py \
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

## 2. 学習（nohup 付き）
`train.log` に出力しつつバックグラウンドで学習します。

```bash
mkdir -p /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_train_random_imp

nohup python3 /mnt/d/WSL_Work/diffcom_roi/diffcom/train_djscc.py \
  --train-images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --train-importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --save-dir /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_train_random_imp \
  --channel-num 2 \
  --image-size 256 \
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --snr-range -10 10 \
  --lambda-corr 0.1 \
  --device cuda \
  > /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_train_random_imp/train.log 2>&1 &
```

進捗確認:

```bash
tail -f /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_train_random_imp/train.log
```

## 3. 学習再開（best.pth から、nohup 付き）
`--epochs` は「最終到達エポック」です（例: epoch 4 で停止後、`--epochs 50` なら 5 から再開）。

```bash
nohup python3 /mnt/d/WSL_Work/diffcom_roi/diffcom/train_djscc.py \
  --resume /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_train_random_imp/best.pth \
  --epochs 50 \
  --train-images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --train-importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --channel-num 2 \
  --image-size 256 \
  --batch-size 8 \
  --lr 1e-4 \
  --snr-range -10 10 \
  --lambda-corr 0.1 \
  --device cuda \
  > /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_train_random_imp/resume.log 2>&1 &
```

## 4. テスト（PSNR / LPIPS / 相関）
`test_djscc.py` では tqdm バーを使わず、読みやすいセクションログを表示します。

### 4.1 平均値のみ表示
```bash
python3 /mnt/d/WSL_Work/diffcom_roi/diffcom/test_djscc.py \
  --checkpoint /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_train_random_imp/best.pth \
  --images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --output-dir /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_test_best \
  --channel-num 2 \
  --snr 10 \
  --report-correlation \
  --device cuda
```

### 4.2 画像ごとの指標も表示（PSNR/LPIPS/CORR）
```bash
python3 /mnt/d/WSL_Work/diffcom_roi/diffcom/test_djscc.py \
  --checkpoint /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_train_random_imp/best.pth \
  --images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
  --importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
  --output-dir /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_test_best \
  --channel-num 2 \
  --snr -5 \
  --print-per-image-psnr \
  --print-per-image-lpips \
  --report-correlation \
  --print-per-image-correlation \
  --device cuda
```

## 5. 補足
- LPIPS には `lpips` パッケージが必要です。未導入時は自動で LPIPS 計算を無効化して継続します。
- 重要度マップが見つからない画像は、学習・テスト側で all-ones マップにフォールバックします。


```
python train_djscc.py \
  --loss-type mse \
  --lambda-corr 0.0 \
  --disable-importance-gating
```
  python3 /mnt/d/WSL_Work/diffcom_roi/diffcom/train_djscc.py \
    --train-images-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k \
    --train-importance-dir /mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance \
    --save-dir /mnt/d/WSL_Work/diffcom_roi/diffcom/results/djscc_train_random_imp \
    --channel-num 2 \
    --image-size 256 \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --disable-importance-gating \
    --snr-range -10 10 \
    --lambda-corr 0.1 \
    --device cuda
