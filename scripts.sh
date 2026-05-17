# uv run main.py simulate \
#   --out data/simulated/test-1 \
#   --split-patients 3000 1000 1000 \
#   --clusters 4 \
#   --min-visits 3 \
#   --max-visits 16 \
#   --followup-days 1095 \
#   --hidden-size 128 \
#   --latent-dim 8 \
#   --attention-layers 4 \
#   --attention-heads 4 \
#   --censoring-rate 0.45 \
#   --seed 2026

uv run main.py train \
  --data data/simulated/test-1/train.pt \
  --val-data data/simulated/test-1/val.pt \
  --test-data data/simulated/test-1/test.pt \
  --clusters 4 \
  --epochs 80 \
  --warmup-epochs 20 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --encoder-hidden-dim 64 \
  --decoder-hidden-dim 64 \
  --latent-dim 16 \
  --n-layers 1 \
  --dropout 0.0 \
  --seed 20260517 \
  --save-dir runs/test-1-sim
