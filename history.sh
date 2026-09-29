#!/usr/bin/env bash
# 按任务块复制执行；此文件用于查阅和复现，请勿整体运行。
# 各块在实际派发前追加；运行状态和代码快照见 remote-run history。

# 2026-09-22：CRC Cox/新Weibull对照的CUDA合成检查（seed=72/20260909），不读取真实数据。
# 主机 tmu-99-rzw-xtcp；目录 /home/rongzw/projects/TRAILS；临时检查代码随本任务保留。
# 输出为控制台检查结果与远端 /tmp/trails-cuda-smoke-*；先按锁文件准备项目环境。
(
  cd /home/rongzw/projects/TRAILS || exit 1
  uv sync --frozen
  uv run python outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/analysis_code/cuda_smoke.py
)

# 2026-09-22：cox完整K=2–10×三seed选择（27次）；与另一生存头并发比较，不重做预处理。
# 主机 tmu-99-rzw-xtcp；目录 /home/rongzw/projects/TRAILS；训练及门槛固定，test仅作描述。
# 任务 crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840-cox；重新训练请另设paths.dir，勿覆盖原产物。
(
  cd /home/rongzw/projects/TRAILS || exit 1
  uv run python -m scripts.crc_yunnan.05_run \
    'model=base' \
    'trainer=full' \
    'split.dir=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-training-s3-w3-20260917-153910/preproc/preproc/attempt-1' \
    'split.seed=20260908' \
    'split.strategy=random' \
    'n_clusters=null' \
    'k_selection.enabled=true' \
    'k_selection.candidate_clusters=[2,3,4,5,6,7,8,9,10]' \
    'k_selection.seeds=[20260909,20260908,20260910]' \
    'k_selection.selection_rule=one_standard_error' \
    'k_selection.require_non_empty=true' \
    'k_selection.min_cluster_fraction=0.05' \
    'k_selection.compute_stability=true' \
    'k_selection.min_mean_pairwise_ari=null' \
    'k_selection.result_dir=k_selection' \
    'model.latent_dim=32' \
    'model.dropout=0.0' \
    'model.survival_head_hidden_layers=1' \
    'model.loss.weighting=uncertainty' \
    'model.loss.reconstruction_weight=1.0' \
    'model.loss.survival_weight=0.2' \
    'model.loss.cluster_weight=0.5' \
    'model.encoder.input.kind=grud' \
    'model.encoder.input.hidden_dim=64' \
    'model.encoder.input.n_heads=2' \
    'model.encoder.input.time_embedding_kind=projection' \
    'model.encoder.input.value_projection_dim=1' \
    'model.encoder.mapping.kind=transformer' \
    'model.encoder.mapping.hidden_dim=64' \
    'model.encoder.mapping.n_layers=2' \
    'model.encoder.mapping.n_heads=2' \
    'model.decoder.kind=transformer' \
    'model.decoder.conditioning=concat_time' \
    'model.decoder.hidden_dim=64' \
    'model.decoder.n_layers=2' \
    'model.decoder.n_heads=2' \
    'trainer.seed=20260909' \
    'trainer.valid_size=0.0' \
    'trainer.batch_size=128' \
    'trainer.learning_rate=0.001' \
    'trainer.warmup_epochs=30' \
    'trainer.gmm_init_iters=5' \
    'trainer.min_epochs=100' \
    'trainer.max_epochs=300' \
    'trainer.gradient_clip_norm=1.0' \
    'trainer.device=cuda:0' \
    'trainer.early_stop=true' \
    'trainer.early_stopping_patience=30' \
    'trainer.early_stopping_min_delta=0.0' \
    'trainer.early_stopping_monitor=loss' \
    '+trainer.risk_horizon=1440' \
    'artifacts.names=[history]' \
    'swanlab.enabled=false' \
    'outputs.summary=run_manifest.json' \
    'model.survival_loss=cox' \
    'trainer.cindex_risk_score=auto' \
    'paths.dir=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/cox'
)

# 2026-09-22：weibull完整K=2–10×三seed选择（27次）；与另一生存头并发比较，不重做预处理。
# 主机 tmu-99-rzw-xtcp；目录 /home/rongzw/projects/TRAILS；训练及门槛固定，test仅作描述。
# 任务 crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840-weibull；重新训练请另设paths.dir，勿覆盖原产物。
(
  cd /home/rongzw/projects/TRAILS || exit 1
  uv run python -m scripts.crc_yunnan.05_run \
    'model=base' \
    'trainer=full' \
    'split.dir=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-training-s3-w3-20260917-153910/preproc/preproc/attempt-1' \
    'split.seed=20260908' \
    'split.strategy=random' \
    'n_clusters=null' \
    'k_selection.enabled=true' \
    'k_selection.candidate_clusters=[2,3,4,5,6,7,8,9,10]' \
    'k_selection.seeds=[20260909,20260908,20260910]' \
    'k_selection.selection_rule=one_standard_error' \
    'k_selection.require_non_empty=true' \
    'k_selection.min_cluster_fraction=0.05' \
    'k_selection.compute_stability=true' \
    'k_selection.min_mean_pairwise_ari=null' \
    'k_selection.result_dir=k_selection' \
    'model.latent_dim=32' \
    'model.dropout=0.0' \
    'model.survival_head_hidden_layers=1' \
    'model.loss.weighting=uncertainty' \
    'model.loss.reconstruction_weight=1.0' \
    'model.loss.survival_weight=0.2' \
    'model.loss.cluster_weight=0.5' \
    'model.encoder.input.kind=grud' \
    'model.encoder.input.hidden_dim=64' \
    'model.encoder.input.n_heads=2' \
    'model.encoder.input.time_embedding_kind=projection' \
    'model.encoder.input.value_projection_dim=1' \
    'model.encoder.mapping.kind=transformer' \
    'model.encoder.mapping.hidden_dim=64' \
    'model.encoder.mapping.n_layers=2' \
    'model.encoder.mapping.n_heads=2' \
    'model.decoder.kind=transformer' \
    'model.decoder.conditioning=concat_time' \
    'model.decoder.hidden_dim=64' \
    'model.decoder.n_layers=2' \
    'model.decoder.n_heads=2' \
    'trainer.seed=20260909' \
    'trainer.valid_size=0.0' \
    'trainer.batch_size=128' \
    'trainer.learning_rate=0.001' \
    'trainer.warmup_epochs=30' \
    'trainer.gmm_init_iters=5' \
    'trainer.min_epochs=100' \
    'trainer.max_epochs=300' \
    'trainer.gradient_clip_norm=1.0' \
    'trainer.device=cuda:0' \
    'trainer.early_stop=true' \
    'trainer.early_stopping_patience=30' \
    'trainer.early_stopping_min_delta=0.0' \
    'trainer.early_stopping_monitor=loss' \
    '+trainer.risk_horizon=1440' \
    'artifacts.names=[history]' \
    'swanlab.enabled=false' \
    'outputs.summary=run_manifest.json' \
    'model.survival_loss=weibull' \
    'trainer.cindex_risk_score=event_probability' \
    'paths.dir=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/weibull'
)

# 2026-09-22：本地合成数据验证现有05和临时比较脚本的54格流程，每格仅1轮预热+1轮主训练。
# 主机本机；目录 /Users/rong/Project/TRAILS；输入为此前临时合成夹具，绝非真实CRC数据。
# 输出 /tmp/trails-cox-comparison-smoke-20260922-1212；检查结果不计入远端54次正式实验或科学结论。
(
  cd /Users/rong/Project/TRAILS || exit 1
  OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/trails-mpl-cache XDG_CACHE_HOME=/tmp/trails-cache uv run 'python' '-m' 'scripts.crc_yunnan.05_run' 'split.dir=/var/folders/75/y2ydps997m3429y_c629qjxr0000gn/T/trails-crc-torchsurv-d7gfcs5_/preproc' 'k_selection.candidate_clusters=[2,3,4,5,6,7,8,9,10]' 'k_selection.seeds=[20260909,20260908,20260910]' 'k_selection.compute_stability=true' 'trainer.device=cpu' 'trainer.max_epochs=1' 'trainer.min_epochs=1' 'trainer.warmup_epochs=1' 'trainer.gmm_init_iters=1' 'trainer.batch_size=16' 'model.latent_dim=4' 'model.encoder.input.hidden_dim=8' 'model.encoder.mapping.kind=gru' 'model.encoder.mapping.hidden_dim=8' 'model.encoder.mapping.n_layers=1' 'model.decoder.kind=gru' 'model.decoder.conditioning=initial_state' 'model.decoder.hidden_dim=8' 'model.decoder.n_layers=1' '+trainer.risk_horizon=1440' 'model.survival_loss=cox' 'paths.dir=/tmp/trails-cox-comparison-smoke-20260922-1212/cox'
  OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/trails-mpl-cache XDG_CACHE_HOME=/tmp/trails-cache uv run 'python' '-m' 'scripts.crc_yunnan.05_run' 'split.dir=/var/folders/75/y2ydps997m3429y_c629qjxr0000gn/T/trails-crc-torchsurv-d7gfcs5_/preproc' 'k_selection.candidate_clusters=[2,3,4,5,6,7,8,9,10]' 'k_selection.seeds=[20260909,20260908,20260910]' 'k_selection.compute_stability=true' 'trainer.device=cpu' 'trainer.max_epochs=1' 'trainer.min_epochs=1' 'trainer.warmup_epochs=1' 'trainer.gmm_init_iters=1' 'trainer.batch_size=16' 'model.latent_dim=4' 'model.encoder.input.hidden_dim=8' 'model.encoder.mapping.kind=gru' 'model.encoder.mapping.hidden_dim=8' 'model.encoder.mapping.n_layers=1' 'model.decoder.kind=gru' 'model.decoder.conditioning=initial_state' 'model.decoder.hidden_dim=8' 'model.decoder.n_layers=1' '+trainer.risk_horizon=1440' 'model.survival_loss=weibull' 'paths.dir=/tmp/trails-cox-comparison-smoke-20260922-1212/weibull'
  PYTHONPATH=/Users/rong/Project/TRAILS OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/trails-mpl-cache XDG_CACHE_HOME=/tmp/trails-cache uv run python .remote-run/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/compare.py --root /tmp/trails-cox-comparison-smoke-20260922-1212 --preproc /var/folders/75/y2ydps997m3429y_c629qjxr0000gn/T/trails-crc-torchsurv-d7gfcs5_/preproc --project /Users/rong/Project/TRAILS --device cpu
)

# 2026-09-22：修正临时汇总读取Hydra元配置的问题后，仅重跑本地合成聚合，不重训。
# 主机本机；目录 /Users/rong/Project/TRAILS；读取已保存54格合成结果，验证配对与产物。
(
  cd /Users/rong/Project/TRAILS || exit 1
  PYTHONPATH=/Users/rong/Project/TRAILS OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/trails-mpl-cache XDG_CACHE_HOME=/tmp/trails-cache uv run python .remote-run/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/compare.py --root /tmp/trails-cox-comparison-smoke-20260922-1212 --preproc /var/folders/75/y2ydps997m3429y_c629qjxr0000gn/T/trails-crc-torchsurv-d7gfcs5_/preproc --project /Users/rong/Project/TRAILS --device cpu
)

# 2026-09-22：54次正式训练结束，加载保存模型做Cox−Weibull配对及三seed聚合，不重训。
# 主机 tmu-99-rzw-xtcp；目录 /home/rongzw/projects/TRAILS；Cox入选K3，Weibull无合格K仍保留全部候选。
# 输出 /home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/summary；临时分析代码随本任务保留，患者数据留远端。
(
  cd /home/rongzw/projects/TRAILS || exit 1
  uv run env PYTHONPATH=/home/rongzw/projects/TRAILS python /home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/analysis_code/compare.py --root /home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840 --preproc /home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-training-s3-w3-20260917-153910/preproc/preproc/attempt-1 --project /home/rongzw/projects/TRAILS --device cuda:0
)

# 2026-09-22：核心配对汇总完成，对Cox入选K3代表seed20260909执行完整06评价，不重训。
# 主机 tmu-99-rzw-xtcp；目录 /home/rongzw/projects/TRAILS；landmark后24/48月，IBS网格1至48月。
# 输出 /home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/evaluation/cox；Weibull无合格K，依预设跳过其06评价。
(
  cd /home/rongzw/projects/TRAILS || exit 1
  uv run 'python' '-m' 'scripts.crc_yunnan.06_evaluate' 'inputs.run_dir=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/cox' 'paths.dir=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/evaluation/cox' 'paths.run_dir=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/logs/evaluation-cox' 'survival.months=[24,48]' 'survival.curve_step_months=1' 'survival.calibration_bins=5' 'seed=20260908' 'device=cuda:0'
)

# 2026-09-22：本地读取已取回的聚合表，核对54格覆盖、逐seed差值、ARI、校准及损失权重。
# 主机本机；目录 /Users/rong/Project/TRAILS；仅聚合数据，检查代码保留于本次任务目录。
(
  cd /Users/rong/Project/TRAILS || exit 1
  uv run python .remote-run/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/inspect_results.py remote-results/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840 > .remote-run/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/inspection.txt
)

# 2026-09-22：本地默认uv缓存无写权限，改用明确的临时缓存重试聚合表核对；不改数据或结果。
# 主机本机；目录 /Users/rong/Project/TRAILS；输出为任务专属inspection.txt。
(
  cd /Users/rong/Project/TRAILS || exit 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/inspect_results.py remote-results/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/outputs/crc_yunnan/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840 > .remote-run/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/inspection.txt
)

# 2026-09-22：训练与评价结束后，核对54个候选固定配置、聚合文件SHA256和Excel结构。
# 主机本机；目录 /Users/rong/Project/TRAILS；仅读取取回的非患者配置和聚合结果，不重训。
# 输入为本任务保留的配置、远端哈希清单；输出remote-results对应目录artifact_checks.json。
(
  cd /Users/rong/Project/TRAILS || exit 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/verify_artifacts.py crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840 > .remote-run/crc-yunnan-os12-d20-cox-vs-weibull-k2-10-s3-20260922-115840/verification.txt
)
