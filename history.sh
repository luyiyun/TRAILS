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

# 2026-09-30：簇特异 TRAILS v3 本地合成验收，核对变分公式、预测隔离、保存加载及短训练/K选择。
# 主机本机；目录 /Users/rong/Project/TRAILS；输入为脚本内固定合成机制，seed=20260930，不读取临床数据。
# 临时核验代码与输出保留在 .remote-run/trails-v3-verification；不属于正式研究实验。
(
  cd /Users/rong/Project/TRAILS || exit 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/trails-v3-verification/verify.py --seed 20260930 --output .remote-run/trails-v3-verification/artifacts > .remote-run/trails-v3-verification/verification.log 2>&1
)

# 2026-09-30：修正临时核验脚本的 K 选择参数名后重跑；首轮已通过数学、梯度、保存加载及三种编码器检查。
# 主机本机；目录 /Users/rong/Project/TRAILS；合成seed=20260930；原脚本保留为 verify-attempt1.py。
(
  cd /Users/rong/Project/TRAILS || exit 1
  UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/trails-v3-verification/verify.py --seed 20260930 --output .remote-run/trails-v3-verification/artifacts-attempt2 > .remote-run/trails-v3-verification/verification-attempt2.log 2>&1
)

# 2026-09-30：CRC OS24月/E60，TRAILS v3按中位生存风险选择K；K2–10×三个seed，共27次训练。
# 远端主机 tmu-99-rzw-xtcp；工作目录 /home/rongzw/projects/TRAILS；复用冻结random划分，test不参与选K。
# 输出 /home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/modeling；有合格K后另行记录并执行06评价。
(
  cd /home/rongzw/projects/TRAILS || exit 1
  uv run 'python' \
    '-m' \
    'scripts.crc_yunnan.05_run' \
    'split.dir=/home/rongzw/projects/TRAILS/data/derived/crc_yunnan/preproc/v2/os/landmark-24m/E60/auto-log1p-robust/random/seed-20260908' \
    'split.seed=20260908' \
    'split.strategy=random' \
    'paths.dir=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/modeling' \
    'n_clusters=null' \
    'k_selection.enabled=true' \
    'k_selection.candidate_clusters=[2,3,4,5,6,7,8,9,10]' \
    'k_selection.seeds=[20260908,20260909,20260910]' \
    'k_selection.selection_rule=one_standard_error' \
    'k_selection.require_non_empty=true' \
    'k_selection.min_cluster_fraction=0.05' \
    'k_selection.compute_stability=false' \
    'k_selection.min_mean_pairwise_ari=null' \
    'model.survival_loss=weibull' \
    'model.latent_dim=32' \
    'model.survival_head_hidden_layers=1' \
    'model.dropout=0.0' \
    'model.encoder.input.kind=grud' \
    'model.encoder.input.hidden_dim=64' \
    'model.encoder.mapping.kind=transformer' \
    'model.encoder.mapping.hidden_dim=64' \
    'model.encoder.mapping.n_layers=2' \
    'model.encoder.mapping.n_heads=2' \
    'model.decoder.kind=transformer' \
    'model.decoder.conditioning=concat_time' \
    'model.decoder.hidden_dim=64' \
    'model.decoder.n_layers=2' \
    'model.decoder.n_heads=2' \
    'model.loss.weighting=uncertainty' \
    'model.loss.reconstruction_weight=1.0' \
    'model.loss.survival_weight=0.2' \
    'model.loss.cluster_weight=0.1' \
    'trainer.seed=20260909' \
    'trainer.valid_size=0.0' \
    'trainer.batch_size=128' \
    'trainer.learning_rate=0.001' \
    'trainer.warmup_epochs=10' \
    'trainer.gmm_init_iters=5' \
    'trainer.min_epochs=100' \
    'trainer.max_epochs=300' \
    'trainer.early_stop=true' \
    'trainer.early_stopping_patience=30' \
    'trainer.early_stopping_min_delta=0.0' \
    'trainer.early_stopping_monitor=loss' \
    'trainer.gradient_clip_norm=1.0' \
    'trainer.device=cuda:0' \
    'trainer.cindex_risk_score=median_survival' \
    'artifacts.names=[history]' \
    'swanlab.enabled=false'
)

# 2026-09-30 CRC TRAILS v3：27候选未通过占簇门槛，整理已取回的聚合训练历史和配置。
# 本地 /Users/rong/Project/TRAILS；远端 tmu-99-rzw-xtcp 的 OS24/E60 固定K2–10、seed20260908/09/10结果，无重训。
(
  cd /Users/rong/Project/TRAILS
  .venv/bin/python -c 'from pathlib import Path; import pandas as pd; p=Path("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/outputs/crc_yunnan/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/modeling/k_selection"); d=pd.concat([pd.json_normalize(pd.read_json(f).to_dict("records")).assign(seed=int(f.parent.parent.name[5:]),n_clusters=int(f.parent.name[1:])) for f in sorted(p.glob("seed-*/k*/history.json"))],ignore_index=True); d.to_csv("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/epoch_metrics.csv",index=False); print(d.groupby("stage").size())'
  .venv/bin/python -c 'from pathlib import Path; import json; import pandas as pd; p=Path("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/outputs/crc_yunnan/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/modeling/k_selection"); d=pd.json_normalize([json.loads(f.read_text()) for f in sorted(p.glob("seed-*/k*/config.json"))]); d.to_csv("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/candidate_configs.csv",index=False); print(d.nunique().to_string())'
)

# 2026-09-30 CRC TRAILS v3：汇总27候选验证损失最佳轮次和各K的三seed指标；仅分析已取回的聚合表。
# 本地 /Users/rong/Project/TRAILS；相同OS24/E60、K2–10及seed20260908/09/10，不选择失败候选充当最终模型。
(
  cd /Users/rong/Project/TRAILS
  .venv/bin/python -c 'import pandas as pd; d=pd.read_csv("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/epoch_metrics.csv"); v=d[d.stage.eq("vade")]; b=v.loc[v.groupby(["seed","n_clusters"])["valid.loss"].idxmin()]; b.to_csv("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/best_epoch_metrics.csv",index=False); print(b.filter(regex="^(global_epoch|train\\.cindex|valid\\.)$|valid\\.").describe().to_string()); print("completed main epochs",v.groupby(["seed","n_clusters"]).epoch.max().value_counts().to_dict())'
  .venv/bin/python -c 'import pandas as pd; d=pd.read_csv("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/outputs/crc_yunnan/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/modeling/k_selection/run_metrics.csv"); print(d.groupby("n_clusters").agg(cindex_mean=("cindex","mean"),cindex_sd=("cindex","std"),empty_min=("cluster_empty_count","min"),min_fraction=("cluster_min_fraction","min")).to_string()); print("unique K/seed",len(d.drop_duplicates(["seed","n_clusters"])))'
)

# 2026-09-30 CRC v3单簇诊断：对齐初始化前后与后续训练的27候选聚合历史，核对损失突变。
# 本地 /Users/rong/Project/TRAILS；OS24/E60、K2–10、seed20260908/09/10，仅读取已取回聚合表。
(
  cd /Users/rong/Project/TRAILS
  .venv/bin/python -c 'import pandas as pd; d=pd.read_csv("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/epoch_metrics.csv"); cols=["train.loss","valid.loss","train.reconstruction_loss","valid.reconstruction_loss","train.survival_loss","valid.survival_loss","train.vade_kl_loss","valid.vade_kl_loss","train.cindex","valid.cindex","valid.reconstruction_loss_weight","valid.survival_loss_weight","valid.vade_kl_loss_weight"]; print(d[d.global_epoch.isin([1,5,10,11,12,15,20,30,60,110,210,310])].groupby("global_epoch")[cols].median().round(5).to_string())'
)

# 2026-09-30 CRC v3单簇诊断：核实初始化后损失突变是否覆盖所有27候选。
# 本地 /Users/rong/Project/TRAILS；仅读取OS24/E60既有聚合历史，不追加训练。
(
  cd /Users/rong/Project/TRAILS
  .venv/bin/python -c 'import pandas as pd; d=pd.read_csv("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/epoch_metrics.csv"); s=d[d.global_epoch.eq(11)]; print(s[["train.vade_kl_loss","train.survival_loss"]].describe().to_string()); print("KL above 1000:",s["train.vade_kl_loss"].gt(1000).sum(),"/",len(s)); print(d.groupby("global_epoch")["valid.cindex"].median().loc[[11,30,110,310]].to_string())'
)

# 2026-10-04 条件生存头本地验收：合成数据、seed20261004，固定K2与K2/3双seed短训练，无远程训练。
# 本机 /Users/rong/Project/TRAILS；临时验证代码保留在本任务.remote-run目录，核验共享z、梯度、后验与格式兼容性。
(
  cd /Users/rong/Project/TRAILS
  UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/conditioned-head-verification-20261004/verify.py --seed 20261004 --output .remote-run/conditioned-head-verification-20261004/artifacts
)

# 2026-10-04 条件生存头远端CUDA前向/反向检查；不读取患者数据。
# 主机 tmu-99-rzw-xtcp；工作目录 /home/rongzw/projects/TRAILS。
(
  cd /home/rongzw/projects/TRAILS
  uv run python -c 'import torch; from trails import DataConfig, ModelConfig; from trails.model import TrailsModel, SURVIVAL_HEAD_ARCHITECTURE; m=TrailsModel(DataConfig(n_features=60), ModelConfig(n_clusters=3, latent_dim=32, survival_head_hidden_layers=1)).cuda(); y=m.survival_head(torch.randn(4,3,35,device='\''cuda'\'')); y.square().mean().backward(); assert y.shape==(4,3,1) and torch.isfinite(y).all(); print(torch.__version__, torch.cuda.get_device_name(0), SURVIVAL_HEAD_ARCHITECTURE, '\''forward/backward passed'\'')'
)

# 2026-10-04 CRC OS24/E60条件生存头：复用冻结数据，K2–10×seed20260908/09/10，比较此前单簇退化结果。
# 主机 tmu-99-rzw-xtcp；工作目录 /home/rongzw/projects/TRAILS；其余协议同2026-09-30，不放宽占簇门槛。
(
  cd /home/rongzw/projects/TRAILS
  'uv' 'run' 'python' '-m' 'scripts.crc_yunnan.05_run' \
    'split.dir=/home/rongzw/projects/TRAILS/data/derived/crc_yunnan/preproc/v2/os/landmark-24m/E60/auto-log1p-robust/random/seed-20260908' \
    'split.seed=20260908' \
    'split.strategy=random' \
    'paths.dir=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/modeling' \
    'n_clusters=null' \
    'k_selection.enabled=true' \
    'k_selection.candidate_clusters=[2,3,4,5,6,7,8,9,10]' \
    'k_selection.seeds=[20260908,20260909,20260910]' \
    'k_selection.selection_rule=one_standard_error' \
    'k_selection.require_non_empty=true' \
    'k_selection.min_cluster_fraction=0.05' \
    'k_selection.compute_stability=false' \
    'k_selection.min_mean_pairwise_ari=null' \
    'model.survival_loss=weibull' \
    'model.latent_dim=32' \
    'model.survival_head_hidden_layers=1' \
    'model.dropout=0.0' \
    'model.encoder.input.kind=grud' \
    'model.encoder.input.hidden_dim=64' \
    'model.encoder.mapping.kind=transformer' \
    'model.encoder.mapping.hidden_dim=64' \
    'model.encoder.mapping.n_layers=2' \
    'model.encoder.mapping.n_heads=2' \
    'model.decoder.kind=transformer' \
    'model.decoder.conditioning=concat_time' \
    'model.decoder.hidden_dim=64' \
    'model.decoder.n_layers=2' \
    'model.decoder.n_heads=2' \
    'model.loss.weighting=uncertainty' \
    'model.loss.reconstruction_weight=1.0' \
    'model.loss.survival_weight=0.2' \
    'model.loss.cluster_weight=0.1' \
    'trainer.seed=20260909' \
    'trainer.valid_size=0.0' \
    'trainer.batch_size=128' \
    'trainer.learning_rate=0.001' \
    'trainer.warmup_epochs=10' \
    'trainer.gmm_init_iters=5' \
    'trainer.min_epochs=100' \
    'trainer.max_epochs=300' \
    'trainer.early_stop=true' \
    'trainer.early_stopping_patience=30' \
    'trainer.early_stopping_min_delta=0.0' \
    'trainer.early_stopping_monitor=loss' \
    'trainer.gradient_clip_norm=1.0' \
    'trainer.device=cuda:0' \
    'trainer.cindex_risk_score=median_survival' \
    'artifacts.names=[history]' \
    'swanlab.enabled=false'
)

# 2026-10-05 条件生存头27候选科学未入围：复用既有聚合命令，整理配置、轮次与验证指标。
# 本地 /Users/rong/Project/TRAILS；远端tmu-99-rzw-xtcp，OS24/E60，K2–10、seed20260908/09/10；不重训。
(
  cd /Users/rong/Project/TRAILS
  .venv/bin/python -c 'from pathlib import Path; import pandas as pd; p=Path("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/outputs/crc_yunnan/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/modeling/k_selection"); d=pd.concat([pd.json_normalize(pd.read_json(f).to_dict("records")).assign(seed=int(f.parent.parent.name[5:]),n_clusters=int(f.parent.name[1:])) for f in sorted(p.glob("seed-*/k*/history.json"))],ignore_index=True); d.to_csv("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/epoch_metrics.csv",index=False); print(d.groupby("stage").size())'
  .venv/bin/python -c 'from pathlib import Path; import json; import pandas as pd; p=Path("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/outputs/crc_yunnan/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/modeling/k_selection"); d=pd.json_normalize([json.loads(f.read_text()) for f in sorted(p.glob("seed-*/k*/config.json"))]); d.to_csv("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/candidate_configs.csv",index=False); print(d.nunique().to_string())'
  .venv/bin/python -c 'import pandas as pd; d=pd.read_csv("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/epoch_metrics.csv"); v=d[d.stage.eq("vade")]; b=v.loc[v.groupby(["seed","n_clusters"])["valid.loss"].idxmin()]; b.to_csv("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/best_epoch_metrics.csv",index=False); print(b.filter(regex="^(global_epoch|train\\.cindex|valid\\.)$|valid\\.").describe().to_string()); print("completed main epochs",v.groupby(["seed","n_clusters"]).epoch.max().value_counts().to_dict())'
  .venv/bin/python -c 'import pandas as pd; d=pd.read_csv("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/outputs/crc_yunnan/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/modeling/k_selection/run_metrics.csv"); print(d.groupby("n_clusters").agg(cindex_mean=("cindex","mean"),cindex_sd=("cindex","std"),empty_min=("cluster_empty_count","min"),min_fraction=("cluster_min_fraction","min")).to_string()); print("unique K/seed",len(d.drop_duplicates(["seed","n_clusters"])))'
)

# 2026-10-05 条件生存头与旧多输出头：同OS24/E60、K2–10、seed20260908/09/10逐格配对，仅汇总已取回表。
# 本地 /Users/rong/Project/TRAILS；差值为新减旧，阶段中位数检查warmup/初始化衔接，不重训或加载旧模型。
(
  cd /Users/rong/Project/TRAILS
  .venv/bin/python -c 'import pandas as pd; a=pd.read_csv("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/outputs/crc_yunnan/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/modeling/k_selection/run_metrics.csv"); b=pd.read_csv("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/outputs/crc_yunnan/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/modeling/k_selection/run_metrics.csv"); d=a.merge(b,on=["seed","n_clusters"],suffixes=("_new","_old"),validate="one_to_one"); d["cindex_difference"]=d.cindex_new-d.cindex_old; d.to_csv("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/paired_comparison.csv",index=False); print(d[["cindex_new","cindex_old","cindex_difference"]].describe()); print(d.groupby("n_clusters").cindex_difference.agg(["mean","std"])); print("improved",(d.cindex_difference>0).sum()); print(d.groupby("seed").cindex_difference.mean())'
  .venv/bin/python -c 'import pandas as pd; d=pd.read_csv("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/epoch_metrics.csv"); s=d[d.global_epoch.isin([10,11,110,310])].groupby("global_epoch").median(numeric_only=True); s.to_csv("remote-results/crc-yunnan-os24-e60-conditioned-head-k2-10-s3-20261004-101554/stage_medians.csv"); print(s.filter(regex="loss|cindex").to_string()); print(d[d.global_epoch.eq(11)][["train.vade_kl_loss","train.survival_loss"]].describe())'
)

# 2026-10-05 配对解释：读取旧多输出头聚合历史，对照初始化前后损失中位数；不运行旧模型。
# 本地 /Users/rong/Project/TRAILS；OS24/E60、K2–10、seed20260908/09/10。
(
  cd /Users/rong/Project/TRAILS
  .venv/bin/python -c 'import pandas as pd; d=pd.read_csv("remote-results/crc-yunnan-os24-e60-trails-v3-k2-10-s3-20260930-004807/epoch_metrics.csv"); print(d[d.global_epoch.isin([10,11,310])].groupby("global_epoch")[["train.vade_kl_loss","train.survival_loss","valid.loss","valid.reconstruction_loss","valid.reconstruction_loss_weight"]].median().to_string())'
)

# 2026-10-06 CRC 07本地合成验证：固定采样20261006；检查192格分片、新增训练分支、诊断RNG、两进程及续跑。
# 本地 /Users/rong/Project/TRAILS；CPU、synthetic train60/validation30/test36，主阶段2轮；与真实患者数据及正式结果隔离。
(
  cd /Users/rong/Project/TRAILS
  UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/verification/crc-search-20261006/verify.py > .remote-run/verification/crc-search-20261006/verify.log 2>&1
)

# 2026-10-06 CRC 07合成验证续跑：修正Hydra组合配置入口后重新执行；参数、seed及隔离目录同上。
(
  cd /Users/rong/Project/TRAILS
  UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/verification/crc-search-20261006/verify.py > .remote-run/verification/crc-search-20261006/verify.log 2>&1
)

# 2026-10-06 CRC 07远端数据准备：现有凭据临时SSH-agent；99直接复制train/validation及预处理元信息到215。
# 两机 /home/rongzw/projects/TRAILS；OS24/E60/random/seed-20260908；核对五个SHA256，test仅留99，患者数据不落本地。
sh /Users/rong/Project/TRAILS/.remote-run/verification/crc-search-20261006/prepare-data.sh

# 2026-10-06 CRC 07合成验证：修正临时脚本的公开from_dataframes调用；参数、seed及输出同上，已完成训练按协议复用。
(
  cd /Users/rong/Project/TRAILS
  MPLCONFIGDIR=/tmp/trails-07-mpl UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/verification/crc-search-20261006/verify.py > .remote-run/verification/crc-search-20261006/verify.log 2>&1
)

# 2026-10-06 CRC 07合成汇总夹具：测试失败K排除、锁模SHA256、一次test评价与重复finalize复用。
# 本地隔离目录 .remote-run/verification/crc-search-20261006/fixtures；受控汇总指标仅验证控制流，不作为模型占簇达标证据。
(
  cd /Users/rong/Project/TRAILS
  MPLCONFIGDIR=/tmp/trails-07-mpl UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/verification/crc-search-20261006/verify-finalize.py > .remote-run/verification/crc-search-20261006/finalize.log 2>&1
)

# 2026-10-06 CRC 07 CLI合成入口与汇总回归：仅复用本地合成完成试验；验证Hydra子进程入口与失败门槛修订。
(
  cd /Users/rong/Project/TRAILS
  MPLCONFIGDIR=/tmp/trails-07-mpl UV_CACHE_DIR=/tmp/uv-cache uv run python -m scripts.crc_yunnan.07_search mode=plan paths.dir=.remote-run/verification/crc-search-20261006/cli-plan
  MPLCONFIGDIR=/tmp/trails-07-mpl UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/verification/crc-search-20261006/verify-finalize.py > .remote-run/verification/crc-search-20261006/finalize-regression.log 2>&1
)

# 2026-10-06 CRC 07真实合成K验证：r000，K2/3×seed20260908/09/10；复用两格K3，新增4次CPU两轮训练。
# 本地 .remote-run/verification/crc-search-20261006/real-validate-*；每片两进程，再次validate验证不重训，保留5%占簇门槛。
(
  cd /Users/rong/Project/TRAILS
  MPLCONFIGDIR=/tmp/trails-07-mpl UV_CACHE_DIR=/tmp/uv-cache uv run python .remote-run/verification/crc-search-20261006/verify-validation.py > .remote-run/verification/crc-search-20261006/validation.log 2>&1
)

# 2026-10-06 CRC 07两机锁定环境准备；主机99/215，工作目录/home/rongzw/projects/TRAILS。
# 同一Git跟踪工作树与uv.lock；新215环境使用Python3.12.12，99按同一锁核对环境；不修改系统Python。
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-99-rzw-xtcp 'cd /home/rongzw/projects/TRAILS && uv sync --locked --python 3.12.12'
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-215-rzw-xtcp 'cd /home/rongzw/projects/TRAILS && uv sync --locked --python 3.12.12'

# 2026-10-06 CRC 07锁定环境恢复：215的NumPy/DuckDB下载长期等待，停止本次安装并从99复用相同版本缓存。
# 服务器直接复制仅numpy1.26.4和duckdb1.4.5的cp312/Linux缓存及各自归档；重新uv sync --locked/Python3.12.12。
sh /Users/rong/Project/TRAILS/.remote-run/crc07-20261006/prepare-dependencies.sh

# 2026-10-06 CRC 07两机独立GPU短试跑：99/215，工作目录/home/rongzw/projects/TRAILS；每机2进程，各4线程/cuda:0。
# K10×seed20260908/09：r036 Transformer2宽64/latent32/batch256/warmup10，r052 mTAN宽64/latent16/batch256/warmup0；主阶段3轮。
# 两机各4次试跑，输出outputs/crc_yunnan/crc07-20261006/pilot/99或215；仅确认显存和执行，不进入正式选优。
.venv/bin/python /Users/rong/Project/skills/remote-run/scripts/remote_run.py group submit --plan .remote-run/crc07-20261006/pilot.json --apply

# 2026-10-06 CRC 07 GPU试跑修复后重试：CUDA统计前显式初始化设备；保留旧pilot失败记录，尚无正式训练。
# 99/215均/home/rongzw/projects/TRAILS；原r036/r052、K10、seed20260908/09、并行2各4线程；输出pilot-r2/99或215。
.venv/bin/python /Users/rong/Project/skills/remote-run/scripts/remote_run.py group submit --plan .remote-run/crc07-20261006/pilot-r2.json --apply

# 2026-10-06 CRC 07 GPU试跑聚合验收：两机r036/r052×seed20260908/09共8格；只读取已取回的统计量。
# 本地/Users/rong/Project/TRAILS；核对协议/数据指纹、完整退出与峰值显存，输出remote-results/crc07-20261006-pilot-r2/pilot_metrics.csv。
.venv/bin/python .remote-run/crc07-20261006/analyse_pilot.py

# 2026-10-06 CRC 07正式初筛：99/215均/home/rongzw/projects/TRAILS；各2进程/cuda:0/每进程4 CPU与BLAS线程，总并行4。
# OS24/E60/auto-log1p-robust/random seed20260908，train1854/validation444；K3，96配方×seed20260908/09=192次；主100–300轮/patience30。
# 负log混合中位生存时间排序；两seed占簇均>=5%且无空簇才入围，test不读取；输出outputs/crc_yunnan/crc07-20261006/screen/shard-0或1。
.venv/bin/python /Users/rong/Project/skills/remote-run/scripts/remote_run.py group submit --plan .remote-run/crc07-20261006/screen-formal.json --apply

# 2026-10-06 CRC 07正式初筛聚合：两机192格终态（191完成、1非有限梯度）；仅读取安全summary，不读取患者或test。
# 本地/Users/rong/Project/TRAILS；原OS24/E60/random-seed20260908、K3×seed20260908/09；保留5%占簇门槛及ARI优先排序。
# 输出remote-results/crc07-20261006/screen-merged；最多3套配方，K2–10×3seed及复用K3的分配在此次冻结。
(
  cd /Users/rong/Project/TRAILS
  MPLCONFIGDIR=/tmp/trails-07-mpl UV_CACHE_DIR=/tmp/uv-cache uv run --locked python -m scripts.crc_yunnan.07_search mode=merge merge_stage=screen trainer.device=cuda:0 trainer.min_epochs=100 trainer.max_epochs=300 trainer.early_stopping_patience=30 trainer.cindex_risk_score=median_survival shard.count=2 search.n_recipes=96 search.sampling_seed=20261006 search.screen_k=3 'search.screen_seeds=[20260908,20260909]' 'search.validation_seeds=[20260908,20260909,20260910]' 'search.candidates=[2,3,4,5,6,7,8,9,10]' search.min_cluster_fraction=0.05 search.top_n=3 split.seed=20260908 split.strategy=random split.dir=data/derived/crc_yunnan/preproc/v2/os/landmark-24m/E60/auto-log1p-robust/random/seed-20260908 'inputs.summaries=[remote-results/crc07-20261006-screen/default/crc07-20261006-screen-99/outputs/crc_yunnan/crc07-20261006/screen/shard-0/summary.json,remote-results/crc07-20261006-screen/tmu-215-rzw-xtcp/crc07-20261006-screen-215/outputs/crc_yunnan/crc07-20261006/screen/shard-1/summary.json]' paths.dir=remote-results/crc07-20261006/screen-merged
)

# 2026-10-06 CRC 07冻结K验证任务分配：r009/r089，K2–10×seed20260908/09/10；仅复制聚合运行输入JSON。
# 两机/home/rongzw/projects/TRAILS；assignment SHA256=71bcce10bec652cb5e4107f8bc72d8a16527ac7a07060e58f0bbeabc67e55931。
scp -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes /Users/rong/Project/TRAILS/remote-results/crc07-20261006/screen-merged/validation_assignment.json tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc07-20261006/validation_assignment.json
scp -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes /Users/rong/Project/TRAILS/remote-results/crc07-20261006/screen-merged/validation_assignment.json tmu-215-rzw-xtcp:/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc07-20261006/validation_assignment.json

# 2026-10-06 CRC 07正式K验证：r009/r089，K2–10×seed20260908/09/10共54单元，复用4格K3，新增50次训练。
# 99/215均/home/rongzw/projects/TRAILS；冻结新增分配24/26；各2进程/cuda:0/4线程；原OS24/E60划分及100–300轮/patience30。
# 输出outputs/crc_yunnan/crc07-20261006/validate/shard-0或1；配方内one-standard-error选K，三seed占簇与ARI，不读取test。
.venv/bin/python /Users/rong/Project/skills/remote-run/scripts/remote_run.py group submit --plan .remote-run/crc07-20261006/validate.json --apply

# 2026-10-06 CRC 07配方表导出：仅在本地生成与正式协议相同的96套参数及protocol.json，不加载患者数据或训练。
# 本地/Users/rong/Project/TRAILS；采样seed20261006，输出remote-results/crc07-20261006/recipe-plan，供聚合结果解释与复现。
(
  cd /Users/rong/Project/TRAILS
  MPLCONFIGDIR=/tmp/trails-07-mpl UV_CACHE_DIR=/tmp/uv-cache uv run --locked python -m scripts.crc_yunnan.07_search mode=plan trainer.device=cuda:0 trainer.min_epochs=100 trainer.max_epochs=300 trainer.early_stopping_patience=30 search.n_recipes=96 search.sampling_seed=20261006 'search.screen_seeds=[20260908,20260909]' 'search.validation_seeds=[20260908,20260909,20260910]' paths.dir=remote-results/crc07-20261006/recipe-plan
)

# 2026-10-06 用户追加CRC 07回顾性诊断：两机CPU各1进程/4线程，已完成K3模型比较p(c|z)和p(c|z,t,e)的占簇及跨seed ARI。
# 仅读取冻结validation444人与远端checkpoint/标签，额外纳入已完成的入围配方第三seed；不训练、不读test，脚本与患者数组不取回。
# 临时脚本.remote-run/crc07-20261006/posterior_diagnostics.py经Ruff/Pyright检查后传送独立目录，输出outputs/crc_yunnan/crc07-20261006/posterior/shard-0/1。
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-99-rzw-xtcp 'set -eu
diag_out=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc07-20261006/posterior/shard-0
mkdir "$diag_out/.launch-lock"
nohup sh /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc07-20261006/posterior-99.sh < /dev/null > "$diag_out/stdout.log" 2> "$diag_out/stderr.log" &
printf '\''%s\n'\'' "$!" > "$diag_out/pid"
cat "$diag_out/pid"'

ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-215-rzw-xtcp 'set -eu
diag_out=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc07-20261006/posterior/shard-1
mkdir "$diag_out/.launch-lock"
nohup sh /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc07-20261006/posterior-215.sh < /dev/null > "$diag_out/stdout.log" 2> "$diag_out/stderr.log" &
printf '\''%s\n'\'' "$!" > "$diag_out/pid"
cat "$diag_out/pid"'

# 2026-10-06 CRC 07已完成初筛的超参数和损失诊断：仅本地读取聚合CSV；按配方计数并比较37个单因素对照。
# 输入remote-results/crc07-20261006-screen及screen-merged/recipe-plan；输出remote-results/crc07-20261006/diagnostics，不读取患者或test。
MPLCONFIGDIR=/tmp/trails-07-mpl .venv/bin/python .remote-run/crc07-20261006/analyze_diagnostics.py

# 2026-10-06 CRC 07后验诊断合并：191个初筛完成模型及两个入围配方的K3第三seed，仅聚合统计。
# 本地/Users/rong/Project/TRAILS；对照outcome-free与outcome-conditioned的占簇/ARI，并汇总损失权重、方差与退化关系；不读test、不改选优。
.venv/bin/python .remote-run/crc07-20261006/merge_posterior.py

# 2026-10-06 CRC 07诊断补充：在同一权重策略内分层比较实际权重，并核对单类退化首次出现的正式轮次。
# 仍只读取原聚合CSV，更新diagnostics目录；不训练、不读取test。
.venv/bin/python .remote-run/crc07-20261006/merge_posterior.py

# 2026-10-07 CRC 07两机K验证汇总：54单元（50新增、4复用）均终态，只读取白名单聚合统计。
# 本地/Users/rong/Project/TRAILS；冻结协议和分配不变，按原占簇、BIC及one-standard-error规则选K；不读取test。
MPLCONFIGDIR=/tmp/trails-07-mpl UV_CACHE_DIR=/tmp/uv-cache uv run --locked python -m scripts.crc_yunnan.07_search mode=merge merge_stage=validate trainer.device=cuda:0 trainer.min_epochs=100 trainer.max_epochs=300 trainer.early_stopping_patience=30 trainer.cindex_risk_score=median_survival shard.count=2 search.n_recipes=96 search.sampling_seed=20261006 search.screen_k=3 'search.screen_seeds=[20260908,20260909]' 'search.validation_seeds=[20260908,20260909,20260910]' 'search.candidates=[2,3,4,5,6,7,8,9,10]' search.min_cluster_fraction=0.05 search.top_n=3 split.seed=20260908 split.strategy=random split.dir=data/derived/crc_yunnan/preproc/v2/os/landmark-24m/E60/auto-log1p-robust/random/seed-20260908 inputs.assignment=remote-results/crc07-20261006/screen-merged/validation_assignment.json 'inputs.summaries=[remote-results/crc07-20261006-validate/default/crc07-20261006-validate-99/outputs/crc_yunnan/crc07-20261006/validate/shard-0/summary.json,remote-results/crc07-20261006-validate/tmu-215-rzw-xtcp/crc07-20261006-validate-215/outputs/crc_yunnan/crc07-20261006/validate/shard-1/summary.json]' paths.dir=remote-results/crc07-20261006/validate-merged

# 2026-10-07 CRC 07一次test终评：锁定r089/K2/seed20260908，checkpoint SHA256 6e3dd63e520280d0c6286ac15e6099d500f1ec50cbb64b6c6d7a2e5b4fefeea8。
# default=tmu-99-rzw-xtcp，工作目录/home/rongzw/projects/TRAILS；IPCW仅来自train，test不参与选优；持久run crc07-20261006-finalize-99。
.venv/bin/python /Users/rong/Project/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc07-20261006-finalize-99 --apply -- python -m scripts.crc_yunnan.07_search mode=finalize shard.index=0 shard.count=2 shard.workers=2 shard.threads=4 trainer.device=cuda:0 trainer.min_epochs=100 trainer.max_epochs=300 trainer.early_stopping_patience=30 trainer.cindex_risk_score=median_survival search.n_recipes=96 search.sampling_seed=20261006 search.screen_k=3 'search.screen_seeds=[20260908,20260909]' 'search.validation_seeds=[20260908,20260909,20260910]' 'search.candidates=[2,3,4,5,6,7,8,9,10]' search.min_cluster_fraction=0.05 search.top_n=3 split.seed=20260908 split.strategy=random split.dir=data/derived/crc_yunnan/preproc/v2/os/landmark-24m/E60/auto-log1p-robust/random/seed-20260908 inputs.selection=outputs/crc_yunnan/crc07-20261006/final_selection.json inputs.checkpoint=/home/rongzw/projects/TRAILS/outputs/crc_yunnan/crc07-20261006/validate/shard-0/trials/r089/k2/seed-20260908/model.pt paths.dir=outputs/crc_yunnan/crc07-20261006/final

# 2026-10-07 用户追加单seed占簇汇总：仅本地读取已完成54格验证聚合CSV，不训练、不读取test。
# /Users/rong/Project/TRAILS；比较两个配方各seed在K2–10的无空簇次数和最小占比≥5%的次数，输出seed-occupancy。
.venv/bin/python .remote-run/crc07-20261006/seed_occupancy_summary.py

# 2026-10-07 用户指定r089/20260908补充比较K2–10：准备9个已完成模型输入，复用原单seed C-index+BIC分数。
# 本地/Users/rong/Project/TRAILS；不训练，后续按验证分数选K，所有K的test生存比较作为探索性展示。
.venv/bin/python .remote-run/crc07-20261006/prepare_single_seed.py

# 2026-10-07 r089/seed20260908补充分析：99经临时SSH-agent直接从215取5个checkpoint，模型不经过本地。
# 其余4个checkpoint已经在99；本次不训练，test仍留99；临时脚本和9模型清单为运行输入。
sh .remote-run/crc07-20261006/prepare-single-seed-models.sh
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-99-rzw-xtcp 'mkdir -p /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc07-20261007-single-seed'
scp -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes .remote-run/crc07-20261006/single_seed_evaluate.py .remote-run/crc07-20261006/single-seed-input.json tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc07-20261007-single-seed/

# 2026-10-07 持久提交r089/20260908单seed K2–10验证/test探索比较；不使用ARI，直接按原混合分数排名。
# default=tmu-99-rzw-xtcp，工作目录/home/rongzw/projects/TRAILS；4线程、cuda:0；只导出聚合指标、KM及风险表。
.venv/bin/python /Users/rong/Project/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc07-20261007-r089-seed08-compare --apply -- env PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-07-single-seed-mpl python .remote-run/diagnostics/crc07-20261007-single-seed/single_seed_evaluate.py --plan .remote-run/diagnostics/crc07-20261007-single-seed/single-seed-input.json --data data/derived/crc_yunnan/preproc/v2/os/landmark-24m/E60/auto-log1p-robust/random/seed-20260908 --output outputs/crc_yunnan/crc07-20261007/r089-seed20260908 --device cuda:0 --threads 4

# 2026-10-07 九K补充评价已exit0；修复KM风险表中文标签缺字，沿用原结果，不重算C-index/BIC或选K。
# 99/home/rongzw/projects/TRAILS，cuda0/4线程；仅确定性预测后重绘英文风险表，患者数组不保存。
scp -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes .remote-run/crc07-20261006/replot_single_seed.py tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc07-20261007-single-seed/
.venv/bin/python /Users/rong/Project/skills/remote-run/scripts/remote_run.py --target default run --apply -- env PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-07-single-seed-mpl python .remote-run/diagnostics/crc07-20261007-single-seed/replot_single_seed.py --plan .remote-run/diagnostics/crc07-20261007-single-seed/single-seed-input.json --data data/derived/crc_yunnan/preproc/v2/os/landmark-24m/E60/auto-log1p-robust/random/seed-20260908 --output outputs/crc_yunnan/crc07-20261007/r089-seed20260908

# 2026-10-07 r089/20260908单seed补充比较汇总：只读已取回18单元聚合指标、KM和簇对检验。
# 本地/Users/rong/Project/TRAILS；生成comparison.csv及报告，按验证混合分数选K，test仅探索展示。
.venv/bin/python .remote-run/crc07-20261006/analyze_single_seed.py

# 2026-10-07 补充比较报告完善：核对九K两评价集人数及模型SHA，解释混合分数与两两生存差异。
# 仅本地读取聚合产物，更新同一comparison.csv和report.md；不训练或重读远端患者数据。
.venv/bin/python .remote-run/crc07-20261006/analyze_single_seed.py

# 2026-10-07：本地核对CRC r089/seed20260908的01–05配置与K2/K5入口。
# /Users/rong/Project/TRAILS；对照原推荐配置，检查四线程/GPU设置及新版本路径衔接；不读患者数据、不训练。
MPLCONFIGDIR=/tmp/trails-mpl PYTHONPATH=. .venv/bin/python .remote-run/crc07-20261007-reproduction/verify_config.py

# 2026-10-07：按用户要求仅保留run.yaml；复核r089单seed的K2–10及01–05路径衔接。
# /Users/rong/Project/TRAILS；配置与运行设置检查，不读取患者数据、不训练。
MPLCONFIGDIR=/tmp/trails-mpl PYTHONPATH=. .venv/bin/python .remote-run/crc07-20261007-reproduction/verify_config.py
