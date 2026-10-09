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

# 2026-10-07：按当前run.yaml在99重跑05；r089、seed20260908、K2–10，train/validation选K后读取test。
# tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS；cuda:0、单进程/4线程，复用OS24/E60冻结v2输入。
# 新输出outputs/crc_yunnan/r089-20261007-current-05-06/modeling；成功后另行记录并提交06，不重用旧预测。
.venv/bin/python /Users/rong/Project/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc05-r089-20261007-current01 --apply -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-r089-20261007-current-mpl python -m scripts.crc_yunnan.05_run split.dir=data/derived/crc_yunnan/preproc/v2/os/landmark-24m/E60/auto-log1p-robust/random/seed-20260908 split.seed=20260908 split.strategy=random n_clusters=null 'k_selection.candidate_clusters=[2,3,4,5,6,7,8,9,10]' 'k_selection.seeds=[20260908]' k_selection.selection_rule=best_mean k_selection.require_non_empty=true k_selection.min_cluster_fraction=0.05 k_selection.compute_stability=false trainer.seed=20260908 trainer.device=cuda:0 parallel.workers=1 parallel.torch_threads=4 paths.dir=outputs/crc_yunnan/r089-20261007-current-05-06/modeling

# 2026-10-07：99重跑05已exit0且选中r089/K2/seed20260908；运行06生成三集临床、轨迹、生存和选择诊断。
# tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS；cuda:0/4线程，输入本次新modeling，不重训练或重选K。
# 输出outputs/crc_yunnan/r089-20261007-current-05-06/evaluation；患者数据、checkpoint、预测均留远端。
.venv/bin/python /Users/rong/Project/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc06-r089-20261007-current01 --apply -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-r089-20261007-current-mpl python -m scripts.crc_yunnan.06_evaluate inputs.run_dir=outputs/crc_yunnan/r089-20261007-current-05-06/modeling 'datasets=[train,validation,test]' seed=20260908 device=cuda:0 paths.dir=outputs/crc_yunnan/r089-20261007-current-05-06/evaluation paths.run_dir=outputs/crc_yunnan/r089-20261007-current-05-06/logs/06_evaluate

# 2026-10-07：本地读取99已成功06的聚合摘要和工作簿，核对表结构及指标；不重新统计患者数据。
# /Users/rong/Project/TRAILS；输入remote-results/default/crc06-r089-20261007-current01，seed20260908/K2。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m json.tool remote-results/default/crc06-r089-20261007-current01/outputs/crc_yunnan/r089-20261007-current-05-06/evaluation/evaluation_summary.json
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c 'import pandas as pd; p="remote-results/default/crc06-r089-20261007-current01/outputs/crc_yunnan/r089-20261007-current-05-06/evaluation/evaluation_tables.xlsx"; t=pd.read_excel(p,sheet_name=None); print({k:(v.shape,list(v.columns)) for k,v in t.items()})'

# 2026-10-07：本地核对06已有聚合表的簇人数、调整HR、临床q值和预设时间点指标；seed20260908/K2。
# /Users/rong/Project/TRAILS；只读取同一evaluation_tables.xlsx，不重拟合、不访问患者数据。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c 'import pandas as pd; p="remote-results/default/crc06-r089-20261007-current01/outputs/crc_yunnan/r089-20261007-current-05-06/evaluation/evaluation_tables.xlsx"; t=pd.read_excel(p,sheet_name=None); print(t["簇概况"].to_string(index=False)); print(t["校正Cox"].to_string(index=False)); print(t["临床总体检验"].to_string(index=False)); print(t["时间预测指标"].loc[lambda x:x["月份"].isin([12,36,60])].to_string()); print(t["簇生存"].to_string(index=False))'

# 2026-10-07：本地读取06已有临床中位数、校准和轨迹聚合表；seed20260908/K2，不新增科学计算。
# /Users/rong/Project/TRAILS；仅取回的evaluation_tables.xlsx，核对测试集描述性差异与观测覆盖。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c 'import pandas as pd; p="remote-results/default/crc06-r089-20261007-current01/outputs/crc_yunnan/r089-20261007-current-05-06/evaluation/evaluation_tables.xlsx"; t=pd.read_excel(p,sheet_name=None); print(t["临床特征"].loc[lambda x:(x["数据集"]=="测试集") & (x["类型"]=="数值")].to_string(index=False)); print(t["生存校准"].loc[lambda x:x["数据集"]=="测试集"].to_string(index=False)); print(t["纵向轨迹"].loc[lambda x:(x["数据集"]=="测试集") & (x["特征"]=="CEA")].to_string(index=False))'

# 2026-10-07：按变量名称读取06测试集数值型临床聚合统计，补充簇间方向解释；seed20260908/K2。
# /Users/rong/Project/TRAILS；只读同一evaluation_tables.xlsx，不重新分析患者数据。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c 'import pandas as pd; p="remote-results/default/crc06-r089-20261007-current01/outputs/crc_yunnan/r089-20261007-current-05-06/evaluation/evaluation_tables.xlsx"; t=pd.read_excel(p,sheet_name="临床特征"); print(t.loc[(t["数据集"]=="测试集") & t["变量"].isin(["术前CEA","术前CA242","年龄"])].to_string(index=False))'

# 2026-10-08：本地合成数据冒烟验证06多K入口和英文绘图；不接触真实CRC数据。
# /Users/rong/Project/TRAILS；合成双特征/三集，seed20260908，CPU；输入与输出均在.remote-run/crc06-multik-20261008。
# 生成临时K2/K5模型（正式训练仅2/3轮）后，06分别评价；用于代码验证，不是研究结果。
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/trails-crc06-multik-smoke PYTHONPATH=. .venv/bin/python .remote-run/crc06-multik-20261008/make_synthetic_fixture.py
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MPLCONFIGDIR=/tmp/trails-crc06-multik-smoke .venv/bin/python -m scripts.crc_yunnan.06_evaluate inputs.run_dir=.remote-run/crc06-multik-20261008/modeling 'n_clusters=[2,5]' 'datasets=[train,validation,test]' seed=20260908 device=cpu plot.dpi=60 paths.dir=.remote-run/crc06-multik-20261008/evaluation paths.run_dir=.remote-run/crc06-multik-20261008/logs

# 2026-10-08：同一合成输入验证06默认入选K2入口，核对与多K运行的测试集统计一致。
# 本地主机 /Users/rong/Project/TRAILS；seed20260908、CPU，输入modeling，输出独立evaluation-default。
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 MPLCONFIGDIR=/tmp/trails-crc06-multik-smoke .venv/bin/python -m scripts.crc_yunnan.06_evaluate inputs.run_dir=.remote-run/crc06-multik-20261008/modeling n_clusters=null 'datasets=[test]' seed=20260908 device=cpu plot.dpi=60 paths.dir=.remote-run/crc06-multik-20261008/evaluation-default paths.run_dir=.remote-run/crc06-multik-20261008/logs-default

# 2026-10-08：按用户要求在99用新06分别评价r089/K2、K5，seed20260908；沿用20261007新05冻结输入及模型。
# tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS；cuda:0/4线程，不训练或重选K，英文绘图及默认统计选项。
# 新输出outputs/crc_yunnan/r089-20261007-current-05-06/evaluation-k2-k5-20261008；仅取回聚合报告及图表。
.venv/bin/python /Users/rong/Project/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc06-r089-20261008-k2k5-01 --apply -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-crc06-k2k5-20261008-mpl python -m scripts.crc_yunnan.06_evaluate inputs.run_dir=outputs/crc_yunnan/r089-20261007-current-05-06/modeling 'n_clusters=[2,5]' 'datasets=[train,validation,test]' seed=20260908 device=cuda:0 paths.dir=outputs/crc_yunnan/r089-20261007-current-05-06/evaluation-k2-k5-20261008 paths.run_dir=outputs/crc_yunnan/r089-20261007-current-05-06/logs/06_evaluate-k2-k5-20261008

# 2026-10-08：06多K重跑exit0后，本地读取K2/K5聚合表，比较占簇、生存预测、调整Cox和临床描述。
# /Users/rong/Project/TRAILS；输入crc06-r089-20261008-k2k5-01取回的workbook，seed20260908；不读取患者数据或重新拟合。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c 'from pathlib import Path; import pandas as pd; p=Path("remote-results/default/crc06-r089-20261008-k2k5-01/outputs/crc_yunnan/r089-20261007-current-05-06/evaluation-k2-k5-20261008/k2/evaluation_tables.xlsx"); t=pd.read_excel(p,sheet_name=None); print(t["簇概况"].to_string(index=False)); print(t["生存预测"].to_string(index=False)); print(t["校正Cox"].to_string(index=False)); print(t["时间预测指标"].loc[lambda x:x["月份"].isin([12,36,60])].to_string(index=False))'
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c 'from pathlib import Path; import pandas as pd; p=Path("remote-results/default/crc06-r089-20261008-k2k5-01/outputs/crc_yunnan/r089-20261007-current-05-06/evaluation-k2-k5-20261008/k5/evaluation_tables.xlsx"); t=pd.read_excel(p,sheet_name=None); print(t["簇概况"].to_string(index=False)); print(t["生存预测"].to_string(index=False)); print(t["校正Cox"].to_string(index=False)); print(t["时间预测指标"].loc[lambda x:x["月份"].isin([12,36,60])].to_string(index=False)); print(t["临床总体检验"].loc[lambda x:x["数据集"]=="测试集"].to_string(index=False)); print(t["生存校准"].loc[lambda x:x["数据集"]=="测试集"].to_string(index=False))'

# 2026-10-08：只读K5聚合轨迹人数/临床数值及K2校准表，核对小簇图形与精确分位统计。
# 本地 /Users/rong/Project/TRAILS；同一seed20260908取回工作簿，不重新拟合、不访问患者数据。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c 'from pathlib import Path; import pandas as pd; p=Path("remote-results/default/crc06-r089-20261008-k2k5-01/outputs/crc_yunnan/r089-20261007-current-05-06/evaluation-k2-k5-20261008/k5/evaluation_tables.xlsx"); t=pd.read_excel(p,sheet_name=None); print({k:(v.shape,list(v.columns)) for k,v in t.items()}); print(t["纵向轨迹"].loc[lambda x:(x["数据集"]=="测试集") & (x["簇"]==4) & x["特征"].isin(["CEA","CA242","CA199"])].to_string(index=False)); print(t["临床特征"].loc[lambda x:(x["数据集"]=="测试集") & x["变量"].isin(["术前CEA","术前CA242","年龄"])].to_string(index=False))'
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c 'import pandas as pd; t=pd.read_excel("remote-results/default/crc06-r089-20261008-k2k5-01/outputs/crc_yunnan/r089-20261007-current-05-06/evaluation-k2-k5-20261008/k2/evaluation_tables.xlsx",sheet_name="生存校准"); print(t.loc[t["数据集"]=="测试集"].to_string(index=False))'

# 2026-10-08：G14实现的合成输入检查，验证范围端点、逐条剔除计数、同日聚合和有效访视筛选。
# 本地 /Users/rong/Project/TRAILS；seed20260908，输入由保留的临时脚本生成，输出.remote-run/g14-20261008/synthetic。
# 不读取真实患者数据，不训练模型；范围使用本轮确认的G14探索界限。
PYTHONPATH=. .venv/bin/python .remote-run/g14-20261008/validate_synthetic.py

# 2026-10-08：按确认的宽界限清洗G14指标，依次运行01→03→04；不改变既有v2患者主划分。
# tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS；OS/720天、G14、auto-log1p/robust、seed20260908、4线程。
# 输出独立v2-g14-clean-20261008数据目录；仅复制聚合摘要、配置和预处理参数到outputs供取回。
.venv/bin/python /Users/rong/.codex/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc-g14-prep-20261008-01 --apply -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-g14-20261008-mpl bash -c 'set -e
python -m scripts.crc_yunnan.01_cohort version=v2-g14-clean-20261008 '\''processing.observation_value_bounds={CEA:{min:0},CA242:{min:0},CA125:{min:0},血清白蛋白:{min:0,max:100},总蛋白:{min:0,max:200},白球比:{min:0,max:10},前白蛋白:{min:0,max:1000},胆碱酯酶:{min:0,max:50000},红细胞:{min:0.1,max:20},红细胞平均体积:{min:20,max:300},平均血小板体积:{min:1,max:50},中性粒细胞绝对值:{min:0,max:200},乳酸脱氢酶:{min:0,max:20000},尿素:{min:0,max:100}}'\'' processing.research_id_seed=20260908 paths.run_dir=outputs/crc_yunnan/g14-20261008/logs/01_cohort
python -m scripts.crc_yunnan.03_split version=v2-g14-clean-20261008 split_version=v2 '\''split.strategies=[random]'\'' '\''split.seeds=[20260908]'\'' paths.dir=data/derived/crc_yunnan/splits/v2-g14-clean-20261008 paths.run_dir=outputs/crc_yunnan/g14-20261008/logs/03_split
python -m scripts.crc_yunnan.04_preproc version=v2-g14-clean-20261008 features.panel=G14 outcome=os landmark_months=24 min_observation_dates=2 preprocessing.log_transform=auto-log1p preprocessing.scaling=robust preprocessing.skew_threshold=1.0 paths.run_dir=outputs/crc_yunnan/g14-20261008/logs/04_preproc
mkdir -p outputs/crc_yunnan/g14-20261008/preparation
cp data/derived/crc_yunnan/cohort/v2-g14-clean-20261008/cohort_summary.json outputs/crc_yunnan/g14-20261008/preparation/cohort_summary.json
cp data/derived/crc_yunnan/cohort/v2-g14-clean-20261008/resolved_config.yaml outputs/crc_yunnan/g14-20261008/preparation/cohort_config.yaml
cp data/derived/crc_yunnan/preproc/v2-g14-clean-20261008/os/landmark-24m/G14/auto-log1p-robust/random/seed-20260908/preproc_manifest.json outputs/crc_yunnan/g14-20261008/preparation/preproc_manifest.json
cp data/derived/crc_yunnan/preproc/v2-g14-clean-20261008/os/landmark-24m/G14/auto-log1p-robust/random/seed-20260908/preprocessing_parameters.csv outputs/crc_yunnan/g14-20261008/preparation/preprocessing_parameters.csv
cp data/derived/crc_yunnan/preproc/v2-g14-clean-20261008/os/landmark-24m/G14/auto-log1p-robust/random/seed-20260908/resolved_config.yaml outputs/crc_yunnan/g14-20261008/preparation/preproc_config.yaml'

# 2026-10-08：上方G14准备命令未提交；配置预览发现Hydra不支持未引号保护的中文字典键。
# 以下使用OmegaConf内置oc.create解析JSON，已核验全部14项阈值；输入/输出、seed及科学规则不变。
.venv/bin/python /Users/rong/.codex/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc-g14-prep-20261008-01 --apply -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-g14-20261008-mpl bash -c 'set -e
python -m scripts.crc_yunnan.01_cohort version=v2-g14-clean-20261008 '\''processing.observation_value_bounds="${oc.create:'\''\'\'''\''{\"CEA\":{\"min\":0},\"CA242\":{\"min\":0},\"CA125\":{\"min\":0},\"血清白蛋白\":{\"min\":0,\"max\":100},\"总蛋白\":{\"min\":0,\"max\":200},\"白球比\":{\"min\":0,\"max\":10},\"前白蛋白\":{\"min\":0,\"max\":1000},\"胆碱酯酶\":{\"min\":0,\"max\":50000},\"红细胞\":{\"min\":0.1,\"max\":20},\"红细胞平均体积\":{\"min\":20,\"max\":300},\"平均血小板体积\":{\"min\":1,\"max\":50},\"中性粒细胞绝对值\":{\"min\":0,\"max\":200},\"乳酸脱氢酶\":{\"min\":0,\"max\":20000},\"尿素\":{\"min\":0,\"max\":100}}'\''\'\'''\''}"'\'' processing.research_id_seed=20260908 paths.run_dir=outputs/crc_yunnan/g14-20261008/logs/01_cohort
python -m scripts.crc_yunnan.03_split version=v2-g14-clean-20261008 split_version=v2 '\''split.strategies=[random]'\'' '\''split.seeds=[20260908]'\'' paths.dir=data/derived/crc_yunnan/splits/v2-g14-clean-20261008 paths.run_dir=outputs/crc_yunnan/g14-20261008/logs/03_split
python -m scripts.crc_yunnan.04_preproc version=v2-g14-clean-20261008 features.panel=G14 outcome=os landmark_months=24 min_observation_dates=2 preprocessing.log_transform=auto-log1p preprocessing.scaling=robust preprocessing.skew_threshold=1.0 paths.run_dir=outputs/crc_yunnan/g14-20261008/logs/04_preproc
mkdir -p outputs/crc_yunnan/g14-20261008/preparation
cp data/derived/crc_yunnan/cohort/v2-g14-clean-20261008/cohort_summary.json outputs/crc_yunnan/g14-20261008/preparation/cohort_summary.json
cp data/derived/crc_yunnan/cohort/v2-g14-clean-20261008/resolved_config.yaml outputs/crc_yunnan/g14-20261008/preparation/cohort_config.yaml
cp data/derived/crc_yunnan/preproc/v2-g14-clean-20261008/os/landmark-24m/G14/auto-log1p-robust/random/seed-20260908/preproc_manifest.json outputs/crc_yunnan/g14-20261008/preparation/preproc_manifest.json
cp data/derived/crc_yunnan/preproc/v2-g14-clean-20261008/os/landmark-24m/G14/auto-log1p-robust/random/seed-20260908/preprocessing_parameters.csv outputs/crc_yunnan/g14-20261008/preparation/preprocessing_parameters.csv
cp data/derived/crc_yunnan/preproc/v2-g14-clean-20261008/os/landmark-24m/G14/auto-log1p-robust/random/seed-20260908/resolved_config.yaml outputs/crc_yunnan/g14-20261008/preparation/preproc_config.yaml'

# 2026-10-08：01→03→04 exit0后，只读核对G14基础患者表和三集主ID归属，与原v2逐字节比较。
# tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS；读取现有聚合摘要，核对14项/OS/720天、剔除数及GPU。
'ssh' '-o' 'BatchMode=yes' '-o' 'ConnectTimeout=15' '-o' 'ClearAllForwardings=yes' 'tmu-99-rzw-xtcp' 'cd /home/rongzw/projects/TRAILS
cmp data/derived/crc_yunnan/cohort/v2/patients.csv data/derived/crc_yunnan/cohort/v2-g14-clean-20261008/patients.csv
for name in train validation test; do cmp data/derived/crc_yunnan/splits/v2/random/seed-20260908/$name/ids.csv data/derived/crc_yunnan/splits/v2-g14-clean-20261008/random/seed-20260908/$name/ids.csv || exit; done
sha256sum data/derived/crc_yunnan/cohort/v2-g14-clean-20261008/patients.csv data/real/crc_yunnan/*csv
cat outputs/crc_yunnan/g14-20261008/preparation/cohort_summary.json outputs/crc_yunnan/g14-20261008/preparation/preproc_manifest.json
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader'

# 2026-10-08：G14单seed K=2–10探索；01→03→04原子exit0，107条原始观测剔除，基础4100人与主归属不变。
# tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS；OS/720天、G14、auto-log1p/robust、r089、seed20260908、cuda:0、4线程。
# 输入1853/444/586人；仅验证集选择K，保留无空簇/最小5%门槛；仅取回聚合结果，患者级产物留在99。
.venv/bin/python /Users/rong/.codex/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc05-g14-20261008-01 --apply -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-g14-20261008-mpl python -m scripts.crc_yunnan.05_run split.dir=data/derived/crc_yunnan/preproc/v2-g14-clean-20261008/os/landmark-24m/G14/auto-log1p-robust/random/seed-20260908 split.seed=20260908 split.strategy=random paths.dir=outputs/crc_yunnan/g14-20261008/modeling n_clusters=null k_selection.enabled=true 'k_selection.candidate_clusters=[2,3,4,5,6,7,8,9,10]' 'k_selection.seeds=[20260908]' k_selection.selection_rule=best_mean k_selection.require_non_empty=true k_selection.min_cluster_fraction=0.05 k_selection.compute_stability=false k_selection.min_mean_pairwise_ari=null trainer.seed=20260908 trainer.device=cuda:0 trainer.valid_size=0 trainer.batch_size=64 trainer.learning_rate=0.001 trainer.warmup_epochs=30 trainer.warmup_objective=reconstruction trainer.min_epochs=100 trainer.max_epochs=300 trainer.early_stopping_monitor=loss trainer.early_stopping_patience=30 model.latent_dim=32 model.dropout=0 model.survival_loss=weibull model.survival_head_hidden_layers=1 model.loss.weighting=fixed model.loss.reconstruction_weight=1 model.loss.survival_weight=0.05 model.loss.cluster_weight=0.1 parallel.workers=1 parallel.torch_threads=4 runtime.torch_interop_threads=4 swanlab.enabled=false

# 2026-10-08：论文Results第三部分，固定原E60 r089/K5/seed20260908；只读生成画像、稀疏性与完整KM聚合。
# 99:/home/rongzw/projects/TRAILS；CPU/4线程，G14训练另在运行，不同步或覆盖其源码；脚本与模型SHA单独记录。
# 已授权的一次性聚合脚本保留在outputs/crc_yunnan/manuscript_results3/reproducibility；患者级数据与模型留99。
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-99-rzw-xtcp 'test ! -e /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-fig3-20261008-01 && mkdir -p /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-fig3-20261008-01 && test ! -e /home/rongzw/projects/TRAILS/outputs/crc_yunnan/manuscript_results3_aggregate-20261008'
scp -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes outputs/crc_yunnan/manuscript_results3/reproducibility/aggregate_k5.py outputs/crc_yunnan/manuscript_results3/reproducibility/launch_aggregate.sh tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-fig3-20261008-01/
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-99-rzw-xtcp 'cd /home/rongzw/projects/TRAILS && test ! -e /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-fig3-20261008-01/pid && { nohup sh /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-fig3-20261008-01/launch_aggregate.sh >/home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-fig3-20261008-01/stdout.log 2>/home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-fig3-20261008-01/stderr.log </dev/null & paper_pid=$!; printf "%s\n" "$paper_pid" >/home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-fig3-20261008-01/pid; printf "%s\n" "$paper_pid"; }'

# 2026-10-08：本地制作K5六面板论文图及Source Data中间表；只用已SHA核验的三集聚合数据。
# /Users/rong/Project/TRAILS；seed20260908，训练集定义画像、测试集KM/Cox；输出manuscript_results3。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 outputs/crc_yunnan/manuscript_results3/reproducibility/make_figure.py

# 2026-10-08：bundled Python未提供Matplotlib；改用项目已有绘图库绘图，输入与科学口径相同。
# 本地 /Users/rong/Project/TRAILS；不安装依赖，Source Data工作簿仍使用bundled Node/artifact-tool。
.venv/bin/python outputs/crc_yunnan/manuscript_results3/reproducibility/make_figure.py

# 2026-10-09：按180×195 mm最终尺寸校对六面板，矢量热图、英文单位、KM风险表及Cox文字位置。
# 本地 /Users/rong/Project/TRAILS；仍用原E60/K5/seed20260908聚合输入，统计结果及显示门槛不变。
env MPLCONFIGDIR=/tmp/trails-paper-fig3-mpl .venv/bin/python outputs/crc_yunnan/manuscript_results3/reproducibility/make_figure.py

# 2026-10-09：将固定K5三集聚合值和K2/K5比较表写入Source Data；无新增模型、检验或患者级数据。
# 本地 /Users/rong/Project/TRAILS；bundled Node/artifact-tool，输入source_data_payload.json，输出SourceData.xlsx。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node outputs/crc_yunnan/manuscript_results3/reproducibility/make_source_data.mjs

# 2026-10-09：只读三集阶段画像，逐项核对C0肝胆指标、C1/C2红细胞指标和C4营养/中性粒细胞方向。
# 本地 /Users/rong/Project/TRAILS；原K5/seed20260908聚合中位数，不拟合、不检验，供论文文字核对。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c 'import json; import pandas as pd; from pathlib import Path; p=json.loads(Path("outputs/crc_yunnan/manuscript_results3/source_data/source_data_payload.json").read_text()); t=pd.DataFrame(p["Atlas"]["records"]); print(t.loc[t.feature.isin(["血红蛋白","红细胞平均体积","平均血红蛋白量","血清白蛋白","前白蛋白","中性粒细胞绝对值"]) & t.phase.eq("0–3m"),["dataset","cluster","feature","n","original_median"]].to_string(index=False)); print(t.loc[t.feature.isin(["碱性磷酸酶","r谷氨酰转移酶"]) & t.phase.eq("3–12m") & t.cluster.isin([0,3]),["dataset","cluster","feature","n","original_median"]].to_string(index=False))'

# 2026-10-09：源表phase使用0/1/2编码，上方文字筛选为空；改为对应的0–3与3–12月整数编码读取。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -c 'import json; import pandas as pd; from pathlib import Path; p=json.loads(Path("outputs/crc_yunnan/manuscript_results3/source_data/source_data_payload.json").read_text()); t=pd.DataFrame(p["Atlas"]["records"]); print(t.loc[t.feature.isin(["血红蛋白","红细胞平均体积","平均血红蛋白量","血清白蛋白","前白蛋白","中性粒细胞绝对值"]) & t.phase.eq(0),["dataset","cluster","feature","n","original_median"]].to_string(index=False)); print(t.loc[t.feature.isin(["碱性磷酸酶","r谷氨酰转移酶"]) & t.phase.eq(1) & t.cluster.isin([0,3]),["dataset","cluster","feature","n","original_median"]].to_string(index=False))'

# 2026-10-09：补充主图观测人数索引和Source Data时间编码说明；重导出同一套最终图与源表。
# 本地 /Users/rong/Project/TRAILS；原E60/K5/seed20260908统计值不变，仅显示与元数据完善。
env MPLCONFIGDIR=/tmp/trails-paper-fig3-mpl .venv/bin/python outputs/crc_yunnan/manuscript_results3/reproducibility/make_figure.py
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node outputs/crc_yunnan/manuscript_results3/reproducibility/make_source_data.mjs

# 2026-10-09：最终输出校对，消除矢量色标接缝、Source Data长列名换行和极小P值显示为0的问题。
# 本地 /Users/rong/Project/TRAILS；只改图形/工作簿显示，固定K5/seed20260908聚合数值不变。
env MPLCONFIGDIR=/tmp/trails-paper-fig3-mpl .venv/bin/python outputs/crc_yunnan/manuscript_results3/reproducibility/make_figure.py
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node outputs/crc_yunnan/manuscript_results3/reproducibility/make_source_data.mjs

# 2026-10-09：原有CEA上轴限250未容纳最大支持IQR 373.325；仅将对数轴上限调至500以完整显示。
# 本地 /Users/rong/Project/TRAILS；固定K5/seed20260908源数据、观测支持门槛及统计值不变。
env MPLCONFIGDIR=/tmp/trails-paper-fig3-mpl .venv/bin/python outputs/crc_yunnan/manuscript_results3/reproducibility/make_figure.py

# 2026-10-09：按用户修订将d改为验证/测试latent z UMAP；固定原E60 r089/K5/seed20260908，训练集拟合。
# 99:/home/rongzw/projects/TRAILS；CPU/4线程，沿用06 UMAP参数；不训练、不新检验、不同步在训G14源码。
# 只取回渲染图、聚合占簇数和参数元数据，患者坐标/模型/预测留99；脚本SHA单独核对。
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-99-rzw-xtcp 'test ! -e /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-01 && mkdir -p /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-01 && test ! -e /home/rongzw/projects/TRAILS/outputs/crc_yunnan/manuscript_results3_umap-20261009'
scp -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes outputs/crc_yunnan/manuscript_results3/reproducibility/render_umap.py outputs/crc_yunnan/manuscript_results3/reproducibility/launch_umap.sh tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-01/
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-99-rzw-xtcp 'cd /home/rongzw/projects/TRAILS && test ! -e .remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-01/pid && { nohup sh .remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-01/launch_umap.sh >.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-01/stdout.log 2>.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-01/stderr.log </dev/null & paper_pid=$!; printf "%s\n" "$paper_pid" >.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-01/pid; printf "%s\n" "$paper_pid"; }'

# 2026-10-09：UMAP初版exit0、人数与SHA一致；去掉重复横轴标题以消除与下图标题重叠，独立输出v2。
# 99:/home/rongzw/projects/TRAILS；原E60/K5/seed20260908与UMAP参数不变；原脚本保留initial副本。
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-99-rzw-xtcp 'test ! -e /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-02 && mkdir -p /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-02 && test ! -e /home/rongzw/projects/TRAILS/outputs/crc_yunnan/manuscript_results3_umap-20261009-v2'
scp -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes outputs/crc_yunnan/manuscript_results3/reproducibility/render_umap.py outputs/crc_yunnan/manuscript_results3/reproducibility/launch_umap.sh tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-02/
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-99-rzw-xtcp 'cd /home/rongzw/projects/TRAILS && test ! -e .remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-02/pid && { nohup sh .remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-02/launch_umap.sh >.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-02/stdout.log 2>.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-02/stderr.log </dev/null & paper_pid=$!; printf "%s\n" "$paper_pid" >.remote-run/diagnostics/crc-paper-r089-k5-umap-20261009-02/pid; printf "%s\n" "$paper_pid"; }'

# 2026-10-09：修订主图180×215 mm，双集UMAP及44×36 mm双KM；仅使用已核验的聚合表与渲染图。
# 本地 /Users/rong/Project/TRAILS；原E60/r089/K5/seed20260908，Cox保持测试集；原稿留versions/v1_20261009。
env MPLCONFIGDIR=/tmp/trails-paper-fig3-mpl .venv/bin/python outputs/crc_yunnan/manuscript_results3/reproducibility/make_figure.py
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 outputs/crc_yunnan/manuscript_results3/reproducibility/compose_figure.py
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node outputs/crc_yunnan/manuscript_results3/reproducibility/make_source_data.mjs

# 2026-10-09：Source Data极小P值导出保留浮点有效数字，不按固定小数位截断；图形与统计口径不变。
# 本地 /Users/rong/Project/TRAILS；最终重导出固定K5/seed20260908主图、14表源数据与两集UMAP图件合成。
env MPLCONFIGDIR=/tmp/trails-paper-fig3-mpl .venv/bin/python outputs/crc_yunnan/manuscript_results3/reproducibility/make_figure.py
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 outputs/crc_yunnan/manuscript_results3/reproducibility/compose_figure.py
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node outputs/crc_yunnan/manuscript_results3/reproducibility/make_source_data.mjs

# 2026-10-09：G14 05原子exit0，9/9候选完整，原门槛选中K2/seed20260908后执行06。
# tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS；OS/720天、G14、r089、cuda:0/4线程；输入modeling，输出独立evaluation。
# 仅评价入选K的三集、landmark后12/36/60月及1–60月积分；模型/患者数据留99，只取回聚合产物。
.venv/bin/python /Users/rong/.codex/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc06-g14-20261008-01 --apply -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-g14-20261008-mpl python -m scripts.crc_yunnan.06_evaluate inputs.run_dir=outputs/crc_yunnan/g14-20261008/modeling n_clusters=null 'datasets=[train,validation,test]' seed=20260908 device=cuda:0 'survival.months=[12,36,60]' survival.curve_step_months=1 survival.calibration_bins=5 paths.dir=outputs/crc_yunnan/g14-20261008/evaluation paths.run_dir=outputs/crc_yunnan/g14-20261008/logs/06_evaluate

# 2026-10-09：只读已授权取回的G14与历史E60 K2聚合工作簿，核对字段和三集评价口径。
# 本地 /Users/rong/Project/TRAILS；两组均seed20260908/OS/720天，不新增训练、检验或患者级数据。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 .remote-run/g14-20261008/inspect_aggregates.py

# 2026-10-09：只核对G14精确白名单95份聚合文件；远端SHA/尺寸由只读ssh取得，患者级数据仍留99。
# 本地 /Users/rong/Project/TRAILS；比较历史E60 K2与G14 K2/seed20260908，输出比较CSV和检查摘要。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 .remote-run/g14-20261008/analyze_aggregates.py --inventory-only

# 2026-10-09：99:/home/rongzw/projects/TRAILS；只读核对准备5份/05聚合35份/06聚合55份的SHA及尺寸。
# 精确95项路径随本任务脚本保留；校验后只在本地比较OS/720天、K2、seed20260908的两组聚合结果。
bash .remote-run/g14-20261008/verify_remote_artifacts.sh
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 .remote-run/g14-20261008/analyze_aggregates.py

# 2026-10-09：补充G14/E60共同14项的原量纲轨迹与覆盖聚合对照，簇编号不作患者匹配。
# 本地 /Users/rong/Project/TRAILS；不新增统计检验，保留各组原有90天分箱及全部时间箱数值。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 .remote-run/g14-20261008/analyze_aggregates.py

# 2026-10-09：用户追加要求查看G14 K4/K6生存差异；只评价05保存的同seed候选，原K2选择不变。
# 99:/home/rongzw/projects/TRAILS；OS/720天、seed20260908、cuda:0/4线程、三集KM/Cox与既有评价；不训练。
# K4/K6原占簇门槛未通过，仅作补充探索；独立输出，不覆盖K2，患者级产物留99。
.venv/bin/python /Users/rong/.codex/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc06-g14-k4k6-20261009-01 --apply -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-g14-k4k6-20261009-mpl python -m scripts.crc_yunnan.06_evaluate inputs.run_dir=outputs/crc_yunnan/g14-20261008/modeling 'n_clusters=[4,6]' 'datasets=[train,validation,test]' seed=20260908 device=cuda:0 'survival.months=[12,36,60]' survival.curve_step_months=1 survival.calibration_bins=5 umap.seed=20260908 paths.dir=outputs/crc_yunnan/g14-20261008/evaluation-k4-k6-20261009 paths.run_dir=outputs/crc_yunnan/g14-20261008/logs/06_evaluate-k4-k6-20261009

# 2026-10-09：上方K4在test调整Cox不收敛，原子exit1；K6尚未执行，现单独完成用户已授权的K6评价。
# 99:/home/rongzw/projects/TRAILS；输入/seed20260908/统计口径不变；独立K6目录，不覆盖K4失败产物。
.venv/bin/python /Users/rong/.codex/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc06-g14-k6-20261009-01 --apply -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-g14-k6-20261009-mpl python -m scripts.crc_yunnan.06_evaluate inputs.run_dir=outputs/crc_yunnan/g14-20261008/modeling n_clusters=6 'datasets=[train,validation,test]' seed=20260908 device=cuda:0 'survival.months=[12,36,60]' survival.curve_step_months=1 survival.calibration_bins=5 umap.seed=20260908 paths.dir=outputs/crc_yunnan/g14-20261008/evaluation-k4-k6-20261009 paths.run_dir=outputs/crc_yunnan/g14-20261008/logs/06_evaluate-k6-20261009

# 2026-10-09：上方独立K6命令未提交；用户确认一次性汇总K4/K6的KM/log-rank及Cox失败标记。
# 99:/home/rongzw/projects/TRAILS；临时脚本复用原函数、固定模型/seed20260908/四协变量；无惩罚或合并。
# 脚本/模型SHA记录，患者级产物留99，原06失败目录保留，独立聚合目录g14-k4k6-survival-20261009。
ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes tmu-99-rzw-xtcp 'test ! -e /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-g14-k4k6-survival-20261009-01 && mkdir -p /home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-g14-k4k6-survival-20261009-01'
scp -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes .remote-run/g14-20261008/aggregate_k4k6_survival.py tmu-99-rzw-xtcp:/home/rongzw/projects/TRAILS/.remote-run/diagnostics/crc-g14-k4k6-survival-20261009-01/
.venv/bin/python /Users/rong/.codex/skills/remote-run/scripts/remote_run.py --target default submit --run-id crc-g14-k4k6-survival-20261009-01 --apply -- env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 MPLCONFIGDIR=/tmp/trails-g14-k4k6-survival-20261009-mpl python .remote-run/diagnostics/crc-g14-k4k6-survival-20261009-01/aggregate_k4k6_survival.py

# 2026-10-09：完整阶段日志纠正短日志判断：原06实际已完成K4，随后K6 test调整Cox失败，整体exit1。
# K6单独06从未提交；用户确认的一次性两K汇总exit0，K4 Cox成功，K6 test失败与train警告明确保留。
# 本地只读16份生存聚合文件，核对SHA/尺寸/图件及原K4工作簿，不新增统计或患者级输出。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 .remote-run/g14-20261008/analyze_k4k6_survival.py




# 2026-10-09：补充核对原06已完整产出的K4工作簿和摘要，尺寸/SHA及三张生存聚合表一致性。
# 本地 /Users/rong/Project/TRAILS；原模型和统计函数不变，Cox失败归属更正为K6 test，未新增检验。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 .remote-run/g14-20261008/analyze_k4k6_survival.py

# 2026-10-09：CSV的三列Torch诊断以float32短小数写出，Excel保留其double表示；恢复原float32后比较。
# 人数、事件、KM和Cox仍按原精度核对；只恢复导出类型，不改统计或重算远端结果。
/Users/rong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 .remote-run/g14-20261008/analyze_k4k6_survival.py

# 2026-10-09：本地合成输入检查05候选汇总及06/07独立流程，不使用患者数据。
# 主机：本机；工作目录：/Users/rong/Project/TRAILS；合成输入/输出：/tmp/trails-crc-postprocess-smoke；seed=20260908/20260909。
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/Users/rong/Project/TRAILS uv run python /tmp/trails_crc_postprocess_smoke.py

# 2026-10-09：复用上述合成模型继续检查06/07；修正临时检查中/tmp与/private/tmp的路径比较，不再训练。
# 主机：本机；工作目录：/Users/rong/Project/TRAILS；输入/输出：/tmp/trails-crc-postprocess-smoke；seed=20260908。
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=/Users/rong/Project/TRAILS uv run python /tmp/trails_crc_postprocess_check.py

# 2026-10-09：复用已保存合成模型核对完整占簇指标、历史兼容、低支持显示及最终矢量版面，不再训练。
# 主机：本机；工作目录：/Users/rong/Project/TRAILS；输入/输出：/tmp/trails-crc-postprocess-smoke；K2/K5/K10，seed20260908。
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/trails-crc-mpl PYTHONPATH=/Users/rong/Project/TRAILS uv run python /tmp/trails_crc_final_check.py

# 2026-10-09：继续最终本地检查，复用已完成的modeling-final；修正检查夹具的cluster_label=None，不再训练或覆盖05产物。
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/trails-crc-mpl PYTHONPATH=/Users/rong/Project/TRAILS uv run python /tmp/trails_crc_final_check.py

# 2026-10-09：最终色条矢量化核对、历史07及单seed分支；复用合成模型，不训练、不访问患者数据或99。
# 主机：本机；工作目录：/Users/rong/Project/TRAILS；输入/输出：/tmp/trails-crc-postprocess-smoke；K2/K5/K10，seed20260908。
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/trails-crc-mpl PYTHONPATH=/Users/rong/Project/TRAILS uv run python /tmp/trails_crc_export_check.py

# 2026-10-09：最终正文版面及14张Source Data表验收，复用合成模型；不训练、不访问患者数据或99。
# 主机：本机；工作目录：/Users/rong/Project/TRAILS；输入/输出：/tmp/trails-crc-postprocess-smoke；K2/K5/K10，seed20260908，600 dpi。
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/trails-crc-mpl PYTHONPATH=/Users/rong/Project/TRAILS uv run python /tmp/trails_crc_acceptance.py

# 2026-10-09：仅用既有r089 Source Data复绘并核对人数/参考簇/画像/轨迹/KM/Cox，不拟合、不读患者表或UMAP坐标；d为明确标注的检查占位。
# 主机：本机；工作目录：/Users/rong/Project/TRAILS；输入：outputs/crc_yunnan/manuscript_results3/SourceData.xlsx；输出：/tmp/trails-crc-real-aggregate-replay；K5，seed20260908。
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/trails-crc-mpl PYTHONPATH=/Users/rong/Project/TRAILS uv run python /tmp/trails_crc_aggregate_replay.py
