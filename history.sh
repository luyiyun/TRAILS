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
