#!/usr/bin/env bash
# 在项目根目录运行：bash scripts/crc_yunnan/04_explore.sh <全新输出目录>
# 仅通过CLI重复调用原版04_run.py；test产物会生成，但不进入调参汇总。
set -euo pipefail

output_root="${1:?请指定全新输出目录}"
shift
smoke_overrides=(trainer.device=cuda:0 "$@")
if [[ -e "$output_root" ]]; then
    printf '拒绝覆盖探索输出：%s\n' "$output_root" >&2
    exit 1
fi
mkdir -p "$output_root"
output_root="$(cd "$output_root" && pwd)"
p0="${CRC_P0_SPLIT:-data/derived/crc_yunnan/splits/v1/dfs/landmark-12m/D20/random/seed-20260908}"
p1="${CRC_P1_SPLIT:-data/derived/crc_yunnan/splits/v1/dfs/landmark-12m/D20/auto-log1p-robust/random/seed-20260908}"
reference_args=(n_clusters=3)
groups=()
completed=0
printf 'completed,stage,group,seed\n' > "$output_root/progress.csv"

fit_group() {
    local stage="$1" label="$2" seed started finished
    shift 2
    local group="$output_root/$stage/$label"
    groups+=("$group")
    for seed in 20260908 20260909; do
        printf 'completed=%s\nbudget_max=38\nstage=%s\ngroup=%s\nseed=%s\n' \
            "$completed" "$stage" "$label" "$seed" > "$output_root/current_run.txt"
        printf '探索进度 %s/38：%s %s seed=%s\n' "$completed" "$stage" "$label" "$seed"
        started=$(date +%s)
        uv run python -m scripts.crc_yunnan.04_run model=base trainer=full \
            trainer.valid_size=0.0 trainer.cindex_risk_score=median_survival \
            swanlab.enabled=false "${reference_args[@]}" "$@" "${smoke_overrides[@]}" \
            n_clusters=3 "trainer.seed=$seed" "paths.dir=$group/seed-$seed"
        finished=$(date +%s)
        printf '%s\n' "$((finished - started))" > "$group/seed-$seed/wall_seconds.txt"
        completed=$((completed + 1))
        printf '%s,%s,%s,%s\n' "$completed" "$stage" "$label" "$seed" >> "$output_root/progress.csv"
        if [[ "$completed" -eq 4 ]]; then
            printf '首批4次完成，可由wall_seconds.txt更新剩余耗时估计。\n'
        fi
    done
}

summarize() {
    local summary="$1" reference="$2"
    shift 2
    # 此处仅汇总04的既有输出，不训练模型、不读取test指标或test预测。
    uv run python - "$output_root/$summary" "$reference" "$@" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import adjusted_rand_score

output = Path(sys.argv[1])
output.mkdir(parents=True, exist_ok=False)
reference_dir = sys.argv[2]
seeds = (20260908, 20260909)
factors = (
    "model.loss.reconstruction_weight",
    "model.latent_dim",
    "trainer.gmm_init_iters",
    "trainer.warmup_epochs",
)
argument_keys = ("split.dir", "trainer.batch_size", "trainer.learning_rate", *factors)
records: list[dict[str, Any]] = []
groups: list[dict[str, Any]] = []
configs: dict[str, DictConfig] = {}
common_ids: list[str] | None = None

# 一、读取每组两个seed的验证指标、训练历史和计时。
for directory in sys.argv[3:]:
    group = Path(directory)
    rows: list[dict[str, Any]] = []
    labels: list[list[int]] = []
    for seed in seeds:
        run_dir = group / f"seed-{seed}"
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        assert manifest["selected_k"] == 3 and manifest["selected_seed"] == seed
        cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")
        assert isinstance(cfg, DictConfig)
        if directory in configs:
            previous = configs[directory]
            left = OmegaConf.to_container(previous.trainer, resolve=True)
            right = OmegaConf.to_container(cfg.trainer, resolve=True)
            assert isinstance(left, dict) and isinstance(right, dict)
            left.pop("seed")
            right.pop("seed")
            assert left == right and previous.model == cfg.model and previous.split == cfg.split
        else:
            configs[directory] = cfg
        # 原04按train/validation/test写表；只解析前两行，test行不读取。
        metrics = pd.read_csv(run_dir / "metrics.csv", nrows=2).set_index("split")
        assert metrics.index.tolist() == ["train", "validation"]
        valid = metrics.loc["validation"]
        prediction = pd.read_csv(
            run_dir / "validation/patient_outputs.csv", dtype={"patient_id": "string"}
        )
        ids = prediction["patient_id"].tolist()
        if common_ids is None:
            common_ids = ids
        else:
            assert ids == common_ids, "验证患者及顺序必须一致"
        labels.append(prediction["pred_cluster"].tolist())
        history = pd.read_csv(run_dir / "training_history.csv")
        saved_epoch = int(history.iloc[-1].get("best_global_epoch", len(history)))
        saved = history.loc[history.global_epoch.eq(saved_epoch)].iloc[0]
        row = {
            "stage": group.parent.name,
            "group": group.name,
            "group_dir": directory,
            "seed": seed,
            "split_dir": manifest["split_dir"],
            "batch_size": cfg.trainer.batch_size,
            "learning_rate": cfg.trainer.learning_rate,
            "reconstruction_weight": cfg.model.loss.reconstruction_weight,
            "latent_dim": cfg.model.latent_dim,
            "gmm_init_iters": cfg.trainer.gmm_init_iters,
            "warmup_budget": cfg.trainer.warmup_epochs,
            "cindex": float(valid["cindex"]),
            "empty_clusters": int(valid["cluster_empty_count"]),
            "min_fraction": float(valid["cluster_min_fraction"]),
            "max_fraction": float(valid["cluster_max_fraction"]),
            "warmup_epochs": int(history.stage.eq("warmup").sum()),
            "vade_epochs": int(history.stage.eq("vade").sum()),
            "saved_global_epoch": saved_epoch,
            "optimizer_updates": len(history)
            * math.ceil(int(metrics.loc["train", "n_patients"]) / int(cfg.trainer.batch_size)),
            "wall_seconds": int((run_dir / "wall_seconds.txt").read_text()),
            **{
                f"effective_{key}": float(saved[f"val_{key}"])
                for key in (
                    "reconstruction_loss_weight",
                    "survival_loss_weight",
                    "vade_kl_loss_weight",
                )
            },
        }
        assert all(math.isfinite(row[key]) for key in ("cindex", "min_fraction", "max_fraction"))
        row["passes_gate"] = row["empty_clusters"] == 0 and row["min_fraction"] >= 0.05
        rows.append(row)
    records.extend(rows)
    groups.append(
        {
            "stage": group.parent.name,
            "group": group.name,
            "group_dir": directory,
            "passed_seeds": sum(row["passes_gate"] for row in rows),
            "worst_empty_clusters": max(row["empty_clusters"] for row in rows),
            "worst_min_fraction": min(row["min_fraction"] for row in rows),
            "mean_cindex": sum(row["cindex"] for row in rows) / 2,
            "cindex_gap": abs(rows[0]["cindex"] - rows[1]["cindex"]),
            "validation_ari": float(adjusted_rand_score(*labels)),
            "wall_seconds": sum(row["wall_seconds"] for row in rows),
            "optimizer_updates": sum(row["optimizer_updates"] for row in rows),
        }
    )


# 二、只用验证集占用门槛和C-index排名；ARI和seed差值作为描述性结果。
def rank(row: dict[str, Any]) -> tuple[float, ...]:
    eligible = row["passed_seeds"] == 2
    return (
        -row["passed_seeds"],
        0 if eligible else row["worst_empty_clusters"],
        0 if eligible else -row["worst_min_fraction"],
        -row["mean_cindex"],
    )


def save_arguments(path: Path, cfg: DictConfig, changes: dict[str, Any]) -> None:
    arguments = [f"{key}={changes.get(key, OmegaConf.select(cfg, key))}" for key in argument_keys]
    path.write_text("\n".join(arguments) + "\n")


groups.sort(key=rank)
best = groups[0]
save_arguments(output / "reference.args", configs[best["group_dir"]], {})
decision = {
    "reference_group": best["group_dir"],
    "passes_gate": best["passed_seeds"] == 2,
    "mean_cindex": best["mean_cindex"],
    "combination_changes": {},
}

# 三、每类只取一个优于原参考组的单因素改动，至少两类才进入组合验证。
if reference_dir != "none":
    baseline = next(row for row in groups if row["group_dir"] == reference_dir)
    baseline_config = configs[reference_dir]
    changes: dict[str, Any] = {}
    for row in groups:
        if rank(row) >= rank(baseline):
            continue
        candidate = configs[row["group_dir"]]
        changed = [
            key
            for key in factors
            if OmegaConf.select(candidate, key) != OmegaConf.select(baseline_config, key)
        ]
        assert len(changed) == 1, "每个单因素组必须恰好改变一项预设参数"
        changes.setdefault(changed[0], OmegaConf.select(candidate, changed[0]))
    if len(changes) >= 2:
        save_arguments(output / "combination.args", baseline_config, changes)
        decision["combination_changes"] = changes
pd.DataFrame(records).to_csv(output / "run_metrics.csv", index=False)
pd.DataFrame(groups).to_csv(output / "comparison.csv", index=False)
(output / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n")
print(
    f"阶段完成：{output.name}，{len(groups)}组，参考={best['group_dir']}，合格={decision['passes_gate']}",
    flush=True,
)
PY
}

# 一、原robust数据：先跑1e-3下的128/256，再跑较低学习率，共12次。
for lr in 1e-3 3e-4 1e-4; do
    for batch in 128 256; do
        fit_group stage1 "bs${batch}-lr${lr}" \
            "split.dir=$p0" "trainer.batch_size=$batch" "trainer.learning_rate=$lr"
    done
done
summarize stage1-summary none "${groups[@]}"

# 二、自动log数据重复同一网格，共12次；合并前两阶段选择参考配置。
for lr in 1e-3 3e-4 1e-4; do
    for batch in 128 256; do
        fit_group stage2 "bs${batch}-lr${lr}" \
            "split.dir=$p1" "trainer.batch_size=$batch" "trainer.learning_rate=$lr"
    done
done
summarize stage2-summary none "${groups[@]}"
reference_group="$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["reference_group"])' \
    "$output_root/stage2-summary/decision.json")"
reference_args=()
while IFS= read -r argument; do reference_args+=("$argument"); done < "$output_root/stage2-summary/reference.args"

# 三、在参考配置上逐项改变初始重建权重、潜维度、初始化和warmup，共12次。
stage3_start=${#groups[@]}
fit_group stage3 weight-0p1 model.loss.reconstruction_weight=0.1
fit_group stage3 weight-0p01 model.loss.reconstruction_weight=0.01
fit_group stage3 latent-16 model.latent_dim=16
fit_group stage3 latent-8 model.latent_dim=8
fit_group stage3 gmm-50 trainer.gmm_init_iters=50
fit_group stage3 warmup-30 trainer.warmup_epochs=30
summarize stage3-summary "$reference_group" "$reference_group" "${groups[@]:$stage3_start}"

# 四、满足条件时追加两个seed的组合验证；最后汇总全部K=3对照，不搜索其他K。
if [[ -f "$output_root/stage3-summary/combination.args" ]]; then
    reference_args=()
    while IFS= read -r argument; do reference_args+=("$argument"); done < "$output_root/stage3-summary/combination.args"
    fit_group stage4 combined
fi
summarize final none "${groups[@]}"
printf 'completed=%s\nstatus=completed\ntest_used_for_selection=false\n' "$completed" > "$output_root/current_run.txt"
printf '探索完成：%s次训练；K=3；汇总：%s/final\n' "$completed" "$output_root"
