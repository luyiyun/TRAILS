#!/usr/bin/env bash
# 在项目根目录运行：bash scripts/crc_yunnan/05_explore_preprocessing.sh <全新输出目录>
# 一次划分、36套预处理、固定K=3、两个训练seed；失败后继续，test产物不进入探索汇总。
set -euo pipefail
umask 077

# 一、检查入口并隔离本轮输出；保留既有源数据、主划分和探索结果。
if [[ "$#" -ne 1 ]]; then
    printf '用法：bash scripts/crc_yunnan/05_explore_preprocessing.sh <全新输出目录>\n' >&2
    exit 1
fi
if [[ ! -f scripts/crc_yunnan/03_split.py || ! -f scripts/crc_yunnan/04_preproc.py || ! -f scripts/crc_yunnan/05_run.py ]]; then
    printf '请在TRAILS项目根目录运行。\n' >&2
    exit 1
fi
command -v uv >/dev/null
output_root="$1"
if [[ -e "$output_root" || -L "$output_root" ]]; then
    printf '拒绝覆盖探索输出：%s\n' "$output_root" >&2
    exit 1
fi
mkdir -p -- "$(dirname -- "$output_root")"
mkdir -- "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"
mkdir -- "$output_root/preproc" "$output_root/runs" \
    "$output_root/logs" "$output_root/summary"
panels=(A2 B8 C12 D20 E60 F88)
transforms=(none auto-log1p)
scalings=(robust standard minmax)
seeds=(20260908 20260909)
split_total=$((${#panels[@]} * ${#transforms[@]} * ${#scalings[@]}))
run_total=$((split_total * ${#seeds[@]}))
splits_finished=0
runs_finished=0
runs_succeeded=0
failed_steps=0
skipped_runs=0
call_exit=0
printf 'stage,panel,log_transform,scaling,seed,status,exit_code,wall_seconds,log_path,output_path,reason\n' \
    > "$output_root/progress.csv"

# Hydra的字符串引号与shell引号分别处理，允许输出路径包含空格、逗号或括号。
hydra_string() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '"%s"' "$value"
}

write_current() {
    printf 'status=%s\nstage=%s\ngroup=%s\nseed=%s\nsplits_finished=%s\nsplits_total=%s\nruns_finished=%s\nruns_total=%s\nruns_succeeded=%s\nfailed_steps=%s\nskipped_runs=%s\n' \
        "$1" "$2" "$3" "$4" "$splits_finished" "$split_total" "$runs_finished" \
        "$run_total" "$runs_succeeded" "$failed_steps" "$skipped_runs" \
        > "$output_root/current_run.txt"
}

run_logged() {
    local stage="$1" seed="$2" output_path="$3" log_path="$4" started elapsed status reason
    shift 4
    mkdir -p -- "$output_root/$(dirname -- "$log_path")"
    write_current running "$stage" "$group" "$seed"
    printf '开始 %s：%s seed=%s；已完成冻结 %s/%s，训练 %s/%s\n' \
        "$stage" "$group" "$seed" "$splits_finished" "$split_total" "$runs_finished" "$run_total"
    started=$(date +%s)
    # 此轮明确要求独立组合失败后继续；保留原命令输出和退出码，不修改输入契约。
    if "$@" 2>&1 | tee "$output_root/$log_path"; then
        call_exit=0
        status=success
        reason=""
    else
        call_exit=$?
        status=failed
        reason=command_failed
        failed_steps=$((failed_steps + 1))
    fi
    elapsed=$(($(date +%s) - started))
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$stage" "$panel" "$transform" "$scaling" "$seed" "$status" "$call_exit" \
        "$elapsed" "$log_path" "$output_path" "$reason" >> "$output_root/progress.csv"
    if [[ "$stage" == preproc ]]; then
        splits_finished=$((splits_finished + 1))
    else
        runs_finished=$((runs_finished + 1))
        if [[ "$call_exit" -eq 0 ]]; then runs_succeeded=$((runs_succeeded + 1)); fi
    fi
    write_current running "$stage" "$group" "$seed"
    printf '结束 %s：%s seed=%s status=%s exit=%s 用时=%ss\n' \
        "$stage" "$group" "$seed" "$status" "$call_exit" "$elapsed"
}

# 二、基础队列仅划分一次；随后复用同一患者归属生成一次划分、36套预处理产物。
uv run python -m scripts.crc_yunnan.03_split \
    'split.strategies=[random]' 'split.seeds=[20260908]' \
    "paths.dir=$(hydra_string "$output_root/raw_splits")" \
    "paths.run_dir=$(hydra_string "$output_root/logs/split_hydra")"
raw_split="$output_root/raw_splits/random/seed-20260908"
for panel in "${panels[@]}"; do
    for transform in "${transforms[@]}"; do
        for scaling in "${scalings[@]}"; do
            variant="$transform-$scaling"
            group="$panel/$variant"
            split_path="preproc/$group/random/seed-20260908"
            split_log="logs/$group/split.log"
            run_logged preproc 20260908 "$split_path" "$split_log" \
                uv run python -m scripts.crc_yunnan.04_preproc \
                outcome=dfs landmark_months=12 "features.panel=$panel" \
                "preprocessing.name=$variant" "preprocessing.log_transform=$transform" \
                preprocessing.skew_threshold=1.0 "preprocessing.scaling=$scaling" \
                "inputs.split_dir=$(hydra_string "$raw_split")" \
                "paths.dir=$(hydra_string "$output_root/$split_path")" \
                "paths.run_dir=$(hydra_string "$output_root/logs/$group/preproc_hydra")"
            if [[ "$call_exit" -ne 0 ]]; then
                for seed in "${seeds[@]}"; do
                    printf 'run,%s,%s,%s,%s,skipped,,,%s,%s,preproc_failed\n' \
                        "$panel" "$transform" "$scaling" "$seed" "$split_log" \
                        "runs/$group/seed-$seed" >> "$output_root/progress.csv"
                    skipped_runs=$((skipped_runs + 1))
                    runs_finished=$((runs_finished + 1))
                done
                continue
            fi
            for seed in "${seeds[@]}"; do
                run_logged run "$seed" "runs/$group/seed-$seed" "logs/$group/seed-$seed.log" \
                    uv run python -m scripts.crc_yunnan.05_run model=base trainer=full \
                    n_clusters=3 split.strategy=random split.seed=20260908 \
                    "split.dir=$(hydra_string "$output_root/$split_path")" \
                    model.latent_dim=32 trainer.batch_size=128 trainer.learning_rate=1e-3 \
                    trainer.warmup_epochs=10 trainer.gmm_init_iters=5 trainer.valid_size=0.0 \
                    trainer.cindex_risk_score=median_survival trainer.device=cuda:0 \
                    swanlab.enabled=false "trainer.seed=$seed" \
                    "paths.dir=$(hydra_string "$output_root/runs/$group/seed-$seed")"
            done
        done
    done
done

# 三、保留全部计划行；仅汇总train/validation指标和验证集预测，不排名或自动选优。
write_current summarizing summary all ""
if uv run python - "$output_root" <<'PY' > "$output_root/logs/summary.log" 2>&1
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import adjusted_rand_score

root = Path(sys.argv[1]).resolve()
output = root / "summary"
progress = pd.read_csv(root / "progress.csv", dtype={"exit_code": "Int64", "wall_seconds": "Int64"})
split_steps = progress.loc[progress.stage.eq("preproc")]
run_steps = progress.loc[progress.stage.eq("run")]
assert len(split_steps) == 36 and len(run_steps) == 72
failure_steps = progress.loc[~progress.status.eq("success")]
failure_steps.to_csv(output / "failures.csv", index=False)
keys = ["panel", "log_transform", "scaling"]
panel_ids: dict[tuple[str, str], pd.Index] = {}
validation_ids: dict[str, pd.Index] = {}
split_info: dict[str, dict[str, Any]] = {}
assignment_manifest: str | None = None
training_config: dict[str, Any] | None = None

# 同面板的预处理变体须保留相同患者；只核对ID，不读取test模型指标或预测。
for step in split_steps.loc[split_steps.status.eq("success")].to_dict(orient="records"):
    group = f"{step['panel']}/{step['log_transform']}-{step['scaling']}"
    bundle = root / step["output_path"]
    manifest = json.loads((bundle / "preproc_manifest.json").read_text())
    assert manifest["panel"] == step["panel"] and len(manifest["feature_order"]) == int(
        step["panel"][1:]
    )
    assert Path(manifest["source_datasets"]["train"]).parent == root / "raw_splits/random/seed-20260908"
    assert manifest["outcome"] == "dfs" and manifest["landmark_days"] == 360
    preprocessing = manifest["preprocessing"]
    assert preprocessing["log_transform"] == step["log_transform"]
    assert preprocessing["scaling"] == step["scaling"] and preprocessing["skew_threshold"] == 1.0
    if assignment_manifest is None:
        assignment_manifest = manifest["assignment_manifests"][0]
    assert manifest["assignment_manifests"][0] == assignment_manifest, "须复用相同患者主划分"
    for name in ("train", "validation", "test"):
        ids = cast(
            pd.Series,
            pd.read_csv(bundle / name / "ids.csv", dtype={"patient_id": "string"})["patient_id"],
        )
        assert ids.notna().all() and ids.is_unique
        ordered = pd.Index(ids).sort_values()
        panel_key = (step["panel"], name)
        if panel_key not in panel_ids:
            panel_ids[panel_key] = ordered
        assert ordered.equals(panel_ids[panel_key]), f"同面板{name}患者名单不一致：{group}"
        if name == "validation":
            validation_ids[group] = ordered
    split_info[group] = {
        f"{name}_{field}": manifest["split_counts"][name][field]
        for name in ("train", "validation")
        for field in ("patients", "events")
    }

metric_names = (
    "cindex",
    "empty_clusters",
    "min_fraction",
    "max_fraction",
    "entropy",
    "passes_gate",
)
cohort_names = ("train_patients", "train_events", "validation_patients", "validation_events")
records: list[dict[str, Any]] = []
predictions: dict[tuple[str, int], pd.Series] = {}
for step in run_steps.to_dict(orient="records"):
    group = f"{step['panel']}/{step['log_transform']}-{step['scaling']}"
    row = {
        **step,
        **dict.fromkeys(cohort_names),
        **split_info.get(group, {}),
        **dict.fromkeys(metric_names),
    }
    if step["status"] == "success":
        run_dir = root / step["output_path"]
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        assert manifest["selected_k"] == 3 and manifest["selected_seed"] == step["seed"]
        assert (
            Path(manifest["split_dir"]).resolve()
            == root / "preproc" / group / "random/seed-20260908"
        )
        cfg = OmegaConf.load(run_dir / ".hydra/config.yaml")
        assert isinstance(cfg, DictConfig)
        protocol = OmegaConf.to_container(
            OmegaConf.create({"model": cfg.model, "trainer": cfg.trainer}), resolve=True
        )
        assert isinstance(protocol, dict) and isinstance(protocol["trainer"], dict)
        assert protocol["trainer"].pop("seed") == step["seed"]
        if training_config is None:
            training_config = cast(dict[str, Any], protocol)
        assert protocol == training_config, "各组须使用同一模型与训练配置，只有seed不同"
        # 05按train/validation/test写表；限定解析前两行，test行不进入数据框。
        metrics = pd.read_csv(run_dir / "metrics.csv", nrows=2).set_index("split")
        assert metrics.index.tolist() == ["train", "validation"]
        for name in ("train", "validation"):
            for field in ("patients", "events"):
                assert metrics.loc[name, f"n_{field}"] == row[f"{name}_{field}"]
        valid = metrics.loc["validation"]
        row.update(
            cindex=float(valid["cindex"]),
            empty_clusters=int(valid["cluster_empty_count"]),
            min_fraction=float(valid["cluster_min_fraction"]),
            max_fraction=float(valid["cluster_max_fraction"]),
            entropy=float(valid["cluster_entropy"]),
        )
        row["passes_gate"] = row["empty_clusters"] == 0 and row["min_fraction"] >= 0.05
        prediction = (
            pd.read_csv(
                run_dir / "validation/patient_outputs.csv",
                usecols=["patient_id", "pred_cluster"],
                dtype={"patient_id": "string", "pred_cluster": "int64"},
            )
            .set_index("patient_id")
            .sort_index()
        )
        assert prediction.index.is_unique and prediction.index.equals(validation_ids[group])
        labels = cast(pd.Series, prediction["pred_cluster"])
        assert labels.isin(range(3)).all()
        predictions[group, int(step["seed"])] = labels
    records.append(row)

runs = pd.DataFrame(records)
comparisons: list[dict[str, Any]] = []
for step in split_steps.to_dict(orient="records"):
    group = f"{step['panel']}/{step['log_transform']}-{step['scaling']}"
    rows = runs.loc[runs[keys].eq(pd.Series({key: step[key] for key in keys})).all(axis=1)]
    assert rows.seed.tolist() == [20260908, 20260909]
    completed = rows.loc[rows.status.eq("success")]
    paired = len(completed) == 2
    comparisons.append(
        {
            **{key: step[key] for key in keys},
            **dict.fromkeys(cohort_names),
            **split_info.get(group, {}),
            "split_status": step["status"],
            "status": "completed" if paired else "incomplete",
            "completed_seeds": len(completed),
            "failed_seeds": int(rows.status.eq("failed").sum()),
            "skipped_seeds": int(rows.status.eq("skipped").sum()),
            "passed_seeds": int(completed.passes_gate.eq(True).sum()),
            "both_seeds_pass_gate": paired and bool(completed.passes_gate.all()),
            "mean_cindex": (completed.cindex.iloc[0] + completed.cindex.iloc[1]) / 2
            if paired
            else None,
            "cindex_gap": abs(completed.cindex.iloc[0] - completed.cindex.iloc[1])
            if paired
            else None,
            "validation_ari": float(
                adjusted_rand_score(predictions[group, 20260908], predictions[group, 20260909])
            )
            if paired
            else None,
            "split_wall_seconds": step["wall_seconds"],
            "training_wall_seconds": int(rows.wall_seconds.sum()),
        }
    )
runs.to_csv(output / "run_metrics.csv", index=False)
pd.DataFrame(comparisons).to_csv(output / "comparison.csv", index=False)
status = {
    "status": "completed" if failure_steps.empty else "incomplete",
    "planned_splits": 36,
    "planned_runs": 72,
    "successful_splits": int(split_steps.status.eq("success").sum()),
    "successful_runs": int(run_steps.status.eq("success").sum()),
    "failed_steps": int(progress.status.eq("failed").sum()),
    "skipped_runs": int(run_steps.status.eq("skipped").sum()),
    "test_used_for_selection": False,
    "automatic_selection": False,
}
(output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(status, ensure_ascii=False, indent=2))
PY
then
    cat "$output_root/logs/summary.log"
else
    summary_exit=$?
    write_current incomplete summary all ""
    cat "$output_root/logs/summary.log" >&2
    printf '汇总失败（exit=%s）；原始产物和progress.csv已保留：%s\n' "$summary_exit" "$output_root" >&2
    exit "$summary_exit"
fi

# 四、成功与不完整明确区分；空簇不改变进程状态，命令失败或跳过则返回非零。
if [[ "$failed_steps" -gt 0 || "$skipped_runs" -gt 0 ]]; then
    write_current incomplete finished all ""
    printf '探索不完整：成功训练%s/%s，失败步骤%s，跳过训练%s；汇总：%s/summary\n' \
        "$runs_succeeded" "$run_total" "$failed_steps" "$skipped_runs" "$output_root" >&2
    exit 1
fi
write_current completed finished all ""
printf '探索完成：%s套split，%s次训练；未自动选优；汇总：%s/summary\n' \
    "$split_total" "$run_total" "$output_root"
