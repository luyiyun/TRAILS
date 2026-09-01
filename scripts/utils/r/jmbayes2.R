#!/usr/bin/env Rscript
# JMbayes2共享基线的隔离R入口；由Python RScriptBackend传入JSON文件路径。

# 1. 命令行、依赖与进度日志
# =============================================================================
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("用法：jmbayes2.R <action> <config.json> <result.json>")
}
action <- args[[1L]]
config_path <- args[[2L]]
result_path <- args[[3L]]
run_started <- proc.time()[["elapsed"]]
report_progress <- function(stage, detail) {
  elapsed <- proc.time()[["elapsed"]] - run_started
  cat(sprintf("[jmbayes2][%s][elapsed=%.1fs] %s\n", stage, elapsed, detail))
  flush.console()
}
report_progress("start", sprintf("开始执行action=%s", action))
required_packages <- c(
  "data.table", "JMbayes2", "jsonlite", "nlme", "R.utils", "survival"
)
missing_packages <- required_packages[!vapply(
  required_packages, requireNamespace, logical(1L), quietly = TRUE
)]
if (length(missing_packages) > 0L) {
  stop(sprintf("缺少R包：%s", paste(missing_packages, collapse = ", ")))
}
config <- jsonlite::read_json(config_path, simplifyVector = TRUE)
report_progress("preflight", "R依赖检查通过")

# 2. 外部动态预测：只读取纵向历史
# =============================================================================
if (action == "predict") {
  required_predict <- c(
    "observations_csv", "features_csv", "model_path", "predictions_csv",
    "entry_time", "prediction_times", "risk_horizon", "seed", "n_samples",
    "n_mcmc", "n_cores", "parallel", "patient_batch_size"
  )
  missing_predict <- setdiff(required_predict, names(config))
  if (length(missing_predict) > 0L) {
    stop(sprintf("JMbayes2 predict配置缺少：%s", paste(missing_predict, collapse = ", ")))
  }
  prediction_times <- as.numeric(config$prediction_times)
  risk_horizon <- as.numeric(config$risk_horizon)
  entry_time <- as.numeric(config$entry_time)
  n_samples <- as.integer(config$n_samples)
  n_mcmc <- as.integer(config$n_mcmc)
  n_cores <- as.integer(config$n_cores)
  patient_batch_size <- as.integer(config$patient_batch_size)
  parallel_method <- as.character(config$parallel)
  integer_options <- c(n_samples, n_mcmc, n_cores, patient_batch_size)
  if (length(prediction_times) == 0L || any(!is.finite(prediction_times)) ||
      any(diff(prediction_times) <= 0) || prediction_times[[1L]] <= 0 ||
      !is.finite(risk_horizon) || tail(prediction_times, 1L) >= risk_horizon ||
      !is.finite(entry_time) || entry_time < 0 || any(is.na(integer_options)) ||
      any(integer_options <= 0L) || !parallel_method %in% c("snow", "multicore")) {
    stop("JMbayes2预测时间、Monte Carlo或并行配置无效")
  }

  report_progress("predict", "读取模型、纵向观测和特征映射")
  bundle <- readRDS(config$model_path)
  model <- bundle$model
  if (!inherits(model, "jm")) {
    stop("模型文件不包含JMbayes2 jm对象")
  }
  training_features <- data.table::as.data.table(bundle$features)
  observations <- data.table::fread(config$observations_csv,
    select = c("subject_id", "time", "feature_id", "value"))
  features <- data.table::fread(config$features_csv,
    select = c("feature_id", "feature_name"))
  data.table::setorder(training_features, feature_id)
  data.table::setorder(features, feature_id)
  data.table::setorder(observations, subject_id, time, feature_id)
  if (!identical(as.integer(training_features$feature_id),
      as.integer(features$feature_id)) ||
      !identical(training_features$feature_name, features$feature_name)) {
    stop("外部数据的特征映射与训练模型不一致")
  }
  subject_ids <- sort(unique(as.integer(observations$subject_id)))
  if (length(subject_ids) == 0L ||
      !identical(subject_ids, seq_along(subject_ids)) ||
      anyDuplicated(observations[, .(subject_id, time, feature_id)]) > 0L ||
      !setequal(as.integer(observations$feature_id), as.integer(features$feature_id)) ||
      any(!is.finite(observations$time)) || any(!is.finite(observations$value)) ||
      any(observations$time < 0) || any(observations$time > entry_time + 1e-8)) {
    stop("外部纵向观测的ID、时间、特征或数值不符合预测契约")
  }
  observations[, marker := sprintf("marker_%d", feature_id)]
  wide <- data.table::dcast(observations,
    subject_id + time ~ marker, value.var = "value")
  data.table::setorder(wide, subject_id, time)
  marker_columns <- sprintf("marker_%d", features$feature_id)
  if (!all(marker_columns %in% names(wide))) {
    stop("至少一个训练marker在外部数据中完全缺失")
  }
  report_progress("predict", sprintf(
    "读取完成：%d名患者，%d条观测，%d个marker",
    length(subject_ids), nrow(observations), nrow(features)
  ))

  # counting-process格式不接受零宽区间；极短无事件区间仅表达landmark时存活。
  conditioning_epsilon <- max(1e-8, abs(entry_time) * 1e-8)
  conditioning_time <- entry_time + conditioning_epsilon
  relative_times <- c(prediction_times, risk_horizon)
  absolute_times <- entry_time + relative_times
  survival_columns <- sprintf("survival_%d", seq_along(prediction_times))
  predictions_path <- normalizePath(config$predictions_csv, mustWork = FALSE)
  dir.create(dirname(predictions_path), recursive = TRUE, showWarnings = FALSE)

  for (batch_start in seq.int(1L, length(subject_ids), by = patient_batch_size)) {
    batch_stop <- min(batch_start + patient_batch_size - 1L, length(subject_ids))
    batch_subjects <- subject_ids[batch_start:batch_stop]
    batch_longitudinal <- wide[subject_id %in% batch_subjects]
    batch_event <- data.frame(
      subject_id = batch_subjects,
      entry_time = rep(entry_time, length(batch_subjects)),
      event_time = rep(conditioning_time, length(batch_subjects)),
      event = integer(length(batch_subjects))
    )
    # value(marker)在事件设计矩阵中只返回全1；占位列不作纵向观测。
    # 真正的历史仍仅来自newdataL，不能将目标结局或插补值放入其中。
    for (marker in marker_columns) {
      batch_event[[marker]] <- 0
    }
    report_progress("predict", sprintf(
      "动态预测患者批次%d-%d/%d", batch_start, batch_stop, length(subject_ids)
    ))
    prediction <- stats::predict(
      model,
      newdata = list(
        newdataL = as.data.frame(batch_longitudinal),
        newdataE = batch_event
      ),
      process = "event",
      times = absolute_times,
      control = list(
        use_Y = TRUE, return_newdata = FALSE, return_mcmc = FALSE,
        n_samples = n_samples, n_mcmc = n_mcmc, cores = n_cores,
        parallel = parallel_method, seed = as.integer(config$seed) + batch_start - 1L
      )
    )
    prediction_frame <- data.table::data.table(
      subject_id = as.integer(prediction$id),
      absolute_time = as.numeric(prediction$times),
      cumulative_risk = as.numeric(prediction$pred)
    )
    prediction_frame[, time_index := match(absolute_time, absolute_times)]
    prediction_frame <- prediction_frame[!is.na(time_index)]
    if (nrow(prediction_frame) != length(batch_subjects) * length(relative_times) ||
        anyDuplicated(prediction_frame[, .(subject_id, time_index)]) > 0L ||
        !setequal(prediction_frame$subject_id, batch_subjects) ||
        any(!is.finite(prediction_frame$cumulative_risk)) ||
        any(prediction_frame$cumulative_risk < -1e-6) ||
        any(prediction_frame$cumulative_risk > 1 + 1e-6)) {
      stop("JMbayes2动态预测的患者、时间或累计风险不符合契约")
    }
    prediction_frame[, cumulative_risk := pmin(pmax(cumulative_risk, 0), 1)]
    risk_wide <- data.table::dcast(prediction_frame,
      subject_id ~ time_index, value.var = "cumulative_risk")
    data.table::setorder(risk_wide, subject_id)
    risk_columns <- as.character(seq_along(relative_times))
    if (!identical(names(risk_wide)[-1L], risk_columns)) {
      stop("JMbayes2动态预测缺少要求的时间点")
    }
    risk_matrix <- as.matrix(risk_wide[, risk_columns, with = FALSE])
    survival_matrix <- 1 - risk_matrix
    if (any(apply(survival_matrix, 1L, diff) > 1e-6)) {
      stop("JMbayes2患者级条件生存曲线不是单调非增")
    }
    output <- data.table::data.table(
      subject_id = risk_wide$subject_id,
      risk_score = risk_matrix[, ncol(risk_matrix)]
    )
    for (index in seq_along(survival_columns)) {
      output[[survival_columns[[index]]]] <- survival_matrix[, index]
    }
    data.table::fwrite(output, predictions_path,
      append = batch_start > 1L, col.names = batch_start == 1L)
  }

  result <- list(
    format_version = 1L,
    method = "JMbayes2::jm",
    n_patients = length(subject_ids),
    n_features = nrow(features),
    prediction_times = prediction_times,
    survival_columns = survival_columns,
    risk_horizon = risk_horizon,
    predictions_csv = predictions_path,
    outcome_columns_consumed = list(),
    prediction_inputs = "longitudinal_only",
    survival_conditioning = "alive_at_entry",
    conditioning_epsilon = conditioning_epsilon,
    monte_carlo = list(n_samples = n_samples, n_mcmc = n_mcmc)
  )
  jsonlite::write_json(result, result_path, auto_unbox = TRUE, pretty = TRUE)
  report_progress("predict", sprintf("预测完成并保存：%s", predictions_path))
  quit(save = "no", status = 0L)
}
if (action != "fit") {
  stop(sprintf("未知action：%s", action))
}
# 3. 拟合配置与输入边界
# =============================================================================
required_fit <- c(
  "patients_csv", "observations_csv", "features_csv", "model_path", "seed",
  "n_chains", "n_iter", "n_burnin", "n_thin", "n_cores", "parallel",
  "lme_max_iterations"
)
missing_fit <- setdiff(required_fit, names(config))
if (length(missing_fit) > 0L) {
  stop(sprintf("JMbayes2 fit配置缺少：%s", paste(missing_fit, collapse = ", ")))
}
n_chains <- as.integer(config$n_chains)
n_iter <- as.integer(config$n_iter)
n_burnin <- as.integer(config$n_burnin)
n_thin <- as.integer(config$n_thin)
n_cores <- as.integer(config$n_cores)
lme_max_iterations <- as.integer(config$lme_max_iterations)
parallel_method <- as.character(config$parallel)
integer_options <- c(n_chains, n_iter, n_thin, n_cores, lme_max_iterations)
if (any(is.na(integer_options)) || any(integer_options <= 0L) ||
    is.na(n_burnin) || n_burnin < 0L || n_iter <= n_burnin ||
    n_cores > n_chains || !parallel_method %in% c("snow", "multicore")) {
  stop("MCMC、LME迭代或并行配置无效")
}
report_progress("input", "读取患者、观测和特征表")
patients <- data.table::fread(config$patients_csv,
  select = c("subject_id", "entry_time", "event_time", "event"))
observations <- data.table::fread(config$observations_csv,
  select = c("subject_id", "time", "feature_id", "value"))
features <- data.table::fread(config$features_csv,
  select = c("feature_id", "feature_name"))
if (nrow(patients) == 0L || nrow(observations) == 0L || nrow(features) == 0L) {
  stop("JMbayes2输入表不能为空")
}
data.table::setorder(patients, subject_id)
data.table::setorder(observations, subject_id, time, feature_id)
data.table::setorder(features, feature_id)
expected_subjects <- seq_len(nrow(patients))
expected_features <- seq_len(nrow(features))
if (!identical(as.integer(patients$subject_id), expected_subjects) ||
    !identical(as.integer(features$feature_id), expected_features) ||
    !setequal(as.integer(observations$subject_id), expected_subjects) ||
    !setequal(as.integer(observations$feature_id), expected_features)) {
  stop("subject_id和feature_id必须分别连续覆盖1..N与1..D")
}
if (anyDuplicated(observations[, .(subject_id, time, feature_id)]) > 0L ||
    any(!patients$event %in% c(0L, 1L)) ||
    any(!is.finite(patients$entry_time)) || any(!is.finite(patients$event_time)) ||
    any(patients$entry_time < 0) || any(patients$event_time <= patients$entry_time) ||
    any(!is.finite(observations$time)) || any(!is.finite(observations$value)) ||
    any(observations$time < 0)) {
  stop("JMbayes2输入包含重复观测、无效结局或非有限数值")
}
if (any(observations$time > patients$entry_time[
  match(observations$subject_id, patients$subject_id)
] + 1e-8)) {
  stop("纵向观测时间不能晚于患者的landmark进入时间")
}
report_progress("input", sprintf(
  "读取完成：%d名患者，%d条观测，%d个marker",
  nrow(patients), nrow(observations), nrow(features)
))
# 4. 每个marker的线性混合子模型
# =============================================================================
observations[, marker := sprintf("marker_%d", feature_id)]
wide <- data.table::dcast(observations,
  subject_id + time ~ marker, value.var = "value")
data.table::setorder(wide, subject_id, time)
marker_columns <- sprintf("marker_%d", features$feature_id)
model_data <- as.data.frame(wide)
fit_marker <- function(marker, index) {
  report_progress("longitudinal", sprintf(
    "拟合marker %d/%d：%s", index, length(marker_columns),
    features$feature_name[[index]]
  ))
  fixed_formula <- stats::as.formula(sprintf("%s ~ time", marker))
  # jm会核对各子模型的原始data；响应缺失仍由各lme的na.omit处理。
  model <- nlme::lme(
    fixed = fixed_formula,
    random = ~time | subject_id,
    data = model_data,
    method = "ML",
    na.action = stats::na.omit,
    control = nlme::lmeControl(
      opt = "optim", msMaxIter = lme_max_iterations, returnObject = FALSE
    )
  )
  model$call$fixed <- fixed_formula
  model
}
longitudinal_models <- lapply(seq_along(marker_columns), function(index) {
  fit_marker(marker_columns[[index]], index)
})
names(longitudinal_models) <- marker_columns

# 5. 左截断生存子模型与多变量joint model
# =============================================================================
report_progress("survival", "拟合左截断Cox生存子模型")
survival_model <- survival::coxph(
  survival::Surv(entry_time, event_time, event) ~ 1,
  data = as.data.frame(patients), x = TRUE, y = TRUE, model = TRUE
)
control <- list(
  n_chains = n_chains, n_iter = n_iter, n_burnin = n_burnin,
  n_thin = n_thin, seed = as.integer(config$seed), cores = n_cores,
  parallel = parallel_method, save_random_effects = FALSE,
  save_logLik_contributions = FALSE
)
joint_arguments <- list(
  Surv_object = survival_model,
  Mixed_objects = longitudinal_models,
  time_var = "time",
  data_Surv = as.data.frame(patients),
  id_var = "subject_id",
  control = control
)
if (length(longitudinal_models) > 1L) {
  joint_arguments$which_independent <- "all"
}
report_progress("joint", sprintf(
  "开始MCMC：%d条链，每链%d次迭代，burn-in=%d，thin=%d",
  n_chains, n_iter, n_burnin, n_thin
))
model <- do.call(JMbayes2::jm, joint_arguments)

# 6. 聚合MCMC诊断与模型产物
# =============================================================================
rhat <- model$statistics$Rhat
rhat_diagnostics <- list(available = FALSE)
if (!is.null(rhat)) {
  rhat <- as.matrix(rhat)
  point_values <- rhat[, 1L]
  upper_values <- if (ncol(rhat) >= 2L) rhat[, 2L] else numeric(0L)
  point_values <- point_values[is.finite(point_values)]
  upper_values <- upper_values[is.finite(upper_values)]
  if (length(point_values) > 0L) {
    rhat_diagnostics <- list(
      available = TRUE,
      n_parameters = length(point_values),
      max_point_estimate = max(point_values),
      max_upper_confidence_limit = if (length(upper_values) > 0L) {
        max(upper_values)
      } else {
        NULL
      }
    )
  }
}
model_path <- normalizePath(config$model_path, mustWork = FALSE)
dir.create(dirname(model_path), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(model = model, features = features), model_path)

result <- list(
  format_version = 1L,
  method = "JMbayes2::jm",
  n_patients = nrow(patients),
  n_observations = nrow(observations),
  n_features = nrow(features),
  n_events = sum(patients$event),
  model_path = model_path,
  running_time_seconds = as.numeric(model$running_time[["elapsed"]]),
  mcmc = list(
    n_chains = n_chains, n_iter = n_iter, n_burnin = n_burnin,
    n_thin = n_thin, retained_draws_per_chain = (n_iter - n_burnin) %/% n_thin
  ),
  rhat = rhat_diagnostics,
  package_versions = as.list(vapply(required_packages,
    function(package) as.character(utils::packageVersion(package)), character(1L)))
)
jsonlite::write_json(result, result_path, auto_unbox = TRUE, pretty = TRUE)
report_progress("fit", sprintf("训练完成并保存：%s", model_path))
