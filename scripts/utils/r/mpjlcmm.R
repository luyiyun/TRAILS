#!/usr/bin/env Rscript
# mpjlcmm共享基线的隔离R入口；由Python RScriptBackend传入JSON文件路径。

# =============================================================================
# 1. 命令行、依赖与公共日志
# =============================================================================
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("用法：mpjlcmm.R <action> <config.json> <result.json>")
}
action <- args[[1L]]
config_path <- args[[2L]]
result_path <- args[[3L]]
run_started <- proc.time()[["elapsed"]]
report_progress <- function(stage, detail) {
  elapsed <- proc.time()[["elapsed"]] - run_started
  cat(sprintf("[mpjlcmm][%s][elapsed=%.1fs] %s\n", stage, elapsed, detail))
  flush.console()
}
report_progress("start", sprintf("开始执行action=%s", action))
required_packages <- c("data.table", "jsonlite", "lcmm", "R.utils", "survival")
missing_packages <- required_packages[!vapply(
  required_packages, requireNamespace, logical(1L), quietly = TRUE
)]
if (length(missing_packages) > 0L) {
  stop(sprintf("缺少R包：%s", paste(missing_packages, collapse = ", ")))
}
report_progress("preflight", "R依赖检查通过")
config <- jsonlite::read_json(config_path, simplifyVector = TRUE)

# =============================================================================
# 2. 外部预测：读取模型和纵向历史
# =============================================================================
if (action == "predict") {
  report_progress("predict", "校验预测配置并读取模型")
  required_predict <- c(
    "observations_csv", "features_csv", "model_path", "predictions_csv",
    "entry_time", "prediction_times", "risk_horizon"
  )
  missing_predict <- setdiff(required_predict, names(config))
  if (length(missing_predict) > 0L) {
    stop(sprintf("mpjlcmm predict配置缺少：%s", paste(missing_predict, collapse = ", ")))
  }
  prediction_times <- as.numeric(config$prediction_times)
  risk_horizon <- as.numeric(config$risk_horizon)
  entry_time <- as.numeric(config$entry_time)
  if (length(prediction_times) == 0L || any(!is.finite(prediction_times)) ||
      any(diff(prediction_times) <= 0) || prediction_times[[1L]] <= 0 ||
      tail(prediction_times, 1L) >= risk_horizon || !is.finite(risk_horizon) ||
      !is.finite(entry_time) || entry_time < 0) {
    stop("预测时间必须正且递增，并严格小于有限risk_horizon；entry_time必须有限")
  }

  bundle <- readRDS(config$model_path)
  model <- bundle$model
  longitudinal_models <- bundle$longitudinal_models
  training_features <- data.table::as.data.table(bundle$features)
  observations <- data.table::fread(config$observations_csv,
    select = c("subject_id", "time", "feature_id", "value"))
  features <- data.table::fread(config$features_csv,
    select = c("feature_id", "feature_name"))
  data.table::setorder(training_features, feature_id)
  data.table::setorder(features, feature_id)
  if (!identical(training_features$feature_id, features$feature_id) ||
      !identical(training_features$feature_name, features$feature_name)) {
    stop("外部数据的特征映射与训练模型不一致")
  }
  subject_ids <- sort(unique(observations$subject_id))
  if (!identical(subject_ids, seq_along(subject_ids))) {
    stop("外部observations的subject_id必须覆盖1..N")
  }
  if (anyDuplicated(observations[, .(subject_id, time, feature_id)]) > 0L) {
    stop("外部observations包含重复的患者-时间-特征观测")
  }
  if (!all(observations$feature_id %in% features$feature_id)) {
    stop("外部observations包含未声明的feature_id")
  }
  if (any(!is.finite(observations$time)) || any(!is.finite(observations$value)) ||
      any(observations$time < 0) || any(observations$time > entry_time + 1e-8)) {
    stop("外部预测只能使用[0, landmark]内的有限纵向观测")
  }
  report_progress("predict", sprintf(
    "读取完成：%d名患者，%d条观测，%d个marker",
    length(subject_ids), nrow(observations), nrow(features)
  ))
  observations[, marker := sprintf("marker_%d", feature_id)]
  wide <- data.table::dcast(observations, subject_id + time ~ marker, value.var = "value")

# =============================================================================
# 3. 外部预测：组合纯纵向类别后验
# =============================================================================
  n_clusters <- as.integer(model$ng)
  n_subjects <- length(subject_ids)
  if (length(longitudinal_models) != nrow(features) ||
      model$N[[1L]] != n_clusters - 1L) {
    stop("模型不满足每个marker一个子模型且classmb仅含截距的后验契约")
  }
  prior_logits <- c(model$best[seq_len(n_clusters - 1L)], 0)
  priors <- exp(prior_logits - max(prior_logits))
  priors <- priors / sum(priors)
  log_scores <- matrix(log(priors),
    nrow = n_subjects, ncol = n_clusters, byrow = TRUE)
  probability_columns <- sprintf("prob%d", seq_len(n_clusters))

  # mpjlcmm条件于类别时各marker独立；无该marker观测的患者得到中性贡献。
  for (index in seq_along(longitudinal_models)) {
    marker <- sprintf("marker_%d", features$feature_id[[index]])
    report_progress("predict", sprintf(
      "计算marker后验 %d/%d：%s",
      index, length(longitudinal_models), features$feature_name[[index]]
    ))
    if (!marker %in% names(wide)) {
      next
    }
    marker_data <- wide[!is.na(get(marker))]
    marker_prediction <- as.data.frame(lcmm::predictClass(
      longitudinal_models[[index]],
      newdata = as.data.frame(marker_data),
      subject = "subject_id"
    ))
    if (!all(probability_columns %in% names(marker_prediction))) {
      stop(sprintf("%s的predictClass结果缺少类别概率", marker))
    }
    positions <- match(as.integer(marker_prediction[[1L]]), subject_ids)
    if (anyNA(positions) || anyDuplicated(positions) > 0L) {
      stop(sprintf("%s的predictClass患者顺序无效", marker))
    }
    probabilities <- as.matrix(marker_prediction[, probability_columns])
    if (any(!is.finite(probabilities)) || any(probabilities < 0) ||
        any(rowSums(probabilities) <= 0)) {
      stop(sprintf("%s产生无效类别概率", marker))
    }
    probabilities <- probabilities / rowSums(probabilities)
    log_scores[positions, ] <- log_scores[positions, ] +
      log(pmax(probabilities, .Machine$double.xmin)) - matrix(
        log(priors), nrow = length(positions), ncol = n_clusters, byrow = TRUE
      )
  }
  log_scores <- log_scores - apply(log_scores, 1L, max)
  posterior <- exp(log_scores)
  posterior <- posterior / rowSums(posterior)

# =============================================================================
# 4. 外部预测：landmark条件生存曲线与产物
# =============================================================================
  report_progress("predict", "计算类别累计发生曲线和患者条件生存曲线")
  absolute_times <- entry_time + c(0, prediction_times, risk_horizon)
  incidence <- as.data.frame(lcmm::cuminc(model, time = absolute_times)[[1L]])
  class_columns <- sprintf("class%d", seq_len(n_clusters))
  incidence <- incidence[incidence$event == 1, c("time", class_columns)]
  if (nrow(incidence) != length(absolute_times) ||
      !isTRUE(all.equal(incidence$time, absolute_times))) {
    stop("cuminc没有返回请求的单事件时间网格")
  }
  unconditional <- 1 - as.matrix(incidence[, class_columns])
  landmark_survival <- unconditional[1L, ]
  if (any(!is.finite(unconditional)) || any(landmark_survival <= 0)) {
    stop("类别生存曲线在landmark处无效")
  }
  class_survival <- sweep(unconditional[-1L, , drop = FALSE], 2L,
    landmark_survival, "/")
  class_survival <- apply(class_survival, 2L, cummin)
  class_survival[class_survival < 0] <- 0
  class_survival[class_survival > 1] <- 1
  # 生存预测还条件于所有入组患者均已存活至landmark，这不是外部结局泄漏。
  survival_posterior <- posterior * matrix(
    landmark_survival, nrow = n_subjects, ncol = n_clusters, byrow = TRUE
  )
  survival_posterior <- survival_posterior / rowSums(survival_posterior)
  patient_survival <- survival_posterior %*% t(class_survival)
  if (any(!is.finite(patient_survival)) ||
      any(apply(patient_survival, 1L, diff) > 1e-8)) {
    stop("患者级条件生存曲线无效")
  }

  survival_columns <- sprintf("survival_%d", seq_along(prediction_times))
  output <- data.table::data.table(
    subject_id = subject_ids,
    cluster = max.col(posterior, ties.method = "first"),
    risk_score = 1 - patient_survival[, ncol(patient_survival)]
  )
  for (index in seq_len(n_clusters)) {
    output[[sprintf("prob_%d", index)]] <- posterior[, index]
  }
  for (index in seq_along(prediction_times)) {
    output[[survival_columns[[index]]]] <- patient_survival[, index]
  }
  predictions_path <- normalizePath(config$predictions_csv, mustWork = FALSE)
  dir.create(dirname(predictions_path), recursive = TRUE, showWarnings = FALSE)
  data.table::fwrite(output, predictions_path)
  result <- list(
    format_version = 1L,
    method = "lcmm::mpjlcmm",
    n_patients = n_subjects,
    n_clusters = n_clusters,
    prediction_times = prediction_times,
    survival_columns = survival_columns,
    risk_horizon = risk_horizon,
    predictions_csv = predictions_path,
    outcome_columns_consumed = list(),
    class_assignment_inputs = "longitudinal_only",
    survival_conditioning = "alive_at_entry"
  )
  jsonlite::write_json(result, result_path, auto_unbox = TRUE, pretty = TRUE)
  report_progress("predict", sprintf("预测完成并保存：%s", predictions_path))
  quit(save = "no", status = 0L)
}
if (!action %in% c("fit", "validate-posterior")) {
  stop(sprintf("mpjlcmm暂不支持action=%s", action))
}

# =============================================================================
# 5. 模型训练：配置与输入数据
# =============================================================================
report_progress("fit", "校验训练配置并读取train输入")
required_config <- c(
  "patients_csv", "observations_csv", "features_csv", "model_path", "n_clusters",
  "seed", "max_iterations", "grid_repetitions", "grid_iterations", "n_processes"
)
missing_config <- setdiff(required_config, names(config))
if (length(missing_config) > 0L) {
  stop(sprintf("mpjlcmm fit配置缺少：%s", paste(missing_config, collapse = ", ")))
}
n_clusters <- as.integer(config$n_clusters)
max_iterations <- as.integer(config$max_iterations)
grid_repetitions <- as.integer(config$grid_repetitions)
grid_iterations <- as.integer(config$grid_iterations)
n_processes <- as.integer(config$n_processes)
if (n_clusters < 2L || max_iterations < 1L || grid_repetitions < 1L ||
    grid_iterations < 1L || n_processes < 1L) {
  stop("簇数至少为2，迭代次数、grid重复数和进程数必须为正整数")
}
set.seed(as.integer(config$seed))

patients <- data.table::fread(config$patients_csv,
  select = c("subject_id", "entry_time", "event_time", "event"))
observations <- data.table::fread(config$observations_csv,
  select = c("subject_id", "time", "feature_id", "value"))
features <- data.table::fread(config$features_csv,
  select = c("feature_id", "feature_name"))
if (nrow(patients) == 0L || nrow(observations) == 0L || nrow(features) == 0L) {
  stop("mpjlcmm输入表不能为空")
}
if (!identical(patients$subject_id, seq_len(nrow(patients)))) {
  stop("patients表的subject_id必须按1..N连续排列")
}
if (!identical(features$feature_id, seq_len(nrow(features)))) {
  stop("features表的feature_id必须按1..D连续排列")
}
if (anyDuplicated(observations[, .(subject_id, time, feature_id)]) > 0L) {
  stop("observations表包含重复的患者-时间-特征观测")
}
if (!setequal(unique(observations$subject_id), patients$subject_id)) {
  stop("observations与patients的患者集合不一致")
}
if (!all(observations$feature_id %in% features$feature_id)) {
  stop("observations包含features表未声明的feature_id")
}
report_progress("fit", sprintf(
  "读取完成：%d名患者，%d条观测，%d个marker",
  nrow(patients), nrow(observations), nrow(features)
))

# =============================================================================
# 6. 模型训练：构造共享的稀疏宽表
# =============================================================================
report_progress("fit", "将不规则纵向长表转换为稀疏marker宽表")
# marker使用稳定的整数列名，原始临床名称仅作为模型审计元数据保存。
observations[, marker := sprintf("marker_%d", feature_id)]
wide <- data.table::dcast(observations, subject_id + time ~ marker, value.var = "value")
model_data <- merge(wide, patients, by = "subject_id", all.x = TRUE, sort = FALSE)
marker_columns <- sprintf("marker_%d", features$feature_id)
if (!all(marker_columns %in% names(model_data))) {
  stop("至少一个声明的marker没有纵向观测")
}

# =============================================================================
# 7. 模型训练：逐marker单类别与多类别纵向子模型
# =============================================================================
fit_one_class <- function(outcome, index) {
  report_progress("fit", sprintf(
    "拟合单类别marker %d/%d：%s",
    index, length(marker_columns), features$feature_name[[index]]
  ))
  fixed_formula <- stats::as.formula(sprintf("%s ~ time", outcome))
  model <- lcmm::hlme(
    fixed = fixed_formula,
    random = ~time,
    subject = "subject_id",
    ng = 1L,
    idiag = TRUE,
    data = model_data,
    maxiter = max_iterations,
    verbose = FALSE,
    nproc = n_processes
  )
  if (model$conv != 1L) {
    stop(sprintf("单变量模型%s未收敛（conv=%d）", outcome, model$conv))
  }
  # mpjlcmm会重新eval子模型call，不能留下helper局部变量名。
  model$call$fixed <- fixed_formula
  model
}
one_class_models <- lapply(
  seq_along(marker_columns),
  function(index) fit_one_class(marker_columns[[index]], index)
)

# =============================================================================
# 8. 模型训练：共享类别与左截断生存过程的联合估计
# =============================================================================
report_progress("fit", "拟合单类别联合模型，作为多类别模型初值")
joint_one <- lcmm::mpjlcmm(
  longitudinal = one_class_models,
  subject = "subject_id",
  ng = 1L,
  survival = survival::Surv(entry_time, event_time, event) ~ 1,
  hazard = "Weibull",
  hazardtype = "Specific",
  data = model_data,
  maxiter = max_iterations,
  verbose = FALSE,
  nproc = n_processes
)
if (joint_one$conv != 1L) {
  stop(sprintf("单类别联合模型未收敛（conv=%d）", joint_one$conv))
}
# lcmm多类别初值通过单类别call中的模型名称恢复各子模型。
one_class_names <- sprintf("initial_marker_%d", seq_along(one_class_models))
list2env(stats::setNames(one_class_models, one_class_names), envir = environment())
joint_one$call$longitudinal <- as.call(c(
  list(as.name("list")), lapply(one_class_names, as.name)
))

report_progress("fit", "构造多类别纵向子模型")
class_models <- lapply(seq_along(marker_columns), function(index) {
  fixed_formula <- stats::as.formula(sprintf("%s ~ time", marker_columns[[index]]))
  model <- lcmm::hlme(
    fixed = fixed_formula,
    mixture = ~time,
    random = ~time,
    subject = "subject_id",
    ng = n_clusters,
    idiag = TRUE,
    data = model_data,
    B = one_class_models[[index]],
    maxiter = 0L,
    verbose = FALSE
  )
  model$call$fixed <- fixed_formula
  model
})
if (grid_repetitions == 1L) {
  report_progress("fit", "从单类别初值拟合最终mpjlcmm模型")
  model <- lcmm::mpjlcmm(
    longitudinal = class_models,
    subject = "subject_id",
    ng = n_clusters,
    survival = survival::Surv(entry_time, event_time, event) ~ 1,
    hazard = "Weibull",
    hazardtype = "Specific",
    data = model_data,
    B = joint_one,
    maxiter = max_iterations,
    verbose = FALSE,
    nproc = n_processes
  )
} else {
  report_progress("fit", sprintf(
    "开始grid search：%d个初值，每个%d次初始迭代",
    grid_repetitions, grid_iterations
  ))
  model <- lcmm::gridsearch(
    rep = grid_repetitions,
    maxiter = grid_iterations,
    minit = joint_one,
    lcmm::mpjlcmm(
      longitudinal = class_models,
      subject = "subject_id",
      ng = n_clusters,
      survival = survival::Surv(entry_time, event_time, event) ~ 1,
      hazard = "Weibull",
      hazardtype = "Specific",
      data = model_data,
      maxiter = max_iterations,
      verbose = FALSE,
      nproc = n_processes
    )
  )
}
if (model$conv != 1L) {
  stop(sprintf("mpjlcmm最终模型未收敛（conv=%d）", model$conv))
}
report_progress("fit", sprintf(
  "最终模型收敛：logLik=%.4f，BIC=%.4f",
  model$loglik, model$BIC
))
updated_models <- stats::update(model)
posterior_validation <- NULL
if (action == "validate-posterior") {
  script_argument <- grep(
    "^--file=", commandArgs(trailingOnly = FALSE), value = TRUE
  )[[1L]]
  script_dir <- dirname(normalizePath(sub("^--file=", "", script_argument)))
  source(file.path(script_dir, "validate_mpjlcmm_posterior.R"))
  report_progress("validation", "比较重构纵向后验与内置pprobY")
  posterior_validation <- validate_training_longitudinal_posterior(
    model, updated_models, wide, features, patients$subject_id,
    function(detail) report_progress("validation", detail)
  )
  report_progress("validation", sprintf(
    "验证通过：max_abs_error=%.3g，class_agreement=%.6f",
    posterior_validation$max_absolute_error,
    posterior_validation$class_agreement_rate
  ))
}

# =============================================================================
# 9. 模型训练：剥离患者级内容并保存聚合产物
# =============================================================================
report_progress("fit", "剥离训练患者级输出并保存模型")
# 外部预测只需模型参数；剥离训练患者级posterior、残差与随机效应。
strip_patient_outputs <- function(value) {
  for (field in c("pprob", "pprobY", "pred", "predRE", "predRE_Y", "data")) {
    value[[field]] <- NULL
  }
  value
}
updated_models <- lapply(updated_models, strip_patient_outputs)
model <- strip_patient_outputs(model)
model_path <- normalizePath(config$model_path, mustWork = FALSE)
dir.create(dirname(model_path), recursive = TRUE, showWarnings = FALSE)
saveRDS(list(
  model = model,
  longitudinal_models = updated_models,
  features = features
), model_path)

result <- list(
  format_version = 1L,
  method = "lcmm::mpjlcmm",
  n_patients = nrow(patients),
  n_observations = nrow(observations),
  n_features = nrow(features),
  n_clusters = n_clusters,
  convergence = model$conv,
  log_likelihood = model$loglik,
  bic = model$BIC,
  model_path = model_path,
  package_versions = as.list(vapply(required_packages,
    function(package) as.character(utils::packageVersion(package)), character(1L)))
)
if (!is.null(posterior_validation)) {
  result$posterior_validation <- posterior_validation
}
jsonlite::write_json(result, result_path, auto_unbox = TRUE, pretty = TRUE)
report_progress("fit", sprintf("训练完成并保存：%s", model_path))
