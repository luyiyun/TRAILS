# mpjlcmm多marker纵向后验的共享重构与训练集一致性验证。

reconstruct_longitudinal_posterior <- function(
    model, longitudinal_models, wide, features, subject_ids, report_progress) {
  n_clusters <- as.integer(model$ng)
  if (length(longitudinal_models) != nrow(features) ||
      model$N[[1L]] != n_clusters - 1L) {
    stop("模型不满足每个marker一个子模型且classmb仅含截距的后验契约")
  }
  prior_logits <- c(model$best[seq_len(n_clusters - 1L)], 0)
  priors <- exp(prior_logits - max(prior_logits))
  priors <- priors / sum(priors)
  log_scores <- matrix(
    log(priors), nrow = length(subject_ids), ncol = n_clusters, byrow = TRUE
  )
  probability_columns <- sprintf("prob%d", seq_len(n_clusters))

  # 条件于类别时各marker独立；完全缺失的marker对该患者作中性贡献。
  for (index in seq_along(longitudinal_models)) {
    marker <- sprintf("marker_%d", features$feature_id[[index]])
    report_progress(sprintf(
      "计算marker后验 %d/%d：%s",
      index, length(longitudinal_models), features$feature_name[[index]]
    ))
    if (!marker %in% names(wide)) {
      next
    }
    marker_data <- wide[!is.na(get(marker))]
    if (nrow(marker_data) == 0L) {
      next
    }
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
  posterior / rowSums(posterior)
}

validate_training_longitudinal_posterior <- function(
    model, longitudinal_models, wide, features, subject_ids,
    report_progress, tolerance = 1e-6) {
  reconstructed <- reconstruct_longitudinal_posterior(
    model, longitudinal_models, wide, features, subject_ids, report_progress
  )
  probability_columns <- sprintf("probY%d", seq_len(model$ng))
  official <- as.data.frame(model$pprobY)
  if (!all(c("class", probability_columns) %in% names(official))) {
    stop("mpjlcmm对象缺少内置纵向后验pprobY")
  }
  positions <- match(subject_ids, as.integer(official[[1L]]))
  if (anyNA(positions) || anyDuplicated(positions) > 0L) {
    stop("mpjlcmm内置pprobY的患者顺序无效")
  }
  expected <- as.matrix(official[positions, probability_columns])
  if (any(!is.finite(expected)) ||
      any(abs(rowSums(expected) - 1) > tolerance)) {
    stop("mpjlcmm内置pprobY包含无效概率")
  }
  absolute_error <- abs(reconstructed - expected)
  class_agreement <- mean(
    max.col(reconstructed, ties.method = "first") ==
      max.col(expected, ties.method = "first")
  )
  summary <- list(
    n_patients = length(subject_ids),
    tolerance = tolerance,
    max_absolute_error = max(absolute_error),
    mean_absolute_error = mean(absolute_error),
    class_agreement_rate = class_agreement
  )
  if (summary$max_absolute_error > tolerance || class_agreement != 1) {
    stop(sprintf(
      "重构后验与pprobY不一致：max_abs_error=%.3g，class_agreement=%.6f",
      summary$max_absolute_error, class_agreement
    ))
  }
  summary
}
