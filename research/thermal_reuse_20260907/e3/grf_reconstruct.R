#!/usr/bin/env Rscript
# Restricted Yoo-style reconstruction core. QA, LASSO, imputation, support and
# projection belong to the caller. This script never searches history columns.

GRF_REQUIRED_VERSION <- "2.0.2"
GRF_SEED <- 20260907L
GRF_THREADS <- 2L
GRF_TREES <- 2000L
OFFICIAL_LL_LAMBDA_PATH <- c(0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1, 10)


assert_grf_runtime <- function() {
  if (!requireNamespace("grf", quietly = TRUE)) stop("grf 2.0.2 is not installed")
  version <- as.character(utils::packageVersion("grf"))
  if (!identical(version, GRF_REQUIRED_VERSION)) {
    stop(paste("Exact grf version required:", GRF_REQUIRED_VERSION, "found", version))
  }
  version
}


validate_matrices <- function(train_x, train_y, predict_x, columns) {
  if (!is.matrix(train_x) || !is.matrix(predict_x) ||
      !is.numeric(train_x) || !is.numeric(predict_x)) stop("X inputs must be numeric matrices")
  if (nrow(train_x) < 2L || nrow(predict_x) < 1L || ncol(train_x) < 1L) {
    stop("At least two training rows, one prediction row and one feature are required")
  }
  if (!is.numeric(train_y) || !is.null(dim(train_y)) || length(train_y) != nrow(train_x)) {
    stop("Y must be one numeric vector aligned to coarse training rows")
  }
  if (any(!is.finite(train_x)) || any(!is.finite(train_y)) || any(!is.finite(predict_x))) {
    stop("All input values must already be finite; no implicit imputation is performed")
  }
  names_x <- colnames(train_x)
  if (is.null(names_x) || anyNA(names_x) || any(names_x == "") || anyDuplicated(names_x)) {
    stop("Training columns must have unique explicit names")
  }
  if (!identical(names_x, colnames(predict_x))) stop("Fine and coarse X column order/names must match exactly")
  if (!is.data.frame(columns) || !identical(names(columns), c("name", "role"))) {
    stop("columns.csv must contain exactly name,role in that order")
  }
  if (!identical(as.character(columns$name), names_x) || anyNA(columns$role)) {
    stop("The explicit column-role list must match every X column in order")
  }
  roles <- as.character(columns$role)
  if (any(!roles %in% c("auxiliary", "thermal"))) stop("Roles must be auxiliary or thermal")
  if (!any(roles == "auxiliary")) stop("S1 requires at least one explicit auxiliary feature")
  list(auxiliary = which(roles == "auxiliary"), thermal = which(roles == "thermal"))
}


forest_arguments <- function(feature_count) {
  # Explicit values of the archived grf 2.0.2 defaults, plus fixed seed/threads.
  list(num.trees = GRF_TREES, clusters = NULL, equalize.cluster.weights = FALSE,
       sample.fraction = 0.5,
       mtry = min(ceiling(sqrt(feature_count) + 20), feature_count),
       min.node.size = 5L, honesty = TRUE, honesty.fraction = 0.5,
       honesty.prune.leaves = TRUE, alpha = 0.05, imbalance.penalty = 0,
       ci.group.size = 2L, tune.parameters = "none", tune.num.trees = 50L,
       tune.num.reps = 100L, tune.num.draws = 1000L,
       num.threads = GRF_THREADS, seed = GRF_SEED)
}


fit_rf <- function(x, y) {
  args <- c(list(X = x, Y = y), forest_arguments(ncol(x)))
  args$compute.oob.predictions <- TRUE
  do.call(grf::regression_forest, args)
}


fit_llf <- function(x, y) {
  args <- c(list(X = x, Y = y), forest_arguments(ncol(x)),
            list(enable.ll.split = FALSE, ll.split.weight.penalty = FALSE,
                 ll.split.lambda = 0.1, ll.split.variables = NULL,
                 ll.split.cutoff = NULL))
  do.call(grf::ll_regression_forest, args)
}


rf_predict <- function(forest, x) {
  as.numeric(stats::predict(forest, newdata = x, num.threads = GRF_THREADS,
                           estimate.variance = FALSE)$predictions)
}


reconstruct_grf <- function(train_x, train_y, predict_x, columns, keep_models = FALSE) {
  index <- validate_matrices(train_x, train_y, predict_x, columns)
  version <- assert_grf_runtime()
  set.seed(GRF_SEED)
  aux_x <- train_x[, index$auxiliary, drop = FALSE]
  aux_fine <- predict_x[, index$auxiliary, drop = FALSE]
  s1_model <- fit_rf(aux_x, train_y)
  s1 <- rf_predict(s1_model, aux_fine)
  no_thermal <- length(index$thermal) == 0L
  lambda <- NULL
  s2_model <- s3_model <- NULL
  if (no_thermal) {
    # Exact same vector, not two independently fitted approximations of S1.
    s2 <- s1
    s3 <- s1
  } else {
    s2_model <- fit_rf(train_x, train_y)
    s2 <- rf_predict(s2_model, predict_x)
    s3_model <- fit_llf(train_x, train_y)
    pred <- stats::predict(s3_model, newdata = predict_x,
                          linear.correction.variables = index$thermal,
                          ll.lambda = NULL, ll.weight.penalty = FALSE,
                          num.threads = GRF_THREADS, estimate.variance = FALSE)
    s3 <- as.numeric(pred$predictions)
    lambda <- unique(as.numeric(pred$ll.lambda))
    if (length(lambda) != 1L || !is.finite(lambda) || !lambda %in% OFFICIAL_LL_LAMBDA_PATH) {
      stop("LLF did not return exactly one value from the archived default OOB lambda path")
    }
  }
  predictions <- data.frame(prediction_index = seq_len(nrow(predict_x)) - 1L,
                            s1_rf = s1, s2_rf = s2, s3_llf = s3)
  if (any(vapply(predictions[-1L], length, integer(1)) != nrow(predict_x)) ||
      any(!is.finite(as.matrix(predictions[-1L])))) stop("Nonfinite or incomplete forest predictions")
  roles <- setNames(as.character(columns$role), as.character(columns$name))
  metadata <- list(schema = "e3-restricted-grf-reconstruction-v1",
    grf_version = version, R_version = R.version.string,
    training_rows = nrow(train_x), prediction_rows = nrow(predict_x),
    full_column_order = as.list(colnames(train_x)),
    auxiliary_column_names = as.list(colnames(train_x)[index$auxiliary]),
    selected_thermal_column_names = as.list(colnames(train_x)[index$thermal]),
    selected_thermal_R_indices = as.list(index$thermal),
    selected_thermal_full_X_indices_zero_based = as.list(index$thermal - 1L),
    column_roles = as.list(roles), seed = GRF_SEED, num_threads = GRF_THREADS,
    s1_training_parameters = forest_arguments(length(index$auxiliary)),
    s2_s3_training_parameters = if (no_thermal) NULL else forest_arguments(ncol(train_x)),
    RF_compute_oob_predictions = TRUE,
    S3_enable_ll_split = FALSE, S3_ll_split_lambda = 0.1,
    S3_ll_split_weight_penalty = FALSE, S3_linear_correction_only_selected_thermal = TRUE,
    S3_requested_ll_lambda = NULL, S3_final_ll_lambda = lambda,
    S3_ll_weight_penalty = FALSE, S3_default_OOB_lambda_path = OFFICIAL_LL_LAMBDA_PATH,
    S3_lambda_selection = if (no_thermal) "not_run_no_thermal" else "official_grf_2.0.2_OOB_MSE_minimum",
    no_selected_thermal_fallback = no_thermal,
    fallback_rule = "S2_and_S3_copy_the_same_S1_prediction_when_no_thermal_columns",
    forest_fits_performed = if (no_thermal) list("S1") else list("S1", "S2", "S3"),
    no_feature_search_or_LASSO_performed = TRUE,
    no_QA_filter_imputation_scaling_or_projection_performed = TRUE,
    input_Y_role = "caller_supplied_coarse_training_response_only",
    prediction_row_order = "unchanged_from_predict_x; prediction_index_is_zero_based",
    runtime_session_info = capture.output(utils::sessionInfo()))
  result <- list(predictions = predictions, metadata = metadata)
  if (keep_models) result$models <- list(s1 = s1_model, s2 = s2_model, s3 = s3_model)
  result
}


# Small serializer for our controlled metadata types; avoids installing another
# R dependency. Unsupported classes and non-finite numeric values are rejected.
json_string <- function(x) {
  if (length(x) != 1L || is.na(x)) stop("Expected a single nonmissing JSON string")
  cp <- utf8ToInt(enc2utf8(x))
  value <- vapply(cp, function(c) {
    if (c == 34L) return('\\"')
    if (c == 92L) return('\\\\')
    if (c < 32L) return(sprintf("\\u%04x", c))
    intToUtf8(c)
  }, character(1))
  paste0('"', paste0(value, collapse = ""), '"')
}


metadata_json <- function(x) {
  if (is.null(x)) return("null")
  if (is.list(x)) {
    keys <- names(x)
    if (!is.null(keys)) {
      if (anyNA(keys) || any(keys == "") || anyDuplicated(keys)) stop("Invalid JSON object names")
      fields <- vapply(seq_along(x), function(i) paste0(json_string(keys[i]), ":", metadata_json(x[[i]])), character(1))
      return(paste0("{", paste0(fields, collapse = ","), "}"))
    }
    return(paste0("[", paste0(vapply(x, metadata_json, character(1)), collapse = ","), "]"))
  }
  if (!is.atomic(x) || !is.null(dim(x))) stop("Unsupported metadata JSON type")
  if (length(x) != 1L) return(paste0("[", paste0(vapply(as.list(x), metadata_json, character(1)), collapse = ","), "]"))
  if (is.character(x)) return(json_string(x))
  if (is.logical(x) && !is.na(x)) return(if (x) "true" else "false")
  if (is.numeric(x) && is.finite(x)) return(sprintf("%.17g", x))
  stop("Unsupported or nonfinite metadata scalar")
}


read_numeric_csv <- function(path) {
  frame <- utils::read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
  if (!ncol(frame) || any(!vapply(frame, is.numeric, logical(1)))) stop("Every X/Y CSV field must be numeric")
  as.matrix(frame)
}


parse_arguments <- function(args) {
  allowed <- c("train-x", "train-y", "predict-x", "columns", "output")
  if (length(args) != 2L * length(allowed)) stop("Required: --train-x CSV --train-y CSV --predict-x CSV --columns CSV --output NEW_DIR")
  keys <- sub("^--", "", args[seq.int(1L, length(args), by = 2L)])
  if (!setequal(keys, allowed) || anyDuplicated(keys) ||
      any(!startsWith(args[seq.int(1L, length(args), by = 2L)], "--"))) stop("Unknown, duplicated or missing CLI flags")
  setNames(as.list(args[seq.int(2L, length(args), by = 2L)]), keys)
}


main <- function() {
  args <- parse_arguments(commandArgs(trailingOnly = TRUE))
  inputs <- unlist(args[c("train-x", "train-y", "predict-x", "columns")], use.names = TRUE)
  if (any(!file.exists(inputs))) stop("A required input file is missing")
  output <- args$output
  if (file.exists(output)) stop("Output already exists; immutable run outputs must not be replaced")
  input_md5 <- tools::md5sum(inputs)
  train_x <- read_numeric_csv(args[["train-x"]])
  train_y <- read_numeric_csv(args[["train-y"]])
  if (ncol(train_y) != 1L) stop("train-y must contain exactly one response column")
  predict_x <- read_numeric_csv(args[["predict-x"]])
  columns <- utils::read.csv(args$columns, check.names = FALSE, stringsAsFactors = FALSE)
  result <- reconstruct_grf(train_x, as.numeric(train_y[, 1L]), predict_x, columns)
  if (!identical(input_md5, tools::md5sum(inputs))) stop("Input file bytes changed during the run")
  if (!dir.create(output, recursive = TRUE)) stop("Unable to create a fresh output directory")
  result$metadata$input_files <- as.list(inputs)
  result$metadata$input_file_md5 <- as.list(setNames(unname(input_md5), names(inputs)))
  result$metadata$digest_note <- "MD5 is an internal input-change check; the parent E3 freeze must additionally bind SHA256."
  result$metadata$completed_utc <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6Z", tz = "UTC")
  old <- options(digits = 17)
  on.exit(options(old), add = TRUE)
  utils::write.csv(result$predictions, file.path(output, "predictions.csv"), row.names = FALSE, na = "")
  saveRDS(result$metadata, file.path(output, "run_metadata.rds"), version = 2)
  writeLines(metadata_json(result$metadata), file.path(output, "run_metadata.json"), useBytes = TRUE)
  writeLines(result$metadata$runtime_session_info, file.path(output, "sessionInfo.txt"), useBytes = TRUE)
  parameters <- data.frame(parameter = c("grf_version", "num_trees", "seed", "num_threads",
                                        "S3_final_ll_lambda", "no_selected_thermal_fallback"),
                           value = c(GRF_REQUIRED_VERSION, GRF_TREES, GRF_SEED, GRF_THREADS,
                                     if (is.null(result$metadata$S3_final_ll_lambda)) "not_run" else result$metadata$S3_final_ll_lambda,
                                     result$metadata$no_selected_thermal_fallback))
  utils::write.csv(parameters, file.path(output, "parameters.csv"), row.names = FALSE)
  writeLines("complete", file.path(output, "COMPLETED"))
  cat("grf reconstruction saved; no QA, LASSO, fine-label scoring or projection performed.\n")
}


if (sys.nframe() == 0L) main()
