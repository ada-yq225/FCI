#!/usr/bin/env Rscript

# Run the CRAN pcalg implementation of FCI+ on the committed STAR panels.
#
# The application deliberately supplies a custom G-square test that mirrors
# src/fci_engine/ci/discrete.py. In particular, it does not use pcalg's
# disCItest n.min shortcut, which can declare high-dimensional tables
# independent before calculating the statistic. This keeps the statistical
# decision rule aligned across the Python and R implementations.

suppressPackageStartupMessages(library(pcalg))

parse_args <- function(arguments) {
  parsed <- list(
    input_dir = "case_studies/tennessee_star/data/processed",
    output_dir = "case_studies/tennessee_star/output",
    alpha = 0.05,
    benchmark_repeats = 3L,
    order_audit = TRUE
  )
  index <- 1L
  while (index <= length(arguments)) {
    flag <- arguments[[index]]
    if (flag == "--no-order-audit") {
      parsed$order_audit <- FALSE
      index <- index + 1L
      next
    }
    if (index == length(arguments)) {
      stop(paste("Missing value for", flag))
    }
    value <- arguments[[index + 1L]]
    if (flag == "--input-dir") {
      parsed$input_dir <- value
    } else if (flag == "--output-dir") {
      parsed$output_dir <- value
    } else if (flag == "--alpha") {
      parsed$alpha <- as.numeric(value)
    } else if (flag == "--benchmark-repeats") {
      parsed$benchmark_repeats <- as.integer(value)
    } else {
      stop(paste("Unknown argument:", flag))
    }
    index <- index + 2L
  }
  if (
    is.na(parsed$alpha) ||
    parsed$alpha <= 0 ||
    parsed$alpha >= 1
  ) {
    stop("--alpha must be between zero and one")
  }
  if (
    is.na(parsed$benchmark_repeats) ||
    parsed$benchmark_repeats < 1L
  ) {
    stop("--benchmark-repeats must be positive")
  }
  parsed
}

compact_g_square <- function(table) {
  row_keep <- rowSums(table) > 0
  col_keep <- colSums(table) > 0
  compact <- table[row_keep, col_keep, drop = FALSE]
  if (nrow(compact) < 2L || ncol(compact) < 2L) {
    return(c(statistic = 0, degrees_freedom = 0))
  }

  total <- sum(compact)
  expected <- outer(rowSums(compact), colSums(compact)) / total
  valid <- compact > 0 & expected > 0
  statistic <- 2 * sum(compact[valid] * log(compact[valid] / expected[valid]))
  degrees_freedom <- (nrow(compact) - 1L) * (ncol(compact) - 1L)
  c(statistic = statistic, degrees_freedom = degrees_freedom)
}

python_equivalent_g_square <- function(x, y, S, suffStat) {
  data <- suffStat$dm
  x_levels <- 0:(suffStat$nlev[[x]] - 1L)
  y_levels <- 0:(suffStat$nlev[[y]] - 1L)

  if (length(S) == 0L) {
    strata <- list(seq_len(nrow(data)))
  } else {
    conditioning <- data[, S, drop = FALSE]
    keys <- apply(conditioning, 1L, paste, collapse = "\r")
    strata <- split(seq_len(nrow(data)), keys, drop = TRUE)
  }

  statistic <- 0
  degrees_freedom <- 0L
  for (rows in strata) {
    contingency <- table(
      factor(data[rows, x], levels = x_levels),
      factor(data[rows, y], levels = y_levels)
    )
    part <- compact_g_square(contingency)
    statistic <- statistic + part[["statistic"]]
    degrees_freedom <- degrees_freedom + part[["degrees_freedom"]]
  }

  if (degrees_freedom <= 0L) {
    return(1)
  }
  pchisq(statistic, df = degrees_freedom, lower.tail = FALSE)
}

fit_pcalg_fci_plus <- function(data, alpha) {
  counter <- 0L
  counted_test <- function(x, y, S, suffStat) {
    counter <<- counter + 1L
    python_equivalent_g_square(x, y, S, suffStat)
  }
  sufficient_statistics <- list(
    dm = data,
    nlev = apply(data, 2L, max) + 1L
  )
  warnings <- character()
  elapsed <- system.time(
    fit <- withCallingHandlers(
      fciPlus(
        suffStat = sufficient_statistics,
        indepTest = counted_test,
        alpha = alpha,
        labels = colnames(data),
        verbose = FALSE,
        selectionBias = TRUE
      ),
      warning = function(condition) {
        warnings <<- c(warnings, conditionMessage(condition))
        invokeRestart("muffleWarning")
      }
    )
  )[["elapsed"]]
  list(
    fit = fit,
    ci_tests = counter,
    elapsed_seconds = as.numeric(elapsed),
    warnings = unique(warnings)
  )
}

endpoint_name <- function(mark) {
  marks <- c("1" = "CIRCLE", "2" = "ARROW", "3" = "TAIL")
  unname(marks[[as.character(mark)]])
}

edge_notation <- function(x, y, endpoint_x, endpoint_y) {
  symbols <- c(
    "CIRCLE:CIRCLE" = "o-o",
    "CIRCLE:ARROW" = "o->",
    "ARROW:CIRCLE" = "<-o",
    "TAIL:ARROW" = "-->",
    "ARROW:TAIL" = "<--",
    "ARROW:ARROW" = "<->",
    "TAIL:TAIL" = "---",
    "CIRCLE:TAIL" = "o--",
    "TAIL:CIRCLE" = "--o"
  )
  key <- paste(endpoint_x, endpoint_y, sep = ":")
  paste(x, symbols[[key]], y)
}

edge_frame <- function(fit, canonical_labels, panel) {
  matrix <- fit@amat
  labels <- colnames(matrix)
  order_index <- setNames(seq_along(canonical_labels), canonical_labels)
  rows <- list()
  row_index <- 1L
  if (ncol(matrix) > 1L) {
    for (i in seq_len(ncol(matrix) - 1L)) {
      for (j in seq.int(i + 1L, ncol(matrix))) {
        if (matrix[i, j] == 0L && matrix[j, i] == 0L) {
          next
        }
        x <- labels[[i]]
        y <- labels[[j]]
        endpoint_x <- endpoint_name(matrix[j, i])
        endpoint_y <- endpoint_name(matrix[i, j])
        if (order_index[[x]] > order_index[[y]]) {
          temporary_label <- x
          x <- y
          y <- temporary_label
          temporary_endpoint <- endpoint_x
          endpoint_x <- endpoint_y
          endpoint_y <- temporary_endpoint
        }
        rows[[row_index]] <- data.frame(
          panel = panel,
          algorithm = "pcalg_fci_plus",
          x = x,
          y = y,
          endpoint_x = endpoint_x,
          endpoint_y = endpoint_y,
          edge = edge_notation(x, y, endpoint_x, endpoint_y),
          stringsAsFactors = FALSE
        )
        row_index <- row_index + 1L
      }
    }
  }
  if (length(rows) == 0L) {
    return(data.frame(
      panel = character(),
      algorithm = character(),
      x = character(),
      y = character(),
      endpoint_x = character(),
      endpoint_y = character(),
      edge = character(),
      stringsAsFactors = FALSE
    ))
  }
  do.call(rbind, rows)
}

edge_signature <- function(edges) {
  if (nrow(edges) == 0L) {
    return(character())
  }
  sort(paste(
    edges$x,
    edges$y,
    edges$endpoint_x,
    edges$endpoint_y,
    sep = "|"
  ))
}

skeleton_signature <- function(edges) {
  if (nrow(edges) == 0L) {
    return(character())
  }
  sort(paste(edges$x, edges$y, sep = "|"))
}

jaccard <- function(left, right) {
  union_values <- union(left, right)
  if (length(union_values) == 0L) {
    return(1)
  }
  length(intersect(left, right)) / length(union_values)
}

run_panel <- function(panel, path, alpha, benchmark_repeats, order_audit) {
  frame <- read.csv(path, check.names = FALSE)
  data <- as.matrix(frame)
  storage.mode(data) <- "integer"
  canonical_labels <- colnames(data)

  fits <- lapply(
    seq_len(benchmark_repeats),
    function(unused) fit_pcalg_fci_plus(data, alpha)
  )
  primary <- fits[[1L]]
  primary_edges <- edge_frame(primary$fit, canonical_labels, panel)
  all_warnings <- unique(unlist(lapply(fits, function(item) item$warnings)))

  audit_rows <- list()
  if (order_audit) {
    primary_edge_signature <- edge_signature(primary_edges)
    primary_skeleton <- skeleton_signature(primary_edges)
    for (shift in seq_along(canonical_labels) - 1L) {
      if (shift == 0L) {
        audit_fit <- primary
        audit_edges <- primary_edges
      } else {
        permutation <- c(
          seq.int(shift + 1L, length(canonical_labels)),
          seq_len(shift)
        )
        audit_data <- data[, permutation, drop = FALSE]
        audit_fit <- fit_pcalg_fci_plus(audit_data, alpha)
        audit_edges <- edge_frame(audit_fit$fit, canonical_labels, panel)
        all_warnings <- unique(c(all_warnings, audit_fit$warnings))
      }
      audit_rows[[shift + 1L]] <- data.frame(
        panel = panel,
        ordering = paste0("cyclic_shift_", shift),
        first_variable = colnames(
          data[, c(
            seq.int(shift + 1L, length(canonical_labels)),
            if (shift > 0L) seq_len(shift) else integer()
          ), drop = FALSE]
        )[[1L]],
        exact_pag_match = identical(
          edge_signature(audit_edges),
          primary_edge_signature
        ),
        skeleton_jaccard = jaccard(
          skeleton_signature(audit_edges),
          primary_skeleton
        ),
        edges = nrow(audit_edges),
        ci_tests = audit_fit$ci_tests,
        elapsed_seconds = audit_fit$elapsed_seconds,
        stringsAsFactors = FALSE
      )
    }
  } else {
    audit_rows[[1L]] <- data.frame(
      panel = panel,
      ordering = "not_run",
      first_variable = canonical_labels[[1L]],
      exact_pag_match = NA,
      skeleton_jaccard = NA_real_,
      edges = nrow(primary_edges),
      ci_tests = primary$ci_tests,
      elapsed_seconds = primary$elapsed_seconds,
      stringsAsFactors = FALSE
    )
  }
  audit <- do.call(rbind, audit_rows)

  run <- data.frame(
    panel = panel,
    algorithm = "pcalg_fci_plus",
    samples = nrow(data),
    nodes = ncol(data),
    edges = nrow(primary_edges),
    ci_tests = primary$ci_tests,
    median_elapsed_seconds = median(vapply(
      fits,
      function(item) item$elapsed_seconds,
      numeric(1)
    )),
    elapsed_runs_seconds = paste(vapply(
      fits,
      function(item) item$elapsed_seconds,
      numeric(1)
    ), collapse = ";"),
    r_version = as.character(getRversion()),
    pcalg_version = as.character(packageVersion("pcalg")),
    ci_test = "Python-equivalent G-square",
    alpha = alpha,
    selection_bias = TRUE,
    orderings_checked = nrow(audit),
    exact_pag_match_rate = mean(audit$exact_pag_match, na.rm = TRUE),
    mean_skeleton_jaccard = mean(audit$skeleton_jaccard, na.rm = TRUE),
    warning_count = length(all_warnings),
    warnings = paste(all_warnings, collapse = " | "),
    stringsAsFactors = FALSE
  )
  list(run = run, edges = primary_edges, audit = audit)
}

arguments <- parse_args(commandArgs(trailingOnly = TRUE))
dir.create(arguments$output_dir, recursive = TRUE, showWarnings = FALSE)

panel_files <- c(
  attrition = "attrition_panel.csv",
  longitudinal = "longitudinal_panel.csv",
  focused_treatment = "focused_treatment_panel.csv"
)
results <- lapply(names(panel_files), function(panel) {
  run_panel(
    panel = panel,
    path = file.path(arguments$input_dir, panel_files[[panel]]),
    alpha = arguments$alpha,
    benchmark_repeats = arguments$benchmark_repeats,
    order_audit = arguments$order_audit
  )
})

runs <- do.call(rbind, lapply(results, function(item) item$run))
edges <- do.call(rbind, lapply(results, function(item) item$edges))
audit <- do.call(rbind, lapply(results, function(item) item$audit))

write.csv(
  runs,
  file.path(arguments$output_dir, "star_pcalg_runs.csv"),
  row.names = FALSE
)
write.csv(
  edges,
  file.path(arguments$output_dir, "star_pcalg_edges.csv"),
  row.names = FALSE
)
write.csv(
  audit,
  file.path(arguments$output_dir, "star_pcalg_order_audit.csv"),
  row.names = FALSE
)

cat("R version:", as.character(getRversion()), "\n")
cat("pcalg version:", as.character(packageVersion("pcalg")), "\n")
cat("runs:", normalizePath(
  file.path(arguments$output_dir, "star_pcalg_runs.csv")
), "\n")
cat("edges:", normalizePath(
  file.path(arguments$output_dir, "star_pcalg_edges.csv")
), "\n")
cat("order audit:", normalizePath(
  file.path(arguments$output_dir, "star_pcalg_order_audit.csv")
), "\n")
