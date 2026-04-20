# =============================================================================
# CRISP-DM: DATA PREPARATION
# German Credit — EDA, train/test split, encoding, scaling
# Run AFTER german_credit_wrangle_JT.R
# Objects expected: outputs/german_credit_clean.rds
# =============================================================================

library(dplyr)
library(caret)
library(fastDummies)
library(ggplot2)
library(reshape2)

dir.create("outputs", showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. LOAD CLEAN DATA FROM RDS
# -----------------------------------------------------------------------------

gc_df <- readRDS("outputs/german_credit_clean.rds")

stopifnot(levels(gc_df$other_debtors)[1] == "None")
stopifnot(levels(gc_df$other_installment_plans)[1] == "None")

cat("Loaded:", nrow(gc_df), "rows,", ncol(gc_df), "columns\n")
cat("Bad credit rate:", round(mean(gc_df$target_bad) * 100, 1), "%\n\n")

str(gc_df)
summary(gc_df)
dim(gc_df)

# -----------------------------------------------------------------------------
# 2. DEFINE FEATURE GROUPS
# -----------------------------------------------------------------------------

ordered_feats <- c(
  "account_status",
  "credit_history",
  "savings",
  "employment_duration",
  "property",
  "occupation",
  "installment_rate",
  "residence_since",
  "existing_credits",
  "people_liable"
)

unordered_feats <- c(
  "loan_purpose",
  "marital_status",
  "other_debtors",
  "other_installment_plans",
  "housing",
  "telephone",
  "foreign_worker"
)

numeric_feats <- c(
  "loan_duration_mths",
  "credit_amount",
  "age_yrs"
)

target_col <- "target_bad"

keep_cols <- c(ordered_feats, unordered_feats, numeric_feats, target_col)
gc_model <- gc_df[, keep_cols]

table(gc_df$target_bad)

# =============================================================================
# NUMERIC DISTRIBUTIONS HISTOGRAMS
# =============================================================================

num_vars <- c("loan_duration_mths", "credit_amount", "age_yrs")

for (v in num_vars) {
  p <- ggplot(gc_df, aes(x = .data[[v]], fill = target)) +
    geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
    scale_fill_manual(values = c("Good" = "#1D9E75", "Bad" = "#D85A30")) +
    theme_minimal() +
    labs(title = v)

  print(p)

  ggsave(
    filename = paste0("outputs/hist_", v, ".png"),
    plot = p,
    width = 7,
    height = 5,
    dpi = 300
  )
}

prop.table(table(gc_df$target_bad))
summary(gc_df$target_bad)

# =============================================================================
# CATEGORICAL VS TARGET (bad rate per level)
# =============================================================================

cat_vars <- c("account_status", "credit_history", "savings",
              "employment_duration", "loan_purpose")

for (v in cat_vars) {
  bad_rate_tbl <- gc_df %>%
    group_by(.data[[v]]) %>%
    summarise(n = n(), bad_rate = mean(target_bad), .groups = "drop") %>%
    arrange(desc(bad_rate))

  print(bad_rate_tbl)

  write.csv(
    bad_rate_tbl,
    file = paste0("outputs/bad_rate_", v, ".csv"),
    row.names = FALSE
  )
}

# =============================================================================
# CHI-SQUARE TESTS
# =============================================================================

cat_vars_chi <- c("account_status", "credit_history", "savings",
                  "employment_duration", "loan_purpose", "marital_status",
                  "other_debtors", "other_installment_plans", "housing",
                  "property", "telephone", "foreign_worker", "occupation")

chi_results <- lapply(cat_vars_chi, function(v) {
  tbl <- table(gc_df[[v]], gc_df$target)
  test <- chisq.test(tbl)
  data.frame(
    feature = v,
    chi_sq = round(test$statistic, 3),
    df = test$parameter,
    p_value = round(test$p.value, 4),
    sig = ifelse(test$p.value < 0.001, "***",
                 ifelse(test$p.value < 0.01, "**",
                        ifelse(test$p.value < 0.05, "*", "")))
  )
})

chi_df <- do.call(rbind, chi_results)
rownames(chi_df) <- NULL
chi_df <- chi_df[order(chi_df$p_value), ]
print(chi_df)
write.csv(chi_df, "outputs/chi_square_results.csv", row.names = FALSE)

# =============================================================================
# FISHER'S EXACT TEST FOR SPARSE VARIABLES
# =============================================================================

sparse_vars <- c("loan_purpose", "marital_status")

fisher_results <- lapply(sparse_vars, function(v) {
  tbl <- table(gc_df[[v]], gc_df$target)
  test <- fisher.test(tbl, simulate.p.value = TRUE, B = 10000)
  data.frame(
    feature = v,
    method = "Fisher (Monte Carlo)",
    p_value = round(test$p.value, 4),
    sig = ifelse(test$p.value < 0.001, "***",
                 ifelse(test$p.value < 0.01, "**",
                        ifelse(test$p.value < 0.05, "*", "")))
  )
})

fisher_df <- do.call(rbind, fisher_results)
print(fisher_df)
write.csv(fisher_df, "outputs/fisher_results.csv", row.names = FALSE)

# =============================================================================
# CORRELATION MATRIX
# =============================================================================

cor_vars <- c("loan_duration_mths", "credit_amount", "age_yrs", "target_bad")
cor_matrix <- cor(gc_df[, cor_vars], method = "pearson")
cor_matrix_rounded <- round(cor_matrix, 3)
print(cor_matrix_rounded)
write.csv(cor_matrix_rounded, "outputs/correlation_matrix.csv", row.names = TRUE)

cor_melted <- melt(cor_matrix_rounded)

label_map <- c(
  "loan_duration_mths" = "Loan Duration",
  "credit_amount" = "Credit Amount",
  "age_yrs" = "Age",
  "target_bad" = "Default (target)"
)

cor_melted$Var1 <- label_map[as.character(cor_melted$Var1)]
cor_melted$Var2 <- label_map[as.character(cor_melted$Var2)]

p_cor <- ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.3f", value)),
            size = 4.5, fontface = "bold") +
  scale_fill_gradient2(
    low = "#D85A30",
    mid = "white",
    high = "#1D9E75",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Pearson r"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title = element_blank(),
    axis.text = element_text(face = "bold"),
    panel.grid = element_blank(),
    legend.position = "right"
  ) +
  labs(
    title = "Pearson correlation matrix - numeric features",
    subtitle = "German Credit dataset (n = 1,000)"
  )

print(p_cor)

ggsave(
  "outputs/correlation_heatmap.png",
  plot = p_cor,
  width = 8,
  height = 6,
  dpi = 300
)

# -----------------------------------------------------------------------------
# 3. STRATIFIED TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------

set.seed(42)
train_idx <- createDataPartition(
  y = gc_model[[target_col]],
  p = 0.70,
  list = FALSE
)

train_raw <- gc_model[train_idx, ]
test_raw <- gc_model[-train_idx, ]

cat("Train rows:", nrow(train_raw), "| Bad rate:",
    round(mean(train_raw$target_bad) * 100, 1), "%\n")
cat("Test rows: ", nrow(test_raw), "| Bad rate:",
    round(mean(test_raw$target_bad) * 100, 1), "%\n\n")

# -----------------------------------------------------------------------------
# 4. ORDINAL INTEGER ENCODING
# -----------------------------------------------------------------------------

encode_ordinal <- function(df, cols) {
  for (col in cols) {
    df[[col]] <- as.integer(df[[col]])
  }
  df
}

train_enc <- encode_ordinal(train_raw, ordered_feats)
test_enc <- encode_ordinal(test_raw, ordered_feats)

cat("Ordinal encoding done:", length(ordered_feats), "features\n")

# -----------------------------------------------------------------------------
# 5. ONE-HOT ENCODING
# -----------------------------------------------------------------------------

train_enc <- dummy_cols(
  train_enc,
  select_columns = unordered_feats,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

test_enc <- dummy_cols(
  test_enc,
  select_columns = unordered_feats,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

missing_cols <- setdiff(names(train_enc), names(test_enc))
for (col in missing_cols) test_enc[[col]] <- 0L
test_enc <- test_enc[, names(train_enc)]

cat("One-hot encoding done:", length(unordered_feats),
    "features →", sum(grepl(paste(unordered_feats, collapse = "|"), names(train_enc))),
    "dummy columns\n")

# -----------------------------------------------------------------------------
# 6. NUMERIC TRANSFORM + SCALE
# -----------------------------------------------------------------------------

for (col in numeric_feats) {
  train_enc[[col]] <- log1p(train_enc[[col]])
  test_enc[[col]] <- log1p(test_enc[[col]])
}

train_mins <- sapply(train_enc[, numeric_feats], min)
train_maxs <- sapply(train_enc[, numeric_feats], max)
train_ranges <- train_maxs - train_mins

minmax_scale <- function(df, cols, mins, ranges) {
  for (col in cols) {
    df[[col]] <- (df[[col]] - mins[col]) / ranges[col]
  }
  df
}

train_enc <- minmax_scale(train_enc, numeric_feats, train_mins, train_ranges)
test_enc <- minmax_scale(test_enc, numeric_feats, train_mins, train_ranges)

cat("Numeric scaling done:", length(numeric_feats), "features → [0, 1] range\n\n")

# -----------------------------------------------------------------------------
# 6b. REMOVE NEAR-ZERO VARIANCE FEATURES
# -----------------------------------------------------------------------------

DEBUG_MODE <- FALSE

predictor_cols <- setdiff(names(train_enc), target_col)
nzv <- nearZeroVar(train_enc[, predictor_cols, drop = FALSE])

if (length(nzv) > 0) {
  nzv_names <- predictor_cols[nzv]

  cat("Removing", length(nzv_names), "near-zero variance predictor features:\n")
  print(nzv_names)

  write.csv(
    data.frame(feature = nzv_names),
    "outputs/near_zero_variance_features.csv",
    row.names = FALSE
  )

  if (DEBUG_MODE) {
    nzv_metrics <- nearZeroVar(
      train_enc[, predictor_cols, drop = FALSE],
      saveMetrics = TRUE
    )
    print(nzv_metrics[nzv_metrics$nzv == TRUE, ])
  }

  train_enc <- train_enc[, !names(train_enc) %in% nzv_names, drop = FALSE]
  test_enc <- test_enc[, !names(test_enc) %in% nzv_names, drop = FALSE]
} else {
  write.csv(
    data.frame(feature = character(0)),
    "outputs/near_zero_variance_features.csv",
    row.names = FALSE
  )
}

# -----------------------------------------------------------------------------
# 7. SEPARATE FEATURES AND TARGET
# -----------------------------------------------------------------------------

feature_cols <- setdiff(names(train_enc), target_col)

X_train <- train_enc[, feature_cols]
y_train <- factor(train_enc[[target_col]], levels = c(0, 1), labels = c("Good", "Bad"))

X_test <- test_enc[, feature_cols]
y_test <- factor(test_enc[[target_col]], levels = c(0, 1), labels = c("Good", "Bad"))

cat("Final feature matrix dimensions:\n")
cat("  X_train:", nrow(X_train), "rows x", ncol(X_train), "columns\n")
cat("  X_test: ", nrow(X_test), "rows x", ncol(X_test), "columns\n")
cat("  y_train: Good =", sum(y_train == "Good"), "| Bad =", sum(y_train == "Bad"), "\n")
cat("  y_test:  Good =", sum(y_test == "Good"), "| Bad =", sum(y_test == "Bad"), "\n\n")

# -----------------------------------------------------------------------------
# 8. FINAL CHECKS
# -----------------------------------------------------------------------------

stopifnot(!anyNA(X_train))
stopifnot(!anyNA(X_test))
stopifnot(all(sapply(X_train, is.numeric)))
stopifnot(all(sapply(X_test, is.numeric)))
stopifnot(identical(names(X_train), names(X_test)))

cat("All checks passed. Ready for modelling.\n\n")
cat("Features going into the model (", ncol(X_train), "):\n", sep = "")
cat(paste(names(X_train), collapse = ", "), "\n\n")