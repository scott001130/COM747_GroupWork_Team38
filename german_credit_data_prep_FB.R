# =============================================================================
# CRISP-DM: DATA PREPARATION 
# German Credit — EDA, train/test split, encoding, scaling
# Run AFTER german_credit_wrangle.R  (script 2)
# Objects expected: german_credit_clean.rds
# Code co-worked with chatGPT and Claude.ai
# =============================================================================

library(dplyr)
library(caret)      # for createDataPartition, preProcess
library(fastDummies) # for dummy_cols()
library(ggplot2)
library(reshape2)  # for melt()
#install.packages(c("caret", "fastDummies", "reshape2")) #if not already installed


# -----------------------------------------------------------------------------
# 1. LOAD CLEAN DATA FROM RDS (factors + ordering preserved)
# -----------------------------------------------------------------------------

gc_df <- readRDS("german_credit_clean.rds")

# Sanity check: "None" must be level [1] for both structural columns
stopifnot(levels(gc_df$other_debtors)[1]          == "None")
stopifnot(levels(gc_df$other_installment_plans)[1] == "None")

cat("Loaded:", nrow(gc_df), "rows,", ncol(gc_df), "columns\n")
cat("Bad credit rate:", round(mean(gc_df$target_bad) * 100, 1), "%\n\n")

# Check the data
str(gc_df)       # confirms factors, levels, and ordering are intact
summary(gc_df)   # quick data profile
dim(gc_df)       # should be 1000 rows × 22 columns

# -----------------------------------------------------------------------------
# 2. DEFINE FEATURE GROUPS
#    Ordered factors   → ordinal integer encoding (rank order is meaningful)
#    Unordered factors → one-hot encoding (no rank order)
#    Numerics          → log1p transform then min-max scale

#    The dataset has two target columns: target — factor: "Good" / "Bad" (for interpretability)
#    target_bad — integer: 1 = bad, 0 = good 
#    For most classification models (logistic regression, random forest, etc.) use target_bad. 
# -----------------------------------------------------------------------------

ordered_feats <- c(
  "account_status",       # 4 levels, low→high balance
  "credit_history",       # 5 levels, worst→best history
  "savings",              # 5 levels, unknown→>=1000 DM
  "employment_duration",  # 5 levels, unemployed→7+ years
  "property",             # 4 levels, none→real estate
  "occupation",           # 4 levels, unskilled→manager
  "installment_rate",     # 4 levels, >=35%→<20% of income
  "residence_since",      # 4 levels, <1yr→7+ years
  "existing_credits",     # 4 levels, 1→6+
  "people_liable"         # 2 levels, 0-2→3+
)

unordered_feats <- c(
  "loan_purpose",             # 10 levels — no meaningful rank
  "marital_status",           # 3 levels
  "other_debtors",            # 3 levels, ref = "None"
  "other_installment_plans",  # 3 levels, ref = "None"
  "housing",                  # 3 levels
  "telephone",                # 2 levels
  "foreign_worker"            # 2 levels
)

numeric_feats <- c(
  "loan_duration_mths",  # skew 1.09 — log transform
  "credit_amount",       # skew 1.95 — log transform
  "age_yrs"              # skew 1.02 — log transform
)

target_col <- "target_bad"

# Keep only modelling features + target
keep_cols <- c(ordered_feats, unordered_feats, numeric_feats, target_col)
gc_model <- gc_df[, keep_cols]


table(gc_df$target_bad) #frequency table of the values in the column target_bad - counts how many times each unique value appears

# =============================================================================
# NUMERIC DISTRIBUTIONS HISTOGRAMS (Good / Bad distribution)
# =============================================================================
library(ggplot2)

num_vars <- c("loan_duration_mths", "credit_amount", "age_yrs")

for (v in num_vars) {
  p <- ggplot(gc_df, aes(x = .data[[v]], fill = target)) +
    geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
    scale_fill_manual(values = c("Good" = "#1D9E75", "Bad" = "#D85A30")) +
    theme_minimal() +
    labs(title = v)
  
  print(p)
}

# check the class balance
prop.table(table(gc_df$target_bad))
summary(gc_df$target_bad)

# =============================================================================
# CATEGORICAL VS TARGET (bad rate per level)
# =============================================================================
cat_vars <- c("account_status", "credit_history", "savings",
              "employment_duration", "loan_purpose")

for (v in cat_vars) {
  print(
    gc_df %>%
      group_by(.data[[v]]) %>%
      summarise(n = n(), bad_rate = mean(target_bad)) %>%
      arrange(desc(bad_rate))
  )
}


# =============================================================================
# CHI-SQUARE TESTS: Categorical features vs target
# Tests whether the distribution of credit outcome differs across factor levels
# =============================================================================

cat_vars_chi <- c("account_status", "credit_history", "savings",
                  "employment_duration", "loan_purpose", "marital_status",
                  "other_debtors", "other_installment_plans", "housing",
                  "property", "telephone", "foreign_worker", "occupation")

chi_results <- lapply(cat_vars_chi, function(v) {
  tbl <- table(gc_df[[v]], gc_df$target)
  test <- chisq.test(tbl)
  data.frame(
    feature   = v,
    chi_sq    = round(test$statistic, 3),
    df        = test$parameter,
    p_value   = round(test$p.value, 4),
    sig       = ifelse(test$p.value < 0.001, "***",
                       ifelse(test$p.value < 0.01,  "**",
                              ifelse(test$p.value < 0.05,  "*", "")))
  )
})

chi_df <- do.call(rbind, chi_results)
rownames(chi_df) <- NULL
chi_df <- chi_df[order(chi_df$p_value), ]   # sort by significance
print(chi_df)


# =============================================================================
# Chi-squared approximation is incorrect" "loan_purpose", "marital_status" in the table have an
#  frequency < 5, at least one cell is zero, which causes the calculation to fail

# For sparse categorical variables where chi-square fails
# Fix for the NaN variables — use Fisher's Exact Test instead, which handles sparse cells correctly
# =============================================================================
sparse_vars <- c("loan_purpose", "marital_status")

fisher_results <- lapply(sparse_vars, function(v) {
  tbl <- table(gc_df[[v]], gc_df$target)
  test <- fisher.test(tbl, simulate.p.value = TRUE, B = 10000)
  # simulate.p.value = TRUE uses Monte Carlo simulation
  # B = 10000 is the number of simulations
  data.frame(
    feature = v,
    method  = "Fisher (Monte Carlo)",
    p_value = round(test$p.value, 4),
    sig     = ifelse(test$p.value < 0.001, "***",
                     ifelse(test$p.value < 0.01,  "**",
                            ifelse(test$p.value < 0.05,  "*", "")))
  )
})

fisher_df <- do.call(rbind, fisher_results)
print(fisher_df)



# =============================================================================
# CORRELATION MATRIX — numeric features + target_bad
# Visualised as a heatmap for the report
# =============================================================================

cor_vars <- c("loan_duration_mths", "credit_amount", "age_yrs", "target_bad")
cor_matrix <- cor(gc_df[, cor_vars], method = "pearson")
cor_matrix_rounded <- round(cor_matrix, 3)
print(cor_matrix_rounded)

# Melt for ggplot
cor_melted <- melt(cor_matrix_rounded)

# Clean variable name labels
label_map <- c(
  "loan_duration_mths" = "Loan Duration",
  "credit_amount"      = "Credit Amount",
  "age_yrs"            = "Age",
  "target_bad"         = "Default (target)"
)

cor_melted$Var1 <- label_map[as.character(cor_melted$Var1)]
cor_melted$Var2 <- label_map[as.character(cor_melted$Var2)]

# Plot heatmap
ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%.3f", value)),
            size = 4.5, fontface = "bold") +
  scale_fill_gradient2(
    low      = "#D85A30",
    mid      = "white",
    high     = "#1D9E75",
    midpoint = 0,
    limits   = c(-1, 1),
    name     = "Pearson r"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    axis.title       = element_blank(),
    axis.text        = element_text(face = "bold"),
    panel.grid       = element_blank(),
    legend.position  = "right"
  ) +
  labs(title    = "Pearson correlation matrix — numeric features",
       subtitle = "German Credit dataset (n = 1,000)")

# =============================================================================
# NUMERICAL CORRELATION WITH TARGET
# =============================================================================
gc_df %>%
  select(loan_duration_mths, credit_amount, age_yrs, target_bad) %>%
  cor() %>%
  round(3)

# -----------------------------------------------------------------------------
# 3. Stratified 70/30 train / test split
#    createDataPartition preserves the 70/30 bad-credit ratio in both sets
# -----------------------------------------------------------------------------

set.seed(42)
train_idx <- createDataPartition(
  y     = gc_model[[target_col]],
  p     = 0.70,
  list  = FALSE
)

train_raw <- gc_model[ train_idx, ]
test_raw  <- gc_model[-train_idx, ]

cat("Train rows:", nrow(train_raw), "| Bad rate:",
    round(mean(train_raw$target_bad) * 100, 1), "%\n")
cat("Test rows: ", nrow(test_raw),  "| Bad rate:",
    round(mean(test_raw$target_bad)  * 100, 1), "%\n\n")


# -----------------------------------------------------------------------------
# 4. Ordinal integer encoding
#    Converts ordered factor levels to consecutive integers (1, 2, 3 ...)
#    Preserves the rank signal without creating many dummy columns
# -----------------------------------------------------------------------------

encode_ordinal <- function(df, cols) {
  for (col in cols) {
    df[[col]] <- as.integer(df[[col]])  # factor level → integer rank
  }
  df
}

train_enc <- encode_ordinal(train_raw, ordered_feats)
test_enc  <- encode_ordinal(test_raw,  ordered_feats)

cat("Ordinal encoding done:", length(ordered_feats), "features\n")


# -----------------------------------------------------------------------------
# 5. One-hot encoding for unordered categoricals
#    drop_first = TRUE removes the reference level to avoid dummy variable trap
#    Reference levels (R uses the first factor level by default):
#      other_debtors          → "None" (set explicitly in wrangle script)
#      other_installment_plans → "None" (set explicitly in wrangle script)
#      All others             → first level alphabetically / by factor order
# -----------------------------------------------------------------------------

train_enc <- dummy_cols(
  train_enc,
  select_columns  = unordered_feats,
  remove_first_dummy  = TRUE,    # drop reference level → avoids multicollinearity
  remove_selected_columns = TRUE # drop original factor columns after encoding
)

test_enc <- dummy_cols(
  test_enc,
  select_columns  = unordered_feats,
  remove_first_dummy  = TRUE,
  remove_selected_columns = TRUE
)

# Align columns: test may lack a dummy if a rare level doesn't appear in test
missing_cols <- setdiff(names(train_enc), names(test_enc))
for (col in missing_cols) test_enc[[col]] <- 0L
test_enc <- test_enc[, names(train_enc)]  # reorder to match train

cat("One-hot encoding done:", length(unordered_feats),
    "features →", sum(grepl(paste(unordered_feats, collapse="|"), names(train_enc))),
    "dummy columns\n")


# -----------------------------------------------------------------------------
# 6. Numeric: log1p transform then min-max scale
#    log1p handles zeros and compresses the long right tail
#    Min-max uses TRAIN statistics only — applied to both train and test
#    This prevents data leakage from test into train
# -----------------------------------------------------------------------------

# Step 6a: log1p transform (applied identically to both sets, no leakage risk)
for (col in numeric_feats) {
  train_enc[[col]] <- log1p(train_enc[[col]])
  test_enc[[col]]  <- log1p(test_enc[[col]])
}

# Step 6b: fit min-max scaler on TRAIN only
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
test_enc  <- minmax_scale(test_enc,  numeric_feats, train_mins, train_ranges)

# Note: test values may fall slightly outside [0,1] if extreme values exist
# This is expected and correct — do NOT clip

cat("Numeric scaling done:", length(numeric_feats), "features → [0, 1] range\n\n")


# -----------------------------------------------------------------------------
# 7. Separate features (X) and target (y)
# -----------------------------------------------------------------------------

feature_cols <- setdiff(names(train_enc), target_col)

X_train <- train_enc[, feature_cols]
y_train <- factor(train_enc[[target_col]], levels = c(0, 1), labels = c("Good", "Bad"))

X_test  <- test_enc[, feature_cols]
y_test  <- factor(test_enc[[target_col]],  levels = c(0, 1), labels = c("Good", "Bad"))

cat("Final feature matrix dimensions:\n")
cat("  X_train:", nrow(X_train), "rows x", ncol(X_train), "columns\n")
cat("  X_test: ", nrow(X_test),  "rows x", ncol(X_test),  "columns\n")
cat("  y_train: Good =", sum(y_train == "Good"), "| Bad =", sum(y_train == "Bad"), "\n")
cat("  y_test:  Good =", sum(y_test  == "Good"), "| Bad =", sum(y_test  == "Bad"), "\n\n")


# -----------------------------------------------------------------------------
# 8. Final checks before modelling
# -----------------------------------------------------------------------------

# No NAs anywhere
stopifnot(!anyNA(X_train))
stopifnot(!anyNA(X_test))

# All values numeric
stopifnot(all(sapply(X_train, is.numeric)))
stopifnot(all(sapply(X_test,  is.numeric)))

# Train and test have identical column sets
stopifnot(identical(names(X_train), names(X_test)))

cat("All checks passed. Ready for modelling.\n\n")

# Optional: inspect the final feature list
cat("Features going into the model (", ncol(X_train), "):\n", sep="")
cat(paste(names(X_train), collapse=", "), "\n\n")


# =============================================================================
# Objects available for the modelling step:
#   X_train, y_train  — training features and labels
#   X_test,  y_test   — test features and labels (held out until evaluation)
#   train_mins, train_ranges — scaling params (needed to score new applicants)
# =============================================================================


