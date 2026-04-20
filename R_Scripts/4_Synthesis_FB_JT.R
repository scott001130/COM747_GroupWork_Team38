# =============================================================================
# CRISP-DM: Synthesis & Hypothesis Evaluation
# German Credit — "What factors most influence credit default risk?"
# Run AFTER german_credit_modelling_FB.R
# =============================================================================

library(dplyr)
library(ggplot2)

dir.create("outputs", showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

save_csv <- function(df, path) {
  write.csv(df, path, row.names = FALSE)
}

save_bad_rate_table <- function(var_name) {
  tbl <- gc_df %>%
    group_by(.data[[var_name]]) %>%
    summarise(
      n = n(),
      bad_rate = round(mean(target_bad), 3),
      .groups = "drop"
    ) %>%
    arrange(bad_rate)

  cat("\n---", var_name, "---\n")
  print(tbl)
  save_csv(tbl, paste0("outputs/synthesis_bad_rate_", var_name, ".csv"))
}

# -----------------------------------------------------------------------------
# Feature importance
# -----------------------------------------------------------------------------

importance_top10 <- importance_df %>%
  arrange(desc(MeanDecreaseAccuracy)) %>%
  slice_head(n = 10)

cat("Top 10 features by mean decrease in accuracy (Random Forest):\n")
print(importance_top10[, c("feature", "MeanDecreaseAccuracy")])

save_csv(
  importance_top10[, c("feature", "MeanDecreaseAccuracy")],
  "outputs/top10_feature_importance.csv"
)

p_top10 <- ggplot(
  importance_top10,
  aes(x = reorder(feature, MeanDecreaseAccuracy), y = MeanDecreaseAccuracy)
) +
  geom_col(fill = "#534AB7", alpha = 0.85) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  labs(
    title = "Top 10 features by importance - Random Forest",
    subtitle = "Mean decrease in accuracy (German Credit, n = 1,000)",
    x = NULL,
    y = "Mean Decrease in Accuracy"
  )

print(p_top10)
ggsave("outputs/top10_feature_importance.png", plot = p_top10, width = 8, height = 5, dpi = 300)

# -----------------------------------------------------------------------------
# Bad rates for key categorical features
# -----------------------------------------------------------------------------

cat("\nBad rates for top categorical features:\n")
lapply(c("account_status", "credit_history", "savings", "property"), save_bad_rate_table)

# -----------------------------------------------------------------------------
# Numeric feature means
# -----------------------------------------------------------------------------

cat("\nNumeric feature means by credit outcome:\n")
numeric_means <- gc_df %>%
  group_by(target) %>%
  summarise(
    mean_loan_duration_mths = round(mean(loan_duration_mths), 1),
    mean_credit_amount = round(mean(credit_amount), 0),
    mean_age_yrs = round(mean(age_yrs), 1),
    .groups = "drop"
  )

print(numeric_means)
save_csv(numeric_means, "outputs/numeric_means_by_outcome.csv")

# -----------------------------------------------------------------------------
# Numeric feature evidence
# -----------------------------------------------------------------------------

numeric_df <- bind_rows(lapply(c("loan_duration_mths", "credit_amount", "age_yrs"), function(v) {
  test <- cor.test(gc_df[[v]], gc_df$target_bad, method = "pearson")

  data.frame(
    feature = v,
    p_value = round(test$p.value, 4),
    sig = case_when(
      test$p.value < 0.001 ~ "***",
      test$p.value < 0.01  ~ "**",
      test$p.value < 0.05  ~ "*",
      TRUE ~ ""
    ),
    statistic_type = "Pearson r",
    statistic_value = round(unname(test$estimate), 3)
  )
}))

print(numeric_df)
save_csv(numeric_df, "outputs/numeric_feature_evidence.csv")

# -----------------------------------------------------------------------------
# Combined evidence table
# -----------------------------------------------------------------------------

chi_clean <- chi_df %>%
  filter(!is.na(chi_sq)) %>%
  transmute(
    feature,
    p_value,
    sig,
    statistic_type = "Chi-square",
    statistic_value = chi_sq
  )

fisher_clean <- fisher_df %>%
  transmute(
    feature,
    p_value,
    sig,
    statistic_type = method,
    statistic_value = NA_real_
  )

all_tests <- bind_rows(chi_clean, fisher_clean, numeric_df)

importance_named <- importance_df %>%
  select(feature, MeanDecreaseAccuracy)

evidence_table <- all_tests %>%
  left_join(importance_named, by = "feature") %>%
  arrange(
    is.na(MeanDecreaseAccuracy),
    desc(coalesce(MeanDecreaseAccuracy, 0)),
    p_value
  )

cat("\nCombined evidence table - hypothesis evaluation:\n")
print(evidence_table)
save_csv(evidence_table, "outputs/evidence_table.csv")

# -----------------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------------

summary_blocks <- list(
  "Tier 1 - Strongest predictors (significant + high importance):" = c(
    "Account status (strongest categorical predictor)",
    "Credit history (high significance and RF importance)",
    "Loan duration (positive Pearson correlation with default)",
    "Savings (clear ordinal risk gradient)",
    "Credit amount (positive Pearson correlation with default)"
  ),
  "Tier 2 - Moderate predictors:" = c(
    "Property",
    "Age",
    "Employment duration",
    "Housing",
    "Loan purpose"
  ),
  "Tier 3 - Weak / non-significant predictors:" = c(
    "Telephone",
    "Occupation"
  )
)

cat("\n", strrep("=", 65), "\n", sep = "")
cat("HYPOTHESIS: What factors most influence credit default risk?\n")
cat(strrep("=", 65), "\n", sep = "")

for (section in names(summary_blocks)) {
  cat("\n", section, "\n", sep = "")
  for (i in seq_along(summary_blocks[[section]])) {
    cat(" ", i, ". ", summary_blocks[[section]][i], "\n", sep = "")
  }
}

cat(strrep("=", 65), "\n", sep = "")
