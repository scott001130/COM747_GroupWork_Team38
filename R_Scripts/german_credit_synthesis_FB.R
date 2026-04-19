# =============================================================================
# CRISP-DM: Synthesis & Hypothesis Evaluation
# German Credit — "What factors most influence credit default risk?"
# Run AFTER germal_credit_modelling_FB.R
# =============================================================================

library(dplyr)
library(ggplot2)

dir.create("outputs", showWarnings = FALSE)

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

importance_top10 <- head(
  importance_df[order(-importance_df$MeanDecreaseAccuracy), ], 10
)

cat("Top 10 features by mean decrease in accuracy (Random Forest):\n")
print(importance_top10[, c("feature", "MeanDecreaseAccuracy")])

write.csv(
  importance_top10[, c("feature", "MeanDecreaseAccuracy")],
  "outputs/top10_feature_importance.csv",
  row.names = FALSE
)

p_top10 <- ggplot(importance_top10,
                  aes(x = reorder(feature, MeanDecreaseAccuracy),
                      y = MeanDecreaseAccuracy)) +
  geom_col(fill = "#534AB7", alpha = 0.85) +
  coord_flip() +
  theme_minimal(base_size = 12) +
  labs(
    title = "Top 10 features by importance — Random Forest",
    subtitle = "Mean decrease in accuracy (German Credit, n = 1,000)",
    x = NULL,
    y = "Mean Decrease in Accuracy"
  )

print(p_top10)
ggsave("outputs/top10_feature_importance.png", plot = p_top10, width = 8, height = 5, dpi = 300)

# =============================================================================
# BAD RATES FOR TOP CATEGORICAL FEATURES
# =============================================================================

cat("\nBad rates for top categorical features:\n")

top_cats <- c("account_status", "credit_history", "savings", "property")

for (v in top_cats) {
  cat("\n---", v, "---\n")

  top_cat_tbl <- gc_df %>%
    group_by(.data[[v]]) %>%
    summarise(
      n = n(),
      bad_rate = round(mean(target_bad), 3),
      .groups = "drop"
    ) %>%
    arrange(bad_rate)

  print(top_cat_tbl)

  write.csv(
    top_cat_tbl,
    file = paste0("outputs/synthesis_bad_rate_", v, ".csv"),
    row.names = FALSE
  )
}

# =============================================================================
# NUMERIC FEATURE MEANS
# =============================================================================

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
write.csv(numeric_means, "outputs/numeric_means_by_outcome.csv", row.names = FALSE)

# =============================================================================
# COMBINED EVIDENCE TABLE
# =============================================================================

chi_clean <- chi_df[!is.na(chi_df$chi_sq), c("feature", "chi_sq", "p_value", "sig")]

fisher_clean <- data.frame(
  feature = fisher_df$feature,
  chi_sq = NA,
  p_value = fisher_df$p_value,
  sig = fisher_df$sig
)

all_tests <- rbind(chi_clean, fisher_clean)
importance_named <- importance_df[, c("feature", "MeanDecreaseAccuracy")]

evidence_table <- merge(all_tests, importance_named, by = "feature", all.x = TRUE)

evidence_table <- evidence_table[
  order(is.na(evidence_table$MeanDecreaseAccuracy),
        -replace(evidence_table$MeanDecreaseAccuracy,
                 is.na(evidence_table$MeanDecreaseAccuracy), 0)),
]

rownames(evidence_table) <- NULL

cat("\nCombined evidence table — hypothesis evaluation:\n")
print(evidence_table)
write.csv(evidence_table, "outputs/evidence_table.csv", row.names = FALSE)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

cat("\n")
cat("=================================================================\n")
cat("HYPOTHESIS: What factors most influence credit default risk?\n")
cat("=================================================================\n")
cat("\nTier 1 — Strongest predictors (significant + high importance):\n")
cat("  1. Account status       (χ²=123.72, p<0.001 | RF importance=27.69)\n")
cat("  2. Credit history       (χ²=61.69,  p<0.001 | RF importance=13.49)\n")
cat("  3. Loan duration        (r=0.215,   p<0.001 | RF importance=13.49)\n")
cat("  4. Savings              (χ²=36.10,  p<0.001 | RF importance=8.60)\n")
cat("  5. Credit amount        (r=0.155,   p<0.001 | RF importance=8.90)\n")

cat("\nTier 2 — Moderate predictors (significant, lower importance):\n")
cat("  6. Property             (χ²=23.72,  p<0.001 | RF importance=8.93)\n")
cat("  7. Age                  (r=-0.091,  p<0.05  | RF importance=7.20)\n")
cat("  8. Employment duration  (χ²=18.37,  p=0.001 | RF importance=~3)\n")
cat("  9. Housing              (χ²=18.20,  p<0.001)\n")
cat(" 10. Loan purpose         (Fisher,    p<0.001)\n")

cat("\nTier 3 — Weak / non-significant predictors:\n")
cat("  Telephone    (χ²=1.17, p=0.279) — no significant association\n")
cat("  Occupation   (χ²=1.89, p=0.597) — no significant association\n")
cat("=================================================================\n")
