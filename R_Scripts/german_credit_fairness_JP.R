# =============================================================================
# CRISP-DM: Fairness & Bias Analysis
# German Credit — Does the model treat gender and age groups equitably?
# Run AFTER germal_credit_modelling_FB.R
# =============================================================================

library(dplyr)
library(ggplot2)
library(tidyr)

dir.create("outputs", showWarnings = FALSE)

stopifnot(exists("gc_df"), exists("rf_model"), exists("X_test"), exists("y_test"))
stopifnot(exists("train_idx"))

cat("Fairness analysis: prerequisite check passed.\n\n")

keep_cols_fair <- c("marital_status", "age_yrs", "target", "target_bad")
test_labels <- gc_df[-train_idx, keep_cols_fair]
test_labels <- test_labels[, keep_cols_fair]

cat("Test set for fairness analysis:", nrow(test_labels), "rows\n")

# gender proxy
test_labels$gender <- ifelse(
  grepl("^Female", test_labels$marital_status), "Female", "Male"
)

cat("\nGender distribution in test set:\n")
print(table(test_labels$gender))
cat("\n")

# age groups
test_labels$age_group <- cut(
  test_labels$age_yrs,
  breaks = c(0, 25, 35, 50, 100),
  labels = c("Under 25", "25-35", "36-50", "Over 50"),
  right = TRUE
)

cat("Age group distribution in test set:\n")
print(table(test_labels$age_group))
cat("\n")

# predictions
test_labels$pred_class <- as.character(predict(rf_model, newdata = X_test))
test_labels$pred_prob <- predict(rf_model, newdata = X_test, type = "prob")[, "Bad"]
test_labels$actual <- as.character(test_labels$target)

cat("Prediction distribution:\n")
print(table(Predicted = test_labels$pred_class, Actual = test_labels$actual))
cat("\n")

# subgroup metrics
calc_subgroup_metrics <- function(df, group_col) {
  df %>%
    group_by(.data[[group_col]]) %>%
    summarise(
      n = n(),
      n_good = sum(actual == "Good"),
      n_bad = sum(actual == "Bad"),
      accuracy = round(mean(pred_class == actual), 3),
      fpr = round(sum(actual == "Good" & pred_class == "Bad") /
                    max(sum(actual == "Good"), 1), 3),
      fnr = round(sum(actual == "Bad" & pred_class == "Good") /
                    max(sum(actual == "Bad"), 1), 3),
      selection_rate = round(mean(pred_class == "Bad"), 3),
      mean_prob_bad = round(mean(pred_prob), 3),
      .groups = "drop"
    )
}

gender_metrics <- calc_subgroup_metrics(test_labels, "gender")
age_metrics <- calc_subgroup_metrics(test_labels, "age_group")

cat("=================================================================\n")
cat("SUBGROUP PERFORMANCE — By Gender\n")
cat("=================================================================\n")
print(as.data.frame(gender_metrics))
cat("\n")
write.csv(gender_metrics, "outputs/fairness_gender_metrics.csv", row.names = FALSE)

cat("=================================================================\n")
cat("SUBGROUP PERFORMANCE — By Age Group\n")
cat("=================================================================\n")
print(as.data.frame(age_metrics))
cat("\n")
write.csv(age_metrics, "outputs/fairness_age_metrics.csv", row.names = FALSE)

# fairness metrics
cat("=================================================================\n")
cat("FAIRNESS METRICS — Gender\n")
cat("=================================================================\n")

cat("\n1. Demographic Parity (selection rate should be similar):\n")
cat("   Male selection rate:  ", gender_metrics$selection_rate[gender_metrics$gender == "Male"], "\n")
cat("   Female selection rate:", gender_metrics$selection_rate[gender_metrics$gender == "Female"], "\n")

dp_ratio <- min(gender_metrics$selection_rate) / max(gender_metrics$selection_rate)
cat("   Ratio (closer to 1 = fairer):", round(dp_ratio, 3), "\n")

if (dp_ratio < 0.8) {
  cat("   WARNING: Ratio below 0.8 — potential adverse impact (4/5ths rule)\n")
} else {
  cat("   Passes the 4/5ths rule threshold\n")
}

cat("\n2. Equal Opportunity (FNR should be similar):\n")
cat("   Male FNR:  ", gender_metrics$fnr[gender_metrics$gender == "Male"], "\n")
cat("   Female FNR:", gender_metrics$fnr[gender_metrics$gender == "Female"], "\n")
cat("   Difference:", abs(diff(gender_metrics$fnr)), "\n")

cat("\n3. Predictive Parity (precision should be similar):\n")
gender_precision <- test_labels %>%
  filter(pred_class == "Bad") %>%
  group_by(gender) %>%
  summarise(
    precision = round(mean(actual == "Bad"), 3),
    n_pred_bad = n(),
    .groups = "drop"
  )
print(as.data.frame(gender_precision))
cat("\n")
write.csv(gender_precision, "outputs/fairness_gender_precision.csv", row.names = FALSE)

# calibration
cat("=================================================================\n")
cat("CALIBRATION CHECK — Actual vs Predicted Bad Rates\n")
cat("=================================================================\n\n")

calibration <- test_labels %>%
  group_by(gender, age_group) %>%
  summarise(
    n = n(),
    actual_bad_rate = round(mean(actual == "Bad"), 3),
    predicted_bad_rate = round(mean(pred_class == "Bad"), 3),
    mean_prob_bad = round(mean(pred_prob), 3),
    .groups = "drop"
  ) %>%
  filter(n >= 10)

print(as.data.frame(calibration))
cat("\n")
write.csv(calibration, "outputs/fairness_calibration.csv", row.names = FALSE)

# =============================================================================
# VISUALISATIONS
# =============================================================================

p1 <- ggplot(gender_metrics, aes(x = gender, y = selection_rate, fill = gender)) +
  geom_col(alpha = 0.85, width = 0.5) +
  geom_text(aes(label = selection_rate), vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = c("Female" = "#D85A30", "Male" = "#1D9E75")) +
  ylim(0, max(gender_metrics$selection_rate) * 1.3) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  labs(
    title = "Selection rate by gender (proportion predicted Bad)",
    subtitle = "German Credit — Random Forest model",
    x = NULL, y = "Selection Rate"
  )
print(p1)
ggsave("outputs/fairness_selection_rate_gender.png", plot = p1, width = 7, height = 5, dpi = 300)

p2 <- ggplot(gender_metrics, aes(x = gender, y = fpr, fill = gender)) +
  geom_col(alpha = 0.85, width = 0.5) +
  geom_text(aes(label = fpr), vjust = -0.5, size = 5, fontface = "bold") +
  scale_fill_manual(values = c("Female" = "#D85A30", "Male" = "#1D9E75")) +
  ylim(0, max(gender_metrics$fpr) * 1.3) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  labs(
    title = "False positive rate by gender",
    subtitle = "Proportion of Good applicants wrongly classified as Bad",
    x = NULL, y = "False Positive Rate"
  )
print(p2)
ggsave("outputs/fairness_fpr_gender.png", plot = p2, width = 7, height = 5, dpi = 300)

p3 <- ggplot(age_metrics, aes(x = age_group, y = accuracy, fill = age_group)) +
  geom_col(alpha = 0.85, width = 0.6) +
  geom_text(aes(label = accuracy), vjust = -0.5, size = 4.5, fontface = "bold") +
  scale_fill_manual(values = c(
    "Under 25" = "#D85A30", "25-35" = "#FF9800",
    "36-50" = "#1D9E75", "Over 50" = "#534AB7")) +
  ylim(0, 1) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none") +
  labs(
    title = "Model accuracy by age group",
    subtitle = "German Credit — Random Forest model",
    x = NULL, y = "Accuracy"
  )
print(p3)
ggsave("outputs/fairness_accuracy_age.png", plot = p3, width = 7, height = 5, dpi = 300)

age_compare <- age_metrics %>%
  select(age_group, actual = n_bad, predicted = selection_rate) %>%
  mutate(actual_rate = round(actual / age_metrics$n, 3)) %>%
  select(age_group, actual_rate, predicted)

age_long <- pivot_longer(
  age_compare,
  cols = c(actual_rate, predicted),
  names_to = "type",
  values_to = "rate"
)
age_long$type <- ifelse(age_long$type == "actual_rate", "Actual Bad Rate", "Predicted Bad Rate")

p4 <- ggplot(age_long, aes(x = age_group, y = rate, fill = type)) +
  geom_col(position = "dodge", alpha = 0.85, width = 0.6) +
  scale_fill_manual(values = c("Actual Bad Rate" = "#1D9E75", "Predicted Bad Rate" = "#D85A30")) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Actual vs predicted bad rate by age group",
    subtitle = "Comparing model predictions against ground truth",
    x = NULL, y = "Rate", fill = NULL
  )
print(p4)
ggsave("outputs/fairness_actual_vs_predicted_age.png", plot = p4, width = 8, height = 5, dpi = 300)

# summary csv
fairness_summary <- data.frame(
  male_selection_rate = gender_metrics$selection_rate[gender_metrics$gender == "Male"],
  female_selection_rate = gender_metrics$selection_rate[gender_metrics$gender == "Female"],
  demographic_parity_ratio = round(dp_ratio, 3),
  male_fnr = gender_metrics$fnr[gender_metrics$gender == "Male"],
  female_fnr = gender_metrics$fnr[gender_metrics$gender == "Female"]
)
write.csv(fairness_summary, "outputs/fairness_summary.csv", row.names = FALSE)

cat("\n=================================================================\n")
cat("FAIRNESS ANALYSIS SUMMARY\n")
cat("=================================================================\n")

cat("\nDataset: German Credit (n = 1,000, test set n =", nrow(test_labels), ")\n")
cat("Model: Random Forest (500 trees, AUC =", round(pROC::auc(rf_roc), 3), ")\n")
cat("Protected attributes: Gender (derived from marital_status), Age\n")

cat("\nGender findings:\n")
cat("  - Male applicants:  ", sum(test_labels$gender == "Male"),
    "in test set, selection rate =", gender_metrics$selection_rate[gender_metrics$gender == "Male"], "\n")
cat("  - Female applicants:", sum(test_labels$gender == "Female"),
    "in test set, selection rate =", gender_metrics$selection_rate[gender_metrics$gender == "Female"], "\n")
cat("  - Demographic parity ratio:", round(dp_ratio, 3), "\n")

cat("\nAge findings:\n")
for (i in seq_len(nrow(age_metrics))) {
  cat("  -", as.character(age_metrics$age_group[i]), ": accuracy =",
      age_metrics$accuracy[i], ", FPR =", age_metrics$fpr[i],
      ", FNR =", age_metrics$fnr[i], "\n")
}

cat("\nKey observation: The marital_status column conflates gender with\n")
cat("marital status, making it impossible to fully disentangle gender-based\n")
cat("bias from marital-status-based effects. This is a known limitation of\n")
cat("the German Credit dataset and highlights why proxy variables present\n")
cat("challenges for algorithmic fairness in credit scoring.\n")
cat("=================================================================\n")