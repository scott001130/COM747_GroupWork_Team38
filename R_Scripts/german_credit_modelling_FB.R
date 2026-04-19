# =============================================================================
# CRISP-DM: Modelling
# German Credit — logistic regression baseline + random forest
# Run AFTER german_credit_data_prep_FB.R
# Objects expected in environment: X_train, y_train, X_test, y_test
# =============================================================================

library(caret)
library(randomForest)
library(pROC)
library(ggplot2)

dir.create("outputs", showWarnings = FALSE)

# =============================================================================
# MODEL 1: Logistic regression
# =============================================================================

train_df <- cbind(X_train, target_bad = as.integer(y_train == "Bad"))
test_df <- cbind(X_test, target_bad = as.integer(y_test == "Bad"))

lr_model <- glm(
  target_bad ~ .,
  data = train_df,
  family = binomial(link = "logit")
)

summary(lr_model)

lr_probs <- predict(lr_model, newdata = test_df, type = "response")
lr_pred <- factor(ifelse(lr_probs >= 0.5, "Bad", "Good"), levels = c("Good", "Bad"))

cat("\n--- Logistic Regression ---\n")
lr_cm <- confusionMatrix(lr_pred, y_test, positive = "Bad")
print(lr_cm)

lr_roc <- roc(test_df$target_bad, lr_probs, quiet = TRUE)
cat("AUC:", round(auc(lr_roc), 3), "\n")

lr_metrics <- data.frame(
  Model = "Logistic Regression",
  Accuracy = round(lr_cm$overall["Accuracy"], 3),
  Sensitivity = round(lr_cm$byClass["Sensitivity"], 3),
  Specificity = round(lr_cm$byClass["Specificity"], 3),
  Precision = round(lr_cm$byClass["Precision"], 3),
  F1 = round(lr_cm$byClass["F1"], 3),
  AUC = round(as.numeric(auc(lr_roc)), 3)
)
write.csv(lr_metrics, "outputs/logistic_regression_metrics.csv", row.names = FALSE)

# =============================================================================
# MODEL 2: Random forest
# =============================================================================

set.seed(42)
rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 500,
  mtry = floor(sqrt(ncol(X_train))),
  importance = TRUE
)

print(rf_model)

rf_probs <- predict(rf_model, newdata = X_test, type = "prob")[, "Bad"]
rf_pred <- predict(rf_model, newdata = X_test, type = "class")

cat("\n--- Random Forest ---\n")
rf_cm <- confusionMatrix(rf_pred, y_test, positive = "Bad")
print(rf_cm)

rf_roc <- roc(test_df$target_bad, rf_probs, quiet = TRUE)
cat("AUC:", round(auc(rf_roc), 3), "\n")

rf_metrics <- data.frame(
  Model = "Random Forest",
  Accuracy = round(rf_cm$overall["Accuracy"], 3),
  Sensitivity = round(rf_cm$byClass["Sensitivity"], 3),
  Specificity = round(rf_cm$byClass["Specificity"], 3),
  Precision = round(rf_cm$byClass["Precision"], 3),
  F1 = round(rf_cm$byClass["F1"], 3),
  AUC = round(as.numeric(auc(rf_roc)), 3)
)
write.csv(rf_metrics, "outputs/random_forest_metrics.csv", row.names = FALSE)

# =============================================================================
# ROC PLOT
# =============================================================================

png("outputs/roc_lr_vs_rf.png", width = 1200, height = 900, res = 150)
plot(lr_roc, col = "#534AB7", lwd = 2,
     main = "ROC curves — German Credit",
     xlab = "False positive rate", ylab = "True positive rate")
plot(rf_roc, col = "#1D9E75", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray70")
legend("bottomright",
       legend = c(
         paste0("Logistic regression (AUC = ", round(auc(lr_roc), 3), ")"),
         paste0("Random forest      (AUC = ", round(auc(rf_roc), 3), ")")
       ),
       col = c("#534AB7", "#1D9E75"),
       lwd = 2)
dev.off()

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

png("outputs/rf_feature_importance.png", width = 1200, height = 900, res = 150)
varImpPlot(rf_model, type = 1, main = "Feature importance — mean decrease accuracy")
dev.off()

importance_df <- as.data.frame(importance(rf_model, type = 1))
importance_df$feature <- rownames(importance_df)
importance_df <- importance_df[order(-importance_df$MeanDecreaseAccuracy), ]
print(head(importance_df, 10))
write.csv(importance_df, "outputs/rf_feature_importance.csv", row.names = FALSE)

# =============================================================================
# THRESHOLD TUNING
# =============================================================================

thresholds <- seq(0.2, 0.7, by = 0.01)

threshold_results <- sapply(thresholds, function(t) {
  pred <- factor(ifelse(rf_probs >= t, "Bad", "Good"), levels = c("Good", "Bad"))
  cm <- confusionMatrix(pred, y_test, positive = "Bad")
  c(
    threshold = t,
    precision = unname(cm$byClass["Precision"]),
    recall = unname(cm$byClass["Recall"]),
    f1 = unname(cm$byClass["F1"])
  )
})

threshold_df <- as.data.frame(t(threshold_results))
threshold_df[] <- lapply(threshold_df, as.numeric)

best_idx <- which.max(threshold_df$f1)
best_t <- threshold_df$threshold[best_idx]
str(threshold_df)

write.csv(threshold_df, "outputs/rf_threshold_tuning.csv", row.names = FALSE)

cat("\nBest threshold (max F1):", best_t, "\n")
cat("At this threshold:\n")
cat("  Precision:", round(threshold_df$precision[best_idx], 3), "\n")
cat("  Recall:   ", round(threshold_df$recall[best_idx], 3), "\n")
cat("  F1:       ", round(threshold_df$f1[best_idx], 3), "\n")

p_thresh <- ggplot(threshold_df, aes(x = threshold)) +
  geom_line(aes(y = precision, colour = "Precision"), linewidth = 1) +
  geom_line(aes(y = recall, colour = "Recall"), linewidth = 1) +
  geom_line(aes(y = f1, colour = "F1"), linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = best_t, linetype = "dotted", colour = "gray50") +
  scale_colour_manual(values = c("Precision" = "#534AB7", "Recall" = "#D85A30", "F1" = "#1D9E75")) +
  labs(title = "Precision / recall tradeoff by threshold",
       x = "Classification threshold", y = "Score", colour = NULL) +
  theme_minimal()

print(p_thresh)
ggsave("outputs/rf_threshold_tradeoff.png", plot = p_thresh, width = 8, height = 5, dpi = 300)

# =============================================================================
# FINAL PREDICTIONS
# =============================================================================

rf_pred_tuned <- factor(ifelse(rf_probs >= best_t, "Bad", "Good"), levels = c("Good", "Bad"))

cat("\n--- Random Forest (tuned threshold =", best_t, ") ---\n")
rf_tuned_cm <- confusionMatrix(rf_pred_tuned, y_test, positive = "Bad")
print(rf_tuned_cm)

rf_tuned_metrics <- data.frame(
  Model = "Random Forest Tuned Threshold",
  Threshold = best_t,
  Accuracy = round(rf_tuned_cm$overall["Accuracy"], 3),
  Sensitivity = round(rf_tuned_cm$byClass["Sensitivity"], 3),
  Specificity = round(rf_tuned_cm$byClass["Specificity"], 3),
  Precision = round(rf_tuned_cm$byClass["Precision"], 3),
  F1 = round(rf_tuned_cm$byClass["F1"], 3)
)
write.csv(rf_tuned_metrics, "outputs/random_forest_tuned_metrics.csv", row.names = FALSE)

# =============================================================================
# SAVE MODEL
# =============================================================================

saveRDS(rf_model, "outputs/german_credit_rf_model.rds")
cat("\nModel saved to outputs/german_credit_rf_model.rds\n")

score_applicant <- function(new_data, model = rf_model, threshold = best_t) {
  probs <- predict(model, newdata = new_data, type = "prob")[, "Bad"]
  data.frame(
    prob_bad = round(probs, 4),
    prediction = ifelse(probs >= threshold, "Bad", "Good"),
    risk_band = cut(
      probs,
      breaks = c(0, 0.2, 0.4, 0.6, 1),
      labels = c("Low", "Medium", "High", "Very high"),
      include.lowest = TRUE
    )
  )
}

cat("\nSample scores for first 5 test applicants:\n")
print(score_applicant(X_test[1:5, ]))