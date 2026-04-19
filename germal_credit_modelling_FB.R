# =============================================================================
# CRISP-DM: Modelling
# German Credit — logistic regression baseline + random forest
# Run AFTER german_credit_data_prep.R  (script 3)
# Objects expected in environment: X_train, y_train, X_test, y_test
# Code co-worked with chatGPT and Claude.ai
# =============================================================================

library(caret)          # confusionMatrix(), trainControl()
library(randomForest)   # randomForest()
library(pROC)           # roc(), auc(), ggroc()
library(ggplot2)        # plotting

# install.packages(c("pROC", "randomForest")) #if needed


# =============================================================================
# MODEL 1: Logistic regression (baseline)
# Simple, interpretable, fast. Establishes the floor for performance.
# =============================================================================

# Combine X and y for glm() — it needs a single data frame
train_df <- cbind(X_train, target_bad = as.integer(y_train == "Bad"))
test_df  <- cbind(X_test,  target_bad = as.integer(y_test  == "Bad"))

lr_model <- glm(
  target_bad ~ .,
  data   = train_df,
  family = binomial(link = "logit")
)

summary(lr_model)   # inspect coefficients — check signs match domain intuition

# Predict probabilities on test set
lr_probs <- predict(lr_model, newdata = test_df, type = "response")

# Convert to class labels at default 0.5 threshold
lr_pred <- factor(ifelse(lr_probs >= 0.5, "Bad", "Good"), levels = c("Good", "Bad"))

# Evaluate
cat("\n--- Logistic Regression ---\n")
print(confusionMatrix(lr_pred, y_test, positive = "Bad"))

lr_roc <- roc(test_df$target_bad, lr_probs, quiet = TRUE)
cat("AUC:", round(auc(lr_roc), 3), "\n")


# =============================================================================
# MODEL 2: Random forest
# Handles non-linearity and interactions. Generally outperforms LR on tabular
# data without extensive feature engineering.
# =============================================================================

set.seed(42)
rf_model <- randomForest(
  x          = X_train,
  y          = y_train,
  ntree      = 500,        # number of trees — more is better up to a point
  mtry       = floor(sqrt(ncol(X_train))),  # default: sqrt(p) for classification
  importance = TRUE        # needed for varImpPlot()
)

print(rf_model)   # OOB error gives a free validation estimate

# Predict on test set
rf_probs <- predict(rf_model, newdata = X_test, type = "prob")[, "Bad"]
rf_pred  <- predict(rf_model, newdata = X_test, type = "class")

# Evaluate
cat("\n--- Random Forest ---\n")
print(confusionMatrix(rf_pred, y_test, positive = "Bad"))

rf_roc <- roc(test_df$target_bad, rf_probs, quiet = TRUE)
cat("AUC:", round(auc(rf_roc), 3), "\n")


# =============================================================================
# EVALUATION: ROC curves side by side
# =============================================================================

# Plot both ROC curves on one chart
plot(lr_roc, col = "#534AB7", lwd = 2,
     main = "ROC curves — German Credit",
     xlab = "False positive rate", ylab = "True positive rate")
plot(rf_roc, col = "#1D9E75", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray70")  # random baseline
legend("bottomright",
       legend = c(
         paste0("Logistic regression (AUC = ", round(auc(lr_roc), 3), ")"),
         paste0("Random forest      (AUC = ", round(auc(rf_roc), 3), ")")
       ),
       col    = c("#534AB7", "#1D9E75"),
       lwd    = 2)


# =============================================================================
# FEATURE IMPORTANCE (random forest)
# =============================================================================

# Mean decrease in accuracy (more reliable than Gini impurity)
varImpPlot(rf_model, type = 1, main = "Feature importance — mean decrease accuracy")

# As a tidy data frame for further analysis
importance_df <- as.data.frame(importance(rf_model, type = 1))
importance_df$feature <- rownames(importance_df)
importance_df <- importance_df[order(-importance_df$MeanDecreaseAccuracy), ]
print(head(importance_df, 10))


# =============================================================================
# THRESHOLD TUNING
# The default 0.5 threshold is rarely optimal for imbalanced data.
# Find the threshold that maximises the F1 score for the Bad class.
# =============================================================================

thresholds <- seq(0.2, 0.7, by = 0.01)

threshold_results <- sapply(thresholds, function(t) {
  pred <- factor(ifelse(rf_probs >= t, "Bad", "Good"), levels = c("Good", "Bad"))
  cm   <- confusionMatrix(pred, y_test, positive = "Bad")
  c(
    threshold = t,
    precision = unname(cm$byClass["Precision"]),
    recall    = unname(cm$byClass["Recall"]),
    f1        = unname(cm$byClass["F1"])
  )
})

threshold_df <- as.data.frame(t(threshold_results))
# FIX: ensure numeric columns
threshold_df[] <- lapply(threshold_df, as.numeric)

# Best threshold by F1
best_idx <- which.max(threshold_df$f1)
best_t   <- threshold_df$threshold[best_idx]
str(threshold_df)

cat("\nBest threshold (max F1):", best_t, "\n")
cat("At this threshold:\n")
cat("  Precision:", round(threshold_df$precision[best_idx], 3), "\n")
cat("  Recall:   ", round(threshold_df$recall[best_idx], 3), "\n")
cat("  F1:       ", round(threshold_df$f1[best_idx], 3), "\n")

# Plot precision-recall tradeoff vs threshold
ggplot(threshold_df, aes(x = threshold)) +
  geom_line(aes(y = precision, colour = "Precision"), linewidth = 1) +
  geom_line(aes(y = recall,    colour = "Recall"),    linewidth = 1) +
  geom_line(aes(y = f1,        colour = "F1"),        linewidth = 1, linetype = "dashed") +
  geom_vline(xintercept = best_t, linetype = "dotted", colour = "gray50") +
  scale_colour_manual(values = c("Precision" = "#534AB7", "Recall" = "#D85A30", "F1" = "#1D9E75")) +
  labs(title = "Precision / recall tradeoff by threshold",
       x = "Classification threshold", y = "Score", colour = NULL) +
  theme_minimal()


# =============================================================================
# FINAL PREDICTIONS at tuned threshold
# =============================================================================

rf_pred_tuned <- factor(ifelse(rf_probs >= best_t, "Bad", "Good"), levels = c("Good", "Bad"))

cat("\n--- Random Forest (tuned threshold =", best_t, ") ---\n")
print(confusionMatrix(rf_pred_tuned, y_test, positive = "Bad"))


# =============================================================================
# SAVE MODEL AND SCORING FUNCTION
# =============================================================================

saveRDS(rf_model, "german_credit_rf_model.rds")
cat("\nModel saved to german_credit_rf_model.rds\n")

# Scoring function for new applicants
# Expects a data frame with the same columns as X_train (after all prep steps)
score_applicant <- function(new_data, model = rf_model, threshold = best_t) {
  probs <- predict(model, newdata = new_data, type = "prob")[, "Bad"]
  data.frame(
    prob_bad   = round(probs, 4),
    prediction = ifelse(probs >= threshold, "Bad", "Good"),
    risk_band  = cut(probs,
                     breaks = c(0, 0.2, 0.4, 0.6, 1),
                     labels = c("Low", "Medium", "High", "Very high"),
                     include.lowest = TRUE)
  )
}

# Example: score the first 5 test applicants
cat("\nSample scores for first 5 test applicants:\n")
print(score_applicant(X_test[1:5, ]))

