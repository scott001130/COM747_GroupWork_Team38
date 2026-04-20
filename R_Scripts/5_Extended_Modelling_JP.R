# =============================================================================
# CRISP-DM: Extended Modelling & Model Comparison
# German Credit — SVM, Decision Tree, KNN, SMOTE, Cross-Validation
# Run AFTER german_credit_modelling_FB.R
# =============================================================================

library(caret)
library(smotefamily)
library(e1071)
library(rpart)
library(class)
library(pROC)
library(ggplot2)

dir.create("outputs", showWarnings = FALSE)

stopifnot(exists("X_train"), exists("y_train"), exists("X_test"), exists("y_test"))
stopifnot(exists("lr_probs"), exists("rf_probs"))
stopifnot(exists("lr_roc"), exists("rf_roc"))
stopifnot(exists("rf_model"))
stopifnot(exists("test_df"))

cat("Prerequisite check passed.\n")
cat("Training set:", nrow(X_train), "rows |",
    sum(y_train == "Bad"), "Bad,", sum(y_train == "Good"), "Good\n\n")

# =============================================================================
# SMOTE
# =============================================================================

smote_input <- data.frame(X_train, target = as.integer(y_train == "Bad"))

cat("Class distribution BEFORE SMOTE:\n")
print(table(smote_input$target))
cat("\n")

set.seed(42)
smote_result <- SMOTE(
  X = smote_input[, -ncol(smote_input)],
  target = smote_input$target,
  K = 5
)

smote_df <- smote_result$data

X_train_bal <- smote_df[, !names(smote_df) %in% "class"]
y_train_bal <- factor(
  ifelse(smote_df$class == 1, "Bad", "Good"),
  levels = c("Good", "Bad")
)

cat("Class distribution AFTER SMOTE:\n")
print(table(y_train_bal))
cat("\n")

clean_names <- make.names(names(X_train_bal))
names(X_train_bal) <- clean_names

X_test_clean <- X_test
names(X_test_clean) <- make.names(names(X_test_clean))

# =============================================================================
# CROSS-VALIDATION
# =============================================================================

ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# =============================================================================
# SVM LINEAR
# =============================================================================

cat("Training SVM (linear kernel) with 10-fold CV...\n")

set.seed(42)
svm_linear_model <- train(
  x = X_train_bal,
  y = y_train_bal,
  method = "svmLinear",
  metric = "ROC",
  trControl = ctrl,
  preProcess = NULL
)

cat("SVM Linear — CV ROC:", round(max(svm_linear_model$results$ROC), 3), "\n")

svm_linear_probs <- predict(svm_linear_model, newdata = X_test_clean, type = "prob")[, "Bad"]
svm_linear_pred <- predict(svm_linear_model, newdata = X_test_clean)

cat("\n--- SVM Linear ---\n")
print(confusionMatrix(svm_linear_pred, y_test, positive = "Bad"))

svm_linear_roc <- roc(test_df$target_bad, svm_linear_probs, quiet = TRUE)
cat("Test AUC:", round(auc(svm_linear_roc), 3), "\n")

# =============================================================================
# SVM RADIAL
# =============================================================================

cat("\nTraining SVM (radial kernel) with 10-fold CV...\n")

set.seed(42)
svm_radial_model <- train(
  x = X_train_bal,
  y = y_train_bal,
  method = "svmRadial",
  metric = "ROC",
  trControl = ctrl,
  preProcess = NULL
)

cat("SVM Radial — CV ROC:", round(max(svm_radial_model$results$ROC), 3), "\n")

svm_radial_probs <- predict(svm_radial_model, newdata = X_test_clean, type = "prob")[, "Bad"]
svm_radial_pred <- predict(svm_radial_model, newdata = X_test_clean)

cat("\n--- SVM Radial ---\n")
print(confusionMatrix(svm_radial_pred, y_test, positive = "Bad"))

svm_radial_roc <- roc(test_df$target_bad, svm_radial_probs, quiet = TRUE)
cat("Test AUC:", round(auc(svm_radial_roc), 3), "\n")

# =============================================================================
# DECISION TREE
# =============================================================================

cat("\nTraining Decision Tree (CART) with 10-fold CV...\n")

set.seed(42)
dt_model <- train(
  x = X_train_bal,
  y = y_train_bal,
  method = "rpart",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 10
)

cat("Decision Tree — CV ROC:", round(max(dt_model$results$ROC), 3), "\n")
cat("Best cp:", dt_model$bestTune$cp, "\n")

dt_probs <- predict(dt_model, newdata = X_test_clean, type = "prob")[, "Bad"]
dt_pred <- predict(dt_model, newdata = X_test_clean)

cat("\n--- Decision Tree ---\n")
print(confusionMatrix(dt_pred, y_test, positive = "Bad"))

dt_roc <- roc(test_df$target_bad, dt_probs, quiet = TRUE)
cat("Test AUC:", round(auc(dt_roc), 3), "\n")

# =============================================================================
# KNN
# =============================================================================

cat("\nTraining KNN with 10-fold CV...\n")

set.seed(42)
knn_model <- train(
  x = X_train_bal,
  y = y_train_bal,
  method = "knn",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = expand.grid(k = seq(3, 25, by = 2)),
  preProcess = NULL
)

cat("KNN — CV ROC:", round(max(knn_model$results$ROC), 3), "\n")
cat("Best k:", knn_model$bestTune$k, "\n")

knn_probs <- predict(knn_model, newdata = X_test_clean, type = "prob")[, "Bad"]
knn_pred <- predict(knn_model, newdata = X_test_clean)

cat("\n--- KNN ---\n")
print(confusionMatrix(knn_pred, y_test, positive = "Bad"))

knn_roc <- roc(test_df$target_bad, knn_probs, quiet = TRUE)
cat("Test AUC:", round(auc(knn_roc), 3), "\n")

# =============================================================================
# MODEL COMPARISON TABLE
# =============================================================================

extract_metrics <- function(pred, actual, roc_obj, model_name) {
  cm <- confusionMatrix(pred, actual, positive = "Bad")
  data.frame(
    Model = model_name,
    Accuracy = round(cm$overall["Accuracy"], 3),
    Sensitivity = round(cm$byClass["Sensitivity"], 3),
    Specificity = round(cm$byClass["Specificity"], 3),
    Precision = round(cm$byClass["Precision"], 3),
    F1 = round(cm$byClass["F1"], 3),
    AUC = round(as.numeric(auc(roc_obj)), 3),
    Kappa = round(cm$overall["Kappa"], 3),
    row.names = NULL
  )
}

lr_pred <- factor(ifelse(lr_probs >= 0.5, "Bad", "Good"), levels = c("Good", "Bad"))
rf_pred <- predict(rf_model, newdata = X_test, type = "class")

comparison_table <- rbind(
  extract_metrics(lr_pred, y_test, lr_roc, "Logistic Regression"),
  extract_metrics(rf_pred, y_test, rf_roc, "Random Forest"),
  extract_metrics(svm_linear_pred, y_test, svm_linear_roc, "SVM (Linear)"),
  extract_metrics(svm_radial_pred, y_test, svm_radial_roc, "SVM (Radial)"),
  extract_metrics(dt_pred, y_test, dt_roc, "Decision Tree"),
  extract_metrics(knn_pred, y_test, knn_roc, "KNN")
)

comparison_table <- comparison_table[order(-comparison_table$AUC), ]

cat("\n=================================================================\n")
cat("MODEL COMPARISON - All classifiers on test set (n = 300)\n")
cat("=================================================================\n\n")
print(comparison_table, row.names = FALSE)
cat("\n")

write.csv(comparison_table, "outputs/model_comparison_table.csv", row.names = FALSE)

# =============================================================================
# ROC CURVES
# =============================================================================

png("outputs/roc_all_models.png", width = 1200, height = 900, res = 150)
plot(lr_roc, col = "#534AB7", lwd = 2,
     main = "ROC curves - all models (German Credit test set)",
     xlab = "False positive rate (1 - Specificity)",
     ylab = "True positive rate (Sensitivity)")
plot(rf_roc,         col = "#1D9E75", lwd = 2, add = TRUE)
plot(svm_linear_roc, col = "#D85A30", lwd = 2, add = TRUE)
plot(svm_radial_roc, col = "#2196F3", lwd = 2, add = TRUE)
plot(dt_roc,         col = "#FF9800", lwd = 2, add = TRUE)
plot(knn_roc,        col = "#800080", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray70")

legend("bottomright", cex = 0.8,
       legend = c(
         paste0("Logistic Regression (AUC = ", round(auc(lr_roc), 3), ")"),
         paste0("Random Forest       (AUC = ", round(auc(rf_roc), 3), ")"),
         paste0("SVM Linear          (AUC = ", round(auc(svm_linear_roc), 3), ")"),
         paste0("SVM Radial          (AUC = ", round(auc(svm_radial_roc), 3), ")"),
         paste0("Decision Tree       (AUC = ", round(auc(dt_roc), 3), ")"),
         paste0("KNN                 (AUC = ", round(auc(knn_roc), 3), ")")
       ),
       col = c("#534AB7", "#1D9E75", "#D85A30", "#2196F3", "#FF9800", "#800080"),
       lwd = 2)
dev.off()

# =============================================================================
# SMOTE IMPACT
# =============================================================================

cat("\n--- SMOTE Impact: Random Forest on balanced vs original data ---\n")

set.seed(42)
rf_bal_model <- train(
  x = X_train_bal,
  y = y_train_bal,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  ntree = 500
)

rf_bal_probs <- predict(rf_bal_model, newdata = X_test_clean, type = "prob")[, "Bad"]
rf_bal_pred <- predict(rf_bal_model, newdata = X_test_clean)

cat("\nRandom Forest (SMOTE-balanced):\n")
print(confusionMatrix(rf_bal_pred, y_test, positive = "Bad"))

rf_bal_roc <- roc(test_df$target_bad, rf_bal_probs, quiet = TRUE)
cat("AUC (balanced):", round(auc(rf_bal_roc), 3), "\n")
cat("AUC (original):", round(auc(rf_roc), 3), "\n\n")

cm_orig <- confusionMatrix(rf_pred, y_test, positive = "Bad")
cm_bal <- confusionMatrix(rf_bal_pred, y_test, positive = "Bad")

cat("Sensitivity (Bad recall) - original:", round(cm_orig$byClass["Sensitivity"], 3), "\n")
cat("Sensitivity (Bad recall) - SMOTE:   ", round(cm_bal$byClass["Sensitivity"], 3), "\n")
cat("Specificity (Good recall) - original:", round(cm_orig$byClass["Specificity"], 3), "\n")
cat("Specificity (Good recall) - SMOTE:   ", round(cm_bal$byClass["Specificity"], 3), "\n")

smote_impact <- data.frame(
  Model = c("RF Original", "RF SMOTE"),
  Sensitivity = c(
    round(cm_orig$byClass["Sensitivity"], 3),
    round(cm_bal$byClass["Sensitivity"], 3)
  ),
  Specificity = c(
    round(cm_orig$byClass["Specificity"], 3),
    round(cm_bal$byClass["Specificity"], 3)
  ),
  AUC = c(
    round(as.numeric(auc(rf_roc)), 3),
    round(as.numeric(auc(rf_bal_roc)), 3)
  )
)

write.csv(smote_impact, "outputs/smote_impact.csv", row.names = FALSE)

cat("\n=================================================================\n")
cat("Extended modelling complete. See comparison_table for results.\n")
cat("=================================================================\n")