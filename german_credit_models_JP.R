# =============================================================================
# CRISP-DM: Extended Modelling & Model Comparison
# German Credit — SVM, Decision Tree, SMOTE, Cross-Validation
# Run AFTER germal_credit_modelling_FB.R  (script 5)
# Objects expected: X_train, y_train, X_test, y_test, lr_probs, rf_probs,
#                   lr_roc, rf_roc, rf_model, train_df, test_df
# Author: Jamie Price
# =============================================================================

library(caret)          # train(), trainControl(), confusionMatrix()
library(smotefamily)    # SMOTE()
library(e1071)          # SVM backend for caret
library(rpart)          # decision tree backend for caret
library(pROC)           # roc(), auc()
library(ggplot2)        # plotting

# install.packages(c("smotefamily", "e1071")) # if not already installed


# =============================================================================
# 1. VERIFY PREREQUISITE OBJECTS
#    Ensures Flora's pipeline (scripts 1-3) has been run first
# =============================================================================

stopifnot(exists("X_train"), exists("y_train"), exists("X_test"), exists("y_test"))
stopifnot(exists("lr_probs"), exists("rf_probs"))
stopifnot(exists("lr_roc"), exists("rf_roc"))

cat("Prerequisite check passed.\n")
cat("Training set:", nrow(X_train), "rows |",
    sum(y_train == "Bad"), "Bad,", sum(y_train == "Good"), "Good\n\n")


# =============================================================================
# 2. SMOTE — Synthetic Minority Over-sampling Technique
#    The training set is imbalanced (70% Good / 30% Bad).
#    SMOTE generates synthetic examples of the minority class (Bad) by
#    interpolating between existing Bad observations and their K nearest
#    neighbours, giving classifiers a more balanced decision boundary.
# =============================================================================

# Combine features + numeric target for SMOTE input
smote_input <- data.frame(X_train, target = as.integer(y_train == "Bad"))

cat("Class distribution BEFORE SMOTE:\n")
print(table(smote_input$target))
cat("\n")

set.seed(42)
smote_result <- SMOTE(
  X    = smote_input[, -ncol(smote_input)],  # features only
  target = smote_input$target,                # 0/1 numeric target
  K    = 5                                    # 5 nearest neighbours
)

smote_df <- smote_result$data

# Separate back into features and labels
X_train_bal <- smote_df[, !names(smote_df) %in% "class"]
y_train_bal <- factor(
  ifelse(smote_df$class == 1, "Bad", "Good"),
  levels = c("Good", "Bad")
)

cat("Class distribution AFTER SMOTE:\n")
print(table(y_train_bal))
cat("\n")

# Clean column names — some dummy columns contain spaces and parentheses
# (e.g. "loan_purpose_Car (used)") which rpart/caret cannot parse.
# make.names() converts these to syntactically valid R names.
clean_names <- make.names(names(X_train_bal))
names(X_train_bal) <- clean_names

# Apply the same name cleaning to the test set so predictions work
X_test_clean <- X_test
names(X_test_clean) <- make.names(names(X_test_clean))


# =============================================================================
# 3. SHARED CROSS-VALIDATION SETUP
#    10-fold cross-validation repeated 3 times for stable performance estimates.
#    Using ROC as the optimisation metric (better than accuracy for imbalanced
#    data because it considers the tradeoff between sensitivity and specificity).
# =============================================================================

ctrl <- trainControl(
  method          = "repeatedcv",
  number          = 10,           # 10 folds
  repeats         = 3,            # repeat 3 times for stability
  classProbs      = TRUE,         # needed for ROC metric
  summaryFunction = twoClassSummary,
  savePredictions  = "final"
)


# =============================================================================
# 4. SVM — Support Vector Machine (Linear Kernel)
#    Finds the optimal hyperplane that maximises the margin between classes.
#    The linear kernel works well when features are already encoded and scaled,
#    and produces interpretable coefficients.
# =============================================================================

cat("Training SVM (linear kernel) with 10-fold CV...\n")

set.seed(42)
svm_linear_model <- train(
  x      = X_train_bal,
  y      = y_train_bal,
  method = "svmLinear",
  metric = "ROC",
  trControl = ctrl,
  preProcess = NULL   # data already scaled in Flora's prep script
)

cat("SVM Linear — CV ROC:", round(max(svm_linear_model$results$ROC), 3), "\n")

# Predict on held-out test set
svm_linear_probs <- predict(svm_linear_model, newdata = X_test_clean, type = "prob")[, "Bad"]
svm_linear_pred  <- predict(svm_linear_model, newdata = X_test_clean)

cat("\n--- SVM Linear ---\n")
print(confusionMatrix(svm_linear_pred, y_test, positive = "Bad"))

svm_linear_roc <- roc(test_df$target_bad, svm_linear_probs, quiet = TRUE)
cat("Test AUC:", round(auc(svm_linear_roc), 3), "\n")


# =============================================================================
# 5. SVM — Support Vector Machine (Radial Kernel)
#    The radial basis function (RBF) kernel maps features into a higher-
#    dimensional space, allowing the SVM to capture non-linear relationships
#    between predictors and the target. Useful if the decision boundary
#    between Good and Bad credit isn't a straight line.
# =============================================================================

cat("\nTraining SVM (radial kernel) with 10-fold CV...\n")

set.seed(42)
svm_radial_model <- train(
  x      = X_train_bal,
  y      = y_train_bal,
  method = "svmRadial",
  metric = "ROC",
  trControl = ctrl,
  preProcess = NULL
)

cat("SVM Radial — CV ROC:", round(max(svm_radial_model$results$ROC), 3), "\n")

# Predict on test set
svm_radial_probs <- predict(svm_radial_model, newdata = X_test_clean, type = "prob")[, "Bad"]
svm_radial_pred  <- predict(svm_radial_model, newdata = X_test_clean)

cat("\n--- SVM Radial ---\n")
print(confusionMatrix(svm_radial_pred, y_test, positive = "Bad"))

svm_radial_roc <- roc(test_df$target_bad, svm_radial_probs, quiet = TRUE)
cat("Test AUC:", round(auc(svm_radial_roc), 3), "\n")


# =============================================================================
# 6. DECISION TREE (CART via rpart)
#    Recursively splits the data on the feature that provides the most
#    information gain at each node. Highly interpretable — the tree structure
#    directly shows which features matter and their thresholds.
#    caret handles pruning via cross-validation over the complexity parameter.
# =============================================================================

cat("\nTraining Decision Tree (CART) with 10-fold CV...\n")

set.seed(42)
dt_model <- train(
  x      = X_train_bal,
  y      = y_train_bal,
  method = "rpart",
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 10     # search 10 values of the complexity parameter (cp)
)

cat("Decision Tree — CV ROC:", round(max(dt_model$results$ROC), 3), "\n")
cat("Best cp:", dt_model$bestTune$cp, "\n")

# Predict on test set
dt_probs <- predict(dt_model, newdata = X_test_clean, type = "prob")[, "Bad"]
dt_pred  <- predict(dt_model, newdata = X_test_clean)

cat("\n--- Decision Tree ---\n")
print(confusionMatrix(dt_pred, y_test, positive = "Bad"))

dt_roc <- roc(test_df$target_bad, dt_probs, quiet = TRUE)
cat("Test AUC:", round(auc(dt_roc), 3), "\n")


# =============================================================================
# 7. MODEL COMPARISON TABLE
#    Collects key metrics from all 5 models (Flora's LR + RF, Jamie's SVM
#    Linear + SVM Radial + Decision Tree) into a single table for the report.
# =============================================================================

# Helper: extract metrics from a confusionMatrix object
extract_metrics <- function(pred, actual, probs, roc_obj, model_name) {
  cm <- confusionMatrix(pred, actual, positive = "Bad")
  data.frame(
    Model       = model_name,
    Accuracy    = round(cm$overall["Accuracy"], 3),
    Sensitivity = round(cm$byClass["Sensitivity"], 3),
    Specificity = round(cm$byClass["Specificity"], 3),
    Precision   = round(cm$byClass["Precision"], 3),
    F1          = round(cm$byClass["F1"], 3),
    AUC         = round(as.numeric(auc(roc_obj)), 3),
    Kappa       = round(cm$overall["Kappa"], 3),
    row.names   = NULL
  )
}

# Recreate Flora's predictions for the table
lr_pred <- factor(ifelse(lr_probs >= 0.5, "Bad", "Good"), levels = c("Good", "Bad"))
rf_pred <- predict(rf_model, newdata = X_test, type = "class")

comparison_table <- rbind(
  extract_metrics(lr_pred,          y_test, lr_probs,          lr_roc,          "Logistic Regression"),
  extract_metrics(rf_pred,          y_test, rf_probs,          rf_roc,          "Random Forest"),
  extract_metrics(svm_linear_pred,  y_test, svm_linear_probs,  svm_linear_roc,  "SVM (Linear)"),
  extract_metrics(svm_radial_pred,  y_test, svm_radial_probs,  svm_radial_roc,  "SVM (Radial)"),
  extract_metrics(dt_pred,          y_test, dt_probs,          dt_roc,          "Decision Tree")
)

# Sort by AUC descending
comparison_table <- comparison_table[order(-comparison_table$AUC), ]

cat("\n=================================================================\n")
cat("MODEL COMPARISON — All classifiers on test set (n = 300)\n")
cat("=================================================================\n\n")
print(comparison_table, row.names = FALSE)
cat("\n")


# =============================================================================
# 8. COMBINED ROC CURVE PLOT
#    Visualises the tradeoff between true positive rate and false positive rate
#    for all models on the same axes, making it easy to see which model
#    provides the best discrimination between Good and Bad credit.
# =============================================================================

plot(lr_roc, col = "#534AB7", lwd = 2,
     main = "ROC curves — all models (German Credit test set)",
     xlab = "False positive rate (1 - Specificity)",
     ylab = "True positive rate (Sensitivity)")
plot(rf_roc,         col = "#1D9E75", lwd = 2, add = TRUE)
plot(svm_linear_roc, col = "#D85A30", lwd = 2, add = TRUE)
plot(svm_radial_roc, col = "#2196F3", lwd = 2, add = TRUE)
plot(dt_roc,         col = "#FF9800", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray70")

legend("bottomright", cex = 0.8,
       legend = c(
         paste0("Logistic Regression (AUC = ", round(auc(lr_roc), 3), ")"),
         paste0("Random Forest       (AUC = ", round(auc(rf_roc), 3), ")"),
         paste0("SVM Linear          (AUC = ", round(auc(svm_linear_roc), 3), ")"),
         paste0("SVM Radial          (AUC = ", round(auc(svm_radial_roc), 3), ")"),
         paste0("Decision Tree       (AUC = ", round(auc(dt_roc), 3), ")")
       ),
       col = c("#534AB7", "#1D9E75", "#D85A30", "#2196F3", "#FF9800"),
       lwd = 2)


# =============================================================================
# 9. SMOTE IMPACT ANALYSIS
#    Compares the best model trained on original data (Flora's RF) with the
#    same model type trained on SMOTE-balanced data, to quantify the effect
#    of class balancing on minority class recall.
# =============================================================================

cat("\n--- SMOTE Impact: Random Forest on balanced vs original data ---\n")

set.seed(42)
rf_bal_model <- train(
  x      = X_train_bal,
  y      = y_train_bal,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  ntree  = 500
)

rf_bal_probs <- predict(rf_bal_model, newdata = X_test_clean, type = "prob")[, "Bad"]
rf_bal_pred  <- predict(rf_bal_model, newdata = X_test_clean)

cat("\nRandom Forest (SMOTE-balanced):\n")
print(confusionMatrix(rf_bal_pred, y_test, positive = "Bad"))

rf_bal_roc <- roc(test_df$target_bad, rf_bal_probs, quiet = TRUE)
cat("AUC (balanced):", round(auc(rf_bal_roc), 3), "\n")
cat("AUC (original):", round(auc(rf_roc), 3), "\n\n")

# Compare sensitivity (recall for Bad class) — key metric for credit risk
# Missing a bad loan (false negative) is more costly than rejecting a good one
cm_orig <- confusionMatrix(rf_pred, y_test, positive = "Bad")
cm_bal  <- confusionMatrix(rf_bal_pred, y_test, positive = "Bad")

cat("Sensitivity (Bad recall) — original:", round(cm_orig$byClass["Sensitivity"], 3), "\n")
cat("Sensitivity (Bad recall) — SMOTE:   ", round(cm_bal$byClass["Sensitivity"], 3), "\n")
cat("Specificity (Good recall) — original:", round(cm_orig$byClass["Specificity"], 3), "\n")
cat("Specificity (Good recall) — SMOTE:   ", round(cm_bal$byClass["Specificity"], 3), "\n")


# =============================================================================
# Objects available for synthesis / fairness analysis:
#   comparison_table  — all 5 models side by side
#   svm_linear_model, svm_radial_model, dt_model — trained caret models
#   rf_bal_model      — RF trained on SMOTE-balanced data
#   X_train_bal, y_train_bal — SMOTE-balanced training set
# =============================================================================

cat("\n=================================================================\n")
cat("Extended modelling complete. See comparison_table for results.\n")
cat("=================================================================\n")
