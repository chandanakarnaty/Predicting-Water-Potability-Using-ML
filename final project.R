
# WATER POTABILITY PROJECT 


# Load packages
library(tidyverse)
library(caret)
library(corrplot)
library(randomForest)
library(rpart)
library(pROC)

# Load data
df <- read.csv("water_potability.csv")

# Convert target to factor
df$Potability <- factor(df$Potability, levels = c(0,1),
                        labels = c("NotPotable","Potable"))

# Missing value check
colSums(is.na(df))


# Train/Test Split 
set.seed(123)
idx <- createDataPartition(df$Potability, p = 0.7, list = FALSE)
train <- df[idx, ]
test  <- df[-idx, ]


# Preprocessing

pre <- preProcess(train %>% select(-Potability),
                  method = c("medianImpute","center","scale"))

train_x <- predict(pre, train %>% select(-Potability))
test_x  <- predict(pre,  test  %>% select(-Potability))

train_pp <- bind_cols(train_x, Potability = train$Potability)
test_pp  <- bind_cols(test_x,  Potability = test$Potability)


# EDA

# Correlation matrix
corrplot(cor(train_x), method = "color")

# Class balance
df %>% count(Potability) %>%
  ggplot(aes(Potability, n, fill = Potability)) +
  geom_col() +
  labs(title="Class balance", y="Count")

# Missing values per column
tibble(var = names(df), n_missing = colSums(is.na(df))) %>%
  arrange(desc(n_missing)) %>%
  ggplot(aes(reorder(var, n_missing), n_missing)) +
  geom_col() +
  coord_flip() +
  labs(title="Missing values per column", x="", y="N missing")

# Feature distributions by class
df %>%
  select(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity,
         Organic_carbon, Trihalomethanes, Turbidity, Potability) %>%
  pivot_longer(-Potability) %>%
  ggplot(aes(value, fill = Potability)) +
  geom_density(alpha=.4) +
  facet_wrap(~name, scales="free", ncol=3) +
  labs(title="Feature distributions by Potability")


# MODEL TRAINING: Logistic Regression, kNN, Random Forest


ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 2,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Logistic Regression
set.seed(123)
mod_glm <- train(
  Potability ~ ., data = train_pp,
  method = "glm",
  family = binomial(),
  metric = "ROC",
  trControl = ctrl
)

# kNN 
set.seed(123)
mod_knn <- train(
  Potability ~ ., data = train_pp,
  method = "knn",
  tuneLength = 10,
  metric = "ROC",
  trControl = ctrl
)

# Random Forest 
set.seed(123)
mod_rf <- train(
  Potability ~ ., data = train_pp,
  method = "rf",
  tuneLength = 5,
  metric = "ROC",
  trControl = ctrl
)


# MODEL COMPARISON (AUC)

cv_auc <- tibble(
  model = c("Logistic","kNN","RandomForest"),
  AUC   = c(max(mod_glm$results$ROC),
            max(mod_knn$results$ROC),
            max(mod_rf$results$ROC))
) %>% arrange(desc(AUC))

cv_auc


# Selecting best model based on CV AUC

best_name <- cv_auc$model[1]

best_model <- switch(
  best_name,
  "Logistic" = mod_glm,
  "kNN" = mod_knn,
  "RandomForest" = mod_rf
)


# Test Set Evaluation (Threshold = 0.5)

test_prob <- predict(best_model, newdata = test_pp, type = "prob")[, "Potable"]

test_pred <- ifelse(test_prob >= 0.5, "Potable", "NotPotable") %>% 
  factor(levels = c("NotPotable","Potable"))

confusionMatrix(test_pred, test_pp$Potability, positive = "Potable")

roc_test <- roc(test_pp$Potability, test_prob, levels = c("NotPotable","Potable"))
auc(roc_test)

# TUNED RANDOM FOREST (FINAL MODEL)

set.seed(123)
rf_grid <- expand.grid(mtry = 2:9)

mod_rf_tuned <- train(
  Potability ~ ., data = train_pp,
  method = "rf",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = rf_grid,
  ntree = 500,
  importance = TRUE
)

# Feature importance plot
varImp(mod_rf_tuned) %>% plot(top = 9)
mod_rf_tuned$bestTune


# FINAL TEST EVALUATION WITH OPTIMAL THRESHOLD

# Clean any old objects
rm(test_prob, test_pred_thr, roc_test, thr)

# 1. Predicted probabilities from tuned RF on test set
test_prob <- predict(mod_rf_tuned,
                     newdata = test_pp,
                     type = "prob")[, "Potable"]

# 2. ROC and optimal threshold (Youden's J)
roc_test <- roc(test_pp$Potability,
                test_prob,
                levels = c("NotPotable","Potable"))

thr <- coords(roc_test,
              "best",
              best.method = "youden",
              ret = "threshold")

# Make sure thresh is a single numeric
thr <- as.numeric(thr[1])

# 3. Final classification using optimal threshold
#    (DO NOT wrap in factor yet)
test_pred_thr <- ifelse(test_prob >= thr,
                        "Potable",
                        "NotPotable")

# Check lengths here
length(test_prob)           
length(test_pred_thr)       
length(test_pp$Potability) 

# 4. Now convert to factor IN A SEPARATE STEP
test_pred_thr <- factor(test_pred_thr,
                        levels = c("NotPotable","Potable"))

# Optional: see distribution
table(test_pred_thr, useNA = "ifany")

# 5. Final confusion matrix and AUC
confusionMatrix(test_pred_thr,
                test_pp$Potability,
                positive = "Potable")

auc(roc_test)
