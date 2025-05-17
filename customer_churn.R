
# Load necessary libraries
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(class)
library(caret)
library(randomForest)
library(corrplot)
library(DataExplorer)

# Import the data
churn_data <- read_csv("C:/Users/Yousra/Desktop/CP1.dataset1.csv")
View(churn_data)

# Display the structure of the data
str(churn_data)

# Display the summary of the data
summary(churn_data)

# Check the column names
colnames(churn_data)
colnames(churn_data) <- make.names(colnames(churn_data), unique = TRUE)
print(colnames(churn_data))

# Update the list of numerical variables with exact column names
numerical_vars <- c('Call..Failure', 'Subscription..Length', 'Charge..Amount', 'Seconds.of.Use', 
                    'Frequency.of.use', 'Frequency.of.SMS', 'Distinct.Called.Numbers', 
                    'Customer.Value')

# Generate summary statistics for numerical variables
summary(churn_data %>% select(all_of(numerical_vars)))

# Frequency tables for categorical variables
categorical_vars <- c('Complains', 'Tariff.Plan', 'Status', 'Churn')
for (var in categorical_vars) {
  print(table(churn_data[[var]]))
}

# Percentiles for numerical variables
percentiles <- churn_data %>% select(all_of(numerical_vars)) %>%
  summarise_all(list(p25 = ~quantile(., 0.25, na.rm = TRUE),
                     p50 = ~quantile(., 0.50, na.rm = TRUE),
                     p75 = ~quantile(., 0.75, na.rm = TRUE)))

print(percentiles)

# Visualization of numerical variables using aes() with tidy evaluation
for (var in numerical_vars) {
  p <- ggplot(churn_data, aes(x = .data[[var]])) + 
    geom_histogram(binwidth = 30, fill = 'blue', alpha = 0.7) +
    labs(title = paste('Histogram of', var))
  print(p)
}

# Visualization for categorical variables
for (var in categorical_vars) {
  p <- ggplot(churn_data, aes(x = .data[[var]])) + 
    geom_bar(fill = 'blue', alpha = 0.7) +
    labs(title = paste('Bar Plot of', var))
  print(p)
}

# Detecting Missing Values
plot_missing(churn_data)

# Box Plot for Numerical Variables to Detect Outliers
for (var in numerical_vars) {
  p <- ggplot(churn_data, aes(x = .data[['Churn']], y = .data[[var]])) + 
    geom_boxplot(fill = 'blue', alpha = 0.7) +
    labs(title = paste('Box Plot of', var, 'by Churn'))
  print(p)
}

# Function to remove outliers based on IQR
remove_outliers <- function(x) {
  q <- quantile(x, probs=c(.25, .75), na.rm = TRUE)
  iqr <- IQR(x, na.rm = TRUE)
  x[x < (q[1] - 1.5 * iqr)] <- NA
  x[x > (q[2] + 1.5 * iqr)] <- NA
  return(x)
}

# Remove outliers from numerical variables
cleaned_data <- churn_data %>%
  mutate(across(all_of(numerical_vars), remove_outliers)) %>%
  drop_na()

# Check summary of cleaned_data for verification
print(summary(cleaned_data))

# Box Plot for Numerical Variables after Removing Outliers
for (var in numerical_vars) {
  p <- ggplot(cleaned_data, aes(x = as.factor(Churn), y = .data[[var]])) + 
    geom_boxplot(fill = 'blue', alpha = 0.7) +
    labs(title = paste('Box Plot of', var, 'by Churn'))
  
  # Print the plot to verify if it's generated
  print(p)
}

# Correlation matrix for numerical variables
cor_matrix <- cor(churn_data %>% select(all_of(numerical_vars)), use = "complete.obs")
corrplot(cor_matrix, method = "number")

# Correlation with target variable "Churn"
churn_numeric <- as.numeric(churn_data$Churn) # Convert "Churn" to numeric for correlation
cor_with_churn <- sapply(churn_data %>% select(all_of(numerical_vars)), function(x) cor(x, churn_numeric, use = "complete.obs"))
print(cor_with_churn)

# Remove variables with very weak correlation
cleaned_data <- churn_data %>%
  select(-`Call..Failure`, -`Subscription..Length`,-`Age`)

# Verify the column removal
print(colnames(cleaned_data))
cleaned_data <- churn_data
# Normalizing numerica data
preprocess_params <- preProcess(churn_data[, numerical_vars], method = c("center", "scale"))
churn_data[, numerical_vars] <- predict(preprocess_params, churn_data[, numerical_vars])

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(churn_data$Churn, p = .8, list = FALSE, times = 1)
trainData <- churn_data[trainIndex,]
testData  <- churn_data[-trainIndex,]

# Convert Churn to a factor
trainData$Churn <- as.factor(trainData$Churn)
testData$Churn <- as.factor(testData$Churn)

# Logistic Regression
model_logistic <- glm(Churn ~ ., data = trainData, family = binomial)
pred_logistic <- predict(model_logistic, testData, type = "response")
pred_logistic <- ifelse(pred_logistic > 0.5, 1, 0)
accuracy_logistic <- mean(pred_logistic == testData$Churn)

# Random Forest
model_rf <- randomForest(Churn ~ ., data = trainData)
pred_rf <- predict(model_rf, testData)
accuracy_rf <- mean(pred_rf == testData$Churn)

# K-Nearest Neighbors
knn_train <- trainData[, -which(names(trainData) == "Churn")]
knn_test <- testData[, -which(names(testData) == "Churn")]
knn_train_labels <- trainData$Churn
knn_test_labels <- testData$Churn

# Normalize the data
preproc <- preProcess(knn_train, method = c("center", "scale"))
knn_train <- predict(preproc, knn_train)
knn_test <- predict(preproc, knn_test)

model_knn <- knn(train = knn_train, test = knn_test, cl = knn_train_labels, k = 5)
accuracy_knn <- mean(model_knn == knn_test_labels)

# Model Evaluation
conf_matrix_logistic <- confusionMatrix(as.factor(pred_logistic), as.factor(testData$Churn))
conf_matrix_rf <- confusionMatrix(as.factor(pred_rf), as.factor(testData$Churn))
conf_matrix_knn <- confusionMatrix(as.factor(model_knn), as.factor(knn_test_labels))

# Print the accuracy of each model
cat("Accuracy of Logistic Regression:", accuracy_logistic, "\n")
cat("Accuracy of Random Forest:", accuracy_rf, "\n")
cat("Accuracy of KNN:", accuracy_knn, "\n")

# Print the confusion matrix for each model
cat("Confusion Matrix for Logistic Regression:\n")
print(conf_matrix_logistic)
cat("Confusion Matrix for Random Forest:\n")
print(conf_matrix_rf)
cat("Confusion Matrix for KNN:\n")
print(conf_matrix_knn)

# Logistic Regression with Tuning
train_control <- trainControl(method = "cv", number = 10)
model_logistic <- train(Churn ~ ., data = trainData, method = "glm", family = "binomial",
                        trControl = train_control)
pred_logistic <- predict(model_logistic, testData)
pred_logistic <- ifelse(pred_logistic == "1", 1, 0)
accuracy_logistic <- mean(pred_logistic == testData$Churn)

# K-Nearest Neighbors with Tuning
preproc <- preProcess(trainData[, -which(names(trainData) == "Churn")], method = c("center", "scale"))
train_knn <- predict(preproc, trainData[, -which(names(trainData) == "Churn")])
test_knn <- predict(preproc, testData[, -which(names(testData) == "Churn")])

tune_grid_knn <- expand.grid(k = 1:20)
model_knn <- train(x = train_knn, y = trainData$Churn, method = "knn",
                   trControl = train_control, tuneGrid = tune_grid_knn)
pred_knn <- predict(model_knn, test_knn)
accuracy_knn <- mean(pred_knn == testData$Churn)

# Fitting models with cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train Random Forest model with cross-validation
tune_grid_rf <- expand.grid(mtry = seq(1, ncol(trainData) - 1, by = 2))
model_rf_cv <- train(Churn ~ ., data = trainData, method = "rf",
                    trControl = train_control, tuneGrid = tune_grid_rf, ntree = 200)
pred_rf_cv <- predict(model_rf_cv, testData)
accuracy_rf_cv <- mean(pred_rf_cv == testData$Churn)

# Model Evaluation
conf_matrix_logistic <- confusionMatrix(as.factor(pred_logistic), as.factor(testData$Churn))
conf_matrix_rf <- confusionMatrix(as.factor(pred_rf_cv), as.factor(testData$Churn))
conf_matrix_knn <- confusionMatrix(as.factor(pred_knn), as.factor(testData$Churn))

# Print the accuracy of each model
cat("Accuracy of Logistic Regression:", accuracy_logistic, "\n")
cat("Accuracy of Random Forest:", accuracy_rf_cv, "\n")
cat("Accuracy of KNN:", accuracy_knn, "\n")

# Print the confusion matrix for each model
cat("Confusion Matrix for Logistic Regression:\n")
print(conf_matrix_logistic)
cat("Confusion Matrix for Random Forest:\n")
print(conf_matrix_rf)
cat("Confusion Matrix for KNN:\n")
print(conf_matrix_knn)


summary(model_logistic)
summary(model_rf)
summary(model_knn)
