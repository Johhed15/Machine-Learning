# This repository contains some of the work completed by me and two other students during a machine learning course in my master's program!

The code is written in R, using Rmarkdown and a pdf with the results have been generated.

- lab 1: KNN and logistic regression
- lab 2: Lasso reagerssion, Decision trees, logistic regression and PCA(principal components)
- lab 3: Kernels, SVM and Neural nets
- block 2: Random forrest and EM-algortihm


#### If you have any questions regarding the work, please contact me

# Some simple examples
## K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a simple and effective algorithm for classification tasks. It works by finding the K nearest points in the training data to a given data point and making predictions based on the majority class among those neighbors.

```R
library(caret)

# Load Iris dataset
data(iris)
set.seed(123445)
train_index <- createDataPartition(iris$Species, p = 0.6, list = FALSE)
train_data <- iris[train_index, ]
temp_data <- iris[-train_index, ]
validation_index <- createDataPartition(temp_data$Species, p = 0.5, list = FALSE)
validation_data <- temp_data[validation_index, ]
test_data <- temp_data[-validation_index, ]

# Define possible values of k
k_values <- seq(1, 20, by = 1)

# Initialize variables to store results
best_k <- NULL
best_accuracy <- 0

# Loop through each k value
for (k in k_values) {
  # Train KNN model
  knn_model <- train(Species ~ ., data = train_data, method = "knn",
 trControl = trainControl(method = "cv", number = 5),tuneGrid = data.frame(k = k))

  # Make predictions on validation set
  knn_predictions <- predict(knn_model, newdata = validation_data)

  # Calculate accuracy
  knn_accuracy <- confusionMatrix(knn_predictions, validation_data$Species)$overall["Accuracy"]
  
  # Update best k and best accuracy if necessary
  if (knn_accuracy > best_accuracy) {
    best_k <- k
    best_accuracy <- knn_accuracy
  }
}

# Train KNN model with best k
final_knn_model <- train(Species ~ ., data = train_data, method = "knn",
 trControl = trainControl(method = "cv", number = 5), tuneGrid = data.frame(k = best_k))

# Make predictions on test set
final_knn_predictions <- predict(final_knn_model, newdata = test_data)

# Calculate final accuracy
final_knn_accuracy <- confusionMatrix(final_knn_predictions, test_data$Species)$overall["Accuracy"]
print(paste("Best K for KNN:", best_k))
print(paste("KNN Accuracy on testdata with Best K:", final_knn_accuracy))

```
"Best K for KNN: 13"<br>
"KNN Accuracy on testdata with Best K: 0.966666666666667"


## Decision tree with pruning

```R
library(caret)
library(rpart)

# Load Iris dataset
data(iris)
set.seed(12345)
train_index <- createDataPartition(iris$Species, p = 0.6, list = FALSE)
train_data <- iris[train_index, ]
temp_data <- iris[-train_index, ]
validation_index <- createDataPartition(temp_data$Species, p = 0.5, list = FALSE)
validation_data <- temp_data[validation_index, ]
test_data <- temp_data[-validation_index, ]

# Train a large decision tree
large_dt_model <- rpart(Species ~ ., data = train_data, method = "class",
 control = rpart.control(cp = 0))

# Prune the tree using cross-entropy on validation set
pruned_cp <- large_dt_model$cptable[which.min(large_dt_model$cptable[,"xerror"]), "CP"]
pruned_dt_model <- prune(large_dt_model, cp = pruned_cp)

# Make predictions on validation set using the pruned tree
pruned_dt_predictions_val <- predict(pruned_dt_model, newdata = validation_data, type = "class")

# Calculate accuracy on validation set
pruned_dt_accuracy_val <- confusionMatrix(pruned_dt_predictions_val,
validation_data$Species)$overall["Accuracy"]

print(paste("Decision Tree Accuracy on Validation Set after Pruning:", pruned_dt_accuracy_val))

# Make predictions on test set using the pruned tree
pruned_dt_predictions_test <- predict(pruned_dt_model, newdata = test_data, type = "class")

# Calculate accuracy on test set
pruned_dt_accuracy_test <- confusionMatrix(pruned_dt_predictions_test,
test_data$Species)$overall["Accuracy"]

print(paste("Decision Tree Accuracy on Test Set after Pruning:", pruned_dt_accuracy_test))
```
"Decision Tree Accuracy on Validation Set after Pruning: 0.9" <br>
"Decision Tree Accuracy on Test Set after Pruning: 0.966666666666667"
<br>
<br>
<br>
<br>
<div align="center">
  <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExaWN4d2h5bGg1emo2MTJmYTl5aXdrZG9zaXlkdHo4NHIzaXVibmZpbSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WmBl8pvjfyYUszw1TS/giphy.gif" width="600" height="600"/>
</div>

