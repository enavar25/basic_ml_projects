##############################################
############# Loading Libraries ##############
##############################################

library(caret)
library (e1071) # This library is needed to use svm()
library(nnet) # needed for multinom
library(rpart)
library(dplyr)
library(class) # needed to use knn()
library(randomForest)
# install.packages("DataExplorer") # Uncomment if not installed
library(DataExplorer)
library(corrplot)
# install.packages("kernlab") # Uncomment if not installed
library(kernlab)
# install.packages("yardstick") # Uncomment if not installed
library(yardstick)
library(ggplot2)



###############################################
############# Preparing Datasets ##############
###############################################

# ------------------------------------------

# Reading in all the datasets 

df1 <- read.csv("dataset_5secondWindow%5B1%5D.csv") # formed by twelve features, four for each sensor.
df1$target <- as.factor(df1$target)

df2 <- read.csv("dataset_5secondWindow%5B2%5D.csv") # formed by eight sensor and thirty-two features.
df2$target <- as.factor(df2$target)

df3 <- read.csv("dataset_5secondWindow%5B3%5D.csv") # all nine relevant sensors and thirty-six features
df3$target <- as.factor(df3$target)

# ------------------------------------------

# Attempt to merge all datasets 

test1 <- merge(df1, df2) # Merging df1 and df2 together
test2 <- merge(test1, df3) # Merging test1 and df3 togsether, all 3 datasets merged together

setdiff(colnames(test2), colnames(df3)) # character(0)

# We see that there is no point in mergeing all 3 datasets since df3 has all possible features
# Hence, we will mostly be using df3 for our models

# ------------------------------------------

# Checking if missing values exist in any of the datasets

na_1 <-sapply(df1, function(y) sum(is.na(y))) # Returns number of NA's in each column
sum(na_1)

na_2 <-sapply(df2, function(y) sum(is.na(y))) # Returns number of NA's in each column
sum(na_2)

na_3 <-sapply(df3, function(y) sum(is.na(y))) # Returns number of NA's in each column
sum(na_3)

# ------------------------------------------

## Getting all of the training and test data from all three datasets

train_idx <- c(1:4000) # indexes of rows to be used in the training set 

# -------------- Using df1 ----------------

ncol_1 <- ncol(df1)

X_train.1 <- df1[train_idx, ] # Gets all row indexes in train_idx, and all columns except the last one
y_train.1 <- df1[train_idx, ncol_1] # Gets all row indexes in train_idx, and just the last column

X_test.1 <- df1[-train_idx, ] # Gets all row indexes not in train_idx, and all columns except the last one
y_test.1 <- df1[-train_idx, ncol_1] # Gets all row indexes not in train_idx, and just the last column

# -------------- Using df2 ----------------

ncol_2 <- ncol(df2)

X_train.2 <- df2[train_idx, ] 
y_train.2 <- df2[train_idx, ncol_2] 


X_test.2 <- df2[-train_idx, ] 
y_test.2 <- df2[-train_idx, ncol_2] 


# -------------- Using df3 ----------------

ncol_3 <- ncol(df3)

X_train.3 <- df3[train_idx, ] 
y_train.3 <- df3[train_idx, ncol_3] 


X_test.3 <- df3[-train_idx, ] 
y_test.3 <- df3[-train_idx, ncol_3] 

# -------------- Normalizing df3 ----------------

scaler <- preProcess(X_train.3[, -ncol_3])

X_train.3_norm <- predict(scaler, X_train.3[, -ncol_3])
X_train.3_norm['target'] <- y_train.3

X_test.3_norm <- predict(scaler, X_test.3[, -ncol_3])
X_test.3_norm['target'] <- y_test.3

# ---------------

mean_norm_minmax <- function(x){
  (x- mean(x)) /(max(x)-min(x))
}

X_train.3_norm <- as.data.frame(lapply(X_train.3[, -ncol_3], mean_norm_minmax))
X_train.3_norm['target'] <- y_train.3

X_test.3_norm <- as.data.frame(lapply(X_test.3[, -ncol_3], mean_norm_minmax))
X_test.3_norm['target'] <- y_test.3

# ---------------

norm_minmax <- function(x){
  (x- min(x)) /(max(x)-min(x))
}

X_train.3_norm <- as.data.frame(lapply(X_train.3[, -ncol_3], norm_minmax))
X_train.3_norm['target'] <- y_train.3

X_test.3_norm <- as.data.frame(lapply(X_test.3[, -ncol_3], norm_minmax))
X_test.3_norm['target'] <- y_test.3

# ---------------


###############################################
############# Analyzing Datasets ##############
###############################################

# ------------------------------------------

# Using DataExplorer

plot_bar(X_train.3) # Helps visualize all categorical variables in the dataset (only target in this case)

# From the above plot, it seems like the proportion of our classes in the training set are
# about the same, which is good. Our data is not biased in any way. 

plot_histogram(X_train.3) # Helps visualize all continous variables in the dataset 

plot_correlation(X_train.3, type = "c") # Correlation plot of all continous variables

?plot_correlation
plot_boxplot(X_train.3, by = "target") # Boxplots plot grouped by target

#################################
###### Logistic Regression ######
#################################

model <- multinom(target~., data=X_train.3)
predicted.classes <- model %>% predict(X_test.3)

mean(predicted.classes == X_test.3$target) # 72.16059 (accuracy)
  
# ----------------------------

# Logistic Regression With Cross Validation

set.seed(4)
train_control <- trainControl(method = "cv", number = 10)
model <- train(target ~ .,
               data = X_train.3,
               trControl = train_control,
               method = "multinom")

predicted.classes <- model %>% predict(X_test.3)
mean(predicted.classes == X_test.3$target) # 72.21342% accuracy 

# ----------------------------

# Logistic Regression after normalizing (use preProcess)

model <- multinom(target~., data=X_train.3_norm)
predicted.classes <- model %>% predict(X_test.3_norm)

accuracy <- mean(predicted.classes == X_test.3$target) # 72.21342% accuracy
accuracy

# ------------ Plotting the Confusion Matrix  -------------

true_predicted <- data.frame(true = y_test.3, predicted = predicted.classes)
true_predicted$true <- as.factor(true_predicted$true)
true_predicted$predicted <- as.factor(true_predicted$predicted) 


cm <- conf_mat(true_predicted, true, predicted)

heading <- paste("Confusion Matrix - Logistic Regression",
                 paste("Accuracy: ",round(accuracy, 6)*100, "%"),
                 sep = "\n")

#  scale_fill_gradient() +
autoplot(cm, type = "heatmap") +
  scale_fill_distiller(palette = "Greens") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle(heading)

#####################################
###### Support Vector Machines ######
#####################################


# ----------------------------

svm.1_1 <- svm(target~., data=X_train.1, 
            method="C-classification", kernel="radial", 
            gamma=0.1, cost=5000)

# summary(svm.1_1)

pred.1 <- predict(svm.1_1, X_test.1)
mean(pred.1 == X_test.1$target) # 0.8151083 (accuracy)

# ----------------------------

svm.1_2 <- svm(target~., data=X_train.1, 
             method="C-classification", kernel="polynomial", 
             gamma=0.1, cost=10000)

summary(svm.1_2)

pred.2 <- predict(svm.1_2, X_test.1)

mean(pred.2 == X_test.1$target) # 0.7744321 (accuracy)

# ----------------------------

# Using svm with tuning

# Trying it on X_train.1 because it's much faster
svm.2_1 = tune.svm(target~., data=X_train.1, kernel="radial", type="C", gamma = c(0.01,0.1),
                   cost=c(0.1,5,10,100))

summary(svm.2_1)
svm.2_1 <- svm.2_1$best.model # best_gamma = 0.1, best_cost = 0.01

pred.2 <- predict(svm.2_1, X_test.1)
mean(pred.2 == X_test.1$target)

# ----------------------------

# Using svm with Cross Validation
svmGrid <-  expand.grid(C=100, sigma = 0.01)

train_control <- trainControl(method = "repeatedcv", repeats = 5)
model <- train(target ~ .,
               data = X_train.3,
               trControl = train_control,
               method = "svmRadial",
               tuneGrid = svmGrid)

predicted.classes <- model %>% predict(X_test.3)
mean(predicted.classes == X_test.3$target) # 89.75172

# ----------  

# Using svm after normalization 

svm.2_1 <- svm(target~., data=X_train.3_norm, 
               type="C", kernel="radial", 
               gamma=0.1, cost=1000) 

summary(svm.2_1)

pred.2 <- predict(svm.2_1, X_test.3_norm)

mean(pred.2 == X_test.3$target) # 0.9244585 (accuracy)

## Sadly, the accuracy did not change even after normzalizing the dataset (preProcess)

# ---------- 

svm.2_1 <- svm(target~., data=X_train.3, 
               type="C", kernel="radial", 
               gamma=0.1, cost=100) # best model for svm 

summary(svm.2_1)

pred.2 <- predict(svm.2_1, X_test.3)

accuracy <- mean(pred.2 == X_test.3$target) # 92.44585% (accuracy)
accuracy

# ------------ Plotting the Confusion Matrix  -------------

true_predicted <- data.frame(true = y_test.3, predicted = pred.2)
true_predicted$true <- as.factor(true_predicted$true)
true_predicted$predicted <- as.factor(true_predicted$predicted) 


cm <- conf_mat(true_predicted, true, predicted)

heading <- paste("Confusion Matrix - SVM",
                 paste("Accuracy: ",round(accuracy, 6)*100, "%"),
                 sep = "\n")

autoplot(cm, type = "heatmap") +
  scale_fill_distiller(palette = "Greens")+
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle(heading) 


#####################################
########## Decision Trees ###########
#####################################


model <- rpart(target~., data=X_train.2, method = "class")

preds <- predict(model, newdata=X_test.2, type = "class")

mean(preds == X_test.2$target) # 63.28579% accuracy 


# -----------------

model <- rpart(target~., data=X_train.3, method = "class")

preds <- predict(model, newdata=X_test.3, type = "class")

accuracy <- mean(preds == X_test.3$target) # 75.43582% accuracy 
accuracy

# confusionMatrix(y_test.3, preds)

# ------------ Plotting the Confusion Matrix  -------------

true_predicted <- data.frame(true = y_test.3, predicted = preds)
true_predicted$true <- as.factor(true_predicted$true)
true_predicted$predicted <- as.factor(true_predicted$predicted) 


cm <- conf_mat(true_predicted, true, predicted)

heading <- paste("Confusion Matrix - Decision Trees",
                 paste("Accuracy: ",round(accuracy, 6)*100, "%"),
                 sep = "\n")

autoplot(cm, type = "heatmap") +
  scale_fill_distiller(palette = "Greens")+
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle(heading)

######################################
######## k-Nearest Neighbors #########
######################################

set.seed(234)
pred.knn <- knn(train=X_train.3[, -ncol_3], test=X_test.3[, -ncol_3], 
                cl=y_train.3, k=1)


mean(pred.knn == X_test.3$target)



pred.knn <- knn(train=X_train.3_norm[, -ncol_3], test=X_test.3_norm[, -ncol_3], 
                cl=y_train.3, k=1)



accuracy <- mean(pred.knn == X_test.3$target)
accuracy

train_control <- trainControl(method = "cv", number = 5)
model <- train(target ~ .,
               data = X_train.3_norm,
               trControl = train_control,
               method = "knn",
               tuneGrid = expand.grid(k = 1:10),
               metric = "Accuracy")

model

predicted.classes <- model %>% predict(X_test.3_norm)
mean(predicted.classes == X_test.3$target) # 93.92499%







# confusionMatrix(pred.knn, y_test.3) # 92.71% accuracy 

# ------------ Plotting the Confusion Matrix -------------

true_predicted <- data.frame(true = y_test.3, predicted = pred.knn)
true_predicted$true <- as.factor(true_predicted$true)
true_predicted$predicted <- as.factor(true_predicted$predicted) 


cm <- conf_mat(true_predicted, true, predicted)

heading <- paste("Confusion Matrix - kNN (k = 1)",
                 paste("Accuracy: ",round(accuracy, 6)*100, "%"),
                 sep = "\n")

autoplot(cm, type = "heatmap") +
  scale_fill_distiller(palette = "Greens") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle(heading)


# ------------------------------------------------------------

####################################
########### Naive Bayes ############
####################################

nb <- naiveBayes(target~., data=X_train.3)
y_pred <- predict(nb, X_test.3, type = "class")
mean(y_pred == X_test.3$target) # 54.0412%


nb <- naiveBayes(target~., data=X_train.3_norm)
y_pred <- predict(nb, X_test.3_norm, type = "class")
mean(y_pred == X_test.3$target) # 57.0523% (mini_max_norm)


nb <- naiveBayes(target~., data=X_train.3_norm) # Best model 
y_pred <- predict(nb, X_test.3_norm, type = "class")
accuracy <- mean(y_pred == X_test.3$target) # 62.38774% (mean_norm)
accuracy

# ------------ Plotting the Confusion Matrix -------------

true_predicted <- data.frame(true = y_test.3, predicted = y_pred)
true_predicted$true <- as.factor(true_predicted$true)
true_predicted$predicted <- as.factor(true_predicted$predicted) 


cm <- conf_mat(true_predicted, true, predicted)

heading <- paste("Confusion Matrix - Naive Bayes",
                 paste("Accuracy: ",round(accuracy, 6)*100, "%"),
                 sep = "\n")

autoplot(cm, type = "heatmap") +
  scale_fill_distiller(palette = "Greens") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle(heading)









#####################################
########## Random Forests ###########
#####################################

set.seed(4)
model.rf <- randomForest(target~., data = X_train.3)
pred.rf <- predict(model.rf, newdata=X_test.3)

accuracy <- mean(pred.rf == X_test.3$target) # 94.98151% accuracy 
accuracy

# ------------ Plotting the Confusion Matrix -------------

true_predicted <- data.frame(true = y_test.3, predicted = pred.rf)
true_predicted$true <- as.factor(true_predicted$true)
true_predicted$predicted <- as.factor(true_predicted$predicted) 


cm <- conf_mat(true_predicted, true, predicted)

heading <- paste("Confusion Matrix - Random Forests",
                 paste("Accuracy: ",round(accuracy, 6)*100, "%"),
                 sep = "\n")

autoplot(cm, type = "heatmap") +
  scale_fill_distiller(palette = "Greens") +
  theme(plot.title = element_text(hjust = 0.5)) +
  ggtitle(heading)



