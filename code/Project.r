data <- read.csv("../data/FinalProjectDataset.csv")
test_data = data[is.na(data$disc_hire),]
test_data_y = test_data[, 1 , drop = FALSE]
train_data = data[!is.na(data$disc_hire),]
train_data_y = train_data[, 1 , drop = FALSE]
View(test_data)


## Logistic Regression

set.seed(1)
glm.fits <- glm(
  disc_hire ~ . ,
  data = train_data , family = binomial
)
n=dim(train_data)[1]
m=dim(test_data)[1]
glm.probs <- predict(glm.fits , type = "response")
glm.pred <- rep(0, n)
glm.pred[glm.probs > 0.5] = 1
mean(glm.pred == train_data$disc_hire)

glm.probs1 <- predict(glm.fits , newdata = test_data, type = "response")
glm.pred1 <- rep(0, m)
glm.pred1[glm.probs1 > .5] = 1
glm

library(pROC)
test_roc = roc(train_data$disc_hire ~ glm.probs, plot = TRUE, print.auc = TRUE) # 0.878


## Random Forests

set.seed(1)
library(randomForest)
train_dataRF = train_data
train_dataRF$disc_hire = as.factor(train_data$disc_hire)
bag.data <- randomForest(disc_hire ~ ., data = train_dataRF, importance = TRUE, proximity=T)
yhat.bag <- predict(bag.data , newdata = train_data, type = "prob")
forest.pred <- rep(0, n)
probs = as.vector(yhat.bag[, 2])
forest.pred[probs > .5] = 1
mean(forest.pred == train_data$disc_hire)
test_roc = roc(train_data$disc_hire ~ probs, plot = TRUE, print.auc = TRUE) # 0.984

## Logistic Regression with Lasso Regularization

library(glmnet)
set.seed (1)
grid <- 10^seq(10, -2, length = 100)
lasso.mod <- glmnet(as.matrix(train_data[, 2:18]), as.factor(train_data[, 1]), alpha = 1,
                    lambda = grid, family = "binomial")
cv.out <- cv.glmnet(as.matrix(train_data[, 2:18]), as.factor(train_data[, 1]), alpha = 1, family = "binomial")
plot(cv.out)
bestlam <- cv.out$lambda.min # 0.003141718
lasso.probs <- predict(lasso.mod , s = bestlam ,
                      newx = as.matrix(train_data[, 2:18]), type= "response")
lasso.pred = rep(0,n)
lasso.pred[lasso.probs > .5] = 1
mean(lasso.pred == train_data$disc_hire) # 0.8910606
test_roc = roc(train_data$disc_hire ~ as.vector(lasso.probs), plot = TRUE, print.auc = TRUE) # 0.878

## KNN

library(caret)
trControl <- trainControl(method  = "cv",
                          number  = 10)
fit <- train(as.factor(disc_hire) ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:10),
             trControl  = trControl,
             metric     = "Accuracy",
             data       = train_data) # 7

knnPredict <- predict(fit,newdata = train_data[, 2:18] )
mean(knnPredict == train_data$disc_hire) # 0.852264
