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
set.seed(1)
trControl <- trainControl(method  = "cv",
                          number  = 10)
fit <- train(as.factor(disc_hire) ~ .,
             method     = "knn",
             tuneGrid   = expand.grid(k = 1:10),
             trControl  = trControl,
             metric     = "Accuracy",
             data       = train_data) # 8

knnPredict <- predict(fit,newdata = train_data[, 2:18], type = "prob")
probs = as.vector(knnPredict[, 2])
mean(knnPredict == train_data$disc_hire) # 0.852264
knn.pred = rep(0,n)
knn.pred[probs>.5]=1
mean(knn.pred == train_data$disc_hire)
test_roc = roc(train_data$disc_hire ~probs, plot = TRUE, print.auc = TRUE) # 0.90

## SVM with radial kernel

library(e1071)
set.seed(1)
train_dataSVM = train_data
train_dataSVM$disc_hire = as.factor(train_data$disc_hire)
tune.out <- tune(svm , disc_hire ~ ., data = train_dataSVM,
                 kernel = "radial",
                 ranges = list(
                   cost = c(0.1 , 1, 10, 100, 1000) ,
                   gamma = c(0.5, 1, 2, 3, 4),
                   probability = TRUE
                 )
)
summary(tune.out)

svm.probs = predict(tune.out$best.model , train_dataSVM, probability = TRUE)
svm.probs=as.vector(attr(svm.probs, "probabilities")[, 1])
svm.pred = rep(0,n)
svm.pred[svm.probs > .5] = 1
mean(svm.pred == train_data$disc_hire)
test_roc = roc(train_data$disc_hire ~ svm.pred, plot = TRUE, print.auc = TRUE) # 0.922

## Single - layer Neural Network

library(keras)
set.seed(1)
x= scale(model.matrix(disc_hire ~ . -1, data = train_data))
modnn <- keras_model_sequential () %>%
   layer_dense(units = 50, activation = "relu",
                input_shape = ncol(x)) %>%
   layer_dropout(rate = 0.4) %>%
   layer_dense(units = 1, activation = "sigmoid")
modnn %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)
history <- modnn %>% fit(
  x, train_data$disc_hire, epochs = 600, batch_size = 32)
nn.pred = rep(0,n)
nn.pred[npred>0.5]=1
mean(nn.pred == train_data$disc_hire)
test_roc = roc(train_data$disc_hire ~ npred, plot = TRUE, print.auc = TRUE) # 0.930

