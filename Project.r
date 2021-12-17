data <- read.csv("C:/Users/anara/OneDrive/Desktop/2021Fall/MAS456/FinalProjectDataset.csv")
test_data = data[is.na(data$disc_hire),]
train_data = data[!is.na(data$disc_hire),]
View(test_data)


## Logistic Regression

glm.fits <- glm(
  disc_hire ??? . ,
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
test_roc = roc(train_data$disc_hire ~ glm.probs, plot = TRUE, print.auc = TRUE)


## Random Forests

library(randomForest)
train_data$disc_hire = as.factor(train_data$disc_hire)
bag.data <- randomForest(disc_hire ??? ., data = train_data, importance = TRUE, proximity=T)
yhat.bag <- predict(bag.data , newdata = train_data, type = "prob")
forest.pred <- rep(0, n)
probs = as.vector(yhat.bag[, 2])
forest.pred[probs > .5] = 1
mean(forest.pred == train_data$disc_hire)
test_roc = roc(train_data$disc_hire ~ probs, plot = TRUE, print.auc = TRUE)

