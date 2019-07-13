# Libraries
  library(readr)
  library(nnet)
  library(gridExtra)


  model_Name <- "DigitRecognizer"

# Importing data
  train <- read.csv("mnist_train_42k.csv")
  test <- read.csv("mnist_test_28k.csv")

# Predictos & target variables
  predictors <- train[,-1]
  target <- train[,1]
  train_label <- as.factor(train[,1])

# PCA
  predictors_reduced <- predictors/255
  predictors_cov <- cov(predictors_reduced)
  pca_predictors <- prcomp(predictors_cov)


  variance_explained <- as.data.frame(pca_predictors$sdev^2/sum(pca_predictors$sdev^2))
  variance_explained <- cbind(c(1:784),variance_explained,cumsum(variance_explained[,1]))
  colnames(variance_explained) <- c("No_of_Principal_Components","Individual_Variance_Explained",
                            "Cumulative_Variance_Explained")

# Plot
  plot(variance_explained$No_of_Principal_Components,variance_explained$Cumulative_Variance_Explained, 
       xlim = c(0,100),type='b',pch=16,xlab = "Principal Components",
       ylab = "Cumulative Variance Explained",main = 'Principal Components vs Cumulative Variance')

  variance_explained_summary <- variance_explained[seq(0,100,5),]

# png("datatable_variance_explained.png",height = 800,width =1000)
# p <-tableGrob(variance_explained_summary)
# 
# grid.arrange(p)
# dev.off()
  predictors_final <- as.matrix(predictors_reduced) %*% pca_predictors$rotation[,1:45]
  
  train_label <- as.factor(train_label)
  target <- class.ind(target)
# print(predictors[1:5,1:5])
# print(target[1:5,])

# Model
  model_final <- nnet(predictors_final,target,size=150,softmax=TRUE,maxit=100,MaxNWts = 80000)

# test_target <- as.factor(test[,1])

  test_reduced <- test/255
  test_final <- as.matrix(test_reduced) %*%  pca_predictors$rotation[,1:45]
  
  prediction <- predict(model_final,test_final,type="class")
  prediction <- as.data.frame(prediction)
  final_prediction<- cbind(as.data.frame(1:nrow(prediction)),prediction)
  save.image(file=paste(model_Name,"-Model.RData",sep=''))

# actual <- as.double(unlist(test[,1]))

# accuracy = round(mean(actual == prediction) * 100, 2)
# accuracy

# as <- cbind(actual,prediction)
  colnames(final_prediction) <- c("ImageId","Label")
  write.csv(final_prediction,file="prediction.csv",row.names=FALSE)
