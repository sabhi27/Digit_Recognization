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
  prin_comp <- prcomp(predictors_cov)

  names(prin_comp)
  
# Output the mean of variables
  View(prin_comp$center)
  
# Output the SD of variables
  prin_comp$scale
  
# Rotation matrix where each column contains PC loading vector
  View(prin_comp$rotation)

# Plot esultant principal components 
  biplot(prin_comp, scale = 0)
  
# Compute SD of each Principal component
  SD <- prin_comp$sdev
  
# Compute Variance
  Var <- SD^2
  
# Proportion of Variance Explained
  variance_explained <- Var/sum(Var)
  variance_explained_df <- data.frame(c(1:784),variance_explained,cumsum(variance_explained))
  colnames(variance_explained_df) <- c("No_of_Principal_Components","Individual_Variance_Explained",
                            "Cumulative_Variance_Explained")

# Scree Plot
  plot(variance_explained,xlab = "Principal Components",
       ylab = "Proportion of Variance Explained",xlim=c(1,100),
       type="b")
  
# Cummulative scree plot
  plot(cumsum(variance_explained), xlab = "Principal Component",
  ylab = "Cumulative Proportion of Variance Explained",
  type = "b")


# png("datatable_variance_explained.png",height = 800,width =1000)
# p <-tableGrob(variance_explained_summary)
# 
# grid.arrange(p)
# dev.off()
  
  predictors_final <- as.matrix(predictors_reduced) %*% prin_comp$rotation[,1:260]
  
  train_label <- as.factor(train_label)
  target <- class.ind(target)
# print(predictors[1:5,1:5])
# print(target[1:5,])

# Neural Network Model
  model_final <- nnet(predictors_final,target,size=150,softmax=TRUE,maxit=100,MaxNWts = 80000)

# test_target <- as.factor(test[,1])

  test_reduced <- test/255
  test_final <- as.matrix(test_reduced) %*%  prin_comp$rotation[,1:260]
  
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

# options(repos = BiocInstaller::biocinstallRepos())
# getOption("repos")
# options(rsconnect.http = "rcurl")
