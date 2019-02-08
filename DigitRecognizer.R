library(readr)
library(nnet)
library(gridExtra)

model_Name <- "DigitRecognizer"
train <- read.csv("mnist_train.csv")
test <- read.csv("mnist_test.csv")


cat(sprintf("Training set has %d rows and %d columns\n", nrow(train), ncol(train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(test), ncol(test)))

X <- train[,-1]
Y <- train[,1]
train_label <- train[,1]


X_reduced <- X/255
X_cov <- cov(X_reduced)
pca_X <- prcomp(X_cov)


variance_explained <- as.data.frame(pca_X$sdev^2/sum(pca_X$sdev^2))
variance_explained <- cbind(c(1:784),variance_explained,cumsum(variance_explained[,1]))
colnames(variance_explained) <- c("No_of_Principal_Components","Individual_Variance_Explained",
                          "Cumulative_Variance_Explained")

plot(variance_explained$No_of_Principal_Components,variance_explained$Cumulative_Variance_Explained, 
     xlim = c(0,100),type='b',pch=16,xlab = "Principal Components",
     ylab = "Cumulative Variance Explained",main = 'Principal Components vs Cumulative Variance')

variance_explained_summary <- variance_explained[seq(0,100,5),]
variance_explained_summary
png("datatable_variance_explained.png",height = 800,width =1000)
p <-tableGrob(variance_explained_summary)

grid.arrange(p)
dev.off()
X_final <- as.matrix(X_reduced) %*% pca_X$rotation[,1:45]

train_label <- as.factor(train_label)
Y <- class.ind(Y)
print(X[1:5,1:5])
print(Y[1:5,])

final_seed <- 150
set.seed(final_seed)
model_final <- nnet(X_final,Y,size=150,softmax=TRUE,maxit=100,MaxNWts = 80000)

testlabel <- as.factor(test[,1])

test_reduced <- test/255
test_final <- as.matrix(test_reduced[,-1]) %*%  pca_X$rotation[,1:45]

prediction <- predict(model_final,test_final,type="class")
prediction <- as.data.frame(prediction)
final_prediction<- cbind(as.data.frame(1:nrow(prediction)),prediction)
save.image(file=paste(model_Name,"-Model.RData",sep=''))

actual <- as.double(unlist(test[,1]))

accuracy = round(mean(actual == prediction) * 100, 2)
accuracy

as <- cbind(actual,prediction)
colnames(final_prediction) <- c("ImageId","Label")
write.csv(final_prediction,file="prediction.csv",row.names=FALSE)