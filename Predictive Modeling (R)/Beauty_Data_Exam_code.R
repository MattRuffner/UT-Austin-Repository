rm(list = ls())
set.seed(1)
# import appropriate libraries
library(plotmo)
require(plotmo)
library(glmnet)
library(kknn)
library(tree)
library(randomForest)
library(gbm)


# importing data
BeautyData <- read.csv("/Users/ruffner/Downloads/BeautyData.csv")

# putting the data into a dataframe
beauty <- data.frame(BeautyData)

# using the attach function on our dataframe to make calling variables easier
attach(beauty)

# plot our two variables of interest to get an idea of what type of model to use
plot(BeautyScore, CourseEvals)

# check distribution of predictor variable:
plot(density(BeautyScore))

# I see there appears to be a linear relationship meaning I will probably want to use a linear regression model, 
# the question now becomes what factors affect beauty scores.
cor(BeautyScore, female)
cor(BeautyScore, lower)
scatter.smooth(female, BeautyScore)
scatter.smooth(lower, BeautyScore)

# separate my data set into training and test sets via 80-20 rule:
n.beauty = dim(beauty)[1]
test.cases.beauty <- sample(1:n.beauty, size=n.beauty*(.2))
training.cases.beauty <- setdiff(1:n.beauty,test.cases.beauty)
training.set.beauty <- beauty[training.cases.beauty,]
test.set.beauty <- beauty[test.cases.beauty,]

# trying a simple linear regression between beauty and course evals:
beauty.lin.reg <- lm(CourseEvals~BeautyScore)
summary(beauty.lin.reg)
plot(BeautyScore,CourseEvals)
abline(beauty.lin.reg)

# predicting CourseEval using BeautyScore in test data set with linear regression:
beauty.lin.reg.train <-lm(CourseEvals~ BeautyScore, data= training.set.beauty)
predicted.y <- predict.lm(beauty.lin.reg.train, test.set.beauty, interval = "confidence", level = 0.95)
test.y = test.set.beauty$CourseEvals
lm.MSE = mean((predicted.y - test.y)^2)
lm.RMSE = sqrt(lm.MSE)

# predicting CourseEval using BeautyScore in test data set with knn using 10-fold cv:
#Define the number of folds
kcv = 10 

#Size of the fold 
n0 = round(n.beauty/kcv, #Number of observations in the fold
           0) #Rounding with 0 decimals

#Number of neighbors for different models to iterate through
kk <- 1:100

#MSE matrix
out_MSE = matrix(0, #matrix filled with zeroes
                 nrow = kcv, #number of rows
                 ncol = 100) #number of columns

#Vector of indices that have already been used inside the for
used = NULL

#The set of indices not used (will be updated removing the used)
set = 1:n.beauty

for(j in 1:kcv){
  
  if(n0<length(set)){ #If the set of 'not used' is > than the size of the fold
    val = sample(set, size = n0) #then sample indices from the set
  }
  
  if(n0>=length(set)){ #If the set of 'not used' is <= than the size of the fold
    val=set #then use all of the remaining indices as the sample
  }
  
  train_i = beauty[-val,] #Every observation except the ones whose indices were sampled
  test_i = beauty[val,] #The observations whose indices sampled
  
  for(i in kk){
    
    #The current model
    near = kknn(CourseEvals ~ BeautyScore, #The formula
                train = train_i, #The train matrix/df
                test = test_i, #The test matrix/df
                k=i, #Number of neighbors
                kernel = "rectangular") #Type of kernel (see help for more)
    
    #Calculating the MSE of current model
    aux = mean((test_i[,1]-near$fitted)^2)
    
    #Store the current MSE
    out_MSE[j,i] = aux
  }
  
  #The union of the indices used currently and previously
  used = union(used,val)
  
  #The set of indices not used is updated
  set = (1:n.beauty)[-used]
  
  #Printing on the console the information that you want
  #Useful to keep track of the progress of your loop
  cat(j,"folds out of",kcv,'\n.beauty')
}

#Calculate the mean of MSE for each k
mMSE = apply(out_MSE, #Receive a matrix
             2, #Takes its columns (it would take its rows if this argument was 1)
             mean) #And for each column, calculate the mean

# finding the value of the lowest RMSE
lowest_RMSE = min(sqrt(mMSE)) 
lowest_RMSE

# k-val of lowest RMSE
k.val <- which.min(sqrt(mMSE))
k.val

# plotting knn model for k=9
beauty.knn.train <-kknn(CourseEvals~ BeautyScore,training.set.beauty, test.set.beauty, k=k.val, kernel = "rectangular")
plot(BeautyScore, CourseEvals, main=paste("k=",k.val),pch=19,cex=0.8,col="black")
ind = order(test.set.beauty$BeautyScore) 
test.set.beauty$BeautyScore = test.set.beauty$BeautyScore[ind]
lines(test.set.beauty$BeautyScore,beauty.knn.train$fitted,col=2,lwd=2)

# calculating MSE and RMSE for knn
test.y = test.set.beauty$CourseEvals
knn.MSE = mean((beauty.knn.train$fitted - test.y)^2)
knn.RMSE = sqrt(knn.MSE)


boostfit = gbm(CourseEvals~., 
               data=beauty, #Data
               distribution='gaussian',
               interaction.depth=2, #Maximum depth of each tree
               n.trees=100, #Number of trees
               shrinkage=.2) #Learning rate

p=ncol(beauty)-1 #Number of covariates (-1 because one column is the response)
vsum=summary(boostfit,plotit=FALSE) #This will have the variable importance info
row.names(vsum)=NULL #Drop variable names from rows.

#The default plot of the package
plot(vsum) #This is the default of the package.
#An alternative is presented below

#Plot variable importance
plot(vsum$rel.inf,axes=F,pch=16,col='red')
axis(1,labels=vsum$var,at=1:p)
axis(2)
for(i in 1:p){
  lines(c(i,i),c(0,vsum$rel.inf[i]),lwd=4,col='blue')
}


#Fit random forest and plot variable importance
rffit = randomForest(CourseEvals~., #Formula (. means all variables are included)
                     data=beauty,#Data frame
                     mtry=3, #Number candidates variables at each split
                     ntree=500)#Number of trees in the forest

#Variance importance for Random Forest model
varImpPlot(rffit)

# tree model to predict beauty score as it handles categorical variables better

# data frame without course evals
beauty2 <- beauty[,-c(1)] 

temp = tree(BeautyScore~., #Formula
            data=beauty2, #Data frame
            mindev=.0001) #The within-node deviance must be at least

#this times that of the root node for the node to be split
cat('First create a big tree size: \n')
print(length(unique(temp$where))) #Number of leaf nodes

#Then prune it down to one with 7 leaves
beauty.tree = prune.tree(temp, #The tree model
                       best=7) #Only the seven best nodes
cat('Pruned tree size: \n')
print(length(unique(CDdf.tree$where))) #Number of new leaf nodes

#Plot the tree
plot(beauty.tree,
     type="uniform") #branches of uniform length
text(beauty.tree,col="blue",label=c("yval"),cex=.8)

#Plot data with fit
beauty.fit = predict(beauty.tree) #Get training fitted values

plot(MPG,Weight, #Data
     cex=.5, #Size of points
     pch=16) #Type of point
oo=order(Weight) #Order the indices of variable lstat
lines(Weight[oo],CDdf.fit[oo], #Fitted values (step function)
      col='red', #Color of line
      lwd=3) #Line width

plot(density(BeautyData$BeautyScore))
plot(density(BeautyData$CourseEvals))
cor(BeautyData$CourseEvals, BeautyData$BeautyScore)
boxplot(BeautyData$CourseEvals, main="Course Evaluations", sub=paste("Outlier Location: ", boxplot.stats(BeautyData$CourseEvals)$out))
boxplot(BeautyData$BeautyScore, main="Beauty Score")
boxplot.stats(BeautyData$CourseEvals)
BeautyDataModel <- lm(BeautyScore ~ CourseEvals, data=BeautyData)
summary(BeautyDataModel)
