rm(list = ls())
# import appropriate libraries
library(plotmo)
require(plotmo)
library(glmnet)
library(kknn)
library(tree)
library(randomForest)
library(gbm)

# setting a seed
set.seed(12345)

# importing our data
CarData <- read.csv("https://raw.githubusercontent.com/brianlukoff/sta371g/master/data/cars.csv")

# save data into a dataframe
CDdf <- data.frame(CarData)

# attaching our dataframe to allow us to call variables more easily
attach(CDdf)

# exploratory analysis
plot(CDdf$MPG, CDdf$After1975)
boxplot(MPG~After1975, data=CDdf)
boxplot(MPG~Origin, data=CDdf)
scatter.smooth(x=CDdf$HP, y=CDdf$MPG)
scatter.smooth(x=CDdf$Acceleration, y=CDdf$MPG)
scatter.smooth(x=CDdf$Cylinders, y=CDdf$MPG)
scatter.smooth(x=CDdf$Displacement, y=CDdf$MPG)
scatter.smooth(x=CDdf$Weight, y=CDdf$MPG)
scatter.smooth(x=CDdf$Weight, y=CDdf$HP)
cor(CDdf$Weight, CDdf$HP)
# Weight and HP are highly correlated so we just need to include one: 0.865
cor(CDdf$Displacement, CDdf$Weight)
# Weight and Displacement are highly correlated so we just need to include one: 0.933
scatter.smooth(x=CDdf$Weight, y=CDdf$Displacement)
cor(CDdf$Displacement, CDdf$HP)
cor(CDdf$Weight, CDdf$Cylinders)
cor(CDdf$Cylinders, CDdf$HP)
CarModel <- lm(MPG ~.,data = CDdf)
summary(CarModel)

# summary of the variable we are predicting
summary(MPG)

### Running a LASSO regression to decide which variables are most important to include in our knn model ###

# Converting a categorical variable into a numeric one
CDdf$After1975 <- unclass(CDdf$After1975)

# Scaling our weight variable
CDdf$Weight <- CDdf$Weight*(1/1000)

# Removing origin data due to its the fact that it is the least important variable to consider when predicting MPG (see better explanation in lm section)
CDdf1 <- CDdf[,-c(8)]

# ceating a matrix of interactions
XXCD <- model.matrix(MPG~., data=data.frame(CDdf1))[,-1]

# number of observations
n = dim(CDdf)[1]

# creating a random sample of size 300 without replacement
tr = sample(1:n, 
            size =  300, 
            replace = FALSE) 

# running the LASSO fit and plotting it
Lasso.Fit = glmnet(XXCD[tr,],MPG[tr],alpha = 1)
plot_glmnet(Lasso.Fit, label = 5)

# doing a 10-fold cv for selecting our parameter lambda
CV.L = cv.glmnet(XXCD[tr,], MPG[tr],alpha= 1)

# lowest RMSE given various values of lambda from LASSO
low_RMSE_LASSO = (sqrt(CV.L$cvm))
low_RMSE_LASSO

# LASSO lambda values
LamL = CV.L$lambda.min

# ploting lambda values
plot(log(CV.L$lambda),sqrt(CV.L$cvm),
     main="LASSO CV (k=10)",xlab="log(lambda)",
     ylab = "RMSE",col=4,type="b",cex.lab=1.2)
abline(v=log(LamL),lty=2,col=2,lwd=2)

# coefficient of the lambda model
coef.L = predict(CV.L,type="coefficients",s=LamL)
coef.L

# making a new dataframe without the predictors Acceleration and Displacement
CDdf2 = CDdf1[,-c(3,6)]

### Performing a K-fold analysis to find the optimal K-value for our model ###

#Define the number of folds
kcv = 10 

#Size of the fold 
n0 = round(n/kcv, #Number of observations in the fold
           0) #Rounding with 0 decimals

#Number of neighbors for different models to iterate through
kk <- 1:150

#MSE matrix
out_MSE = matrix(0, #matrix filled with zeroes
                 nrow = kcv, #number of rows
                 ncol = length(kk)) #number of columns

#Vector of indices that have already been used inside the for
used = NULL

#The set of indices not used (will be updated removing the used)
set = 1:n

for(j in 1:kcv){
  
  if(n0<length(set)){ #If the set of 'not used' is > than the size of the fold
    val = sample(set, size = n0) #then sample indices from the set
  }
  
  if(n0>=length(set)){ #If the set of 'not used' is <= than the size of the fold
    val=set #then use all of the remaining indices as the sample
  }
  
  #Create the train and test matrices
  train_i = CDdf2[-val,] #Every observation except the ones whose indices were sampled
  test_i = CDdf2[val,] #The observations whose indices sampled
  
  for(i in kk){
    
    #The current model
    near = kknn(MPG ~., #The formula
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
  set = (1:n)[-used]
  
  #Printing on the console the information that you want
  #Useful to keep track of the progress of your loop
  cat(j,"folds out of",kcv,'\n')
}

#Calculate the mean of MSE for each k
mMSE = apply(out_MSE, #Receive a matrix
             2, #Takes its columns (it would take its rows if this argument was 1)
             mean) #And for each column, calculate the mean

# finding the value of the lowest RMSE
lowest_RMSE = min(sqrt(mMSE))
lowest_RMSE

#Complexity x RMSE graph
plot(log(1/kk),sqrt(mMSE),
     xlab="Complexity (log(1/k))",
     ylab="out-of-sample RMSE",
     col=4, #Color of line
     lwd=2, #Line width
     type="l", #Type of graph = line
     cex.lab=1.2, #Size of labs
     main=paste("kfold(",kcv,")")) #Title of the graph

#Find the index of the minimum value of mMSE
best = which.min(mMSE)

#Inclusing text at specific coordinates of the graph
text(log(1/kk[best]),sqrt(mMSE[best])+0.1, #Coordinates
     paste("k=",kk[best]), #The actual text
     col=2, #Color of the text
     cex=1.2) #Size of the text
text(log(1/2),sqrt(mMSE[2])+0.3,paste("k=",2),col=2,cex=1.2)
text(log(1/150)+0.4,sqrt(mMSE[150]),paste("k=",150),col=2,cex=1.2)

### Our best KNN model given a set k-value and given specific parameters ###

train = data.frame(CDdf2)
test = data.frame(CDdf2)
ind = order(test[,1]) #saving the indexes of the first column ordered, ordering the MPG column by smallest to largest
test = test[ind,]

near = kknn(MPG~.,train,test,k=best,kernel = "rectangular")
plot(CDdf2,MPG,main=paste("k=",best),pch=19,cex=0.8,col="black")

### Trees attempt ###

#First get a big tree using a small value of mindev (which forces big trees)
temp = tree(MPG~., #Formula
            data=CDdf, #Data frame
            mindev=.0001) #The within-node deviance must be at least

#this times that of the root node for the node to be split
cat('First create a big tree size: \n')
print(length(unique(temp$where))) #Number of leaf nodes

#Then prune it down to one with 7 leaves
CDdf.tree = prune.tree(temp, #The tree model
                       best=7) #Only the seven best nodes
cat('Pruned tree size: \n')
print(length(unique(CDdf.tree$where))) #Number of new leaf nodes

#Plot the tree
plot(CDdf.tree,
     type="uniform") #branches of uniform length
text(CDdf.tree,col="blue",label=c("yval"),cex=.8)

#Plot data with fit
CDdf.fit = predict(CDdf.tree) #Get training fitted values

plot(MPG,Weight, #Data
     cex=.5, #Size of points
     pch=16) #Type of point
oo=order(Weight) #Order the indices of variable lstat
lines(Weight[oo],CDdf.fit[oo], #Fitted values (step function)
      col='red', #Color of line
      lwd=3) #Line width

#cvals=c(9.725,4.65,3.325,5.495,16.085,19.9) #Cutpoints from tree

### random forest ###

#Get rf fits for different number of trees
#Note: to get this to work I had to use maxnodes parameter of randomForest!!!
ntreev = c(10,500,2000) #Different numbers of trees
nset = length(ntreev) #size of the for loop
fmat = matrix(0,n,nset) #Matrix of fits

for(i in 1:nset) {
  cat('Random Forest model: ',i,"; Number of Trees: ", ntreev[i],'\n')
  rffit = randomForest(MPG~Displacement, #Formula
                       data=CDdf1, #Data frame
                       ntree=ntreev[i]) #Number of trees in the forest
                       #maxnodes=15) #Maximum number of nodes in each tree
  fmat[,i] = predict(rffit) #Predicted values for the fits
  #min_error_tree = min(ntreev[i])
  #aux = mean((test_i[,1]-predict(rffit))^2)
  #out_MSE[j,i] = aux
}



par(mfrow=c(1,1)) #Plot window: 1 row, 1 column
#plot oob error using last fitted rffit which has the largest ntree. out-of-bag = oob
plot(rffit)

test.cases <- sample(1:392,80)
training.cases <- setdiff(1:392,test.cases)
training.set <- CDdf1[training.cases,]
test.set <- CDdf1[test.cases,]



yhat = predict(rffit,newdata = test.set)
sqrt(mean((test.set$MPG - yhat)^2))

#Plot fits
par(mfrow=c(1,3))#Plot window: 1 row, 1 column
oo = order(CDdf)
for(i in 1:nset) {
  plot(CDdf,MPG, #Plot data
       xlab='xvars',ylab='MPG')
  lines(CDdf[oo],fmat[oo,i], #Plot ordered fitted values
        col=(i+1), #Line color
        lwd=3) #Line width
  title(main=paste('Bagging ntrees = ',ntreev[i]))
}

### variable importance ###

#Fit boost and plot  variable importance
boostfit = gbm(MPG~., #Formula (. means that all variables of df are included)
               data=CDdf, #Data
               distribution='gaussian',
               interaction.depth=2, #Maximum depth of each tree
               n.trees=100, #Number of trees
               shrinkage=.2) #Learning rate

par(mfrow=c(1,1)) #Plot window: 1 row, 1 column
p=ncol(CDdf)-1 #Number of covariates (-1 because one column is the response)
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
rffit = randomForest(MPG~., #Formula (. means all variables are included)
                     data=CDdf,#Data frame
                     mtry=3, #Number candidates variables at each split
                     ntree=500)#Number of trees in the forest

#Variance importance for Random Forest model
varImpPlot(rffit)


####


