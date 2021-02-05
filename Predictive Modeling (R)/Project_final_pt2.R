rm(list=ls())

cars <- read.csv("https://raw.githubusercontent.com/brianlukoff/sta371g/master/data/cars.csv")
View(cars)


test.cases <- sample(1:392, 80 )
training.cases <- setdiff(1:392, test.cases)
training.set <- cars[training.cases,]
test.set <- cars[test.cases, ]


attach(training.set)

# Can we avoid categorical variable?
boxplot(MPG ~ Origin, data=cars, ylab="MPG",
        xlab="Origin", col='darkgray')
boxplot(MPG ~ After1975, data=cars, ylab="MPG",
        xlab="After 1975", col='darkgray')


# Stepwise: first round of elimination-> acceleration

library(MASS)
origin.model <- lm(MPG ~ Cylinders + Displacement + HP + Weight +Acceleration+ After1975 + Origin, data = cars)
summary(origin.model)
anova(origin.model)  #both categorical variables are statistically significant with p-val < 0.05
step <- stepAIC(origin.model, direction="both")
step$anova # display results



origin.model_1 <- lm(MPG ~ HP + Weight + Displacement + Cylinders + After1975 + Origin, data=cars)
summary(origin.model)
anova(origin.model)
# good R2, but a really complex model
# Would like to get a better model


# Forward:
library(leaps)
par(bg = "white")
plot(regsubsets(MPG ~ Cylinders + Displacement + HP + Weight + After1975 + Origin), scale = "adjr2", col = 7)
# Looks like model with only weight or with weight and After1975 is most parsimonious

# But why is that?

cor(cars$HP, cars$Weight)
cor(cars$Displacement, cars$Weight)
cor(cars$Cylinders, cars$Weight)

# Check for multicolinearity 
library(car)
modelx <- lm(MPG ~ Cylinders + Displacement + HP + Weight, data = cars)
vif(modelx)


# Closer look at Dummy variable: After1975
n=dim(cars)[1]
dn1 = rep(0, #rep will repeat the first argument (0 in this case)
          n) #n times (the second argument)
dn1[After1975=="Yes"]=1 #Now, conditioning the values to be 1 if After1975=="YES"

dn2 = rep(0,n)
dn2[After1975=="No"]=1

CarsModel = lm(MPG~ dn1+ Weight)
plot(Weight,MPG,xlab="lbs")
abline(coef=CarsModel$coef[c(1,3)], #the coefficient fit
       col=2, #Line color
       lwd=2) #Line width
points(Weight[dn2==1],MPG[dn2==1], #The points of that group
       col=2, #Color of points
       pch=19) #Type of point

abline(coef=c(CarsModel$coef[1]+CarsModel$coef[2],CarsModel$coef[3]),col=4,lwd=2)
points(Weight[dn1==1],MPG[dn1==1],col=4,pch=19)

abline(lsfit(Weight,MPG), #OLS Fit with the dummies
       col=1,lwd=2,lty=2)

legend("topleft",
       c("After1975==Yes","After1975==No","No Dummies"),
       lty=c(1,1,2), #Line type
       lwd=c(2,2,2), #Line width
       col=c(4,2,1)) #Color of lines

summary(CarsModel)
anova(CarsModel)


# Are these models the best we can get?
Model <- lm(MPG ~ Weight)
summary(Model)
qqnorm(resid(Model))

plot(Weight, Model$res, #x variable against residuals
     pch=18, #Type of point
     col=4,
     xlab="Weight",ylab="Std. Res")

# Is this the best model? 
Model1 <- lm(MPG ~ Weight + After1975)
summary(Model1)


# Check for normality
qqnorm(resid(Model1))


# Residual Plot:
plot(Weight, Model1$res, #x variable against residuals
     pch=16, #Type of point
     col=4,
     xlab="Weight",ylab="Std. Res")


#Let us add x^2 among the covariates and analyze the model again
Weight2 = Weight^2
Model2 = lm(MPG~Weight+Weight2+After1975)
summary(Model2)

qqnorm(resid(Model2))

plot(Weight,Model2$res, #x variable against residuals
     pch=18, #Type of point
     col=4,
     xlab="months",ylab="Std. Res")


#Let is see the fitted curves for each model
plot(Weight,MPG, #Scatterplot of data
     pch=18) #Type of point
abline(Model2)
lines(Weight,Model2$fitted,lwd=2,col=3) #Model 2 curve
lines(Weight,Model1$fitted,lwd=2,col=2) #Model 1 curve
legend("bottomright",
       legend = c("Model 1", "Model 2"),
       col = c(2,3), #Color of lines
       lty = c(1,1), #Line type
       lwd = c(2,2)) #Line width

# Does adding interaction work? 
Model3 <- lm(MPG ~ Weight*After1975)
summary(Model3)
plot(Weight, Model3$res, #x variable against residuals
     pch=18, #Type of point
     col=4,
     xlab="Weight",ylab="Std. Res")


# Use Log-log model to fix issues;
attach(training.set)
Logmodel = lm(log(MPG)~log(Weight) + After1975)

summary(Logmodel)
anova(Logmodel)

plot(log(Weight),Logmodel$res, #x variable against residuals
     pch=16, #Type of point
     col=4,
     xlab="Weight",ylab="Std. Res")


par(mfrow=c(1,2))

plot(Weight,MPG,#Scatterplot
     pch=19, #Type of point
     col=4, #Color of point
     main="Plug-in Prediction")

i <- order(Weight[After1975 == "Yes"])
ii <- order(Weight[After1975 == "No"])


#Fitted line : After1975 = yes
lines((Weight[After1975 == "Yes"])[i],exp(Logmodel$fitted[After1975 == "Yes"])[i],
      col=2, #Color of point
      lwd=2) #Line width

#95% intervals:  After1975 = yes
# SE of 0.1321 could be difference since the training set is randomly picked
lines((Weight[After1975 == "Yes"])[i],exp((Logmodel$fitted[After1975 == "Yes"])[i] - 2*0.1295),col=2,lwd=2,lty=2)
lines((Weight[After1975 == "Yes"])[i],exp((Logmodel$fitted[After1975 == "Yes"])[i] + 2*0.1295),col=2,lwd=2,lty=2)

# Do a separate plot
plot(Weight,MPG,#Scatterplot
     pch=19, #Type of point
     col=4, #Color of point
     main="Plug-in Prediction")

#Fitted line: After1975 = No
lines((Weight[After1975 == "No"])[ii],exp(Logmodel$fitted[After1975 == "No"])[ii],
      col=17, #Color of point
      lwd=2) #Line width


#95% intervals: After1975 = no
# SE of 0.1321 could be difference since the training set is randomly picked
lines((Weight[After1975 == "No"])[ii],exp((Logmodel$fitted[After1975 == "No"])[ii] - 2*0.1295),col=17,lwd=2,lty=2)
lines((Weight[After1975 == "No"])[ii],exp((Logmodel$fitted[After1975 == "No"])[ii] + 2*0.1295),col=17,lwd=2,lty=2)

#Combined look:
plot(Weight,MPG,#Scatterplot
     pch=19, #Type of point
     col=4, #Color of point
     main="Plug-in Prediction")

#Fitted lines: 
lines((Weight[After1975 == "Yes"])[i],exp(Logmodel$fitted[After1975 == "Yes"])[i],
      col=2, #Color of point
      lwd=2) #Line width
lines((Weight[After1975 == "No"])[ii],exp(Logmodel$fitted[After1975 == "No"])[ii],
      col=17, #Color of point
      lwd=2) #Line width

#95% intervals: After1975 = yes
lines((Weight[After1975 == "Yes"])[i],exp((Logmodel$fitted[After1975 == "Yes"])[i] + 2*0.1295),col=2,lwd=2,lty=2)
#95% intervals: After1975 = no
lines((Weight[After1975 == "No"])[ii],exp((Logmodel$fitted[After1975 == "No"])[ii] - 2*0.1295),col=17,lwd=2,lty=2)

legend("topright",
       c("After1975==Yes","After1975==No"),
       lty=c(1,1), #Line type
       lwd=c(2,2), #Line width
       col=c(2,17)) #Color of lines


#Also, let us evaluate on the log scale
plot(log(Weight),log(MPG), #Scatterplot
     pch=19, #Type of point
     col=4, #Color of point
     main="Plug-in Prediction")
abline(lsfit(log(Weight),log(MPG)),lwd=2,col=2)
abline(c(Logmodel$coef[1]-2*0.1295,Logmodel$coef[2]),lwd=2,col=2,lty=2)
abline(c(Logmodel$coef[1]+2*0.1295,Logmodel$coef[2]),lwd=2,col=2,lty=2)


par(mfrow=c(1,1)) #Plot window: 1 row, 1 column
#Now, the residual plot looks good
plot(log(Weight),Logmodel$res,
     pch=19,col=2,ylab="residuals",main="Residual Plot")
abline(0,0)


# Would a log-log model with interaction make our model even better?
Logmodel_1 = lm(log(MPG)~log(Weight)*After1975)
summary(Logmodel_1)
anova(Logmodel_1)


# Validation
rm(list=ls())
cars <- read.csv("https://raw.githubusercontent.com/brianlukoff/sta371g/master/data/cars.csv")
View(cars)


test.cases <- sample(1:392, 80 )
training.cases <- setdiff(1:392, test.cases)
training.set <- cars[training.cases,]
test.set <- cars[test.cases, ]

attach(test.set)

Logmodel = lm(log(MPG)~log(Weight) + After1975)
summary(Logmodel)
anova(Logmodel)


#Now, the residual plot looks good
plot(log(Weight),Logmodel$res,
     pch=19,col=2,ylab="residuals",main="Residual Plot")
abline(0,0)


par(mfrow=c(1,2))

plot(Weight,MPG,#Scatterplot
     pch=19, #Type of point
     col=4, #Color of point
     main="Plug-in Prediction")

i <- order(Weight[After1975 == "Yes"])
ii <- order(Weight[After1975 == "No"])

lines((Weight[After1975 == "Yes"])[i],exp(Logmodel$fitted[After1975 == "Yes"])[i],
      col=2, #Color of point
      lwd=2) #Line width
lines((Weight[After1975 == "No"])[ii],exp(Logmodel$fitted[After1975 == "No"])[ii],
      col=17, #Color of point
      lwd=2) #Line width


#95% intervals: After1975 = yes
lines((Weight[After1975 == "Yes"])[i],exp((Logmodel$fitted[After1975 == "Yes"])[i] - 2*0.1335),col=2,lwd=2,lty=2)
lines((Weight[After1975 == "Yes"])[i],exp((Logmodel$fitted[After1975 == "Yes"])[i] + 2*0.1335),col=2,lwd=2,lty=2)

#95% intervals: After1975 = no
lines((Weight[After1975 == "No"])[ii],exp((Logmodel$fitted[After1975 == "No"])[ii] - 2*0.1335),col=17,lwd=2,lty=2)
lines((Weight[After1975 == "No"])[ii],exp((Logmodel$fitted[After1975 == "No"])[ii] + 2*0.1335),col=17,lwd=2,lty=2)



#Also, let us evaluate on the log scale
plot(log(Weight),log(MPG), #Scatterplot
     pch=19, #Type of point
     col=4, #Color of point
     main="Plug-in Prediction")
abline(lsfit(log(Weight),log(MPG)),lwd=2,col=2)
abline(c(Logmodel$coef[1]-2*0.1335,Logmodel$coef[2]),lwd=2,col=2,lty=2)
abline(c(Logmodel$coef[1]+2*0.1335,Logmodel$coef[2]),lwd=2,col=2,lty=2)


par(mfrow=c(1,1)) #Plot window: 1 row, 1 column




#################################################################3
# RMSE & MSE 
#################################################################

attach(test.set)


Logmodel = lm(log(MPG)~log(Weight) + After1975)

MSE_val <- (exp(Logmodel$fitted.values) - MPG)^2
MSE <- sum(MSE_val)/(80)
#Mean square error = 
MSE

RMSE <- sqrt(MSE)
# Root mean square error =
RMSE

#########################################################################
# Prediction with log-log model on the entire dataset
#########################################################################


Logmodel = lm(log(MPG)~log(Weight) + After1975)
summary(Logmodel)
anova(Logmodel)

#Creating data frame for prediction (After 1975 = Yes)
#Weight size from 1500 to 8000
X_future1 <- data.frame(Weight=seq(1500,8000,by=10), After1975 ="Yes")

#Calculating 95% and 99% prediction interval
Future1_1 = predict(Logmodel, X_future1,
                    interval = "prediction",se.fit=T)

Future1_2 = predict(Logmodel, X_future1,
                    interval = "prediction",se.fit=T,level=0.99)

par(mfrow=c(1,1)) #Plot window: 1 row, 1 column

#Plotting the model(After 1975 = Yes)
plot(Weight,MPG, #The data
     xlim=c(1500,8000), #the range of my predicted X
     ylim=range(exp(Future1_1$fit)), #Range of my fit
     pch=19, #Type of point
     cex.lab=1.3) #Size of lab


lines(X_future1$Weight,exp(Future1_1$fit[,1]), #Lines of fitted
      col=4, #Color of line
      lwd=4,) #Line width
lines(X_future1$Weight,exp(Future1_1$fit[,2]), #Lines of 95% prediction interval
      col=14, #Color of line
      lwd=2, #Line width
      lty=2) #Line type
lines(X_future1$Weight,exp(Future1_1$fit[,3]), #Lines of 95% prediction interval
      col=14,lwd=2,lty=2)
lines(X_future1$Weight,exp(Future1_2$fit[,2]), #Lines of 99% prediction interval
      col=8,lwd=2,lty=2)
lines(X_future1$Weight,exp(Future1_2$fit[,3]), #Lines of 99% prediction interval
      col=8,lwd=2,lty=2)



#Creating data frame for prediction (After 1975 = No)
#Weight size from 1500 to 8000
X_future2 <- data.frame(Weight=seq(1500,8000,by=10), After1975 ="No")

#Calculating 95% and 99% prediction interval
Future2_1 = predict(Logmodel, X_future2,
                    interval = "prediction",se.fit=T)

Future2_2 = predict(Logmodel, X_future2,
                    interval = "prediction",se.fit=T,level=0.99)


#Plotting the model(After 1975 = No)
plot(Weight,MPG, #The data
     xlim=c(1500,8000), #the range of my predicted X
     ylim=range(exp(Future2_1$fit)), #Range of my fit
     pch=19, #Type of point
     cex.lab=1.3) #Size of lab


lines(X_future2$Weight,exp(Future2_1$fit[,1]), #Lines of fitted
      col=2, #Color of line
      lwd=4,) #Line width
lines(X_future2$Weight,exp(Future2_1$fit[,2]), #Lines of 95% prediction interval
      col=4, #Color of line
      lwd=2, #Line width
      lty=2) #Line type
lines(X_future2$Weight,exp(Future2_1$fit[,3]), #Lines of 95% prediction interval
      col=4,lwd=2,lty=2)
lines(X_future2$Weight,exp(Future2_2$fit[,2]), #Lines of 99% prediction interval
      col=5,lwd=2,lty=2)
lines(X_future2$Weight,exp(Future2_2$fit[,3]), #Lines of 99% prediction interval
      col=5,lwd=2,lty=2)


#Plotting the model(After 1975 = all)
plot(Weight,MPG, #The data
     xlim=c(1500,8000), #the range of my predicted X
     ylim=range(exp(Future1_1$fit)), #Range of my fit
     pch=19, #Type of point
     cex.lab=1.3) #Size of lab

lines(X_future1$Weight,exp(Future1_1$fit[,1]), #Lines of fitted(After 1975 = Yes)
      col=4, #Color of line
      lwd=4,) #Line width


lines(X_future2$Weight,exp(Future2_1$fit[,1]), #Lines of fitted(After 1975 = No)
      col=2, #Color of line
      lwd=4,) #Line width

legend("topleft",
       c("After1975==Yes","After1975==No"),
       lty=c(1,1), #Line type
       lwd=c(2,2), #Line width
       col=c(4,2)) #Color of lines

###Example for linear regression 
attach(cars)
mean(MPG)
K = 3.741/23.44592
K

# Ferrari Ferrari 308 GTS (man. 5) , model year 1980, version for North America U.S.
exp(10.79444 - 0.98018*log(1468) + 0.207542)
# Actual mpg = 13 mpg
# How many SD away?
abs((13 - 47.22136)/3.741)

# 1971 Chevrolet Vega 2300 Hatchback Coupe 140-1
exp(10.79444 - 0.98018*log(2249) + 0.207542*0)
# Actual mpg = 21.8
# how many SD away? 
(25.2588 - 21.8)/3.741



##########################################################################
#LASSO
##########################################################################
# Loading the library
library(glmnet)


x_vars <- model.matrix(MPG~.,data=cars)[,-1]
y_var <- cars$MPG
lambda_seq <- 10^seq(2, -2, by = -.1)

# Splitting the data into test and train
set.seed(86)
train = sample(1:nrow(x_vars), nrow(x_vars)/2)
x_test = (-train)
y_test = y_var[x_test]

cv_output <- cv.glmnet(x_vars[train,], y_var[train],
                       alpha = 1, lambda = lambda_seq, nfolds = 5)
plot(cv_output)
# identifying best lamda
best_lam <- cv_output$lambda.min
best_lam

# Rebuilding the model with best lamda value identified
lasso_best <- glmnet(x_vars[train,], y_var[train], alpha = 1, lambda = best_lam)
pred <- predict(lasso_best, s = best_lam, newx = x_vars[x_test,])

final <- cbind(y_var[y_test], pred)
# Checking the first six obs
head(final)

#getting the list of important variables
# Inspecting beta coefficients
coef(lasso_best)

######
#Trees and Random Forest
######
rm(list=ls()) #Removes every object from your environment
cars <- read.csv("https://raw.githubusercontent.com/brianlukoff/sta371g/master/data/cars.csv")

attach(cars)

library(tree)
library(rpart)
library(MASS)
library(randomForest) #Package for the RF model
library(MASS)

#MPG~Displacement
#Displcaement is the variable with highest importance
#First get a big tree using a small value of Displacment (which forces big trees)
temp = tree(MPG~Displacement, #Formula
            data=cars, #Data frame
            mindev=.0001) #The within-node deviance must be at least
#this times that of the root node for the node to be split
cat('First create a big tree size: \n')
print(length(unique(temp$where))) #Number of leaf nodes

#Then prune it down to one with 7 leaves
cars.tree=prune.tree(temp, #The tree model
                       best=7) #Only the seven best nodes

cat('Pruned tree size: \n')
print(length(unique(cars.tree$where))) #Number of new leaf nodes


#Plot the tree and the fits.
par(mfrow=c(1,2)) #Plot window: 1 row, 2 columns

#Plot the tree
plot(cars.tree,
     type="uniform") #branches of uniform length
text(cars.tree,col="blue",label=c("yval"),cex=.8)

#Plot data with fit
cars.fit = predict(cars.tree) #Get training fitted values

#Scartterplot
plot(Displacement,MPG, #Data
     cex=.5, #Size of points
     pch=16) #Type of point
oo=order(Displacement) #Order the indices of variable lstat
lines(Displacement[oo],cars.fit[oo], #Fitted values (step function)
      col='red', #Color of line
      lwd=3) #Line width

################################################################################
## Fit a regression tree to MPG~Displacement+HP from the cars data.
## The tree is plotted as well as the corresponding partition of the two-dimensional
## x=(Displacement,HP) space.
################################################################################

#BEWARE
rm(list=ls()) #Removes every object from your environment

library(MASS)


cars <- read.csv("https://raw.githubusercontent.com/brianlukoff/sta371g/master/data/cars.csv")

attach(cars)

library(tree)

#Create new data frame
df2=cars[,c(3,4,1)] #Pick variables Displacement,HP,MPG
print(names(df2)) #Names of variables

#First get a big tree
temp = tree(MPG ~ .,#Formula (the point includes every variable)
            data=df2, #Data frame
            mindev=.0001) #The within-node deviance must be at least
#this times that of the root node for the node to be split
cat('First create a big tree size: \n')
print(length(unique(temp$where))) #Number of leaf nodes


#Then prune it down to one with 7 leaves (for example)
cars.tree=prune.tree(temp, #The tree model
                       best=7) #Only the seven best nodes
cat('Pruned tree size: \n')
print(length(unique(cars.tree$where))) #Number of new leaf nodes


#Plot tree and partition in x.
par(mfrow=c(1,2)) #Plot window: 1 row, 2 columns
#Plot the tree
plot(cars.tree,
     type="uniform") #branches of uniform length
text(cars.tree,col="blue",label=c("yval"),cex=.8)
#Plotting the covariate space
partition.tree(cars.tree)

################################################################################
## Fit a big tree to MPG~Displacement in the cars data using rpart instead of tree
##  and using cross-validation.
## Use rpart plotcp so do cross-validation.
## Plot: rpart plotcp cross-validation.
## Big off min loss cp value and plot tree for that cp value as well
## as a bigger cp value (smaller tree) and a smaller cp value (bigger tree).
## Plot: the three trees (from the three cp values) as well as the fitted function.
## Plot: the best tree using rpart.
################################################################################

#BEWARE
rm(list=ls()) #Removes every object from your environment
cars <- read.csv("https://raw.githubusercontent.com/brianlukoff/sta371g/master/data/cars.csv")
library(tree)
library(rpart)
library(MASS)


attach(cars)

#For this example, let us use another package for estimating trees
#The rpart package


#Reduce df to just lmed and lrat
bdf = cars[,c(3,1)] #Displacement and MPG

#Fit a big tree using rpart.control
big.tree = rpart(MPG~Displacement,
                 method="anova", #split maximizes the sum-of-squares between the new partitions
                 data=bdf, #data frame
                 control=rpart.control(minsplit=5, #the minimum number of obs in each leaf
                                       cp=.0005)) #complexity parameter (see rpart.control help)
nbig = length(unique(big.tree$where))
cat('Number of leaf nodes: ',nbig,'\n')


#Look at cross-validation for proning
par(mfrow=c(1,1)) #plot window: 1 row, 1 column
plotcp(big.tree) #possible cost-complexity prunings


#plot best tree

oo=order(bdf$Displacement) #Order the indices of the variable
bestcp=big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"] #Find best CP
#Printing Best CP
cat('Bestcp: ',bestcp,'\n')

#Pruning the tree using the cp
best.tree = prune(big.tree,cp=bestcp)

par(mfrow=c(1,2))
pfit = predict(best.tree) #Predict of the train values
plot(bdf, #Plot the data
     pch=16, #Type of point
     col='blue', #Color of the points
     cex=.5) #Size of the points
lines(bdf$Displacement[oo],pfit[oo], #Ordered data
      col='red', #Color of line
      lwd=2) #Line width
prp(best.tree) #Plot tree


################################################################################
## Random Forests: fit MPG~Displacement using random forests.
## Plot: oob error esitmation.
## Plot: fit from random forests for three different number of trees in forest.
################################################################################

#BEWARE
rm(list=ls()) #Removes every object from your environment
cars <- read.csv("https://raw.githubusercontent.com/brianlukoff/sta371g/master/data/cars.csv")
library(randomForest) #Package for the RF model
library(MASS)

attach(Boston)


#Get rf fits for different number of trees
#Note: to get this to work I had to use maxnodes parameter of randomForest!!!
set.seed(99) #To guarantee the same results
n = nrow(cars) #Sample size
ntreev = c(10,500,5000) #Different numbers of trees
nset = length(ntreev) #size of the for loop

fmat = matrix(0,n,nset) #Matrix of fits

for(i in 1:nset) {
  cat('Random Forest model: ',i,"; Number of Trees: ", ntreev[i],'\n')
  rffit = randomForest(MPG~Displacement, #Formula
                       data=cars, #Data frame
                       ntree=ntreev[i], #Number of trees in the forest
                       maxnodes=15) #Maximum number of nodes in each tree
  fmat[,i] = predict(rffit) #Predicted values for the fits
}


par(mfrow=c(1,1)) #Plot window: 1 row, 1 column
#plot oob error using last fitted rffit which has the largest ntree.
plot(rffit)


#Plot fits
par(mfrow=c(1,3))#Plot window: 1 row, 1 column
oo = order(cars$Displacement)
for(i in 1:nset) {
  plot(cars$Displacement,cars$MPG, #Plot data
       xlab='Displacement',ylab='MPG')
  lines(cars$Displacement[oo],fmat[oo,i], #Plot ordered fitted values
        col=(i+1), #Line color
        lwd=3) #Line width
  title(main=paste('Bagging ntrees = ',ntreev[i]))
}


################################################################################
## Fit MPG~Displacement, Ccars using boosting.
## Plot: fits for three different values of number of trees.
################################################################################

#BEWARE
rm(list=ls()) #Removes every object from your environment

library(gbm) #boost package
library(MASS)

cars <- read.csv("https://raw.githubusercontent.com/brianlukoff/sta371g/master/data/cars.csv")
attach(cars)


#Fit boosting for various number of trees
set.seed(99) #Seed to guarantee the same results
n = nrow(cars) #Sample size
ntreev = c(5,20,100) #Number of trees on boosting
nset = length(ntreev) #size of the for loop

fmat = matrix(0,n,nset) #Matrix of fits

for(i in 1:nset) {
  cat('Boosting model: ',i,"; Number of Trees: ", ntreev[i],'\n')
  boostfit = gbm(MPG~Displacement, #Formula
                 data=cars, #Data frame
                 distribution='gaussian',
                 interaction.depth=2, #Maximum depth of each tree
                 n.trees=ntreev[i], #Number of trees
                 shrinkage=.2) #Learning rate
  fmat[,i] = predict(boostfit,n.trees=ntreev[i]) #Predicted values for the fits
}


#Plot fits
par(mfrow=c(1,3))#Plot window: 1 row, 1 column
oo = order(cars$Displacement)
for(i in 1:nset) {
  plot(cars$Displacement,cars$MPG, #Plot data
       xlab='Displacement',ylab='MPG')
  lines(cars$Displacement[oo],fmat[oo,i], #Plot ordered fitted values
        col=(i+1), #Line color
        lwd=3) #Line width
  title(main=paste('Boosting ntrees = ',ntreev[i]))
}


