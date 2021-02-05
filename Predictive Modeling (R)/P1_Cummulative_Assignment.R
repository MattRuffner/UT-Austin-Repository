set.seed(12355)
library(ISLR)
library(MASS)
library(glmnet)
library(plotmo)
require(plotmo)
library(pls)
library(kknn)
library(class)
library(tree)
library(randomForest)
library(gbm)
data(Boston)
?Boston
#attach(Boston)

### A ###

rows <- dim(Boston)[1]
# each row holds values of different variables for a specific suburb in Boston

columns <- dim(Boston)[2]
# each column represents a different variable of interest for suburbs in Boston

### B ###

# pairwise scatterplot
pairs(~medv+age+ptratio+dis+crim,data=Boston,
      main="Boston Scatterplot Matrix")
# I made a pairwise scatterplot using columns:
#   1) median value of owner-occupied homes in \$1000s
#   2) proportion of owner-occupied units built prior to 1940
#   3) pupil-teacher ratio by town
#   4) weighted mean of distances to five Boston employment centres
# I noticed that the higher the proportion of units built before 1940 the more likely the unit is close to employment centres
# I noticed at a ptratio of about 20 there appears to be a much higher crime rate
# I noticed that there doesn't appear to be much correlation between the age and ptratio variables

### C ###

pairs(crim~zn+indus+chas+nox,data=Boston,
      main="crim Scatterplot Matrix")


scatter.smooth(zn, crim)
scatter.smooth(indus, crim)
scatter.smooth(chas, crim)
scatter.smooth(nox, crim)
scatter.smooth(rm, crim)
scatter.smooth(age, crim)
scatter.smooth(dis, crim)
scatter.smooth(rad, crim)
scatter.smooth(tax, crim)
scatter.smooth(ptratio, crim)
scatter.smooth(black, crim)
scatter.smooth(lstat, crim)
scatter.smooth(medv, crim)

# crime rate drops to almost zero once the median home value is $20000
# crime rate increases from zero once lower class population proportion increases past roughly 10%
# there is a lower crime rate in towns with higher proportions of blacks
# at a pupil-teacher ratio of just over 20 there is a large range of crime rates
# higher crime rate at higher property tax rates
# higher crime rate at higher property tax rates
# better accessibility to radial highways leads to higher crime rates
# there seems to be almost no crime when further away from employement centers
# higher crime rate in areas of  nitrogen oxide concentrations of about .55-.75

### D ###

plot(density(crim))
plot(density(tax))
plot(density(ptratio))
boxplot(crim, main="crime rate")
boxplot(tax, main="tax rate")
boxplot(ptratio, main="pupil teacher ratio")

# looking at boxplots for crime rate, tax rate, and pupil-teacher ratio:
# we dont see any significantly large outliers for either the tax rate and pupil-teacher ratios
# however, there are many significantly large outliers for the crime rate predictor
# the range of the tax-rate is from 187 to 711
# the range of the crim-rate is from 0.00632 to 88.9762
# the range of the pupil-teacher ratio is from 12.6 to 22

### E ###

sort(table(chas), decreasing = TRUE)[2]

# there are 35 suburbs that bound the Charles River

### F ###

summary(ptratio)[3]

# median student teacher ratio is 19.05

### G ### 

min_val = which.min(medv)
Boston[min_val,]

# there are two suburbs that are tied for the lowest median value of owner-occupied homes: they are located at indexes 399 and 406
# both have relatively high student teacher ratios, for idx 399: 20.2, for idx 406: 20.2.
# both have large black proportions, for idx 399: 396.90, for idx 406: 384.97.
# both have high property tax-rates, for idx 399: 666, for idx 406: 666.
# both have a lower number of rooms per dwelling, for idx 399: 5.453, for idx 406: 5.683.
# both show that all homes were buit prior to 1940, for idx 399: 100.0, for idx 406: 100.0.
# both have the highest index of accessibility to radial highways, for idx 399: 24, for idx 406: 24.
# both are close to employement centers, for idx 399: 1.4896, for idx 406: 1.4254.
# both have high proportions of non-retail business acres, for idx 399: 18.10, for idx 406: 18.10.
# both are not on the Charles River, for idx 399: 0, for idx 406: 0.
# both have no proportion of residential land zoned for lots over 25,000 sq. ft., for idx 399: 0, for idx 406: 0.
# both have relatively high nitrogen oxide concentrations, for idx 399: 0.6930, for idx 406: 0.6930.
# will they do differ a little in this regard, both have high proportion of lower class population, for idx 399: 30.59, for idx 406: 22.98. 
#
# where they differ quite significantly is the crime rates of the two suburbs. While they are both much higher than average the crime rate for suburb with index 399 is 38.35180 while the crime rate for suburb with index 406 is 67.92080 which is almost double
### H ###

subs_with_7ormore = dim(Boston[rm > 7, ])[1]
# there are 64 suburbs with more than 7 rooms per dwelling on average

eightormore = Boston[rm > 8, ]
subs_with_8ormore = dim(Boston[rm > 8, ])[1]
# there are 13 suburbs with more than 8 rooms per dwelling on average

# the suburbs with more than 8 rooms per dwelling on average:
# have higher median value of owner-occupied homes
# have lower proportions of lower class population
# have low crime rates
# skewed towards having more homes built before 1940
# skewed towards lower proportions of non-retail business acres


###################
### Chap 3 Q 15 ###
###################


# try to predict per capita crime rate using the other variables in this data set:

Czn = lm(crim~zn)
summary(Czn)
plot(zn,crim)
abline(Czn)
# coeffs: intercept= 4.45, zn= -0.074
# low p-val(5.5e-06) and relatively high t-stat(-4.6) so this would be classified as a statistically significant predictor of crime rate
# as can be seen from the plot, there is a slight negative relationship between the proportion of residential land zoned for lots over 25,000 sq.ft. and crime rate

Cindus = lm(crim~indus)
summary(Cindus)
plot(indus,crim)
abline(Cindus)
 #coeffs: intercept= -2.06, indus= 0.51
 #very low p-val(<2e-16) and very high t-stat(9.99) so this would be classified as a statistically significant predictor of crime rate
 #as can be seen by the plot, crime rate does not appear to change much depending on whether or not the suburb is on the river

Cchas = lm(crim~chas)
summary(Cchas)
plot(chas,crim)
abline(Cchas)
 #coeffs: intercept= 3.74, chas= -1.89
# p-val(0.21) and t-stat(-1.26) so this predictor is not a statistically significant predictor of crime rate
# as can be seen from the plot, there is a positive relationship between the proportion of non-retail business acres and crime rate

Cnox = lm(crim~nox)
summary(Cnox)
plot(nox,crim)
abline(Cnox)
# coeffs: intercept= -13.72, nox=31.249
# very low p-val(<2e-16) and very high t-stat(10.4) indicates that this is a statistically significant predictor of crime rate
# as can be seen from the plot, there is a positive relationship between crime rate and nitrogen oxide concentration

Crm = lm(crim~rm)
summary(Crm)
plot(rm,crim)
abline(Crm)
# coeffs: intercept= 20.48, rm= -2.68
# low p-val(6.35e-07) and high t-stat(-5.05) indicates that this is a statistically significant predictor of crime rate
# as can be seen from the plot, there is a negative relationship between average number of rooms and crime rate

Cage = lm(crim~age)
summary(Cage)
plot(age,crim)
abline(Cage)
# coeffs: intercept= -3.78, age= 0.11
# very low p-val(2.85e-16) and high t-stat(8.46) indicates that this is a statistically significant predictor of crime rate
 #as can be seen from the plot, there is a positive relationship between unit age and crime rate


Cdis = lm(crim~dis)
summary(Cdis)
plot(dis,crim)
abline(Cdis)
# coeffs: intercept= 9.5, dis= -1.55
# very low p-val(<2e-16) and high t-stat(-9.213) indicates that this is a statistically significant predictor of crime rate
# as can be seen from the plot, there is a negative relationship between distance to employment centers and crime rate

Crad = lm(crim~rad)
summary(Crad)
plot(rad,crim)
abline(Crad)
 #coeffs: intercept= -2.29, rad= 0.62
# very low p-val(<2e-16) and very high t-stat(18) indicates that this is a statistically significant predictor of crime rate
# as can be seen from the plot, there is a positive relationship between accessibility to radial highways and crime rate

Ctax = lm(crim~tax)
summary(Ctax)
plot(tax,crim)
abline(Ctax)
# coeffs: intercept= -8.53, tax= 0.03
# very low p-val(<2e-16) and very high t-stat(16.1) indicates that this is a statistically significant predictor of crime rate
# as can be seen from the plot, there is a positive relationship between crime rate and property tax rate

Cptratio = lm(crim~ptratio)
summary(Cptratio)
plot(ptratio,crim)
abline(Cptratio)
# coeffs: intercept= -17.65, ptratio= 1.15
# low p-val(2.94e-11) and high t-stat(6.8) indicates that this is a statistically significant predictor of crime rate
# as can be seen from the plot, there is a positive relationship between crime rate and pupil-teacher ratio

Cblack = lm(crim~black)
summary(Cblack)
plot(black,crim)
abline(Cblack)
# coeffs: intercept= 16.55, black= -0.036
# very low p-val(<2e-16) and high t-stat(-9.37) indicates that this is a statistically significant predictor of crime rate
# as can be seen from the plot, there is a negative relationship between crime rate and proportion of blacks by town

Clstat = lm(crim~lstat)
summary(Clstat)
plot(lstat,crim)
abline(Clstat)
#coeffs: intercept= -3.33, lstat= 0.55
# very low p-val(<2e-16) and very high t-stat(11.5) indicates that this is a statistically significant predictor of crime rate
# as can be seen from the plot, there is a positive relationship between crime rate and the proportion of lower class population

Cmedv = lm(crim~medv)
summary(Cmedv)
plot(medv,crim)
abline(Cmedv)
# coeffs: intercept= 11.8, medv= -0.36
# very low p-val(<2e-16) and high t-stat(-9.46) indicates that this is a statistically significant predictor of crime rate
# as can be seen from the plot, there is a negative relationship between median home value and crime rate

# B ###

mult_reg <- lm(crim~., data=Boston)
summary(mult_reg )
confint(mult_reg)
# looking at the summary of the model, I can see that the multiple R-squared value (0.45) is higher than the adjusted R-squared value (0.44)
# this indicates that our model's complexity outweighs its prediction accuracy. I will look to drop variables that don't contribute as much as others
# In addition, looking at the residuals, I can see that the median residual of -0.353 indicates that our model does a good job of correctly predicting crime rate
 #looking at the t-statistics and p-values, we can reject the null hypothesis for the following variables: rad, dis, medv, zn, and black

### C ###

# I look at the variable coefficients of each univariate linear regression model 
univar_reg_coeffs <- lm(crim ~ zn, data = Boston)$coefficients[2]
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ indus, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ chas, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ nox, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ rm, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ age, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ dis, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ rad, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ tax, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ ptratio, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ black, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ lstat, data = Boston)$coefficients[2])
univar_reg_coeffs <- append(univar_reg_coeffs, lm(crim ~ medv, data = Boston)$coefficients[2])

# I plot univariate vs. multiple linear regression coeffs:
plot(univar_reg_coeffs, mult_reg$coefficients[2:14], main = "Univariate vs. Multiple Linear Regression Coefficients", 
     xlab = "Univariate", ylab = "Multiple")


### D ###

# define a list with each predictor in it

# looking for evidence of a non-lin relationship with predictor:zn
model1 = lm(crim~ I(zn) + I(zn^2) + I(zn^3), data= Boston)
summary(model1)
# we can see that there is some sort of quadratic relationship here


# looking for evidence of a non-lin relationship with predictor:indus
model2 = lm(crim~ I(indus) + I(indus^2) + I(indus^3), data= Boston)
summary(model2)
# we can see that there is both a statistically significant quadratic and cubic term here


# looking for evidence of a non-lin relationship with predictor:chas
model3 = lm(crim~ I(chas) + I(chas^2) + I(chas^3), data= Boston)
summary(model3)
# we can see there is no quadratic or cubic term whatsoever indicating no such relationship with the chas predictor


# looking for evidence of a non-lin relationship with predictor:nox
model4 = lm(crim~ I(nox) + I(nox^2) + I(nox^3), data= Boston)
summary(model4)
# we can see that there is both a statistically significant quadratic and cubic term here indicating a non-linear relationship with this predictor


# looking for evidence of a non-lin relationship with predictor:rm
model5 = lm(crim~ I(rm) + I(rm^2) + I(rm^3), data= Boston)
summary(model5)
# there is a weak non-linear relationship here


# looking for evidence of a non-lin relationship with predictor:age
model6 = lm(crim~ I(age) + I(age^2) + I(age^3), data= Boston)
summary(model6)
# here there is a strong cubic relationship as well as a mild quadratic relationship with the age predictor


# looking for evidence of a non-lin relationship with predictor:dis
model7 = lm(crim~ I(dis) + I(dis^2) + I(dis^3), data= Boston)
summary(model7)
# we can see that there is both a statistically significant quadratic and cubic term here indicating a non-linear relationship with this predictor


# looking for evidence of a non-lin relationship with predictor:rad
model8 = lm(crim~ I(rad) + I(rad^2) + I(rad^3), data= Boston)
summary(model8)
# there is a weak non-linear relationship here


# looking for evidence of a non-lin relationship with predictor:tax
model9 = lm(crim~ I(tax) + I(tax^2) + I(tax^3), data= Boston)
summary(model9)
# there is a weak non-linear relationship here


# looking for evidence of a non-lin relationship with predictor:ptratio
model10 = lm(crim~ I(ptratio) + I(ptratio^2) + I(ptratio^3), data= Boston)
summary(model10)
# there is a strong non-linear relationship here


# looking for evidence of a non-lin relationship with predictor:black
model11 = lm(crim~ I(black) + I(black^2) + I(black^3), data= Boston)
summary(model11)
# there is a weak non-linear relationship here


# looking for evidence of a non-lin relationship with predictor:lstat
model12 = lm(crim~ I(lstat) + I(lstat^2) + I(lstat^3), data= Boston)
summary(model12)
# there is a weak non-linear relationship here


# looking for evidence of a non-lin relationship with predictor:medv
model13 = lm(crim~ I(medv) + I(medv^2) + I(medv^3), data= Boston)
summary(model13)
# we can see that there is both a statistically significant quadratic and cubic term here indicating a non-linear relationship with this predictor

#we conclude that predictors that have a significant non-linear relationship with crime rate are: medv, dis, nox, indus. the variables ptratio, and age also demonstrate a non-linear relationship

##############
### CH6 P9 ###
##############

colleges <- read.csv("/Users/ruffner/Documents/Matthew Ruffner Texas/Rstudio Stuff/College.csv")
#attach(colleges)


### A ###

# I was having trouble loading in the CSV correctly so I had to remove the names of each college to get the lin. model to work
colleges <- colleges[,-c(1)]

# defining training and test sets
n = dim(colleges)[1]
test.cases.colleges <- sample(1:n, size=n*(.2))
training.cases.colleges <- setdiff(1:n,test.cases.colleges)
training.set.colleges <- colleges[training.cases.colleges,]
test.set.colleges <- colleges[test.cases.colleges,]


### B ###

colleges.lin.mod = lm(Apps~ ., data=training.set.colleges)
summary(colleges.lin.mod)

# I now use the prediction from the training set and test it with the test set

test.colleges.MSE = mean((test.set.colleges[,2]-predict.lm(colleges.lin.mod, test.set.colleges))^2)
lm.test.RMSE.colleges = sqrt(test.colleges.MSE)
# I find an MSE of 1,581,753 or RMSE of 1257  which seems decent indicating linear regression might be a feasible model for prediction

### C ###

#Create a full matrix of interactions for training and test x-vals
XXcol.train <- model.matrix(Apps~., data=training.set.colleges)[,-1]
XXcol.test <- model.matrix(Apps~., data=test.set.colleges)[,-1]

# running a ridge fit and plotting it
Ridge.Fit.col = glmnet(XXcol.train, training.set.colleges$Apps,alpha=0)
plot_glmnet(Ridge.Fit.col, label = 5)

# doing a 10-fold cv for selecting our parameter lambda and plotting the result
CV.R.col = cv.glmnet(XXcol.train, training.set.colleges$Apps,alpha=0)
plot(CV.R.col)

# selecting lambda that minimizes MSE
LamR.col = CV.R.col$lambda.min
LamR.col
# we can see here that the best lambda for ridge regression is 343.6

# coefficients of the ridge model
coef.R.col = predict(CV.R.col,type="coefficients",s=LamR.col)

# creating a ridge model
ridge.model =glmnet(XXcol.train,training.set.colleges$Apps,alpha=0,lambda=LamR.col)

# finding MSE and RMSE
ridge.MSE=mean((test.set.colleges[,2] - predict(ridge.model,s=LamR.col ,newx=XXcol.test))^2)
ridge.test.RMSE.col = sqrt(ridge.MSE)
# I find an MSE of 3,470,528 or an RMSE of 1863 which is worse than the linear model error


### D ###

# running a LASSO fit and plotting it
LASSO.Fit.col = glmnet(XXcol.train, training.set.colleges$Apps,alpha=1)
plot_glmnet(LASSO.Fit.col, label = 5)

# doing a 10-fold cv for selecting our parameter lambda and plotting the result
CV.L.col = cv.glmnet(XXcol.train, training.set.colleges$Apps, alpha=1)
plot(CV.L.col)

# selecting lambda that minimizes MSE
LamL.col = CV.L.col$lambda.min
LamL.col
# the best lambda for LASSO appears to be 9.79

# coefficient of the LASSO model
coef.L.col = predict(CV.L.col,type="coefficients",s=LamL.col)
coef.L.col
# we can see that the variables Enroll and Personal get zeroed out
# we are left with 15 non-zero coefficient estimates

# creating a ridge model
LASSO.model =glmnet(XXcol.train,training.set.colleges$Apps,alpha=1,lambda=LamL.col)

# finding MSE and RMSE
LASSO.MSE=mean((test.set.colleges[,2] - predict(LASSO.model,s=LamR.col ,newx=XXcol.test))^2)
LASSO.test.RMSE.col = sqrt(LASSO.MSE)
# I find an MSE of 1,707,003 or an RMSE of 1306 which is an improvement upon the ridge model error but a little worse than the linear model error


### E ###

# pcr model with standardized predictors
pcr.model.col <- pcr(Apps~., data = training.set.colleges, scale = TRUE, validation = "CV")
summary(pcr.model.col)

# plot cross-validation MSE
validationplot(pcr.model.col ,val.type="MSEP")
# as can be seen by the plot, inlcuding all possible components gives us the lowest MSE but I will choose an M value of 5 as is gives us comparable error with much lower complexity


pcr.MSE<- mean((test.set.colleges$Apps - predict(pcr.model.col, test.set.colleges, ncomp=5))^2)
pcr.test.RMSE.col = sqrt(pcr.MSE)
# I find an MSE of 5,226,165 or an RMSE of 2286 which is less accurate than any of the previous models


### F ###

# pls model with standardized predictors
pls.model.col <- plsr(Apps~., data = training.set.colleges, scale = TRUE, validation = "CV")
summary(pls.model.col)

# plot cross-validation MSE
validationplot(pls.model.col ,val.type="MSEP")
# as can be seen by the plot, inlcuding 9 or more components gives us roughly the lowest MSE but I will choose an M value of 5 as is gives us comparable error with much lower complexity


pls.MSE<- mean((test.set.colleges$Apps - predict(pls.model.col, test.set.colleges, ncomp=5))^2)
pls.test.RMSE.col = sqrt(pls.MSE)
# I find an MSE of 2,447,294 or an RMSE of 1564 which is less accurate than any of the previous models except for the pcr model


### G ###

summary(Apps)
# Looking at all our results from each model and keeping in mind the large range of number of applications from colleges I infer the following:
# Using RMSE as a judge of how accurate our models are, all models gave me relatively similar RMSEs (arguably excluding the pcr model) I tend to prefer simpler and more interpretable models at the expense of a little higher RMSE.
# For that reason I would probably utilise PLS model as it is provides an error rate of 1564 which is comparable to the best possible error rate (1257) while only requiring 5 components
# Given the very large outliers that exist in the application data range, an average error rate of ~1564 or ~1257 between predicted values and true values of number of applications indicates relatively high model prediction accuracy.


###############
### CH6 P11 ###
###############

# using our previously loaded Boston data

### A ###

# constructing training and test sets:
n.bost = dim(Boston)[1]
test.cases.boston <- sample(1:n.bost, size=n.bost*(.2))
training.cases.boston <- setdiff(1:n.bost,test.cases.boston)
training.set.boston <- Boston[training.cases.boston,]
test.set.boston <- Boston[test.cases.boston,]

# using Ridge regression to predict crime rate:

#Create a full matrix of interactions for training and test x-vals
XXbos.train <- model.matrix(crim~., data=training.set.boston)[,-1]
XXbos.test <- model.matrix(crim~., data=test.set.boston)[,-1]

# running a ridge fit and plotting it
Ridge.Fit.bos = glmnet(XXbos.train, training.set.boston$crim,alpha=0)
plot_glmnet(Ridge.Fit.bos, label = 5)

# doing a 10-fold cv for selecting our parameter lambda and plotting the result
CV.R.bos = cv.glmnet(XXbos.train, training.set.boston$crim,alpha=0)
plot(CV.R.bos)

# selecting lambda that minimizes MSE
LamR.bos = CV.R.bos$lambda.min
LamR.bos
# we can see here that the best lambda for ridge regression is 0.52

# coefficients of the ridge model
coef.R.bos = predict(CV.R.bos,type="coefficients",s=LamR.bos)

# creating a ridge model
ridge.model.bos =glmnet(XXbos.train,training.set.boston$crim,alpha=0,lambda=LamR.bos)

# finding MSE and RMSE
ridge.MSE.bos= mean((test.set.boston[,2] - predict(ridge.model.bos,s=LamR.bos ,newx=XXbos.test))^2)
ridge.test.RMSE.bos = sqrt(ridge.MSE.bos)
# I find a cross validation estimate of MSE to be 617.14 or for RMSE to be 24.84


# using a LASSO regression to predict crime rate:


# running a LASSO fit and plotting it
LASSO.Fit.bos = glmnet(XXbos.train, training.set.boston$crim,alpha=1)
plot_glmnet(LASSO.Fit.bos, label = 5)

# doing a 10-fold cv for selecting our parameter lambda and plotting the result
CV.L.bos = cv.glmnet(XXbos.train, training.set.boston$crim, alpha=1)
plot(CV.L.bos)

# selecting lambda that minimizes MSE
LamL.bos = CV.L.bos$lambda.min
LamL.bos
# the best lambda for LASSO appears to be 0.072

# coefficient of the LASSO model
coef.L.bos = predict(CV.L.bos,type="coefficients",s=LamL.bos)
coef.L.bos
# we can see that the variables age and tax get zeroed out
# we are left with 11 non-zero coefficient estimates

# creating a ridge model
LASSO.model.bos =glmnet(XXbos.train,training.set.boston$crim,alpha=1,lambda=LamL.bos)

# finding MSE and RMSE
LASSO.MSE.bos= mean((test.set.boston[,2] - predict(LASSO.model.bos,s=LamR.bos ,newx=XXbos.test))^2)
LASSO.test.RMSE.bos = sqrt(LASSO.MSE.bos)
# I find the cross validation estimate of MSE to be 616.11 and for RMSE 24.82. This extremely similar to ridge regression CV error


# using a PCR model to predict crime rate:


# pcr model with standardized predictors
pcr.model.bos <- pcr(crim~., data = training.set.boston, scale = TRUE, validation = "CV")
summary(pcr.model.bos)

# looking at the principal components and how the predictors are weighted within each one
pcr.model.bos$coefficients

# plot cross-validation MSE
validationplot(pcr.model.bos ,val.type="MSEP")
# as can be seen by the plot, inlcuding all possible components gives us the lowest MSE but I will choose an M value of 3 as is gives us comparable error with much lower complexity


pcr.MSE.bos <- mean((test.set.boston$crim - predict(pcr.model.bos, test.set.boston, ncomp=3))^2)
pcr.test.RMSE.bos = sqrt(pcr.MSE.bos)
# I find the cross validation estimate comes out to 27.31 for MSE or 5.23 for RMSE which is less accurate than any of the previous models

# Something I noticed: LASSO and Ridge gave almost identical cross validation errors which was something I have personally not seen yet


### B ###

# comparing these three approaches, I will choose the best model by selcting the one with the lowest cross validation error.
# we see that the most accurate model is by far the principal component regression with a RMSE of just 5.23 compared to 24.82 for LASSO and 24.84 for Ridge


### C ###

# The PCR model only required 3 principal components in order to achieve a low CV RMSE. While the 3 component PCR does include all predictors within its components,
# each predictor within the components was weighted differently in an attempt to lower the overall CV RMSE
# It should be noted that only LASSO didn't include all predictors (and even this regression still include most of them: 11). Ridge also found its lowest RMSE using all predictors
# this suggests that all predictors are of some importance to the prediction of crime rate


###############
### CH4 P10 ###
###############

#attach(Weekly)
summary(Weekly)
Weekly[0:10,]

### A ###

# looking at numerical and graphical summaries of weekly trying to identify trends:

# looking at volume data:
plot(density(Volume))
plot(Volume)
cor(Volume, Year)
boxplot(Volume, main="Volume")

# looking at today data:
plot(density(Today))
plot(Year, Today)
cor(Today,Year)
boxplot(Today, main="Today")
summary(Today)

# looking at lag1:
plot(density(Lag1))
cor(Lag1, Year)
boxplot(Lag1, main="Lag1")
summary(Lag1)

# looking at lag3:
plot(density(Lag3))
cor(Lag1, Lag2)
boxplot(Lag3, main="Lag3")
summary(Lag3)


# looking at lag5:
plot(density(Lag5))
boxplot(Lag5, main="Lag5")
summary(Lag5)


# I notice that most predictors are barely related to one another except Volume and Year (.84).
# In addition, it appears the variables Lag(1,2,3,4,5) and Today are approximately normally distributed
# the Volume variable seems to skewed towards lower values
# when plotting the variable Today against the variable Year we notice a sort of sinusoidal relationship


### B ###

# I first create a dummy variable for the categorical variable: Direction
# I specify the family = binomial so that R knows I am trying to run a logistic regression
log.reg.week <- glm(Direction~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data = Weekly, family = binomial)
summary(log.reg.week)
summary(log.reg.week)$coef[,4]
# the only predictor that seems mildly statistically significant is the Lag2 predictor
# Lag2 has a P-val of 0.0296 which is less than


### C ###

# calculating the confusion matrix:
# from textbook
glm.probs.week=predict(log.reg.week,type="response")
glm.pred.week=rep("Down",1089)
glm.pred.week[glm.probs.week >.5]= "Up"
confuse.matrix = table(glm.pred.week ,Direction)
perc.correct.week = mean(glm.pred.week==Direction)
perc.inc.week = 1 - perc.correct.week
# so looking at the percentage the model got correct is: 56%
# percentage incorrect is: 44%
# Simply put, this suggests a possible strategy of buying on weeks when the model predicts an increasing market, and avoiding trades on weeks when a decrease is predicted
# In addition, since most predictors were not correlated including them brings down the accuracy of the overall model



### D ###

Weekly1 <- Weekly[,c(1,3,6,9)]
# defining training and test years:
training.years <- Weekly1[Year <= 2008,]
test.years <- Weekly1[Year > 2008,]

# logistic regression model:
log.reg.test <- glm(Direction~ Lag2, data = training.years, family = binomial)
summary(log.reg.test)

# confusion matrix:
glm.probs.week1=predict(log.reg.test, test.years, type="response")
glm.pred.week1=rep("Down",104)
glm.pred.week1[glm.probs.week1 >.5]= "Up"
confuse.matrix1 = table(glm.pred.week1 ,test.years$Direction)
perc.correct.week1 = mean(glm.pred.week1 == test.years$Direction)
perc.inc.week1 = 1 - perc.correct.week1
# percentage of correct predictions on the test data set is 62.5%
# percentage incorrect 37.5%

### skipping parts E and F ###
### G ###

# knn model:
knn.weekly <- knn(as.matrix(training.years$Lag2), as.matrix(test.years$Lag2), training.years$Direction, k = 1)
confuse.matrix2 = table(knn.weekly, test.years$Direction)
perc.correct.week2 = mean(knn.weekly == test.years$Direction)
perc.inc.week2 = 1 - perc.correct.week2
# this model is makes the correct prediction on the test data set 50.96% of the time
# percentage incorrect 49.04%
# this is barely a discernable difference and probably not a reliable investment strategy


### H ###

# the logistic regression model provided the best results on this data as it made the correct prediction on the test data set 62.5 percent of the time. 
# compare this to the simple knn model which made the correct prediction on the test data around 50% of the time which is not reliable


### I ###

# different logistic regression attempts:
# logistic regression with a cubic transformation
log.reg.weektrans <- glm(Direction~ Lag2^3, data = training.years, family = binomial)
glm.probs.weektrans=predict(log.reg.weektrans, test.years, type="response")
glm.pred.weektrans=rep("Down",104)
glm.pred.weektrans[glm.probs.weektrans >.5]= "Up"
confuse.matrixtrans = table(glm.pred.weektrans ,test.years$Direction)
perc.correct.weektrans = mean(glm.pred.weektrans == test.years$Direction)
perc.inc.weektrans = 1 - perc.correct.weektrans
# interestingly, the cubic transformation gave decent results as it made the correct prediction 62.5% of the time

# logistic regression with a interaction between Lag2 and Lag5
log.reg.weekint <- glm(Direction~ Lag2*Lag5, data = training.years, family = binomial)
glm.probs.weekint=predict(log.reg.weekint, test.years,type="response")
glm.pred.weekint=rep("Down",104)
glm.pred.weekint[glm.probs.weekint >.5]= "Up"
confuse.matrixint = table(glm.pred.weekint ,test.years$Direction)
perc.correct.weekint = mean(glm.pred.weekint == test.years$Direction)
perc.inc.weekint = 1 - perc.correct.weekint
# interestingly, the interaction between the two Lags also gave good results as it made the correct prediction 58.65% of the time



# different k-values for knn model, choose k = 5, 15, 50:
# k = 5
knn.weekly3 <- knn(as.matrix(training.years$Lag2), as.matrix(test.years$Lag2), training.years$Direction, k = 5)
confuse.matrix4 = table(knn.weekly3, test.years$Direction)
perc.correct.week4 = mean(knn.weekly3 == test.years$Direction)
perc.inc.week4 = 1 - perc.correct.week4
# increasing k nearest neighbors to 5 increases the model's correct prediction accuracy on the test data to 52.88%

# k = 15
knn.weekly2 <- knn(as.matrix(training.years$Lag2), as.matrix(test.years$Lag2), training.years$Direction, k = 15)
confuse.matrix3 = table(knn.weekly2, test.years$Direction)
perc.correct.week3 = mean(knn.weekly2 == test.years$Direction)
perc.inc.week3 = 1 - perc.correct.week3
# increasing k nearest neighbors to 15 increases the model's correct prediction accuracy on the test data to 58.65%

# k = 50
knn.weekly4 <- knn(as.matrix(training.years$Lag2), as.matrix(test.years$Lag2), training.years$Direction, k = 50)
confuse.matrix5 = table(knn.weekly4, test.years$Direction)
perc.correct.week5 = mean(knn.weekly4 == test.years$Direction)
perc.inc.week5 = 1 - perc.correct.week5
# increasing k nearest neighbors to 50 increases the model's correct prediction accuracy on the test data to 56.73%


# The best model from these expiremental models was actually a tie between the knn model with a k-value of 15 and the logistic regression with a interaction model as they both had correct prediction accuracy on the test data of 58.65%
# best confusion matrix see:
confuse.matrix3


##############
### CH8 P8 ###
##############

#attach(Carseats)
Carseats[0:5,]


### A ###

n.seats = dim(Carseats)[1]
test.cases.seats <- sample(1:n.seats, size=n.seats*(.2))
training.cases.seats <- setdiff(1:n.seats,test.cases.seats)
training.set.seats <- Carseats[training.cases.seats,]
test.set.seats <- Carseats[test.cases.seats,]


### B ### 

# build a big tree
train.reg.tree = tree(Sales~., #Formula
            data=training.set.seats) #Data frame
summary(train.reg.tree)

#plot the tree
plot(train.reg.tree)
text(train.reg.tree,col="blue",label=c("yval"),cex=.8)

# calculating MSE and RMSE
pred.y = predict(train.reg.tree,newdata = test.set.seats)
test.y = test.set.seats$Sales
tree.test.MSE = mean((pred.y - test.y)^2)
tree.test.RMSE = sqrt(tree.test.MSE)
# I get an test MSE of 3.94 and a test RMSE of 1.96


### C ###

# I cross validate my tree model from above and plot the deviance as a function of tree size
cv.train.tree = cv.tree(train.reg.tree)
plot(cv.train.tree$size, cv.train.tree$dev)

# I select the size that returns the lowest deviance
best.treesize = which.min(cv.train.tree$dev)

# now I prune the tree to see if there is improvement in the MSE
pruned.train.tree = prune.tree(train.reg.tree, #The tree model
                       best= best.treesize) #Only the seven best nodes
plot(pruned.train.tree)
text(pruned.train.tree,col="blue",label=c("yval"),cex=.8)

# calculating MSE and RMSE
pred.pruned.y = predict(pruned.train.tree,newdata = test.set.seats)
test.pruned.y = test.set.seats$Sales
pruned.tree.test.MSE = mean((pred.pruned.y - test.pruned.y)^2)
pruned.tree.test.RMSE = sqrt(pruned.tree.test.MSE)
# I get an test MSE of 4.7 and a test RMSE of 2.17
# this is not an improvement on the bigger tree's MSE


### D ###

# I run a random forest model randomnly sampling from 10 variables per split
train.seats.bag = randomForest(Sales~.,data=training.set.seats,mtry = 10, importance = TRUE)
pred.bag.y = predict(train.seats.bag,newdata=test.set.seats)
test.bag.y = test.set.seats$Sales
bag.tree.test.MSE = mean((pred.bag.y - test.bag.y)^2)
bag.tree.test.RMSE = sqrt(bag.tree.test.MSE)
# I get an test MSE of 2.37 and a test RMSE of 1.54
# this is the best test MSE yet

# calling the importance function as asked in the problem
importance(train.seats.bag)
plot(importance(train.seats.bag))
text(72,660,"Price")
text(70,749,"ShelveLoc")
text(45,284,"CompPrice")
# the most important variables in determining sales are ShelveLoc and Price followed later by CompPrice


### E ###

# random forest with m= 3,7
# m=3
train.seats.forest3 = randomForest(Sales~.,data=training.set.seats, mtry = 3, importance = TRUE)
pred.forest.y3 = predict(train.seats.forest3,newdata=test.set.seats)
test.forest.y3 = test.set.seats$Sales
forest.tree.test.MSE3 = mean((pred.forest.y3 - test.forest.y3)^2)
forest.tree.test.RMSE3 = sqrt(forest.tree.test.MSE3)
# MSE of 2.752 and RMSE of 1.66
importance(train.seats.forest3)

# m=4
train.seats.forest4 = randomForest(Sales~.,data=training.set.seats, mtry = 4, importance = TRUE)
pred.forest.y4 = predict(train.seats.forest4,newdata=test.set.seats)
test.forest.y4 = test.set.seats$Sales
forest.tree.test.MSE4 = mean((pred.forest.y4 - test.forest.y4)^2)
forest.tree.test.RMSE4 = sqrt(forest.tree.test.MSE4)
# MSE of 2.53 and RMSE of 1.59
importance(train.seats.forest4)


# larger m values appear to decrease the error rate, ie. MSE decreases as the number of predictors considered at each split increases


###############
### CH8 P11 ###
###############

#attach(Caravan)
Caravan[0:5,]

### A ###

n.caravan = dim(Caravan)[1]

# create dummy variable for purchase:
Caravan$Purchase = ifelse(Caravan$Purchase == "Yes", 1, 0)

training.cases.caravan <- 1:1000
training.set.caravan <- Caravan[training.cases.caravan,]
test.set.caravan <- Caravan[-training.cases.caravan,]


### B ###


# fit boosting model with purchase as response against other vars:
train.boosted.caravan = gbm(Purchase ~ ., data = training.set.caravan, distribution = "gaussian", n.trees = 1000, shrinkage = 0.01)
summary(train.boosted.caravan)
# the three most important variables are PPERSAUT (rel.inf.=13.98) and MKOOPKLA(rel.inf.=9.26)
# approximately half the predictors are of no relative importance at all


### C ###

# boosting:
predicted.boosted.car <- predict(train.boosted.caravan, newdata=test.set.caravan, n.trees = 1000, type = "response")
boost.pred <- ifelse(predicted.boosted.car > 0.2, 1, 0)
confuse.mat.car1 <- table(test.set.caravan$Purchase, boost.pred, dnn=c("Actual","Predicted"))
confuse.mat.car1
predicted.to.purchase1 <- confuse.mat.car1[2,2]/sum(confuse.mat.car1[,2])
predicted.to.purchase1
# looks like about 24.5% of the people predicted to make the purchase actually end up making it.

# knn (k=3) comparison:
knn.caravan <- knn(training.set.caravan, test.set.caravan, cl= training.set.caravan$Purchase, k = 3)
confuse.mat.car3 = table(knn.caravan, test.set.caravan$Purchase)
confuse.mat.car3
predicted.to.purchase3 <- confuse.mat.car3[2,2]/sum(confuse.mat.car3[,2])
predicted.to.purchase3
# looks like about 3.5% of the people predicted to make the purchase actually end up making it which is very low

# logistic regression comparison:
log.reg.caravan <-  glm(Purchase ~ ., data= training.set.caravan, family= binomial)
log.reg.caravan.pred <-  predict(log.reg.caravan, test.set.caravan, type="response")
log.reg.pred <-  ifelse(log.reg.caravan.pred > 0.2, 1, 0)
confuse.mat.car2 <- table(test.set.caravan$Purchase, log.reg.pred)
confuse.mat.car2
predicted.to.purchase2 <- confuse.mat.car2[2,2]/sum(confuse.mat.car2[,2])
predicted.to.purchase2
# looks like about 14.2% of the people predicted to make the purchase actually end up making it, better than knn but still worse than boosting
# Overall, boosting gives us the best results at 24.5% correct accuracy

