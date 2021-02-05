# Neural Network: Information in passed through interconnected units analogous to 
#information passage through neurons in humans

#The first layer of the neural network receives the raw input, processes it and
#passes the processed information to the hidden layers.
set.seed(1)
library(nnet)
n.boston = dim(Boston)[1]
test.cases.boston <- sample(1:n.boston, size=n.boston*(.2))
training.cases.boston <- setdiff(1:n.boston,test.cases.boston)

# scale data
max.boston = apply(Boston , 2 , max)
min.boston = apply(Boston, 2 , min)
scaled.data.boston <- scale(Boston, center = min.boston, scale = max.boston - min.boston)
scaled.boston = as.data.frame(scaled.data.boston)

#training and test set with scaled data
scaled.training.set.boston <- scaled.boston[training.cases.boston,]
scaled.test.set.boston <- scaled.boston[test.cases.boston,]

#neural network for boston
NN.boston = nnet(crim~., data=scaled.training.set.boston, size=5, decay = 0.1 , linout = T)


zlm = lm(crim~.,scaled.boston) #Estimating price using OLS
fzlm = predict(zlm,scaled.boston) #Gets the OLS fits for the data
temp = data.frame(y=scaled.boston$crim,fnn=fit.NN.boston,flm=fzlm) #Data frame of results
pairs(temp) #Matrix of scatterplots
print(cor(temp)) #Correlation matrix

summary(NN.boston)
NeuralNetTools::plotnet(NN.boston)
NN.pred.y = predict(NN.boston,scaled.test.set.boston)
NN.test.y = scaled.test.set.boston$crim
NN.test.MSE = mean((NN.pred.y - NN.test.y)^2)
NN.test.RMSE = sqrt(NN.test.MSE)

#cross validate for a few choices of Size and decay parameters
NN.boston1 = nnet(crim~.,scaled.boston,size=3,decay=.5,linout=T)
NN.boston2 = nnet(crim~.,scaled.boston,size=3,decay=.00001,linout=T)
NN.boston3 = nnet(crim~.,scaled.boston,size=15,decay=.5,linout=T)
NN.boston4 = nnet(crim~.,scaled.boston,size=15,decay=.00001,linout=T)

#The predictions of each model for the data
znnf1 = predict(NN.boston1,scaled.boston)
znnf2 = predict(NN.boston2,scaled.boston)
znnf3 = predict(NN.boston3,scaled.boston)
znnf4 = predict(NN.boston4,scaled.boston)

NeuralNetTools::plotnet(NN.boston1)
NeuralNetTools::plotnet(NN.boston2)
NeuralNetTools::plotnet(NN.boston3)
NeuralNetTools::plotnet(NN.boston4)
