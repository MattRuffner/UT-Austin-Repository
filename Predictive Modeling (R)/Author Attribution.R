# clearing environment
rm(list=ls())

# setting seed to get reproducible results
set.seed(123)

# importing libraries to read in the authors and their text files
library(tm)
library(e1071)
library(plyr)
library(caret)
#library(doMC)
#registerDoMC(cores=detectCores())

# tm 'reader' function
# This wraps another function around readPlain to read
# plain text documents in English.
readerPlain = function(fname){
  readPlain(elem=list(content=readLines(fname)), 
            id=fname, language='en') }

# initiating the file lists and labels for when we read it in the test and train files
file_list.train = NULL
labels.train = NULL
file_list.test = NULL
labels.test = NULL

# characters of each path up to the *
x = nchar('/Users/ruffner/Documents/Matthew Ruffner Texas/Rstudio Stuff/STA380-master/data/ReutersC50/C50test/*')
y = nchar('/Users/ruffner/Documents/Matthew Ruffner Texas/Rstudio Stuff/STA380-master/data/ReutersC50/C50train/*')

# loading in training files
author_directories.train = Sys.glob('/Users/ruffner/Documents/Matthew Ruffner Texas/Rstudio Stuff/STA380-master/data/ReutersC50/C50train/*')
for(author in author_directories.train) {
  author_name.train = substring(author, first=y)
  files_to_add.train = Sys.glob(paste0(author, '/*.txt'))
  file_list.train = append(file_list.train, files_to_add.train)
  labels.train = append(labels.train, rep(author_name.train, length(files_to_add.train)))
}

# creating a pre-corpus object for training files
all_docs.train = lapply(file_list.train, readerPlain) 
names(all_docs.train) = file_list.train
names(all_docs.train) = sub('.txt', '', names(all_docs.train))

# actually defining the corpus
my_corpus.train = Corpus(VectorSource(all_docs.train))

# preprocessing with what is happening for each line to the right of the code:
my_corpus.train = tm_map(my_corpus.train, content_transformer(tolower)) # make everything lowercase
my_corpus.train = tm_map(my_corpus.train, content_transformer(removeNumbers)) # remove numbers
my_corpus.train = tm_map(my_corpus.train, content_transformer(removePunctuation)) # remove punctuation
my_corpus.train = tm_map(my_corpus.train, content_transformer(stripWhitespace)) ## remove excess white-space
my_corpus.train = tm_map(my_corpus.train, content_transformer(removeWords), stopwords("SMART")) # removing words like 'the','and', etc...

# creating DTM for the training data
DTMtrain = DocumentTermMatrix(my_corpus.train)
class(DTMtrain)


# Now lets do the same for the test data:
author_directories.test = Sys.glob('/Users/ruffner/Documents/Matthew Ruffner Texas/Rstudio Stuff/STA380-master/data/ReutersC50/C50test/*')
for(author in author_directories.test) {
  author_name.test = substring(author, first=x)
  files_to_add.test = Sys.glob(paste0(author, '/*.txt'))
  file_list.test = append(file_list.test, files_to_add.test)
  labels.test = append(labels.test, rep(author_name.test, length(files_to_add.test)))
}
# creating a pre-corpus object for test files
all_docs.test = lapply(file_list.test, readerPlain) 
names(all_docs.test) = file_list.test
names(all_docs.test) = sub('.txt', '', names(all_docs.test))

# actually defining the corpus
my_corpus.test = Corpus(VectorSource(all_docs.test))

# preprocessing
my_corpus.test = tm_map(my_corpus.test, content_transformer(tolower)) # make everything lowercase
my_corpus.test = tm_map(my_corpus.test, content_transformer(removeNumbers)) # remove numbers
my_corpus.test = tm_map(my_corpus.test, content_transformer(removePunctuation)) # remove punctuation
my_corpus.test = tm_map(my_corpus.test, content_transformer(stripWhitespace)) ## remove excess white-space
my_corpus.test = tm_map(my_corpus.test, content_transformer(removeWords), stopwords("SMART"))

# creating DTM for the test data
DTMtest = DocumentTermMatrix(my_corpus.test)
class(DTMtest)

# inspecting entries in training data
inspect(DTMtrain[1:10,1:20])

# ...find words with greater than a min count...
findFreqTerms(DTMtrain, 10000)
# looks like 'character' is the most common word

# ...or find words whose count correlates with a specified word.
# the top entries here look like they go with "genetic"
findAssocs(DTMtrain, "genetic", .5)


# Below removes terms that have count 0 in >97.5% of docs.  
DTMtrain = removeSparseTerms(DTMtrain, 0.975)
DTMtrain # now ~ 1414 terms 

# since there are some words used in training set and not in the test set, we set the training set words as the control
DTMtest <- DocumentTermMatrix(my_corpus.test, control=list(dictionary=Terms(DTMtrain)))

# construct TF-IDF weights for training and test set
tfidf.train = weightTfIdf(DTMtrain)
tfidf.test = weightTfIdf(DTMtest)
# Tf-idf stands for term frequency-inverse document frequency 
# This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. 
# The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. 

# using the weighted training and test DMTs we create matrices for each:
weight.train.matrix = as.matrix(tfidf.train)
weight.test.matrix = as.matrix(tfidf.test)

# drop columns which are equal to zero
weight.train.matrix = weight.train.matrix[,-which(colSums(weight.train.matrix)==0)]
weight.test.matrix = weight.test.matrix[,-which(colSums(weight.test.matrix)==0)]

# run a pca analysis
pca.train = prcomp(weight.train.matrix, scale=TRUE)
summary(pca.train)$importance[3,]

# plotting the first few components
plot(pca.train,type='line')
# looks like to get to around 50% we would need the first 221 pricipal components 

# redifining our training matrix to use the first 221 principal components
weight.train.matrix <- pca.train$x[,1:221]
#train.loadings <- pca.train$rotation[,1:221]

# making sure the same PCA results have been selected for our test set
confirmed.test = predict(pca.train,newdata =weight.test.matrix)[,1:221]

# what we are trying to predict for training and test sets, essentially the authors
y.train = file_list.train %>%
  { strsplit(., '/', fixed=TRUE) } %>%
  { lapply(., tail, n=2) } %>%
  { lapply(., head, n=1) } %>%
  { lapply(., paste0, collapse = '') } %>%
  unlist

y.test = file_list.test %>%
  { strsplit(., '/', fixed=TRUE) } %>%
  { lapply(., tail, n=2) } %>%
  { lapply(., head, n=1) } %>%
  { lapply(., paste0, collapse = '') } %>%
  unlist

# importing the random forest library
library(randomForest)


# looks like a 40 tree forest is good enough to minimize error
random.forest.model.final = randomForest(x=as.data.frame(pca.train$x[,1:221]), y=as.factor(y.train), ntree=40) 
random.forest.predict = predict(random.forest.model.final,confirmed.test)

# building a truth table for random forest
accuracy.table.forest = table("Predictions" = random.forest.predict,  "Actual" = y.test)
total.prediction.accuracy.forest = sum(diag(accuracy.table.forest))/sum(accuracy.table.forest)
total.prediction.accuracy.forest
# we get a 45.92% accuracy which is solid considering we are only using 221 of the original 2500 components to train our model

# run a naive bayes
naive.bayes.model = naiveBayes(x=as.data.frame(pca.train$x[,1:221]), y=as.factor(y.train)) 
naive.bayes.predict = predict(naive.bayes.model,confirmed.test)

# build a truth table for naive bayes
accuracy.table.bayes = table("Predictions" = naive.bayes.predict,  "Actual" = y.test)
total.prediction.accuracy.bayes = sum(diag(accuracy.table.bayes))/sum(accuracy.table.bayes)
total.prediction.accuracy.bayes
# we get a slight improvement upon our tree model with a 48% accuracy rating



