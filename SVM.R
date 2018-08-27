
rm(list = ls())

library(e1071)
library(LiblineaR)

##################################################

#  Support Vector Classifier (linear seperable)  #

##################################################


# generating the observations, which belong to two classes.
set.seed(1)
x <- matrix(rnorm(20*2), ncol = 2)
y <- c(rep(-1,10), rep(1,10))
x[y==1, ] <- x[y==1, ] + 1

# checking whether the classes are linearly separable
plot(x, col = (3-y))


# we fit the support vector classifier
dat <- data.frame(x=x, y=as.factor(y))

svmfit <- svm(y ~., data=dat, kernel='linear', cost=10, scale = FALSE)
# scale=FALSE tells the svm() function not to scale each feature to have mean zero or standard deviation one

plot(svmfit, dat)

# there are seven support vectors. We can determine their identities as follows:
svmfit$index

summary(svmfit)


# we instead used a smaller value of the cost parameter
svmfit <- svm(y ~., data=dat, kernel='linear', cost=0.1, scale = FALSE)
plot(svmfit, dat)
svmfit$index 
#we obtain a larger number of support vectors, because the margin is now wider. 



# The e1071 library includes a built-in function, tune(), to perform crossvalidation. 
# By default, tune() performs ten-fold cross-validation on a set of models of interest.
set.seed(1)
tune.out <- tune(svm, y~., data=dat, kernel='linear', ranges = list(cost=c(0.001,0.01,0.1,1,5,10,100)))

summary(tune.out)

# We see that cost=0.1 results in the lowest cross-validation error rate. 

bestmod = tune.out$best.model
summary(bestmod)


# predict() function can be used to predict the class label on a set of test observations, at any given value of the cost parameter. 
xtest <- matrix(rnorm(20*2), ncol = 2)
ytest <- sample(c(-1,1),20,rep=TRUE)
xtest[ytest==1, ]=xtest[ytest==1, ]+1
testdat = data.frame(x=xtest, y = as.factor(ytest))

# Here we use the best model obtained through cross-validation in order to make predictions
ypred = predict(bestmod, testdat)
table(predict=ypred, truth=testdat$y)



##################################################

# Support Vector Machiine (non-linear seperable) #

##################################################

# In order to fit an SVM using a non-linear kernel, we once again use the svm() function. 
# However, now we use a different value of the parameter kernel.
# To fit an SVM with a polynomial kernel we use kernel="polynomial", we use the degree argument (d) to specify a degree for the polynomial kernel
# and to fit an SVM with a radial kernel we use kernel="radial". we use gamma to specify a value of γ for the radial basis kernel

set.seed(1)
x <- matrix(rnorm(200*2), ncol = 2)
x[1:100, ] = x[1:100, ] + 2
x[101:150, ] = x[101:150, ] - 2
y = c(rep(1,150), rep(2,50))
dat = data.frame(x=x, y=as.factor(y))

plot(x, col=y)


# We fit the training data using the svm() function with a radial kernel and γ = 1

train = sample(200,100)
svmfit = svm(y~., data=dat[train, ], kernel='radial', gamma=1, cost=1)
plot(svmfit, dat[train, ])
summary(svmfit)


# We perform cross-validation using tune() to select the best choice of γ and cost for an SVM with a radial kernel:
set.seed(1)
tune.out <- tune(svm, y~., data=dat[train, ], kernel='radial'
                 , ranges = list(cost=c(0.1,1,10,100,1000)
                                 ,gamma=c(0.5,1,2,3,4)))
summary(tune.out)

# the best choice of parameters involves cost=1 and gamma=2.

# We can view the test set predictions for this model by applying the predict() function to the data
table(true=dat[-train,'y'], pred=predict(tune.out$best.model, newx=dat[-train,]))
