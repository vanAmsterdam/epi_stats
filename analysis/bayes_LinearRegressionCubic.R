# This code runs a Bayesian linear regression analysis using the package rjags.
# To start, import the data set LinearRegressionCubic.csv.

MyData <- read.csv(file="D:\\Files from the baby laptop\\Bayes course, March 17\\Exercise data sets\\rjags\\LinearRegressionCubic.csv", header=TRUE, sep=",")
N<-dim(MyData)[1]

# writing out the .txt file with the model

modelstring <- as.character("
model{
beta.0 ~ dnorm(0, .001); # prior for the intercept
beta.1 ~ dnorm(0, .001); # prior for b1
beta.2 ~ dnorm(0, .001); # prior for b2
beta.3 ~ dnorm(0, .001); # prior for b3
tau.e ~ dgamma(.5, .5); # prior for the error precision for Y

sigma2.e<-1/tau.e
sigma.e<-sqrt(sigma2.e)

# Conditional probability of the data
# A regression model

for(i in 1:N){
y.prime[i] <- beta.0 + beta.1*x[i]+ beta.2*x[i]*x[i]+ beta.2*x[i]*x[i]*x[i]; # predicted value of Y
y[i] ~ dnorm(y.prime[i], tau.e); # conditional distribution of y
}
}
") # closes the model as string

model.file.name <- "Linear Regression.txt"
write(x=modelstring, file=model.file.name, append=FALSE)

library('rjags')

jags <- jags.model('Linear Regression.txt',
                   data = list('x' = MyData$X,
                               'y' = MyData$Y,
                               'N' = N),
                   n.chains = 3)

out=coda.samples(jags, variable.names=c("beta.0","beta.1","beta.2", "beta.3", "tau.e"),
                       n.iter=100)

summary(out)

# Coda
library(coda)

model.as.mcmc.list <- as.mcmc.list(out)

gelman.diag(model.as.mcmc.list)
gelman.plot(model.as.mcmc.list)
plot(model.as.mcmc.list, trace=TRUE, density=FALSE)

# Running additional iterations
out2=coda.samples(jags, variable.names=c("beta.0","beta.1","beta.2", "beta.3", "tau.e"),
                 n.iter=1000)

model.as.mcmc.list <- as.mcmc.list(out2)

gelman.diag(model.as.mcmc.list)
gelman.plot(model.as.mcmc.list)
plot(model.as.mcmc.list, trace=TRUE, density=FALSE)

summary(out2)

# After the chains have converged
# combine draws from chains
draws.to.analyze.as.one.list <- 
  as.mcmc(do.call(rbind,model.as.mcmc.list))

#obtain mean, medians, and quantiles
summary.stats <- summary(draws.to.analyze.as.one.list)

#obtain highest-posterior density interval
HPD.interval <- HPDinterval(draws.to.analyze.as.one.list, 
                            prob=.95)