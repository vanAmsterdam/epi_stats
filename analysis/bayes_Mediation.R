# This code runs a Bayesian mediation analysis using the package rjags.
# To start, import the data set waterconsumption.csv.

MyData <- read.csv(file="D:\\Files from the baby laptop\\Bayes course, March 17\\Exercise data sets\\rjags\\waterconsumption.csv", header=TRUE, sep=",")
N<-dim(MyData)[1]

N<-dim(waterconsumption)[1]

modelstring <- as.character("
model {
############################################
  # Prior distributions
############################################
  beta.0.m ~ dnorm(1, .001); # prior for the intercept for M
  beta.0.y ~ dnorm(1, .001); # prior for the intercept for Y

  a ~ dnorm(5, .11); # prior for a
  b ~ dnorm(5, .11); # prior for b

  cp ~ dnorm(0, .11); # prior for c´
  tau.e.M ~ dgamma(.5, .5); # prior for the error precision for M
  tau.e.Y ~ dgamma(.5, .5); # prior for the error precision for Y
  
  ab <-a*b

############################################
  # Conditional probability of the data
  # A regression model
############################################
  for(i in 1:N){
    m.prime[i] <- beta.0.m + a*(x[i]-70.18); # predicted value of M, predictor is mean-centered
    y.prime[i] <- beta.0.y + cp*(x[i]-70.18) + b*(m[i]-3.06) ; # predicted value of Y, predictor is mean-centered
    m[i] ~ dnorm(m.prime[i], tau.e.M); # conditional distribution of m
    y[i] ~ dnorm(y.prime[i], tau.e.Y); # conditional distribution of y
  }  
}
") # closes the model as string

#############################################################################################################################
# Write out the BUGS code to a file
#############################################################################################################################
model.file.name <- "Single Mediator Model.txt"
write(x=modelstring, file=model.file.name, append=FALSE)

library(rjags)

mediation.model <- jags.model('Single Mediator Model.txt',
                   data = list('x' = MyData$x,
                               'm' = MyData$m,
                               'y' = MyData$y,
                               'N' = N),
                 n.chains = 3)
out=coda.samples(mediation.model,
                 variable.names=c("a", "ab","b", "cp", "beta.0.m", 
                   "beta.0.y", "tau.e.M", "tau.e.Y"),
                 n.iter=10000)

summary(out)

library(coda)

model.as.mcmc.list <- as.mcmc.list(out)
draws.to.analyze.as.one.list <- 
  as.mcmc(do.call(rbind,model.as.mcmc.list)) 

summary(draws.to.analyze.as.one.list)

# diagnostics of convergence
gelman.diag(model.as.mcmc.list)
gelman.diag(model.as.mcmc.list)
gelman.plot(model.as.mcmc.list)
plot(model.as.mcmc.list, trace=TRUE, density=TRUE)

# running additional iterations
out2=coda.samples(mediation.model,
                 variable.names=c("a", "ab","b", "cp", "beta.0.m", 
                                  "beta.0.y", "tau.e.M", "tau.e.Y"),
                 n.iter=10000)

model.as.mcmc.list <- as.mcmc.list(out2)
draws.to.analyze.as.one.list <- 
  as.mcmc(do.call(rbind,model.as.mcmc.list)) 

#obtain mean, medians, and quantiles
summary.stats <- summary(draws.to.analyze.as.one.list)

#obtain highest-posterior density interval
HPD.interval <- HPDinterval(draws.to.analyze.as.one.list, 
                            prob=.95)
