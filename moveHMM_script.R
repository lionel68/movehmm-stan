#HMM for animal movement

#load libraries
library(moveHMM)
library(rstan)

#simulate some data
n_obs <- 200
ss <- simData(stepDist = "gamma",angleDist = "wrpcauchy",stepPar = c(1,10,1,5),anglePar = c(0,pi/2,0.5,0.7),obsPerAnimal = n_obs)
plot(ss)

#fit a model
data <- list(N=(n_obs-2),turn=ss$angle[!is.na(ss$angle)],
             dist=ss$step[!is.na(ss$angle)],K=2,nCovs=1,
             X=matrix(rep(1,n_obs-2),ncol=1))
m <- stan("Documents/Programming_stuff/STAN/moveHMM/hmm_covariates.stan",
          data=data,control=list(adapt_delta=0.95))
post <- extract(m)
post_state <- melt(post$state)
ggplot(subset(post_state,iterations < 100),aes(x=Var2,y=value,color=iterations))+
  geom_path()
