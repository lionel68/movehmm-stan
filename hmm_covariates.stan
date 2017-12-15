functions{
  //Function that returns the log pdf of the wrapped-Cauchy
  real wrappedCauchy(real aPhi, real aRho, real aMu) {
    return(- log(2*pi()) + log((1-aRho^2)/(1+aRho^2-2*aRho*cos(aPhi-aMu))));
  }
  
  //Function that returns a random sample from wrapped-Cauchy by first
  //sampling from the Cauchy, then using modular arithmetic
  real wrappedCauchy_rng(real aRho, real aMu){
    real aCauchyRand;
    real aWCRand;
    
    aCauchyRand = cauchy_rng(pi()+aMu,-log(aRho));
    if (aCauchyRand>0){
      aWCRand = fmod(aCauchyRand,2*pi()) - pi();
    } else{
      aWCRand = -fmod(-aCauchyRand,2*pi()) + pi();   
    }

    return(aWCRand);
  }
  
}

data {
  int<lower=0> N; //number of observations
  vector<lower=-pi(), upper=pi()>[N] turn; //observed turning angles
  vector<lower=0>[N] dist; //observed step length
  int<lower=1> K; //number of states
  
  // Out-of-sample test
  //int N1;
  //vector<lower=-pi(), upper=pi()>[N1] turnTest;
  //vector<lower=0>[N1] distTest;
  
  // Covariate
  int nCovs;
  matrix[N,nCovs] X;
}

parameters {
  vector<lower=-pi(), upper=pi()>[K] mu;
  vector<lower=0,upper=1>[K] rho;
  positive_ordered[K] alpha;
  matrix[K*(K-1),nCovs] beta;
}

transformed parameters {
  ## Gamma here is a matrix of the log probabilities not probabilities as in the moveHMM doc
  matrix[K,K] Gamma[N];
  matrix[K,K] Gamma_tr[N];
  
  for(n in 1:N){
    int aCount;
    aCount = 1;
    for(k_from in 1:K){
      for(k in 1:K){
        if(k_from==k){
          Gamma[n,k_from,k] = 1;
      }
	else{
          Gamma[n,k_from,k] = exp(beta[aCount] * to_vector(X[n]));
          aCount = aCount + 1;
      }
    }
    }
    Gamma[n] = log(Gamma[n]/sum(Gamma[n]));
  }
  
  for(n in 1:N)
    for(k_from in 1:K)
      for(k  in 1:K)
        Gamma_tr[n, k, k_from] = Gamma[n,k_from,k];
  
}

model {
  vector[K] lp;
  vector[K] lp_p1;
  
  lp = rep_vector(-log(K), K);
  
  //Forwards algorithm
  for (n in 1:N) {
    for (k in 1:K){
      lp_p1[k] = log_sum_exp(to_vector(Gamma_tr[n,k]) + lp)
        	+ wrappedCauchy(turn[n] , rho[k], mu[k])
        	+ exponential_lpdf(dist[n] | alpha[k]);
    }
    lp = lp_p1;
  }
  target += log_sum_exp(lp);

}

generated quantities{

  //Viterbi to estimate most likely state vector
  int<lower=1,upper=K> state[N];
  real log_p_y_star;
  {
    int back_ptr[N, K];
    real best_logp[N, K];
    real best_total_logp;
    for (k in 1:K)
      best_logp[1, K] = wrappedCauchy(turn[1] , rho[k], mu[k])
        + exponential_lpdf(dist[1] | alpha[k]);
    for (t in 2:N) {
      for (k in 1:K) {
      best_logp[t, k] = negative_infinity();
        for (j in 1:K) {
          real logp;
          logp = best_logp[t-1, j]
            + Gamma[t,j,k] + wrappedCauchy(turn[t] , rho[k], mu[k])
            + exponential_lpdf(dist[t] | alpha[k]);
          if (logp > best_logp[t, k]) {
            back_ptr[t, k] = j;
            best_logp[t, k] = logp;
          }
        }
      }
    }
    log_p_y_star = max(best_logp[N]);
    for (k in 1:K)
      if (best_logp[N, k] == log_p_y_star)
      state[N] = k;
      for (t in 1:(N - 1))
      state[N - t] = back_ptr[N - t + 1,
      				state[N - t + 1]];
  }
}
