functions {
  real log_Z_terms(int j, real lambda, real nu){
    return(j * log(lambda) - nu * lgamma(j + 1));
  }
}
data {
  int<lower=0> n;   // number of observations
  int<lower=0> S1;  // sum of X_i's
  int<lower=0> X[n]; // data vectors for log_lik construction
  real<lower=0> S2; // sum of log(X_i!)
  real<lower=0> a;  // hyper-parameter
  real<lower=0> b;  // hyper-parameter
  real<lower=0> c;  // hyper-parameter
}
parameters {
  real<lower=0> lambda;
  real<lower=0> nu;
}
model {
  real logZ[101];
  for (j in 0:100)
    logZ[j+1] = log_Z_terms(j, lambda, nu);
  target += (a + S1 - 1)*log(lambda) - nu*(b + S2) - (c + n)*log_sum_exp(logZ);
}
generated quantities {
  real logZ[101];
  real log_lik[n];
  for (j in 0:100)
    logZ[j+1] = log_Z_terms(j, lambda, nu);
  for (i in 1:n)
    log_lik[i] = X[i]*log(lambda) - nu*lgamma(X[n] + 1) - n*log_sum_exp(logZ);
}
