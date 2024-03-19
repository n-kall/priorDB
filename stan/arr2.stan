data {
  int<lower=1> T; // number of time points
  vector[T] Y; // observations
  int<lower=0> p; // AR order
  // concentration vector of the Dirichlet prior
  vector<lower=0>[p] cons;
  // data for the R2D2 prior
  real<lower=0> mean_R2;  // mean of the R2 prior
  real<lower=0> prec_R2;  // precision of the R2 prior
  real<lower=0> sigma_sd; // sd of sigma prior
  // variance estimates of y
  real<lower=0> var_y;
}

parameters {
  vector[p] phi; // AR coefficients
  simplex[p] psi; // decomposition simplex
  real<lower=0, upper=1> R2; // coefficient of determination
  real<lower=0> sigma; // observation model sd
}

transformed parameters {
  real<lower=0> tau2 = R2 / (1 - R2); // Equation 18
  vector[T] mu = rep_vector(0.0, T);
  for (t in (p+1):T) {
    for (i in 1:p) {
      mu[t] += phi[i] * Y[t-i]; // Equation 16
    }
  }
}

model {
  // priors
  phi ~ normal(0, sqrt(sigma^2/var_y * tau2 * psi));  // Equation 17
  R2 ~ beta(mean_R2 * prec_R2, (1 - mean_R2) * prec_R2);  // Equation 19
  sigma ~ normal(0, sigma_sd);  // Equation 20
  psi ~ dirichlet(cons);        // Equation 21
  // likelihood
  Y ~ normal_lpdf(mu, sigma);   // Equation 15
}
