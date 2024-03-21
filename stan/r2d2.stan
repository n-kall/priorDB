functions {
  /* Efficient computation of the R2D2 prior
   * Args:
   *   z: standardized population-level coefficients
   *   phi: local weight parameters
   *   tau2: global scale parameter
   * Returns:
   *   population-level coefficients following the R2D2 prior
   */
  vector R2D2(vector z, vector phi, real tau2) {
    return z .* sqrt(phi * tau2);
  }
}

data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  vector[N] Y;
  int prior_only;  // should the likelihood be ignored?
  // concentration vector of the D2 prior
  vector<lower=0>[K-1] R2D2_cons_D2;
  // data for the R2D2 prior
  real<lower=0> R2D2_mean_R2;  // mean of the R2 prior
  real<lower=0> R2D2_prec_R2;  // precision of the R2 prior
}
transformed data {
  int Kc = K - 1;
  matrix[N, Kc] Xc; // centered version of X without an intercept
  vector[Kc] means_X; // column means of X before centering
  real sd_Y = sd(Y);
  for (i in 2 : K) {
    means_X[i - 1] = mean(X[ : , i]);
    Xc[ : , i - 1] = X[ : , i] - means_X[i - 1];
  }
}
parameters {
  // local parameters for the R2D2 prior
  vector[Kc] zb;
  simplex[Kc] R2D2_phi;
  // R2D2 shrinkage parameters
  real<lower=0,upper=1> R2D2_R2;  // R2 parameter
  real<lower=0> sigma;  // dispersion parameter
  real a_c;
}

transformed parameters {
  vector[Kc] b;  // population-level effects
  real R2D2_tau2;  // global R2D2 scale parameter
  array[Kc+4] real lprior;
  R2D2_tau2 = sigma^2 * R2D2_R2 / (1 - R2D2_R2);
  // compute actual regression coefficients
  b = R2D2(zb, R2D2_phi, R2D2_tau2);
  lprior[1] = student_t_lpdf(a_c | 4, 0, sd_Y);
  for (k in 1:Kc) {
    lprior[k+1] = std_normal_lpdf(zb[k]);
  }
  lprior[Kc+2] = beta_lpdf(R2D2_R2 | R2D2_mean_R2 * R2D2_prec_R2, (1 - R2D2_mean_R2) * R2D2_prec_R2);
  lprior[Kc+3] = dirichlet_lpdf(R2D2_phi | R2D2_cons_D2);
  lprior[Kc+4] = student_t_lpdf(sigma | 3, 0, sd_Y);
  
}

model {
  // likelihood including constants
  if (!prior_only) {
    target += normal_id_glm_lpdf(Y | Xc, a_c, b, sigma);
  }
  // priors including constants
  target += sum(lprior);
}

