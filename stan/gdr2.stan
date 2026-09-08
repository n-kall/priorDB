functions {
  /* Scale-free coefficient computation under global-local shrinkage */
  vector gdr2_beta(vector z, vector phi, real tau2, real sigma) {
    return z * sigma .* sqrt(phi * tau2);
  }
}

data {
  int<lower=0> N;                  // Number of observations
  int<lower=1> K;                  // Number of predictors
  matrix[N, K] X;                  // Design matrix
  vector[N] Y;                     // Response vector
  int<lower=0, upper=1> prior_only;// Skip likelihood if 1
  
  // Hyperparameters for R2 prior
  real<lower=0, upper=1> mean_R2;
  real<lower=0> prec_R2;
}

transformed data {
  matrix[N, K] Xc;
  vector[K] means_X;
  vector[K] sds_X;
  real mean_Y = mean(Y);
  real sd_Y = sd(Y);

  for (k in 1:K) {
    means_X[k] = mean(X[, k]);
    sds_X[k] = sd(X[, k]);
    Xc[, k] = (X[, k] - means_X[k]) / sds_X[k];
  }
}

parameters {
  real a_c;                        // Centered intercept
  real<lower=0> sigma;             // Residual standard deviation
  vector[K] zb;                    // Standardized coefficient primitives
  vector[K - 1] eta;               // Unconstrained parameters (K-1 dimensions)
  real<lower=0, upper=1> R2;       // Explained variance proportion
}

transformed parameters {
  vector[K] phi;
  vector[K] b;
  real tau2;

  // Add reference category and map to simplex via Logistic-Normal ALR transform
  phi = softmax(append_row(eta, 0.0));
  
  tau2 = R2 / (1.0 - R2);
  b = gdr2_beta(zb, phi, tau2, sigma);
}

model {
  // Priors
  sigma ~ student_t(3, 0, sd_Y);
  a_c ~ student_t(4, mean_Y, sd_Y);
  
  zb ~ std_normal();
  
  // Scales KL (LNS) prior-matching default: independent N(0, pi) for a_pi = 0.5
  eta ~ normal(0, pi());
  
  R2 ~ beta_proportion(mean_R2, prec_R2);

  // Likelihood
  if (!prior_only) {
    target += normal_id_glm_lpdf(Y | Xc, a_c, b, sigma);
  }
}

generated quantities {
  // Uncenter and rescale parameters back to original X scale
  real alpha = a_c - dot_product(means_X ./ sds_X, b);
  vector[K] beta_raw = b ./ sds_X;
}