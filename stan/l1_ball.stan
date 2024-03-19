functions {
  real signum(real x) {
    return (x >= 0) ? 1 : -1;
  }

  vector L1ball_project(vector beta, real r, int p) {
    /* Projection on to L1-ball
     * Args:
     * beta: weight parameters
     * r: radius
     */
    vector[p] beta_abs;
    vector[p] theta;
    vector[p] sorted_beta_abs;
    vector[p] mu;
    vector[p] mu_tilde;
    int c = 0;

    // if norm of beta is within radius, keep as is
    if (norm1(beta) <= r) {
      theta = beta;
    } else {
      // sort beta by descending absolute values
      beta_abs = abs(beta);
      sorted_beta_abs = sort_desc(beta_abs);

      // calculate cumulative sum for thresholding
      mu = cumulative_sum(sorted_beta_abs) - r;

      // get the index corresponding to the smallest abs beta above the
      // threshold
      for (i in 1:p) {
        mu_tilde[i] = mu[i] / i; // threshold
        if (sorted_beta_abs[i] < mu_tilde[i]) {
          c = i - 1;
          break;
        }
      }

      // do the projection and keep track of the sign
      for (i in 1:p) {
        theta[i] = signum(beta[i]) * max({beta_abs[i] - mu_tilde[c], 0});
      }
    }
    return theta;
  }
}

data {

  int<lower=1> T; // number of time points
  vector[T] Y; // observations
  int<lower=0> p; // AR order

  // sigma prior sd
  real<lower=0> sigma_sd; // sd of sigma prior

  // phi prior sd
  real<lower=0> phi_sd; // sd of phi prior

  // intercept prior
  real alpha_mean;
  real<lower=0> alpha_sd;

  // l1 ball radius alpha
  real<lower=0> r_alpha;
}

transformed data {
  matrix[(T-p), p] Y_matrix;
  vector[T-p] Y_lagged = Y[(p+1):T]; // Subset only once for efficiency

  for(t in 1:(T-p)) {
    for(i in 1:p) {
      Y_matrix[t, p-i+1] = Y[t + (i-1)];
    }
  }
}

parameters {
  real<lower=0> r;
  vector[p] phi_o; // original phi
  real alpha; // intercept
  real<lower=0> sigma;
}

transformed parameters {
  vector[p] phi; // projected phi
  phi = L1ball_project(phi_o, r, p);
}

model {
  // priors
  alpha ~ normal(alpha_mean, alpha_sd);
  phi_o ~ normal(0, phi_sd);
  sigma ~ normal(0, sigma_sd);
  r ~ exponential(r_alpha);

  // likelihood
  Y_lagged ~ normal_id_glm(Y_matrix, alpha, phi, sigma);
}
