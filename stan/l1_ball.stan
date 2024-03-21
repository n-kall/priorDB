functions {
  real signum(real x) {
    return (x >= 0) ? 1 : -1;
  }

  vector L1ball_project(vector beta, real r, int K) {
    /* Projection on to L1-ball
     * Args:
     * beta: weight parameters
     * r: radius
     * K: number of weights
     */
    vector[K] beta_abs;
    vector[K] theta;
    vector[K] sorted_beta_abs;
    vector[K] mu;
    vector[K] mu_tilde;
    int c = 0;

    // if norm of beta is within radius, keep as is
    if (norm1(beta) <= r) {
      theta = beta;
    } else {
      // sort beta by descending absolute values
      beta_abs = abs(beta);
      sorted_beta_abs = sort_desc(beta_abs);

      // calculate cumulative sum for thresholding
      mu = fdim(cumulative_sum(sorted_beta_abs), r);
      // calculate thresholds
      for (i in 1:K) {
	mu_tilde[i] = mu[i] / i;
      }
      
      // get the index corresponding to the smallest abs beta above
      // the corresponding threshold
      for (i in 1:K) {
        if (sorted_beta_abs[i] <= mu_tilde[i]) {
	  c = i - 1;
          break;
        }
      }
      
      // do the projection and keep track of the sign
      for (i in 1:K) {
	if (c != 0) { 
	  theta[i] = signum(beta[i]) * fdim(beta_abs[i], (mu_tilde[c]));
	} else { // handle no betas being above the threshold
	   theta[i] = 0;
	}
      }
    }
    return theta;
  }
}
data {

  int<lower=1> N; // number of observations
  vector[N] Y; // response variable
  int<lower=0> K; // number of covariates
  matrix[N, K] X; // design matrix

  // sigma prior sd
  real<lower=0> sigma_sd; // sd of sigma prior

  // phi prior sd
  real<lower=0> beta_sd; // sd of beta prior

  // intercept prior
  real alpha_mean;
  real<lower=0> alpha_sd;

  // l1 ball radius alpha
  real<lower=0> r_alpha;
}

parameters {
  real<lower=0> r; // radius
  vector[K] beta_o; // original beta
  real alpha; // intercept
  real<lower=0> sigma; // residual sd
}

transformed parameters {
  vector[K] beta; // projected beta
  beta = L1ball_project(beta_o, r, K);
}

model {
  // priors
  alpha ~ normal(alpha_mean, alpha_sd);
  beta_o ~ normal(0, beta_sd);
  sigma ~ normal(0, sigma_sd);
  r ~ exponential(r_alpha);

  // likelihood
  Y ~ normal_id_glm(X, alpha, beta, sigma);
}
