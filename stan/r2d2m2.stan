functions {

  vector R2D2(vector z, vector sds_X, vector phi, real tau2) {
    /* Efficient computation of the R2D2 prior
     * Args:
     *   z: standardized population-level coefficients
     *   phi: local weight parameters
     *   tau2: global scale parameter (sigma is inside tau2)
     * Returns:
     *   population-level coefficients following the R2D2 prior
     */
    return  z .* sqrt(phi * tau2) ./ sds_X ;
  }

}
data {
  int<lower=1> N;  // total number of observations
  vector[N] Y;  // response variable
  int<lower=1> D;  // number of population-level effects including intercept
  matrix[N, D] X;  // population-level design matrix including column of 1s
  int<lower=0> K; // number of groups
  vector[D-1] sds_X; // column sd of X before centering. Pre estimate before or real values.

  //---- data for group-level effects

  int<lower=1> Lg;  // number of  levels per group (constant)
  int<lower=1> Dg; // number of coefficients per level per group (D_g constant per group)
  int<lower=1> J[N,K]; // grouping indicator matrix per observation per group K


  //---- group-level predictor values
  matrix[Dg,N] Z[K];

  //data for shrinkage factors
  vector[D-1] ri;
  matrix[D,Lg] rigj[K];

  //---- data for the R2D2 prior
  vector<lower=0>[ (D-1)+K+(Dg-1)*K] R2D2_alpha;
  real<lower=0> R2D2_mean_R2;  // mean of the R2 prior
  real<lower=0> R2D2_prec_R2;  // precision of the R2 prior
  int prior_only;  // should the likelihood be ignored?
}

transformed data {
  int Dc = D - 1;
  matrix[N, Dc] Xc;  // centered version of X without an intercept
  vector[Dc] means_X;  // column means of X before centering
  vector[Dc] var_X;
  vector[N] Yc;
  real Ymean;
  for (i in 2:D) {
    means_X[i - 1] = mean(X[, i]);
    var_X[i-1]= sds_X[i-1]^2;
    Xc[, i - 1] = (X[, i] - means_X[i - 1]) ;
    //Xc[, i - 1] = (X[, i] - means_X[i - 1]) / sds_X[i-1] ;
  }

  Ymean= mean(Y);
  for (i in 1:N) {
    Yc[i]= Y[i]-mean(Y);
  }

}

parameters {
  real Intercept;  // temporary intercept for centered predictors
  vector[Dc] zb; // standardized population-level effects
  matrix[Dg,Lg] z[K]; // standardized group-level effects
  real<lower=0> sigma;  // residual error

  // local parameters for the R2D2M2 prior
  simplex[Dc+K+(Dg-1)*K] R2D2_phi;
  // R2D2 shrinkage parameters
  // Convention of indexing: First Dc for overall effects, group of K for varying intercepts,
  // Batches of Dc for each group.
  real<lower=0,upper=1> R2D2_R2;  // R2 parameter

}

transformed parameters {

  vector[Dc] b;  // population-level effects

  matrix[Dg,Lg] r[K]; // actual group-level effects (includes varying intercept)

  real R2D2_tau2;  // global R2D2 scale parameter
  R2D2_tau2 =  R2D2_R2 / (1 - R2D2_R2);

  // compute actual regression coefficients
  b = R2D2(zb, sds_X, R2D2_phi[1:Dc], (sigma^2) * R2D2_tau2);

  for(k in 1:K){
    // varying intercepts
    // Dc+k is the kth varying intercept
    r[k,1,] = (sigma * sqrt(R2D2_tau2 * R2D2_phi[Dc+k ]) * (z[k,1,]));
    for(d in 2: Dg){
      // group level effects
      // (k-1)Dc indexes the beginning of the kth batch of scales
      r[k,d,]= sigma /(sds_X[(d-1)]) * sqrt(R2D2_tau2 * R2D2_phi[Dc+K+ (k-1)*(Dg-1) +(d-1) ]) * (z[k,d,]);
      //careful with sds_X
    }
  }
}

model {
  // likelihood including constants

  if (!prior_only) {
    // initialize linear predictor term
    vector[N] mu = Intercept + rep_vector(0.0, N);
    for (n in 1:N) {
      // add more terms to the linear predictor

      for(k in 1:K){
	mu[n]+=dot_product(r[k,,J[n,k]], Z[k,,n]) ;
      }

    }
    target += normal_id_glm_lpdf(Yc | Xc, mu, b, sigma); // mu+ Xc*b
  }
  // priors including constants

  target += beta_lpdf(R2D2_R2 | R2D2_mean_R2 * R2D2_prec_R2, (1 - R2D2_mean_R2) * R2D2_prec_R2); // R^2
  target += dirichlet_lpdf(R2D2_phi | R2D2_alpha); //phi

  target += normal_lpdf(Intercept | 0, 10);  // Intercept
  target += std_normal_lpdf(zb); //zb: overall effects

  for(k in 1:K){
    for(d in 1: Dg){
      target += std_normal_lpdf(z[k,d,]); // z
    }
  }

  target += student_t_lpdf(sigma | 3, 0, sd(Yc));  //  sigma: scale awareness is important!

}
generated quantities {
  //---actual population-level intercept
  real b_Intercept = Ymean+Intercept - dot_product(means_X, b);

  //---y_tilde quantities of interest

  vector[N] log_lik;
  real y_tilde[N];
  vector[N] mu_tilde = rep_vector(0.0, N)+Ymean+Intercept +Xc*b;
  vector<lower=0>[(D-1)+K+(Dg-1)*K] lambdas;

  //---shrinkage factors

  vector[Dc] kappa; //overall coefs
  matrix[Dg,Lg] kappaigj[K]; //varying coeffs
  real<lower=0> meffoc=0; //effective number of overall coeffs
  real<lower=0> meffvc=0; //effective number of varying coeffs
  real<lower=0> meff=0; //effective number of coeffs

  //---y_tilde calc

  for (n in 1:N) {
    for(k in 1:K){
      mu_tilde[n]+=dot_product(r[k,,J[n,k]], Z[k,,n]) ;
    }
    log_lik[n] =normal_lpdf( Y[n] | mu_tilde[n], sigma);
    y_tilde[n]=normal_rng(mu_tilde[n], sigma);  //copy and paste model (executed once per sample)
  }

  //--- shrinkage factors calc

  for(i in 1: Dc){
    kappa[i]=inv(1+ri[i]*R2D2_phi[i]*R2D2_tau2);
  }

  meffoc=Dc-sum(kappa);

  for(k in 1:K){
    for(i in 1:Dg){
      for(j in 1:Lg){
	if(i==1){
	  kappaigj[k,i,j]=inv(1+ rigj[k,i,j] *R2D2_phi[Dc+k]*R2D2_tau2);
	  meffvc=meffvc+1-kappaigj[k,i,j];
	}
	else{
	  kappaigj[k,i,j]=inv(1+ rigj[k,i,j]*R2D2_phi[Dc+K+(k-1)*(Dg-1) +(i-1)]*R2D2_tau2);
	  meffvc=meffvc+1-kappaigj[k,i,j];
	}
      }
    }
  }

  meff=meffoc+meffvc;

  //--- lambdas

  lambdas[1:Dc]= sigma^2*R2D2_phi[1:Dc]./ var_X *R2D2_tau2 ; //overall variances
  lambdas[(Dc+1):(Dc+K)]= sigma^2*R2D2_phi[(Dc+1):(Dc+K)]*R2D2_tau2; //varying int variances

  for(k in 1:K){
    // group level variances
    // (k-1)(Dg-1) indexes the beginning of the kth batch of scales
    lambdas[(Dc+K+(k-1)*(Dg-1)+1):(Dc+K+(k-1)*(Dg-1)+Dg-1)]= sigma^2*R2D2_phi[(Dc+K+(k-1)*(Dg-1)+1):(Dc+K+(k-1)*(Dg-1)+Dg-1)]./ var_X*R2D2_tau2;
  }

}
