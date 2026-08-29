import numpy as np
import pymc as pm
import pytensor.tensor as pt

# center and scale predictors
# Assume X is your predictor matrix and y is your outcome vector
X_mean = X.mean(axis=0)
X_sd = X.std(axis=0, ddof=1)
Xc = (X - X_mean) / X_sd
sd_y = y.std(ddof=1)

# HyperParameters for the GDR2 prior
# Adjust as needed for your specific problem
K = X.shape[1]
mu_phi = np.zeros(K - 1)
mean_R2 = 0.5
prec_R2 = 2



with pm.Model() as model:
    # prior for the intercept and error term
    # not part of the GDR2 prior, but needed for the model
    sigma = pm.HalfStudentT("sigma", nu=3, sigma=sd_y)
    a_c = pm.StudentT("a_c", nu=4, mu=y.mean(), sigma=sd_y)

    # priors
    zb = pm.Normal("zb", mu=0.0, sigma=1.0, shape=K)
    eta = pm.Normal("eta", mu=mu_phi, sigma=np.pi, shape=K - 1)
    eta_full = pt.concatenate([eta, pt.zeros(1)])
    phi = pm.Deterministic("phi", pm.math.softmax(eta_full))
    R2 = pm.Beta("R2", mu=mean_R2, nu=prec_R2)

    # compute coefficients following the GDR2 prior
    tau2 = R2 / (1 - R2)
    b = pm.Deterministic("b", zb * sigma * pt.sqrt(phi * tau2))

    mu = a_c + Xc @ b
    pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)