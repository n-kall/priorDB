import numpy as np
import pymc as pm
import pytensor.tensor as pt

# center and scale predictors
# Assume X is your predictor matrix and y is your outcome vector
X_mean = X.mean(axis=0)
X_sd = X.std(axis=0, ddof=1)
Xc = (X - X_mean) / X_sd
sd_y = y.std(ddof=1)

# HyperParameters for the R2D2 prior
# Adjust as needed for your specific problem
K = X.shape[1]
cons_D2 = np.full(K, 1/K)
mean_R2 = 0.3
prec_R2 = 3

with pm.Model() as model:
    # prior for the intercept and error term
    # not part of the R2D2 prior, but needed for the model
    sigma = pm.HalfStudentT("sigma", nu=3, sigma=sd_y)
    a_c = pm.StudentT("a_c", nu=4, mu=y.mean(), sigma=sd_y)

    # R2D2 prior
    zb = pm.Normal("zb", mu=0.0, sigma=1.0, shape=K)
    phi = pm.Dirichlet("phi", a=cons_D2)
    R2 = pm.Beta("R2", mu=mean_R2, nu=prec_R2)

    # compute coefficients following the R2D2 prior
    tau2 = sigma**2 * R2 / (1 - R2)
    b = pm.Deterministic("b", zb * pt.sqrt(phi * tau2))

    mu = a_c + Xc @ b
    pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)