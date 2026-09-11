import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Assume y is your observed time series
# Construct lag matrix for AR(p) model
p = 2
T = len(y)
var_y = np.var(y, ddof=1)
ys = y[p:]
Y_lag = np.stack([y[p - k : T - k] for k in range(1, p + 1)], axis=1)

# Hyperparameters for the ARR2 prior
cons = np.full(p, 1.0)
mean_R2 = 0.5
prec_R2 = 10.0

with pm.Model() as model:
    sigma = pm.HalfStudentT("sigma", nu=3, sigma=2.5 * np.sqrt(var_y))

    zb = pm.Normal("zb", mu=0.0, sigma=1.0, shape=p)
    psi = pm.Dirichlet("psi", a=cons)
    R2 = pm.Beta("R2", mu=mean_R2, nu=prec_R2)

    tau2 = R2 / (1 - R2)
    phi = pm.Deterministic("phi", zb * pt.sqrt(sigma**2 / var_y * tau2 * psi))

    mu = Y_lag @ phi
    pm.Normal("y_obs", mu=mu, sigma=sigma, observed=ys)
