"""Linear Regression: code from the ML Zero to Hero explainer.

Video: https://youtu.be/esFUXmIsJTo
"""

import numpy as np

rng = np.random.default_rng(0)
n, d = 500, 4
X = rng.normal(size=(n, d))
w_true = np.array([3.0, -2.0, 0.5, 0.0])
y = X @ w_true + 1.5 + 0.3 * rng.normal(size=n)

Xb = np.hstack([np.ones((n, 1)), X])               # bias column

w_normal = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)    # normal equations (fine when well-conditioned)
w_lstsq, *_ = np.linalg.lstsq(Xb, y, rcond=None)   # SVD-based, robust default

resid = y - Xb @ w_lstsq
r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
print("normal eq:", w_normal.round(3))
print("lstsq    :", w_lstsq.round(3), " R^2 =", round(r2, 4))
print("residuals orthogonal to X:", np.allclose(Xb.T @ resid, 0, atol=1e-8))

# --------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
model = LinearRegression().fit(X_tr, y_tr)
pred = model.predict(X_te)
print("coef:", model.coef_.round(3), "intercept:", round(model.intercept_, 3))
print("test RMSE:", round(mean_squared_error(y_te, pred) ** 0.5, 4), "test R^2:", round(r2_score(y_te, pred), 4))
