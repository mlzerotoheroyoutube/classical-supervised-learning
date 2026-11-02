"""Logistic Regression: code from the ML Zero to Hero explainer.

Video: https://youtu.be/ZRmTV1eZn30
"""

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

rng = np.random.default_rng(0)
n, d = 1000, 3
X = rng.normal(size=(n, d))
w_true = np.array([2.0, -1.0, 0.5])
y = (rng.uniform(size=n) < sigmoid(X @ w_true - 0.5)).astype(float)
Xb = np.hstack([np.ones((n, 1)), X])

# Newton / IRLS with a tiny ridge penalty
lam = 1e-3
w = np.zeros(d + 1)
for it in range(10):
    p = sigmoid(Xb @ w)
    grad = Xb.T @ (p - y) + lam * w
    H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + lam * np.eye(d + 1)
    w -= np.linalg.solve(H, grad)
    nll = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    print(f"iter {it}  NLL {nll:.5f}  |grad| {np.linalg.norm(grad):.2e}")
print("w (bias first):", w.round(3))   # ≈ [-0.5, 2.0, -1.0, 0.5]

# --------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
clf = LogisticRegression(C=1e4, max_iter=1000).fit(X_tr, y_tr)   # large C ≈ unregularised
p = clf.predict_proba(X_te)[:, 1]
print("coef:", clf.coef_.round(3), "intercept:", clf.intercept_.round(3))
print("odds ratios:", np.exp(clf.coef_).round(3))
print("test log loss:", round(log_loss(y_te, p), 4), " AUC:", round(roc_auc_score(y_te, p), 4))
