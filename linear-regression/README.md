# Linear Regression

**Level:** Beginner  ·  **Topic:** Classical Supervised Learning

▶️ **Watch the video:** https://youtu.be/esFUXmIsJTo

> Predict a number as a weighted sum of input features, choosing the weights that minimise the squared error. It has a closed-form solution (the normal equations), a clean geometric meaning (orthogonal projection), and a probabilistic one (Gaussian maximum likelihood), which is why every other model is first compared against it.

## The intuition
Picture a scatter of dots on graph paper: each dot is a house, its left-right position is the size of the house, and its height is the price it sold for. You want a ruler-straight line through the cloud so that, for a new house, you can read the price off the line. Linear regression is the rule for choosing **the best line**.

"Best" means this: for every dot, measure how far it sits above or below the line (that gap is called the **error** or **residual**). Square each gap so negatives don't cancel positives and so big misses count a lot more than small ones, then add them all up. The best line is the one with the smallest total.

Tiny example. Three houses: 1 room costs 2, 2 rooms costs 4, 3 rooms costs 5 (in hundreds of thousands). Try the line "price = 1.5 × rooms + 0.67". Its guesses are 2.17, 3.67, 5.17. The gaps are −0.17, +0.33, −0.17; squared and added they give about 0.17. No other straight line gets a smaller total, so that's the answer. With more features (rooms, age, distance to school) the "line" becomes a flat sheet in more dimensions, but the rule is the same: one weight per feature, chosen to make the squared gaps as small as possible.

## Core notes
- Model: $\hat y = w^\top x + b$; absorb $b$ by appending a constant-1 feature.
- Loss: mean squared error (Loss Functions); minimiser solves the **normal equations** $X^\top X w = X^\top y$.
- Geometry: $\hat y$ is the orthogonal projection of $y$ onto the column space of $X$; residuals are orthogonal to every feature (Orthogonality and Projections).
- Probability: OLS = MLE under Gaussian noise; adding a Gaussian prior gives ridge (MAP).
- Never invert $X^\top X$ explicitly; use QR, Cholesky, or the SVD (`np.linalg.lstsq`).
- Cost: $O(nd^2 + d^3)$ closed-form; for huge $n$ or $d$ use SGD or conjugate gradient.
- Collinearity inflates coefficient variance (check VIF / condition number); ridge fixes it.
- Coefficients are only interpretable "holding other features fixed" and only if features are on comparable scales (Feature Scaling and Normalization).
- Evaluate with RMSE/MAE/$R^2$ on held-out data (Regression Metrics, Train, Validation and Test Splits).
- It is the baseline for every regression task; beat it before reaching for Gradient Boosting or neural nets.

## The math
**Setup.** $X \in \mathbb{R}^{n\times d}$ (rows $x_i^\top$), $y \in \mathbb{R}^n$, weights $w \in \mathbb{R}^d$. Objective:
$$
J(w) = \|y - Xw\|_2^2 = (y - Xw)^\top(y - Xw) = y^\top y - 2w^\top X^\top y + w^\top X^\top X w.
$$

**Derivation of the normal equations.** Using $\nabla_w (a^\top w) = a$ and $\nabla_w (w^\top A w) = (A + A^\top)w = 2Aw$ for symmetric $A$ (Matrix Calculus):
$$
\nabla_w J = -2X^\top y + 2X^\top X w = 0 \quad\Longrightarrow\quad X^\top X\,\hat w = X^\top y.
$$
The Hessian $\nabla^2 J = 2X^\top X \succeq 0$, so $J$ is convex and any solution is a global minimum (Positive Definite Matrices). If $\operatorname{rank}(X) = d$, $X^\top X \succ 0$ and
$$
\hat w = (X^\top X)^{-1} X^\top y.
$$
If $X$ is rank-deficient, the minimum-norm solution is $\hat w = X^+ y = V\Sigma^{+}U^\top y$ from the SVD $X = U\Sigma V^\top$.

**Geometry.** The normal equations say $X^\top(y - X\hat w) = 0$: the residual is orthogonal to every column of $X$. Hence $\hat y = Hy$ with $H = X(X^\top X)^{-1}X^\top$, $H^2 = H = H^\top$, $\operatorname{tr}(H) = d$ (degrees of freedom).

**Probabilistic view.** With $y_i = w^\top x_i + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d.:
$$
\log p(y \mid X, w) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|y - Xw\|^2,
$$
so maximising likelihood over $w$ is exactly minimising squared error, and $\hat\sigma^2_{\text{MLE}} = \tfrac{1}{n}\|y - X\hat w\|^2$ (the unbiased estimate divides by $n - d$).

**Sampling distribution.** Substituting $y = Xw^* + \varepsilon$:
$$
\hat w = w^* + (X^\top X)^{-1}X^\top \varepsilon, \qquad \mathbb{E}[\hat w] = w^*, \qquad \operatorname{Cov}(\hat w) = \sigma^2 (X^\top X)^{-1}.
$$

**Coefficient of determination.** $R^2 = 1 - \dfrac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$; with an intercept, $R^2 \in [0,1]$ in-sample, but it can be negative on test data.

## Run the code
```bash
pip install numpy scikit-learn
python linear_regression.py
```

## Check yourself
<details><summary>Derive the closed-form solution of linear regression.</summary>

Write the loss as $J(w) = \|y - Xw\|^2 = y^\top y - 2w^\top X^\top y + w^\top X^\top Xw$. Its gradient is $-2X^\top y + 2X^\top Xw$; setting it to zero gives the normal equations $X^\top Xw = X^\top y$. The Hessian $2X^\top X$ is positive semidefinite, so this is a global minimum, and if $X$ has full column rank the solution is unique: $\hat w = (X^\top X)^{-1}X^\top y$. In practice you solve it with QR or SVD rather than forming the inverse, because forming $X^\top X$ squares the condition number.
</details>

<details><summary>What assumptions does linear regression make, and which ones matter for prediction vs inference?</summary>

Linearity of $\mathbb{E}[y\mid x]$ in the parameters, exogeneity ($\mathbb{E}[\varepsilon\mid X] = 0$), no perfect collinearity, homoscedastic and uncorrelated errors, and optionally Gaussian errors. For pure prediction, only "is a linear function a good approximation" really matters and you check it on held-out data. For inference (p-values, confidence intervals) homoscedasticity and independence matter a lot; violations make standard errors wrong, which you can fix with robust (sandwich) or clustered errors. Normality only matters for exact small-sample tests; the CLT covers large samples.
</details>

<details><summary>Why is MSE the "right" loss for linear regression?</summary>

Under i.i.d. Gaussian noise, the negative log-likelihood is proportional to the sum of squared residuals, so minimising MSE is maximum likelihood. MSE is also smooth and convex, giving a closed form, and its minimiser is the conditional mean $\mathbb{E}[y\mid x]$. If the noise is heavy-tailed, MAE (conditional median, Laplace likelihood) or Huber loss is more robust; MSE's quadratic penalty lets a few outliers dominate.
</details>

<details><summary>What happens when features are highly correlated?</summary>

$X^\top X$ becomes nearly singular, so $(X^\top X)^{-1}$ has huge entries and the coefficient variance $\sigma^2(X^\top X)^{-1}$ explodes. Predictions inside the data range may still be fine, but individual coefficients flip sign and become uninterpretable, and extrapolation is unstable. Remedies: drop or combine features, use PCA, or add an $\ell_2$ penalty (ridge), which replaces $X^\top X$ by $X^\top X + \lambda I$ and bounds the condition number.
</details>

<details><summary>What happens to OLS when $d > n$?</summary>

$X^\top X$ is singular, so infinitely many weight vectors fit the training data exactly. The pseudoinverse picks the minimum-norm one, which is also what gradient descent from zero converges to. Its test error depends on the spectrum of the feature covariance: it can be terrible near $d \approx n$ (the double-descent peak) and surprisingly good for $d \gg n$ when the covariance has many small eigenvalues that absorb noise (benign overfitting). Usually you add ridge or lasso regularisation instead.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.