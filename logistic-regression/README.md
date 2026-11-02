# Logistic Regression

**Level:** Beginner  ·  **Topic:** Classical Supervised Learning

▶️ **Watch the video:** https://youtu.be/ZRmTV1eZn30

> A linear model for binary classification: compute a weighted sum of features, squash it through the sigmoid to get a probability, and fit the weights by maximising the likelihood (minimising cross-entropy). Convex, fast, calibrated-ish, interpretable, and the mathematical template for every neural-network output layer.

## The intuition
Suppose you want to predict whether an email is spam. You could add up clues: +2 points if it contains "free money", +1.5 if it has lots of exclamation marks, −3 if it's from someone in your contacts. The total "spam score" could be any number, like −4 or +7. But you want a *probability*, a number between 0 and 1.

Logistic regression adds up weighted clues exactly like linear regression, then feeds the score through an S-shaped squashing function called the **sigmoid**. Very negative scores come out near 0 ("definitely not spam"), very positive scores near 1 ("definitely spam"), and a score of 0 comes out as exactly 0.5 ("coin flip").

Example: an email with "free money" and lots of exclamation marks, from a stranger, scores $2 + 1.5 = 3.5$. The sigmoid of 3.5 is about 0.97, so a 97% chance of spam. From a contact, the score is $3.5 - 3 = 0.5$, giving about 62%.

How are the clue weights learned? From thousands of labelled emails. The computer adjusts weights so that real spam gets high probabilities and real non-spam gets low ones. It especially punishes being confidently wrong: saying "1% spam" for an actual spam email costs a lot more than saying "40%".

## Core notes
- Model: $p(y=1\mid x) = \sigma(w^\top x + b)$; decision boundary $w^\top x + b = 0$ is a hyperplane.
- Loss: binary Cross-Entropy = negative Bernoulli log-likelihood (Log Loss and Brier Score).
- Gradient has the famous form $\nabla = X^\top(p - y)$: prediction error times input, the same as linear regression.
- Convex: any local minimum is global; no closed form, so use Newton/IRLS, L-BFGS, or SGD.
- Coefficient $w_j$: one-unit increase in $x_j$ multiplies the odds by $e^{w_j}$.
- Separable data makes unregularised weights diverge; always add an $\ell_2$ (or $\ell_1$) penalty (Ridge, Lasso and Elastic Net).
- Threshold 0.5 is not sacred: choose it from costs or the Precision-Recall Curve / ROC Curve and AUC.
- Handle class imbalance with class weights or threshold tuning, not by trusting accuracy (Imbalanced Data).
- Multiclass: one-vs-rest, or properly with Softmax Regression.
- A single neuron with sigmoid activation is logistic regression (Perceptron, Multilayer Perceptron).

## The math
**Model.** $x \in \mathbb{R}^d$ (with a constant feature for the bias), $y \in \{0, 1\}$, $p_i = \sigma(w^\top x_i)$, $\sigma(z) = \frac{1}{1 + e^{-z}}$.

**Useful sigmoid identities.** $1 - \sigma(z) = \sigma(-z)$, and
$$
\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)\,(1 - \sigma(z)).
$$

**Likelihood and loss.** Each label is Bernoulli: $p(y_i \mid x_i, w) = p_i^{y_i}(1 - p_i)^{1 - y_i}$. The negative log-likelihood is the binary cross-entropy
$$
\mathcal{L}(w) = -\sum_{i=1}^n \big[\, y_i \log p_i + (1 - y_i)\log(1 - p_i) \,\big].
$$
With labels $\tilde y_i \in \{-1, +1\}$ this is equivalently $\sum_i \log(1 + e^{-\tilde y_i w^\top x_i})$, the **logistic loss** of the margin.

**Gradient derivation.** For one example, with $z_i = w^\top x_i$:
$$
\frac{\partial \ell_i}{\partial z_i} = -\frac{y_i}{p_i}\,p_i(1-p_i) + \frac{1-y_i}{1-p_i}\,p_i(1-p_i) = -y_i(1-p_i) + (1-y_i)p_i = p_i - y_i.
$$
By the chain rule, $\partial z_i/\partial w = x_i$, so
$$
\nabla_w \mathcal{L} = \sum_{i=1}^n (p_i - y_i)\,x_i = X^\top(p - y).
$$

**Hessian and convexity.** Differentiating again, $\partial p_i / \partial w = p_i(1-p_i)x_i$:
$$
\nabla^2_w \mathcal{L} = \sum_i p_i(1-p_i)\,x_i x_i^\top = X^\top S X, \qquad S = \operatorname{diag}\big(p_i(1-p_i)\big).
$$
Since $0 < p_i(1-p_i) \le \tfrac14$, $v^\top X^\top S X v = \sum_i p_i(1-p_i)(x_i^\top v)^2 \ge 0$: the loss is convex (strictly if $X$ has full column rank).

**Newton / IRLS.** $w_{t+1} = w_t - (X^\top S X)^{-1}X^\top(p - y)$, which can be rewritten as a weighted least-squares solve
$$
w_{t+1} = (X^\top S X)^{-1}X^\top S\,z, \qquad z = Xw_t + S^{-1}(y - p),
$$
the "working response" $z$ giving iteratively reweighted least squares.

**Separability.** If some $w$ satisfies $\tilde y_i w^\top x_i > 0$ for all $i$, then $\mathcal{L}(cw) \to 0$ as $c \to \infty$, so the infimum is not attained. With $\ell_2$ penalty $\tfrac{\lambda}{2}\|w\|^2$, the objective becomes strongly convex and the minimiser is unique and finite.

**Log-odds interpretation.** $\log\frac{p}{1-p} = w^\top x$, so increasing $x_j$ by one unit adds $w_j$ to the log-odds, i.e. multiplies the odds by $e^{w_j}$.

## Run the code
```bash
pip install numpy scikit-learn
python logistic_regression.py
```

## Check yourself
<details><summary>Derive the gradient of the logistic regression loss.</summary>

With $p_i = \sigma(w^\top x_i)$ and loss $-\sum_i[y_i\log p_i + (1-y_i)\log(1-p_i)]$, use $\sigma'(z) = \sigma(z)(1-\sigma(z))$. The derivative with respect to the logit $z_i$ simplifies to $p_i - y_i$, because the $p_i(1-p_i)$ from the sigmoid cancels the denominators from the logs. Chain rule with $\partial z_i/\partial w = x_i$ gives $\nabla_w\mathcal{L} = \sum_i (p_i - y_i)x_i = X^\top(p - y)$. The Hessian is $X^\top SX$ with $S = \operatorname{diag}(p_i(1-p_i))$, which is PSD, proving convexity.
</details>

<details><summary>Why not use MSE loss with a sigmoid output for classification?</summary>

MSE on sigmoid outputs is non-convex in $w$, and its gradient contains a factor $\sigma'(z) = p(1-p)$ that vanishes when the model is confidently wrong ($p \approx 0$ for a positive), so learning stalls exactly when it should be fastest. Cross-entropy's gradient $p - y$ has no such factor. Cross-entropy is also the correct Bernoulli likelihood and a proper scoring rule, giving calibrated probabilities.
</details>

<details><summary>What happens when the training data is linearly separable?</summary>

The likelihood can be pushed arbitrarily close to 1 by scaling $w$ up, so the MLE doesn't exist: coefficients grow without bound and software warns about non-convergence. Gradient descent still converges in direction to the hard-margin SVM separator (slowly, at rate $1/\log t$). In practice you add $\ell_2$ regularisation, which makes the problem strongly convex with a finite unique solution, or use Firth's penalised likelihood in statistics.
</details>

<details><summary>How do you interpret a logistic regression coefficient?</summary>

The model is linear in log-odds, so $w_j$ is the change in log-odds of $y=1$ per unit increase in $x_j$, holding other features fixed; $e^{w_j}$ is the odds ratio. For example $w_j = 0.7$ means the odds roughly double per unit. It is not a change in probability, which depends on where you are on the sigmoid (largest effect near $p = 0.5$). Comparisons across features require standardised features, and collinearity or regularisation can distort them.
</details>

<details><summary>How would you handle a 1:1000 class imbalance with logistic regression?</summary>

First, evaluate with PR-AUC, log loss and recall at a fixed precision rather than accuracy. Options: class weights (`class_weight="balanced"`) which reweight the likelihood, or subsampling negatives and correcting the intercept by the log of the sampling rate. Tune the decision threshold on validation data according to costs. If you reweight or resample, probabilities are no longer calibrated to the true base rate, so recalibrate (Platt or isotonic) if you need real probabilities.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.