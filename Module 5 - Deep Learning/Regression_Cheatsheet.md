
# 📘 Regression Cheatsheet: From Linear to Regularization

---

## 📈 Linear Regression

- **Goal**: Predict target `y` from input `x` using a straight line.
- **Equation**: `y = w0 + w1 * x`
- **Assumes** a linear relationship between features and output.
- ✅ Best for: Simple trends, like salary increasing with experience.

---

## 🔄 Polynomial Regression

- **Goal**: Fit a curve (not just a straight line).
- **Equation (degree 2)**: `y = w0 + w1*x + w2*x^2`
- **Still linear in parameters**, but nonlinear in input.
- ✅ Best for: U-shaped, curved patterns in data.

---

## 🧠 Underfitting vs Overfitting

| Term         | Meaning                              | Symptoms                      |
|--------------|--------------------------------------|-------------------------------|
| Underfitting | Model too simple                     | Low accuracy on train/test    |
| Overfitting  | Model too complex (memorizes noise)  | High train, low test accuracy |

---

## 🛡️ Regularization

- **Purpose**: Prevent overfitting by penalizing large weights.
- **Modified Loss**: `Loss = MSE + λ * penalty`

---

## 🔹 Ridge Regression (L2)

- **Penalty**: Sum of squared weights: `λ * ∑w²`
- **Effect**: Shrinks weights but **keeps all features**.
- ✅ Best for: Many **correlated** features.

---

## 🔸 Lasso Regression (L1)

- **Penalty**: Sum of absolute weights: `λ * ∑|w|`
- **Effect**: Shrinks some weights to **exactly zero**.
- ✅ Best for: **Feature selection**, sparse models.

---

## 🟢 Elastic Net Regression

- **Penalty**: `α * L1 + (1 - α) * L2`
- **Effect**: Combines Ridge + Lasso
- ✅ Best for: Large datasets with many, possibly correlated features.

---

## 🧪 Example Choices

| Scenario | Best Regression |
|----------|-----------------|
| Straight-line trend | Linear Regression |
| Curved data | Polynomial Regression |
| Many correlated features | Ridge |
| Want fewer features | Lasso |
| Combo case | Elastic Net |

---
