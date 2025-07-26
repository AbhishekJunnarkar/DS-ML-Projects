
# Regression Error Metrics Cheatsheet

Understand the differences between **MAE**, **MSE**, and **RMSE** — key evaluation metrics for regression models.

---

## Metrics & Formulas

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MAE** (Mean Absolute Error) | \( \text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i| \) | Average of absolute errors |
| **MSE** (Mean Squared Error) | \( \text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2 \) | Average of squared errors |
| **RMSE** (Root Mean Squared Error) | \( \text{RMSE} = \sqrt{\text{MSE}} \) | Square root of MSE |

---

## Visual Behavior

- MAE: 📘 Linear increase with error
- MSE: 🟧 Quadratic increase (large errors hurt more)
- RMSE: 🟢 Grows like MSE but is easier to interpret

---

## Summary Table

| Metric | Penalizes Large Errors? | Output Units | Easy to Understand? |
|--------|--------------------------|---------------|----------------------|
| **MAE** | ❌ No | Same as target | ✅ Yes |
| **MSE** | ✅ Yes (squares errors) | Squared units | ❌ No |
| **RMSE** | ✅ Yes (squares + root) | Same as target | ✅ Yes |

---

## When to Use What

- **Use MAE** if you want average error, treating all mistakes equally.
- **Use MSE** if you want to strongly penalize larger errors.
- **Use RMSE** for a balanced metric that’s sensitive to large errors but easy to interpret.

---

