
# 🌳 Decision Trees & Boosting – ML Cheatsheet

This guide covers essential concepts and Python code examples for understanding **Decision Trees**, **Boosting**, and how they perform on real-world datasets.

---

## 🌳 What is a Decision Tree?

A **Decision Tree** is a model that makes decisions by splitting data based on questions.

**Example**:
- Is it raining? → Yes → Take umbrella
- No → Don't take it

It works like a **flowchart** and is great for:
- Classification (e.g., spam detection)
- Regression (e.g., house price prediction)

---

## ⚡ What is Boosting?

**Boosting** is an ensemble technique that builds **multiple weak learners** (like small trees) and combines them to form a **strong model**.

**How it works**:
1. Train a small decision tree.
2. Build another tree to correct previous mistakes.
3. Combine all trees' predictions.

✅ Boosting reduces overfitting and increases accuracy.

---

## 💻 Python Example

### Dataset: Iris
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
```

- Decision Tree Accuracy: 100%
- Boosting Accuracy: 100%

---

## 🍷 Dataset: Wine (More Challenging)
```python
from sklearn.datasets import load_wine
```

- **Decision Tree Accuracy**: ~96.3%
- **Boosting Accuracy**: ~90.7%

---

## 🧪 Dataset: Breast Cancer (Realistic)

| Model            | Accuracy |
|------------------|----------|
| Decision Tree    | ~94.15%  |
| Gradient Boosting| ~95.91%  ✅ |

---

## 📈 Feature Importance (Top 10 from Boosting)
Most important features:
- `worst concave points`
- `mean concave points`
- `mean perimeter`
- (others...)

---

## 🔲 Confusion Matrix (Gradient Boosting)
Shows correct vs incorrect predictions:
- Top-left: True Negatives
- Bottom-right: True Positives
- Off-diagonal: Errors (False Positives/Negatives)

---

## 📊 Classification Report

| Metric    | Gradient Boosting | Decision Tree |
|-----------|-------------------|----------------|
| Precision | 0.963             | 0.971          |
| Recall    | 0.972             | 0.935          |
| F1 Score  | 0.968             | 0.953          |

---

## ✅ Summary

- Decision Trees are easy to understand and fast to train.
- Boosting improves accuracy by correcting mistakes step-by-step.
- On realistic data (like breast cancer), Boosting provides a more balanced performance.
