
# 🌳 Gini Index vs Entropy – Decision Tree Cheatsheet

A handy reference for understanding how decision trees use **Gini Index** and **Entropy** to make splits.

---

## 📘 What is Gini Index?

- Measures **impurity** in a node.
- Formula:  
  \[
  Gini = 1 - \sum p_i^2
  \]
- **Gini = 0**: Perfectly pure node (all one class)  
- **Gini = 0.5**: Max impurity in binary classification

---

## 🔥 What is Entropy?

- Measures **uncertainty or disorder** in a node.
- Formula:  
  \[
  Entropy = -\sum p_i \cdot \log_2(p_i)
  \]
- **Entropy = 0**: No uncertainty  
- **Entropy = 1**: Maximum uncertainty (binary)

---

## 🆚 Gini vs Entropy Comparison

| Feature        | Gini Index                  | Entropy                     |
|----------------|-----------------------------|-----------------------------|
| Measures       | Impurity                    | Uncertainty                 |
| Range          | 0 to 1                      | 0 to 1                      |
| Speed          | Slightly faster             | Slightly slower (uses log)  |
| Used in        | CART (scikit-learn default) | ID3, C4.5                   |
| Interpretability| Easy to understand          | More theoretical            |

---

## 🧪 Python Accuracy Comparison (Breast Cancer Dataset)

```python
# Gini
DecisionTreeClassifier(criterion='gini')
Accuracy: ~94.15%

# Entropy
DecisionTreeClassifier(criterion='entropy')
Accuracy: ~96.49%
```

✅ In this case, **Entropy performed slightly better**.

---

## ✅ When to Use

- Use **Gini** for slightly faster training and simpler logic.
- Use **Entropy** when interpretability or information gain is key.

---

