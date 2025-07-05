# Scikit-learn API Reference Guide for Machine Learning

This guide covers the most commonly used scikit-learn (sklearn) APIs for data preprocessing, model selection, training, and evaluation.

---

## 1. Importing scikit-learn Modules

```python
from sklearn import datasets, model_selection, preprocessing, metrics
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
```

---

## 2. Loading Sample Datasets

```python
from sklearn.datasets import load_iris, load_boston, load_digits
data = load_iris()
X, y = data.data, data.target
```

---

## 3. Data Preprocessing

- **Train-test split:**  
  `X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)`

- **Standardization (zero mean, unit variance):**  
  `scaler = preprocessing.StandardScaler()`  
  `X_scaled = scaler.fit_transform(X)`

- **Normalization (scaling to [0, 1]):**  
  `normalizer = preprocessing.MinMaxScaler()`  
  `X_norm = normalizer.fit_transform(X)`

- **Label encoding:**  
  `le = preprocessing.LabelEncoder()`  
  `y_encoded = le.fit_transform(y)`

- **One-hot encoding:**  
  `ohe = preprocessing.OneHotEncoder()`  
  `X_ohe = ohe.fit_transform(X)`

---

## 4. Model Selection

- **List of models:**
  - `LinearRegression()`
  - `LogisticRegression()`
  - `DecisionTreeClassifier()`, `DecisionTreeRegressor()`
  - `RandomForestClassifier()`, `RandomForestRegressor()`
  - `SVC()` for classification, `SVR()` for regression
  - `KMeans()` for clustering

- **Instantiate a model:**  
  `model = RandomForestClassifier(n_estimators=100, random_state=42)`

---

## 5. Model Training

```python
model.fit(X_train, y_train)
```

---

## 6. Prediction

```python
y_pred = model.predict(X_test)
```

---

## 7. Model Evaluation

- **Classification metrics:**
  - Accuracy: `metrics.accuracy_score(y_test, y_pred)`
  - Confusion Matrix: `metrics.confusion_matrix(y_test, y_pred)`
  - Classification Report: `metrics.classification_report(y_test, y_pred)`

- **Regression metrics:**
  - Mean Squared Error: `metrics.mean_squared_error(y_test, y_pred)`
  - R2 Score: `metrics.r2_score(y_test, y_pred)`

---

## 8. Hyperparameter Tuning

- **Grid search:**  
  ```python
  from sklearn.model_selection import GridSearchCV
  grid = GridSearchCV(estimator=model, param_grid={'n_estimators': [50, 100]}, cv=5)
  grid.fit(X_train, y_train)
  best_model = grid.best_estimator_
  ```

---

## 9. Cross-Validation

- **K-Fold:**  
  `scores = model_selection.cross_val_score(model, X, y, cv=5)`

---

## 10. Pipelines

```python
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', preprocessing.StandardScaler()),
    ('clf', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

---

## 11. Clustering Example (KMeans)

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
```

---

## References

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Scikit-learn API Reference](https://scikit-learn.org/stable/modules/classes.html)

---