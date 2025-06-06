from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, classification_report

# 1. Load example dataset (binary classification: cancer detection)
data = load_breast_cancer()
X = data.data
y = data.target  # 1 = malignant, 0 = benign

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. Make predictions
y_pred = clf.predict(X_test)

# 5. Evaluate recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# 6. Optional: Print full classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Optional: Show confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
