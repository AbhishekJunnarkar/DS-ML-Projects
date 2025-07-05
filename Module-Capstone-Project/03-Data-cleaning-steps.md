# Key Steps for Data Cleaning in Machine Learning (with Python Examples)

This guide outlines the essential data cleaning steps for machine learning projects, with sample Python code for each.

---

## 1. Remove Duplicates

```python
import pandas as pd

df = pd.read_csv('data.csv')
df = df.drop_duplicates()
```

---

## 2. Handle Missing Data

**2.1 Detect Missing Values**

```python
print(df.isnull().sum())
```

**2.2 Remove Rows/Columns with Missing Data**

```python
df = df.dropna()               # Remove rows with any missing values
df = df.dropna(axis=1)         # Remove columns with any missing values
```

**2.3 Impute Missing Values**

```python
df = df.fillna(0)                              # Fill with constant (e.g. 0)
df['col'] = df['col'].fillna(df['col'].mean()) # Fill with column mean
```

---

## 3. Fix Data Types

```python
df['col'] = df['col'].astype(int)
df['date'] = pd.to_datetime(df['date'])
```

---

## 4. Handle Outliers

```python
# Remove outliers beyond 3 standard deviations
import numpy as np
for col in ['feature1', 'feature2']:
    df = df[(np.abs(df[col] - df[col].mean()) <= (3*df[col].std()))]
```

---

## 5. Standardize and Normalize Data

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
df[['num_col1', 'num_col2']] = scaler.fit_transform(df[['num_col1', 'num_col2']])

minmax = MinMaxScaler()
df[['num_col1', 'num_col2']] = minmax.fit_transform(df[['num_col1', 'num_col2']])
```

---

## 6. Encode Categorical Variables

```python
# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# One-Hot Encoding
df = pd.get_dummies(df, columns=['category'])
```

---

## 7. Rename Columns and Standardize Format

```python
df = df.rename(columns={'Old Name': 'new_name'})
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
```

---

## 8. Remove Unnecessary Columns

```python
df = df.drop(['unwanted_col1', 'unwanted_col2'], axis=1)
```

---

## 9. Handle Inconsistent Data (String Cleaning)

```python
df['city'] = df['city'].str.strip().str.lower()
```

---

## 10. Save Cleaned Data

```python
df.to_csv('clean_data.csv', index=False)
```

---

**References:**
- [Pandas Documentation](https://pandas.pydata.org/)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)