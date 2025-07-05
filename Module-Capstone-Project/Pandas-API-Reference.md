# Pandas API Reference Guide for Data Collection & Exploration

## 1. Importing Pandas
```python
import pandas as pd
```

## 2. Data Collection (Loading Data)
- **CSV:**  
  `pd.read_csv('filename.csv')`
- **Excel:**  
  `pd.read_excel('filename.xlsx', sheet_name='Sheet1')`
- **JSON:**  
  `pd.read_json('filename.json')`
- **SQL:**  
  `pd.read_sql(query, connection)`
- **Clipboard:**  
  `pd.read_clipboard()`
- **From dict/list:**  
  `pd.DataFrame(data)`

## 3. Basic Data Exploration
- **View top/bottom rows:**  
  `df.head(n)` / `df.tail(n)`
- **Shape of DataFrame:**  
  `df.shape`
- **Column names:**  
  `df.columns`
- **Data types:**  
  `df.dtypes`
- **General info:**  
  `df.info()`
- **Quick summary stats:**  
  `df.describe()`
- **Sample random rows:**  
  `df.sample(n)`

## 4. Data Selection & Filtering
- **Select column:**  
  `df['col']` or `df.col`
- **Select multiple columns:**  
  `df[['col1', 'col2']]`
- **Select rows by index:**  
  `df.iloc[0:5]`
- **Select rows by label:**  
  `df.loc[0:5]`
- **Boolean filtering:**  
  `df[df['col'] > value]`

## 5. Handling Missing Data
- **Check for missing values:**  
  `df.isnull().sum()`
- **Drop missing values:**  
  `df.dropna()`
- **Fill missing values:**  
  `df.fillna(value)`

## 6. Data Type Conversion
- **Convert column type:**  
  `df['col'] = df['col'].astype('float')`

## 7. Basic Data Cleaning
- **Rename columns:**  
  `df.rename(columns={'old':'new'}, inplace=True)`
- **Remove duplicates:**  
  `df.drop_duplicates()`
- **Replace values:**  
  `df.replace(to_replace, value)`

## 8. Value Counts & Unique Values
- **Unique values:**  
  `df['col'].unique()`
- **Count unique:**  
  `df['col'].nunique()`
- **Value counts:**  
  `df['col'].value_counts()`

## 9. Sorting Data
- **Sort by column:**  
  `df.sort_values('col', ascending=False)`

## 10. Grouping and Aggregation
- **Group by column:**  
  `df.groupby('col').mean()`
- **Aggregate multiple stats:**  
  `df.groupby('col').agg({'col2': ['mean', 'sum']})`

## 11. Data Visualization (Quickly with Pandas)
- **Histogram:**  
  `df['col'].hist()`
- **Boxplot:**  
  `df.boxplot(column='col')`
- **Scatter plot:**  
  `df.plot.scatter(x='col1', y='col2')`

## 12. Export Data
- **To CSV:**  
  `df.to_csv('filename.csv', index=False)`
- **To Excel:**  
  `df.to_excel('filename.xlsx', index=False)`

---

**Tip:** For large datasets, use `.sample(n)` and `.info()` to quickly inspect without loading all rows.