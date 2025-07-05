# Key Steps for Data Collection in Machine Learning (with Python Examples)

This guide outlines the essential steps for collecting data in machine learning projects, with sample Python code for each step.

---

## 1. Identifying Data Sources

**Description:**  
Determine where your data will come from (CSV, databases, APIs, web scraping, etc.).

**Sample Code:**  
_List possible sources for reference._

```python
# Local CSV
csv_path = 'data/my_data.csv'

# Database connection string
db_conn_str = 'postgresql://user:password@localhost:5432/mydb'

# API endpoint
api_url = 'https://api.example.com/data'

# Website URL for scraping
website_url = 'https://example.com/data'
```

---

## 2. Acquiring Data

**Description:**  
Load or collect the data from the identified sources.

**Sample Code:**

```python
import pandas as pd
import requests

# From CSV
df_csv = pd.read_csv(csv_path)

# From Excel
df_excel = pd.read_excel('data/my_data.xlsx')

# From SQL database
import sqlalchemy
engine = sqlalchemy.create_engine(db_conn_str)
df_sql = pd.read_sql('SELECT * FROM tablename', engine)

# From API
response = requests.get(api_url)
data = response.json()
df_api = pd.DataFrame(data)

# Web scraping (with BeautifulSoup)
from bs4 import BeautifulSoup
import requests

response = requests.get(website_url)
soup = BeautifulSoup(response.text, 'html.parser')
# Example: Extract table data
tables = pd.read_html(response.text)
df_web = tables[0]
```

---

## 3. Data Inspection

**Description:**  
Preview and explore the collected data for validation.

**Sample Code:**

```python
print(df_csv.head())
print(df_csv.info())
print(df_csv.describe())
```

---

## 4. Data Cleaning

**Description:**  
Handle missing values, duplicates, inconsistent formats, etc.

**Sample Code:**

```python
# Remove duplicates
df_clean = df_csv.drop_duplicates()

# Fill missing values
df_clean = df_clean.fillna(0)

# Drop rows with missing values
df_clean = df_clean.dropna()

# Convert data types
df_clean['column'] = df_clean['column'].astype(int)
```

---

## 5. Data Storage

**Description:**  
Save the cleaned data for future use.

**Sample Code:**

```python
df_clean.to_csv('data/clean_data.csv', index=False)
df_clean.to_excel('data/clean_data.xlsx', index=False)
```

---

## 6. Documentation & Versioning

**Description:**  
Document the data collection steps and keep versions of your datasets.

**Sample Code:**

```python
# Save a log of collection steps
with open('data_collection_log.txt', 'w') as f:
    f.write('Data collected from: {}\n'.format(csv_path))
    f.write('Duplicates removed, missing values filled with 0.\n')

# Versioning file name example
df_clean.to_csv('data/clean_data_v1.csv', index=False)
```

---

## 7. Automating Data Collection (Optional)

**Description:**  
Schedule data collection scripts for periodic updates.

**Sample Code:**

```python
# Example: Use cron (Linux/macOS) or Task Scheduler (Windows), or Python's schedule package
import schedule
import time

def job():
    print("Collecting data...")
    # Place data collection and cleaning code here

schedule.every().day.at("09:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

**References:**
- [Pandas Documentation](https://pandas.pydata.org/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Requests Documentation](https://docs.python-requests.org/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)