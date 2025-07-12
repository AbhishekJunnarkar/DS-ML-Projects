As a machine learning developer, I want to have the machine learning process defined and have a sample python code
for each step of the machine learning.

## Machine Learning Process

A typical machine Learning process includes the below 6 building blocks.

### 1. Data Collection

- Accuracy: Ensure the data is accurate, especially for supervised learning, as it serves as the ground truth for training models.
- Relevance: Collect data that is relevant to the problem you are trying to solve. Irrelevant data can negatively impact model performance.
- Quantity: The amount of data needed varies by algorithm. Some require more data to provide meaningful results.
- Variability: Include diverse data to help the model understand different scenarios and improve its predictions.
- Ethics: Address ethical issues such as privacy, security, informed consent, and bias during data collection to avoid harmful biases in predictions.


#### 1.1 How to import data in Python

- Pandas Package: Python's pandas package is powerful and easy to use for data analysis, providing functions for creating, structuring, and importing data.
- Importing Data: You can import data into pandas using the pd.read_csv function for CSV files and pd.read_excel function for Excel files. For multi-sheet Excel files, specify the sheet name if you need a sheet other than the first one.
- Data Structures: Pandas represents data as Series (one-dimensional) and DataFrames (two-dimensional), which can be created from lists, dictionaries, or external files like CSV and Excel.

------

### Data Exploration

- Identifying Missing Data: Use the isnull method in Pandas to create a mask that identifies rows with missing values.
- Removing Missing Data: Use the dropna method to remove rows or columns with missing values. You can specify conditions to remove only certain rows or columns.
- Replacing Missing Data: Use the fillna method to replace missing values with specified values or functions, such as the median of non-missing values.

### Data Preparation

**Common data quality issues**

- Data Preparation: Ensuring data is suitable for machine learning is crucial. Poor quality input leads to unreliable output.
- Common Issues: Missing data, outliers, and class imbalance are typical data quality problems.
- Addressing Issues: Techniques like imputation for missing data, understanding outliers, and under-sampling the majority class to handle class imbalance are essential steps in data preparation.

**Normalizing the data**

- Normalization Purpose: Ensures data values share a particular property, often scaling them to a specified range, which reduces model complexity and makes results easier to interpret.
- Z-Score Normalization: Transforms data to have a mean of zero and a standard deviation of one. Useful when there are no significant outliers.
- Min-Max Normalization: Scales data to a user-defined range, typically between 0 and 1. Suitable when data needs to have specific lower and upper bounds.
- Log Transformation: Replaces data values with their logarithms to minimize the impact of outliers. Works only for positive values.

**How to Normalize data in Python**

- Normalization Purpose: Transform data to make it suitable for machine learning by ensuring it conforms to specific characteristics.
- Min-Max Normalization: Uses the MinMaxScaler from scikit-learn to scale data to a range between 0 and 1.
- Z-Score Normalization: Uses the StandardScaler from scikit-learn to scale data so that it has a mean of 0 and a standard deviation of 1.

**Sampling the data**

- Purpose of Sampling: Sampling is used to reduce the size of a dataset or to split it into training and test sets for model evaluation.
- Random Sampling Without Replacement: Selects a subset from the population without reselecting the same instance.
- Random Sampling With Replacement: Allows reselection of the same instance, useful for techniques like bootstrapping.
- Stratified Random Sampling: Ensures the sample maintains the same distribution of a particular feature as the overall population.

**How to Sample data using Python**

- Splitting Data: Before training a machine learning model, data is split into training and test sets using sampling approaches.
- Simple Random Sampling: The train_test_split function from the sklearn.model_selection package is used to split data into training and test sets, typically allocating 25% of data to the test set by default.
- Stratified Random Sampling: Ensures the distribution of values for a specific column is maintained between the original, training, and test data by using the stratify parameter in the train_test_split function.

**Dimensionality Reduction**
- Dimensionality Reduction: This process reduces the number of features or dimensions in a dataset, which helps in reducing processing time, storage requirements, and improving model interpretability.
- Curse of Dimensionality: Increasing the number of features in a dataset without a corresponding increase in data instances can degrade model performance. Dimensionality reduction helps mitigate this issue.

Approaches: 
- There are two common approaches:
 - Feature Selection: Identifies a minimal set of features that result in model performance close to that obtained using all features.
 - Feature Extraction: Uses mathematical transformations to create new features that are projections of the original ones, though these new features may be less interpretable.


### Modelling


### Evaluation


### Actionable Insights

