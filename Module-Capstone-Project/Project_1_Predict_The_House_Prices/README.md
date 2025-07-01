# Objective

**Build a model to predict the house prices. The columns are self-explanatory**

## Expected Outcomes
- Programme Directors are interested to see the following:
- How you approach the dataset.
- EDA performed if any
- Different Models built
- Parameters you optimized and your final model
- Final recommendations wrt to the model

## Steps for Modelling

### Data Collection

- Accuracy: Ensure the data is accurate, especially for supervised learning, as it serves as the ground truth for training models.
- Relevance: Collect data that is relevant to the problem you are trying to solve. Irrelevant data can negatively impact model performance.
- Quantity: The amount of data needed varies by algorithm. Some require more data to provide meaningful results.
- Variability: Include diverse data to help the model understand different scenarios and improve its predictions.
- Ethics: Address ethical issues such as privacy, security, informed consent, and bias during data collection to avoid harmful biases in predictions.

### Data Exploration


#### Step 1: Clean the Data

 - Use df.head(), df.info(), and df.describe() to get a feel for:

    - Missing values 
    - Data types 
    - Value ranges

#### Step 2: Identify the type of problem

| Problem Type   | Target Output     | Example                           |
|----------------| ----------------- | --------------------------------- |
| Classification | Categories/Labels | Spam detection, Sentiment, Fraud  |
| **Regression** | Continuous Values | Price prediction, Demand forecast |

#### Step 3: One hot encoding: Convert Categorical to binary

http://localhost:8888/notebooks/PycharmProjects/DS-ML-Projects/Module-Capstone-Project/Project_1_Predict_The_House_Prices/notebook/PROJECT-1-PREDICT-HOUSE-PRICES.ipynb