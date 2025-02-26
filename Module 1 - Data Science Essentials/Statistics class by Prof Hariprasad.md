
## Skewness

## Gaussian Distribution

## Gaussian bell curve

## Tail of the curve

## Kurtosis

Kurtosis is a statistical measure that describes the shape of the tails of a data distribution compared to a normal distribution. It indicates whether the data has more or fewer extreme values (outliers) than expected.

Types of Kurtosis:
1️⃣ Leptokurtic (High Kurtosis, > 3)

Sharp peak, heavy tails → More extreme values (outliers).
Example: Latency spikes in a cloud network, where most responses are fast, but a few requests take way longer than expected.
2️⃣ Mesokurtic (Normal Kurtosis, ≈ 3)

Similar to a normal distribution.
Example: Average CPU usage over time, where it mostly fluctuates within a predictable range.

3️⃣ Platykurtic (Low Kurtosis, < 3)

Flat peak, light tails → Data is more evenly spread out with fewer extreme values.
Example: User login times in a global system, where logins are evenly distributed across different time zones without sudden spikes.
💡 In IT & Data Science Context:

High kurtosis may indicate performance anomalies, security risks (DDoS attacks), or critical failures.
Low kurtosis suggests evenly distributed, stable data without unexpected deviations.

### Excess Kurtosis

 - Excess Kurtosis of Gaussian distribution is zero


## Outlier Detection

## Bernoulli
   - Success/failure

## Binomial
   - Multiple success failure

## Poisson
   - Rare events 
     - Earthquake
     - DDoS Attack
     - Market Risk

## Three methods to find the outliers

### 3S

### Box Plot

### Median Absolute Deviation approach (MAD)


## Discrete distribution

## Imbalanced data

## SMOTE
 - SMOTE (Synthetic Minority Over-sampling Technique) is a data augmentation technique used to handle imbalanced datasets 
in machine learning. Instead of simply duplicating minority class samples, SMOTE generates synthetic (new) data points to balance the dataset.
How SMOTE Works?
Identify the Minority Class 

It first detects which class has fewer samples in a dataset.
Find Nearest Neighbors 🤝

For each minority class sample, SMOTE finds its K-nearest neighbors.
Generate Synthetic Samples 

New synthetic data points are created between existing samples using interpolation.
Example Use Case (Fraud Detection)
Suppose you have 1,000 legitimate transactions and only 50 fraudulent transactions in your dataset.
A machine learning model might favor the majority class (legitimate transactions) and ignore fraud cases.
Applying SMOTE: It generates synthetic fraud samples, balancing the dataset to 1,000 vs. 1,000, improving model learning.

When to Use SMOTE?
✔ When you have an imbalanced dataset (one class is much smaller).
✔ When you don’t want to duplicate existing data but need more variety.
✔ When you want to prevent overfitting caused by simple oversampling.

Limitations of SMOTE
❌ Can introduce noise if the minority class has outliers.
❌ Doesn’t work well if the dataset is highly skewed.
❌ Doesn’t improve results if the dataset is already balanced.

### Borderline SMOTE

### ADASYN
 - Algorithmic improvement
 - Adaptive Synthetic Generation


## Hypothesis Testing

- Gaussian distribution
  - Standard Normal distribution
  - Chi Squared Distribution
  - T distribution
  - F Distribution




