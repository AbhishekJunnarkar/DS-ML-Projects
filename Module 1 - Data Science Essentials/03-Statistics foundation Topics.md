# Statistics foundation 1: The Basics

# Mean, Median, and Mode  

## 1. Mean (Average)  
The **mean** is the **sum of all values divided by the total number of values**. 
It represents the central value of a data set.  

**Formula:**  

Mean = (Σx) / n

where:  
- **Σx** = Sum of all values  
- **n** = Total number of values  

**Example:**  
For the numbers **5, 10, 15, 20**:  

Mean = (5+10+15+20) / 4 = 50 / 4 = 12.5


---

## 2. Median (Middle Value)  
The **median** is the **middle number** in an **ordered data set** (from smallest to largest).  

- If the number of values is **odd**, the median is the middle number.  
- If the number of values is **even**, the median is the **average of the two middle numbers**.  

**Example 1 (Odd Set):**  
Numbers: **3, 7, 9** → **Median = 7** (middle value)  

**Example 2 (Even Set):**  
Numbers: **2, 4, 6, 8** → **Median = (4+6) ÷ 2 = 5**  

---

## 3. Mode (Most Frequent Value)  
The **mode** is the **number that appears most frequently** in a dataset.  

- A dataset **can have more than one mode**.  
- If no number repeats, the dataset has **no mode**.  

**Example 1 (Single Mode):**  
Numbers: **2, 3, 3, 5, 7** → **Mode = 3** (appears twice)  

**Example 2 (Multiple Modes - Bimodal):**  
Numbers: **1, 2, 2, 3, 3, 4** → **Modes = 2 and 3** (both appear twice)  

---

## Quick Summary  

| Measure  | Definition | When to Use |
|----------|------------|-------------|
| **Mean** | Sum of values ÷ number of values | Best for balanced data without extreme values (outliers) |
| **Median** | Middle value when ordered | Best for skewed data or when outliers are present |
| **Mode** | Most frequently occurring value | Best for categorical data or when identifying common values |

----

### Variability

Variability refers to how spread out or dispersed the data points in a data set are. 
While measures like mean, median, and mode provide central values, they don't show how much the data points
differ from each other. Variability helps us understand this spread. 


###### Introduction to variability
Two common measures of variability discussed in the course are:

- Range: The difference between the highest and lowest values in a data set.
- Standard Deviation: A measure of how much the data points deviate from the mean.


###### Range

The range is a measure of variability that looks at the spread of your data set by considering the edges. 
To calculate the range, you subtract the smallest data point from the largest data point in the data set. 
For example, if your data set has a minimum value of 10 and a maximum value of 80, 
the range would be 70 (80 - 10). The range helps to understand the overall spread of the data, 
but it can be influenced by outliers, as it only considers the two extreme values.

###### Standard Deviation

Standard deviation is a measure of how spread out the numbers in a data set are around the mean (average). 
It quantifies the amount of variation or dispersion in the data. A low standard deviation means that the 
data points are close to the mean, while a high standard deviation indicates that the data points are 
spread out over a wider range.

**Z Score**

A Z score, is a measure that describes how far a 
specific data point is from the mean of the data set, in terms of standard deviations. 
The formula to calculate a Z score is:

[ Z = \frac{(X - \mu)}{\sigma} ]

where ( X ) is the data point, ( \mu ) is the mean of the data set, and ( \sigma ) is the standard deviation.



###### Empirical Rule 

![Imperical Rule](Imperical-rule.png)

The empirical rule, is a useful guideline for understanding the distribution of data points in 
a normally distributed data set. Here are the key points:

- Normal Distribution: The data points are symmetrically distributed around the mean, forming a bell-shaped curve.
- 68% Rule: Approximately 68% of the data points fall within one standard deviation of the mean.
- 95% Rule: About 95% of the data points fall within two standard deviations of the mean.
- 99.7% Rule: Nearly 99.7% of the data points fall within three standard deviations of the mean.

This rule helps you quickly understand how data points are spread out in a normal distribution and identify outliers.

###### Outliers

----

# Statistics foundation 2: Probability
- The importance of probabilities
- Probability Definition: Probability is the percent chance of something happening, calculated by dividing the desired outcome by the total number of possible outcomes.

### Types of Probability:
- Classical Probability: Based on known possible outcomes (e.g., flipping a coin).
- Empirical Probability: Based on experimental or historical data (e.g., a basketball player's free throw success rate).
- Subjective Probability: Based on personal belief or experience (e.g., a CEO's confidence in launching a new product).

Application: Understanding probability helps in making informed decisions and predictions in various scenarios.

### Types of Probability:

- Even Odds: Scenarios like coin flips, rolling dice, picking cards, and raffles where each outcome has an equal chance.
- Weighted Odds: Real-world scenarios like weather, science, medicine, sports, and business where outcomes have different probabilities.
- Sum of Probabilities: The total probability of all possible outcomes always equals 100%.
- Sample Space: The set of all possible outcomes in a probability scenario.


### Permutations

- Definition: Permutations consider the order of items. For example, AB and BA are different permutations.
- Formula: The number of permutations of n objects is calculated using n! (n factorial). 

For example, 5! = 5 x 4 x 3 x 2 x 1 = 120.

- Partial Permutations: When selecting x items from n objects, use the formula n! / (n - x)!. 

For example, selecting the top 3 finishers out of 8 runners is calculated as 8! / 5! = 336.

### Combinations

- Definition: Combinations are used when the order of selection does not matter. For example, selecting a team from a group of students.
- Formula: The formula for combinations is N! / (N - X)! X!, where N is the total number of objects and X is the number of objects chosen at one time.

### Percentiles

- Definition: Percentiles indicate the relative standing of a value within a dataset. For example, being in the 98th percentile means you scored better than 98% of the participants.
- Calculation: Percentile rank can be calculated using the formula: (Number of values below the score / Total number of values) * 100.
- Application: Percentiles are commonly used in standardized testing and income distribution to understand how a particular value compares to the rest of the dataset.

### Multiple events Probabilities

### Discrete and continuous probabilities

----

# Statistics Foundations 3: Using Datasets

### Sampling

### Sample size

### Standard Error

#### standard error for proportions
- Standard Error Definition: The standard error is the standard deviation of the proportion distribution in a sample.
- Impact of Sample Size: The size of the standard error depends on the sample size; larger sample sizes result in smaller standard errors.
- Application in Real-World Scenarios: Standard errors help understand large populations through simple random samples, and deviations beyond the standard error can indicate unique local conditions or potential flaws in data collection.

#### Sampling distribution of the mean

- Central Limit Theorem: This theorem helps in approximating the population mean by using simple random samples.
- Sample Size and Accuracy: Increasing the number of samples and the sample size leads to a more accurate approximation of the true population mean.
- Application Example: By taking multiple random samples of basketball players' weights, the mean of the sample means closely approximates the true population mean.

#### Standard error for means

- Standard Error Calculation: The standard error for sample means is calculated using the standard deviation of the sample means and the sample size.
- Impact of Sample Size: Larger sample sizes result in smaller standard errors, making the sample mean a more accurate estimate of the population mean.
- Practical Example: The video uses the example of coffee order times to illustrate how to calculate the standard error and understand its implications for estimating the population mean.

#### Questions
80% of customers pay with a debit or credit card. 
25 customers are chosen at random each day. 
68% of the samples would have p-hats between:

(0.80*0.20)/25 = 0.0064. Next take the square root of 0.0064, this provides a standard dev. of 0.08 or 8%. The p-hats would be between both 80%-8% and 80%+8%.

### Confidence Intervals

#### Introduction to confidence Intervals

- Confidence Intervals: A confidence interval provides a range within which we can be certain the population mean lies, based on a single random sample.
- 95% Confidence Level: If we create multiple confidence intervals from different samples, 95% of these intervals will contain the true population mean.
- Efficiency: Confidence intervals allow statisticians to make reliable estimates about a population mean using just one random sample, making it a powerful and resource-efficient method.

#### Components of confidence Intervals


- Formula Components: The confidence interval is calculated using the sample proportion (p-hat), the Z-score, and the standard error.
- Upper and Lower Limits: The upper limit is p-hat plus the Z-score times the standard error, while the lower limit is p-hat minus the Z-score times the standard error.
- Example Calculation: Using a sample proportion of 0.55, a Z-score of 2.0, and a standard error of 0.05, the confidence interval for the population proportion is between 0.45 and 0.65.

#### Creating a 95% confidence interval for a population

To calculate a 95% confidence interval for a population, follow these steps:

Identify the Sample Proportion (p-hat): This is the proportion of your sample that has the characteristic of interest. For example, if 55 out of 100 voters support candidate A, p-hat is 0.55.

Find the Z-Score for 95% Confidence: The Z-score for a 95% confidence interval is 1.96. This means that 95% of the data falls within 1.96 standard deviations from the mean.

Calculate the Standard Error (SE): Use the formula for standard error:
[
SE = \sqrt{\frac{p-hat \times (1 - p-hat)}{n}}
]
where ( n ) is the sample size. For example, with p-hat = 0.55 and n = 100, SE = 0.05.

Determine the Confidence Interval:

Upper Limit: ( p-hat + (Z \times SE) )
Lower Limit: ( p-hat - (Z \times SE) )

Using the example values:

Upper Limit: ( 0.55 + (1.96 \times 0.05) = 0.648 )
Lower Limit: ( 0.55 - (1.96 \times 0.05) = 0.452 )


So, the 95% confidence interval is between 0.452 and 0.648. This means you can be 95% confident that the true population proportion lies within this range.

#### Alternative confidence intervals

#### Confidence intervals with unexpected outcomes


# Statistics Foundations 4: Advanced Statistics