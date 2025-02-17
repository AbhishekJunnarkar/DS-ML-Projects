# Statistics foundation 1: The Basics


### Data and Charts

### The Middle

----

### Variability

###### Introduction to variability

###### Range

###### Standard Deviation

###### Empirical Rule 

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