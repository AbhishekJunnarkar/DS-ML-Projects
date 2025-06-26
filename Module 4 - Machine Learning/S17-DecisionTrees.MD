# Decision Trees and Classification

- An introduction to decision trees and their applications in supervised machine learning.
- How to build and visualize classification and regression trees in Python.
- Techniques for measuring impurity and pruning decision trees to improve model performance.
- Practical tips for when and why to use decision trees in various scenarios.
- An overview of next steps for applying decision trees in different domains.

## Recall 
Recall is especially important in evaluating classification models like decision trees.

Recall (also called Sensitivity or True Positive Rate) is:

The proportion of actual positive cases that were correctly identified by the model.

In Simple Terms:
- Imagine you're trying to detect fraudulent transactions.

Recall tells you:
- Of all the actual frauds, how many did your model catch?

### Confusion Matrix Example:

|                     | Predicted Positive       | Predicted Negative       |
| ------------------- | ------------------------ | ------------------------ |
| **Actual Positive** | True Positive (TP) = 80  | False Negative (FN) = 20 |
| **Actual Negative** | False Positive (FP) = 10 | True Negative (TN) = 90  |


### In Decision Trees
Recall can be tuned by adjusting tree depth, split criteria (e.g. Gini vs Entropy), or pruning.

High recall means fewer false negatives, but sometimes at the cost of precision (i.e., more false positives).

### When to Prioritize Recall?
Use recall when:

Missing a positive is costly or dangerous
E.g., medical diagnosis, fraud detection, cancer screening, intrusion detection.