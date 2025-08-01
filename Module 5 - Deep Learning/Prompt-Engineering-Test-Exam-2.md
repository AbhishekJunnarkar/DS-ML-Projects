# S20: Introduction to Neural Network

Prompt:

I am preparing for the data science and machine learning exam on topic 
"""
Introduction to Neural Network
"""
I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

# 📘 Introduction to Neural Networks – MCQ Set 1 (With Answers & Explanations)

### ✅ 1. What is the primary purpose of a neural network?
**Answer:** C) Detect complex patterns and relationships in data  
**Explanation:** Neural networks excel at learning **non-linear** and **high-dimensional** patterns, which is why they work well in vision, language, and prediction tasks.

---

### ✅ 2. The output of a perceptron is:
**Answer:** B) A binary value based on a threshold  
**Explanation:** The perceptron applies a **step function** and returns either 0 or 1 based on whether the weighted sum of inputs exceeds a threshold.

---

### ✅ 3. Which activation function is commonly used in hidden layers and helps with vanishing gradient issues?
**Answer:** C) ReLU  
**Explanation:** ReLU (Rectified Linear Unit) is preferred in deep networks because it avoids **saturation** and helps **gradient flow** effectively.

---

### ✅ 4. What is the main role of the activation function?
**Answer:** C) To introduce non-linearity into the model  
**Explanation:** Without activation functions, all layers would collapse into a **linear transformation**, limiting the model's power.

---

### ✅ 5. In a neural network, weights are:
**Answer:** C) Parameters that are learned during training  
**Explanation:** Weights are updated via **backpropagation and gradient descent** and determine the influence of inputs on outputs.

---

### ✅ 6. What is the name of the algorithm commonly used to update weights in neural networks?
**Answer:** A) Backpropagation  
**Explanation:** Backpropagation calculates **gradients of the loss** with respect to each weight and updates them to **reduce loss**.

---

### ✅ 7. What is the purpose of backpropagation?
**Answer:** C) Minimize the loss by updating weights through gradients  
**Explanation:** It propagates error backward from output to input, allowing for **efficient weight updates**.

- Back propagation example in simple explanation with mathematical example
https://www.youtube.com/watch?v=QZ8ieXZVjuE

---

### ✅ 8. Which of the following is a common loss function for classification tasks?
**Answer:** B) Cross Entropy  
**Explanation:** Cross-entropy loss compares **predicted class probabilities** with true labels, ideal for classification.

---

### ✅ 9. Overfitting in a neural network can be reduced by:
**Answer:** C) Applying dropout  
**Explanation:** Dropout **randomly deactivates neurons** during training, preventing over-reliance and improving generalization.

---

### ✅ 10. What does the bias term in a neuron do?
**Answer:** C) It allows shifting the activation function  
**Explanation:** Bias helps the activation function to **shift left or right**, improving the model’s flexibility.

---

### ✅ 11. Which of the following is *not* a benefit of using a multilayer perceptron?
**Answer:** C) Always interpretable  
**Explanation:** MLPs are powerful but are often **black boxes**, lacking transparency in their decision-making.

---

### ✅ 12. Gradient descent is used to:
**Answer:** C) Find the optimal weights by minimizing the loss  
**Explanation:** Gradient descent helps reach the **minimum of the loss function**, guiding weight updates effectively.

---

# 📘 Introduction to Neural Networks – MCQ Set 2 (With Answers & Explanations)

---

### ✅ 1. What happens during the forward pass in a neural network?
**A)** Gradients are calculated  
**B)** Inputs are transformed layer by layer to generate output  
**C)** Weights are updated  
**D)** Loss is minimized  

**Correct Answer:** B  
💡 **Explanation:** In the forward pass, data moves through the layers and generates predictions. Gradients are only calculated during the backward pass.

---

### ❌ 2. In backpropagation, the gradients are calculated using:
**A)** Random initialization  
**B)** Chain rule of calculus  
**C)** Taylor series  
**D)** Second-order derivatives  

**Correct Answer:** B  
💡 **Explanation:** Backpropagation relies on the **chain rule** to compute gradients layer by layer.

---

### ✅ 3. Which of the following best describes the learning rate?
**A)** Number of neurons in each layer  
**B)** Size of the dataset  
**C)** Step size used to update weights  
**D)** Total number of training epochs  

**Correct Answer:** C  
💡 **Explanation:** The learning rate determines **how large a step** the optimizer takes during each weight update.

---

### ❌ 4. What problem can occur if the learning rate is too high?
**A)** Model learns very slowly  
**B)** Loss never decreases  
**C)** Model may overshoot minima and never converge  
**D)** Training becomes regularized  

**Correct Answer:** C  
💡 **Explanation:** A high learning rate can cause the optimizer to **bounce around the minimum** or diverge entirely.

---

### ✅ 5. For a binary classification task, the most appropriate output activation and loss function pair is:
**A)** ReLU + MSE  
**B)** Sigmoid + Binary Cross-Entropy  
**C)** Softmax + Categorical Cross-Entropy  
**D)** Tanh + Hinge Loss  

**Correct Answer:** B  
💡 **Explanation:** **Sigmoid** compresses outputs to [0, 1], and **Binary Cross-Entropy** compares them with actual binary labels.

---

### ✅ 6. Xavier initialization is used to:
**A)** Introduce regularization  
**B)** Set bias terms  
**C)** Initialize weights to maintain variance across layers  
**D)** Normalize inputs  

**Correct Answer:** C  
💡 **Explanation:** Xavier (Glorot) initialization helps **keep activations stable** across layers, aiding convergence.

---

### ✅ 7. Which of the following is true about a deep neural network with too many layers?
**A)** Always more accurate  
**B)** Trains faster  
**C)** May suffer from vanishing gradients  
**D)** Requires fewer parameters  

**Correct Answer:** C  
💡 **Explanation:** Deeper networks can experience **vanishing gradients**, especially with sigmoid or tanh activations.

---

### ✅ 8. Which technique randomly disables neurons during training?
**A)** Normalization  
**B)** Max Pooling  
**C)** Dropout  
**D)** Activation Clipping  

**Correct Answer:** C  
💡 **Explanation:** Dropout helps prevent overfitting by **randomly turning off neurons** during each training pass.

---

### ❌ 9. Which of the following does NOT affect the capacity of a neural network?
**A)** Number of layers  
**B)** Number of neurons per layer  
**C)** Activation function used  
**D)** Size of the input data  

**Correct Answer:** D  
💡 **Explanation:** Capacity depends on **architecture**, not the input data's dimensionality.

---

### ✅ 10. When is softmax activation usually applied?
**A)** Hidden layers  
**B)** Input layer  
**C)** Binary classification output  
**D)** Multi-class classification output  

**Correct Answer:** D  
💡 **Explanation:** **Softmax** is applied in the **final layer** of multi-class classification models to produce probabilities across all classes.

---

### ✅ 11. L2 regularization (Ridge) helps by:
**A)** Forcing weights to be binary  
**B)** Encouraging sparsity  
**C)** Penalizing large weights to reduce overfitting  
**D)** Reducing the number of neurons  

**Correct Answer:** C  
💡 **Explanation:** L2 adds a **penalty term** to the loss function to discourage large weights and improve generalization.

---

### ❌ 12. What is the output of a softmax function?
**A)** A binary value  
**B)** Probabilities that sum to 1  
**C)** One-hot encoded labels  
**D)** Loss gradients  

**Correct Answer:** B  
💡 **Explanation:** Softmax outputs a **vector of probabilities** that sum to 1, ideal for multi-class predictions.

---

---
## Perceptron

### Prompt
I am preparing for the data science and machine learning exam on topic 
"""
Perceptron
"""
I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me a short summary of the topic, followed by MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.
---

##  Perceptron – MCQ Set 1 with Answers & Explanations

---

### ✅ 1. What type of problem can a single-layer perceptron solve?
**A)** Non-linear classification  
**B)** Regression  
**C)** Linearly separable classification  
**D)** Multi-class classification  

**Correct Answer:** C  
💡 **Explanation:** A perceptron can solve problems where a straight line or hyperplane can separate the classes — i.e., linearly separable data.

---

### ✅ 2. The perceptron algorithm updates weights when:
**A)** The model predicts correctly  
**B)** The weights are negative  
**C)** The prediction is incorrect  
**D)** The bias is zero  

**Correct Answer:** C  
💡 **Explanation:** Weight updates only happen when the model makes a **wrong prediction**.

---

### ✅ 3. Which function is typically used as the activation function in a basic perceptron?
**A)** Sigmoid  
**B)** ReLU  
**C)** Tanh  
**D)** Step function  

**Correct Answer:** D  
💡 **Explanation:** A **step function** is used in the perceptron to produce binary outputs like 0/1 or -1/+1.

---

### ✅ 4. What happens if the data is not linearly separable in a perceptron?
**A)** The algorithm converges slowly  
**B)** The perceptron cannot converge  
**C)** It overfits  
**D)** It ignores the bias term  

**Correct Answer:** B  
💡 **Explanation:** The perceptron algorithm does **not converge** if the data isn't linearly separable.

---

### ✅ 5. The perceptron algorithm guarantees convergence only if:
**A)** The data is normalized  
**B)** The learning rate is small  
**C)** The data is linearly separable  
**D)** The bias is non-zero  

**Correct Answer:** C  
💡 **Explanation:** According to the **Perceptron Convergence Theorem**, convergence is guaranteed only for **linearly separable** data.

---

### ❌ 6. What is the role of the bias in a perceptron model?
**A)** To scale the output  
**B)** To allow shifting the decision boundary  
**C)** To normalize inputs  
**D)** To speed up convergence  

**Correct Answer:** B  
💡 **Explanation:** The bias term **shifts the decision boundary**, just like the intercept in linear equations.

---

### ✅ 7. The perceptron learning rule updates weights using which of the following?
**A)** Gradient of the loss function  
**B)** (True label - Predicted label)  
**C)** Backpropagation  
**D)** Mean squared error  

**Correct Answer:** B  
💡 **Explanation:** The update rule is:  
```w = w + η * (y_true - y_pred) * x```

---

### ✅ 8. Which of the following problems cannot be solved using a single-layer perceptron?
**A)** AND  
**B)** OR  
**C)** XOR  
**D)** NAND  

**Correct Answer:** C  
💡 **Explanation:** The XOR function is **not linearly separable**. A single-layer perceptron can't solve it.

---

### ❌ 9. What is the output range of the basic perceptron?
**A)** Continuous values  
**B)** [0, 1]  
**C)** {-1, +1} or {0, 1} depending on the step function  
**D)** [-1, 1]  

**Correct Answer:** C  
💡 **Explanation:** Output is **discrete**, not continuous. Depends on the step function used.

---

### ❌ 10. The perceptron algorithm terminates when:
**A)** Loss becomes zero  
**B)** All weights become positive  
**C)** It converges or reaches maximum iterations  
**D)** All input features are zero  

**Correct Answer:** C  
💡 **Explanation:** The algorithm stops when it **classifies all points correctly** or reaches the **maximum number of iterations**.

---

## 🤖 Perceptron – MCQ Set 2 (Advanced) with Answers & Explanations

---

### ✅ 1. Which of the following logical functions can be represented by a single-layer perceptron?
**A)** XOR  
**B)** NAND  
**C)** XNOR  
**D)** NOT AND (NAND) and OR simultaneously  

**Correct Answer:** B  
💡 **Explanation:** Perceptrons can solve problems like **AND, OR, NOT, NAND** — which are linearly separable. **XOR/XNOR** are **not linearly separable**.

---

### ✅ 2. What does the learning rate (η) control in the perceptron algorithm?
**A)** The complexity of the model  
**B)** How far weights are updated on each error  
**C)** Number of features used  
**D)** Bias initialization  

**Correct Answer:** B  
💡 **Explanation:** Learning rate determines the **step size** in each weight update. It doesn’t change model structure or bias.

---

### ✅ 3. Consider a perceptron with:  
x₁ = 1, x₂ = 2, w₁ = 0.5, w₂ = –1, b = 1  
**What is the output using a step function?**

**A)** 0  
**B)** 1  
**C)** –1  
**D)** Depends on learning rate  

**Correct Answer:** A  
💡 **Explanation:**  
Compute weighted sum:  
`z = (0.5*1) + (–1*2) + 1 = –0.5`  
Step function → Output = 0 (since z < 0)

---

### ✅ 4. Which of the following best describes the decision boundary of a perceptron with two inputs?
**A)** A curve  
**B)** A vertical line  
**C)** A linear line (hyperplane in higher dims)  
**D)** A parabola  

**Correct Answer:** C  
💡 **Explanation:** Perceptrons create **linear decision boundaries** in input space.

---

### ✅ 5. What is the time complexity of the perceptron training algorithm (in worst case)?
**A)** O(n)  
**B)** O(mn)  
**C)** O(1)  
**D)** O(m²)  

**Correct Answer:** B  
💡 **Explanation:**  
m = samples, n = features → Each sample can require looping over n features → O(mn)

---

### ❌ 6. Which factor does *not* influence the convergence of a perceptron?
**A)** Linear separability of data  
**B)** Choice of activation function  
**C)** Learning rate  
**D)** Bias initialization  

**Correct Answer:** B  
💡 **Explanation:** Basic perceptron uses a **fixed step function**. Activation function doesn't change — so it doesn't affect convergence.

---

### ✅ 7. What kind of data transformation can help a perceptron solve non-linearly separable problems like XOR?
**A)** Feature duplication  
**B)** Feature scaling  
**C)** Feature mapping to higher dimensions  
**D)** Decreasing learning rate  

**Correct Answer:** C  
💡 **Explanation:** By projecting data into a **higher-dimensional space**, we may make it linearly separable (like in kernel methods or deep networks).

---

### ✅ 8. A perceptron fails to converge even after many iterations. What is the most likely cause?
**A)** Too many features  
**B)** Low bias  
**C)** Non-linearly separable data  
**D)** High learning rate  

**Correct Answer:** C  
💡 **Explanation:** The perceptron convergence theorem only applies to **linearly separable** data.

---

### ✅ 9. Which of the following best describes how a perceptron learns?
**A)** Stores all examples and averages them  
**B)** Uses loss minimization to update weights  
**C)** Updates weights only on incorrect predictions  
**D)** Uses backpropagation to compute gradients  

**Correct Answer:** C  
💡 **Explanation:** The **perceptron rule** updates weights **only when a mistake is made**. No gradient, no loss function involved.

---

### ✅ 10. In practical terms, why is the perceptron rarely used for modern classification tasks?
**A)** It is too slow  
**B)** It lacks activation functions  
**C)** It only works for linearly separable data  
**D)** It requires labeled data  

**Correct Answer:** C  
💡 **Explanation:** Most real-world datasets are **not linearly separable**, making the basic perceptron ineffective.

---


---
## Multi layer Perceptron

## 🤖 Multi-Layer Perceptron (MLP) – MCQ Set 1 with Answers & Explanations

---

### ✅ 1. Which of the following problems can a multi-layer perceptron solve that a single-layer perceptron cannot?

**A)** Linearly separable problems  
**B)** Regression problems  
**C)** XOR problem  
**D)** Binary classification  

**Correct Answer:** C  
💡 **Explanation:** The **XOR problem** is not linearly separable, and a single-layer perceptron fails. MLPs with hidden layers can solve it using non-linear transformations.

---

### ✅ 2. What role does the activation function play in hidden layers of an MLP?

**A)** It increases model size  
**B)** It introduces linearity  
**C)** It reduces training time  
**D)** It introduces non-linearity  

**Correct Answer:** D  
💡 **Explanation:** Activation functions like ReLU, sigmoid, etc., make MLPs capable of learning **non-linear** mappings. Without them, the model is just linear.

---

### ✅ 3. Which activation function is most commonly used in the hidden layers of an MLP?

**A)** Softmax  
**B)** ReLU  
**C)** Sigmoid  
**D)** Step Function  

**Correct Answer:** B  
💡 **Explanation:** **ReLU (Rectified Linear Unit)** is computationally efficient and avoids the vanishing gradient problem, making it ideal for hidden layers.

---

### ✅ 4. Why is ReLU preferred over sigmoid in hidden layers of MLPs?

**A)** It requires fewer neurons  
**B)** It prevents overfitting  
**C)** It avoids vanishing gradients  
**D)** It computes probabilities  

**Correct Answer:** C  
💡 **Explanation:** ReLU maintains stronger gradients for large positive inputs, avoiding the vanishing gradient issue found in sigmoid/tanh.

---

### ❌ 5. Which loss function is most appropriate for multi-class classification using MLP?

**A)** Mean Absolute Error  
**B)** Mean Squared Error  
**C)** Binary Cross-Entropy  
**D)** Categorical Cross-Entropy  

**Correct Answer:** D  
💡 **Explanation:** For multi-class classification with Softmax outputs, use **Categorical Cross-Entropy**.  
❌ Binary Cross-Entropy is for **2-class problems** only.

---

### ✅ 6. What is the purpose of backpropagation in MLP training?

**A)** Generate new data  
**B)** Prevent overfitting  
**C)** Compute gradients for weight updates  
**D)** Normalize inputs  

**Correct Answer:** C  
💡 **Explanation:** Backpropagation applies the **chain rule** to calculate gradients for all weights so they can be updated via gradient descent.

---

### ✅ 7. In an MLP with 2 hidden layers, how many sets of weights will exist?

**A)** 1  
**B)** 2  
**C)** 3  
**D)** Depends on neurons, not layers  

**Correct Answer:** C  
💡 **Explanation:**  
- Input → Hidden Layer 1  
- Hidden Layer 1 → Hidden Layer 2  
- Hidden Layer 2 → Output  
So, **3 sets of weights** exist.

---

### ✅ 8. What type of learning does an MLP use?

**A)** Unsupervised  
**B)** Reinforcement  
**C)** Supervised  
**D)** Self-supervised  

**Correct Answer:** C  
💡 **Explanation:** MLPs require **labeled data** during training, hence it is a **supervised learning** method.

---

### ✅ 9. The universal approximation theorem states that:

**A)** MLP can replicate CNNs  
**B)** A single hidden layer MLP can approximate any continuous function  
**C)** Only deep networks can solve complex tasks  
**D)** Perceptrons outperform MLPs  

**Correct Answer:** B  
💡 **Explanation:** Even a **single hidden layer MLP** (with enough neurons) can approximate any continuous function on compact input spaces.

---

### ❌ 10. Which of the following is *not* a hyperparameter in MLPs?

**A)** Number of hidden layers  
**B)** Learning rate  
**C)** Weights  
**D)** Activation function  

**Correct Answer:** C  
💡 **Explanation:**  
- **Weights** are learned during training → not hyperparameters.  
- Hyperparameters are things you **set before training** (layers, learning rate, activation).

---

## Multi layer Perceptron - Structure 

### Input Layer, Output Layer and Hidden Layer

### Epoch, Batch and Mini Batch

### Nodes, Neurons and Weights in each Layer



---
## Classical Activation function and when to use them

---
### Sigmoid Activation Function

---
### tanh Activation Function

---
### ReLU Activation Function

---
### Linear Activation Function

---
### When to use Softmax over sigmoid activation function?

---
## Loss Functions - Regression
  - Question on Loss functions (MAE, MSE, and RMSE)

## Loss function - Classification
- Cross Entropy

## Hyperparameter Tuning

## Shallow Learning Model

## Deep Learning Model

## Back Propagation


## Back Propagation - Training multilayer perceptrons

## Boston Dataset Example

### Random state

## Deep Learning Libraries

### Karas

### Karas with Tensorflow

### Pytorch

---
---

# S22: Deep Learning

---
---
#


---
---
#


---
---


### Loss Functions - Classification MCQ questions


### Hyperparameter Tuning


### Back propagation 


### Confusion matrix and Precision, accuracy and recall


  - Question on Back-propogation: training multilayer perceptrons
  - Dropout can be a exam question
  - Array of sigmoid - softmax will be a question in exam.

  - Multi Layer perceptron and its structure
  - Epoch, batch and mini batch

   - LSTM (Long short Term Memory Neural Network) 


# Topics that can have numerical questions asked:

1. GINI
2. Entropy