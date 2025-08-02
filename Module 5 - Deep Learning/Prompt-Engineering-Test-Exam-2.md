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

I am preparing for the data science and machine learning exam on topic 

"""
Introduction to Neural Network
"""

I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

### Input Layer, Output Layer and Hidden Layer

I am preparing for the data science and machine learning exam on topic 

"""

Input Layer, Output Layer and Hidden Layer

"""

I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

### Epoch, Batch and Mini Batch

I am preparing for the data science and machine learning exam on topic 

"""

Epoch, Batch and Mini Batch

"""

I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

### Nodes, Neurons and Weights in each Layer

I am preparing for the data science and machine learning exam on topic 

"""

Nodes, Neurons and Weights in each Layer

"""

I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

# 🧠 Revision: Nodes, Neurons & Weights in Neural Network Layers

---

## ✅ Topic Summary

- **Neuron (or Node):** Basic unit of a neural network layer that performs:
  \[
  z = \sum (w_i \cdot x_i) + b
  \]
  followed by an **activation function**.
- **Weights (w):** Determine how much influence a given input has.
- **Bias (b):** Helps shift the activation function output.
- **Layers:**
  - **Input Layer:** Receives features
  - **Hidden Layers:** Perform transformations
  - **Output Layer:** Provides predictions

---

## 📘 MCQs with Answers & Explanations

---

### **1. In a fully connected layer with 5 input features and 4 neurons, how many weights are there (excluding bias)?**

**Options:**
- A) 9  
- B) 20  
- C) 5  
- D) 4  

✅ **Answer:** B  
📘 **Explanation:** 5 inputs × 4 neurons = **20 weights**.

---

### **2. What does a neuron in a neural network compute?**

**Options:**
- A) A sum of inputs only  
- B) A linear function only  
- C) A weighted sum + activation  
- D) Only bias addition  

✅ **Answer:** C  
📘 **Explanation:** A neuron computes a **weighted sum of inputs plus bias**, passed through an activation function.

---

### **3. What is the total number of parameters (weights + biases) in a layer with 10 inputs and 3 neurons?**

**Options:**
- A) 30  
- B) 33  
- C) 13  
- D) 10  

✅ **Answer:** B  
📘 **Explanation:** Each neuron has 10 weights + 1 bias → (10 + 1) × 3 = **33 parameters**.

---

### **4. What is the role of a weight in a neural network?**

**Options:**
- A) It activates the neuron  
- B) It stores the input values  
- C) It determines the importance of each input  
- D) It reduces model complexity  

✅ **Answer:** C  
📘 **Explanation:** Weights represent how **important** an input is to a neuron.

---

### **5. If a layer has 8 neurons and each receives input from 6 neurons in the previous layer, how many total weights are there?**

**Options:**
- A) 14  
- B) 48  
- C) 64  
- D) 86  

❌ **Correct Answer:** **C**  
📘 **Explanation:** 6 inputs × 8 neurons = **48 weights**. Biases not included here.

---

### **6. Which part of a neuron helps shift the activation function left or right?**

**Options:**
- A) Input  
- B) Output  
- C) Weight  
- D) Bias  

✅ **Answer:** D  
📘 **Explanation:** The **bias** term helps shift the activation function.

---

### **7. In a neural network, how are weights learned?**

**Options:**
- A) They are set manually  
- B) Using cross-validation  
- C) By minimizing loss using optimization (e.g., gradient descent)  
- D) They are fixed after initialization  

✅ **Answer:** C  
📘 **Explanation:** Weights are **updated during training** by minimizing the loss function using optimizers.

---

### **8. If the number of neurons in a layer increases, what happens to the number of weights?**

**Options:**
- A) Decreases  
- B) Remains the same  
- C) Increases  
- D) Only bias increases  

✅ **Answer:** C  
📘 **Explanation:** More neurons mean more **connections and weights**.

---

### **9. What does it mean if a weight is close to zero after training?**

**Options:**
- A) It adds more non-linearity  
- B) The input it connects is not important  
- C) Model accuracy improves  
- D) It overfits  

❌ **Correct Answer:** **B**  
📘 **Explanation:** A weight close to zero means the **connected input feature isn’t contributing much**.

---

### **10. A neural network layer with 7 inputs and 2 neurons will have how many biases?**

**Options:**
- A) 7  
- B) 2  
- C) 1  
- D) 9  

✅ **Answer:** B  
📘 **Explanation:** One bias per neuron → **2 biases** total.

---

## ✅ Final Score: **8 / 10**

Focus areas for improvement:
- **Weight interpretation after training (Q9)**
- **Weight calculation in multi-neuron layers (Q5)**

---



---
## Classical Activation function and when to use them

I am preparing for the data science and machine learning exam on topic 

"""

Classical activation function and when to use them

"""

I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

# 🔢 Classical Activation Functions – MCQ Revision (IIT Delhi Prep)

---

## 📘 Topic Summary

**Classical Activation Functions Covered:**
- **Linear (Identity):** Used in regression, no non-linearity  
- **Sigmoid:** Output range (0, 1), used in binary classification output  
- **Tanh:** Output range (–1, 1), zero-centered  
- **ReLU:** Popular in hidden layers, outputs max(0, x)  
- **Leaky ReLU:** Variant of ReLU that avoids dying neurons by allowing small gradient for negative inputs

---

## ✅ MCQs with Correct Answers & Explanations

---

### **1. Which of the following activation functions is linear?**

**Options:**
- A) Sigmoid  
- B) Tanh  
- C) ReLU  
- D) Identity  

✅ **Answer:** D  
📘 **Explanation:** The identity function \( f(x) = x \) is linear, used in regression tasks.

---

### **2. Which activation function is commonly used in the output layer of a binary classification task?**

**Options:**
- A) ReLU  
- B) Tanh  
- C) Sigmoid  
- D) Linear  

✅ **Answer:** C  
📘 **Explanation:** Sigmoid outputs values between 0 and 1 — ideal for binary classification.

---

### **3. What is a key reason to avoid sigmoid in deep hidden layers?**

**Options:**
- A) High computation cost  
- B) Exploding gradient  
- C) Vanishing gradient  
- D) Too many parameters  

✅ **Answer:** C  
📘 **Explanation:** Sigmoid saturates for large inputs, causing gradients to vanish and slow learning.

---

### **4. What is the output range of the tanh function?**

**Options:**
- A) [0, 1]  
- B) (–1, 1)  
- C) (0, ∞)  
- D) [–∞, ∞]  

✅ **Answer:** B  
📘 **Explanation:** tanh outputs are in the range (–1, 1), centered around 0.

---

### **5. Why is ReLU preferred over sigmoid/tanh in hidden layers of deep networks?**

**Options:**
- A) It’s smoother  
- B) It avoids saturation for positive inputs  
- C) It outputs probabilities  
- D) It centers data around zero  

✅ **Answer:** B  
📘 **Explanation:** ReLU avoids saturation for positive values, helping gradient flow better during training.

---

### **6. What is the output of a ReLU activation for input x = –5?**

**Options:**
- A) –5  
- B) 0  
- C) 1  
- D) Undefined  

✅ **Answer:** B  
📘 **Explanation:** ReLU returns 0 for all negative inputs.

---

### **7. Leaky ReLU differs from ReLU because it:**

**Options:**
- A) Adds noise  
- B) Is used in output layers  
- C) Passes small negative gradients  
- D) Requires more memory  

✅ **Answer:** C  
📘 **Explanation:** Leaky ReLU allows a small slope for negative inputs to avoid neuron death.

---

### **8. Which function is most likely to produce sparse activations (i.e., many zeros)?**

**Options:**
- A) Sigmoid  
- B) Tanh  
- C) ReLU  
- D) Linear  

✅ **Answer:** C  
📘 **Explanation:** ReLU outputs 0 for all negative inputs, making many activations zero — hence "sparse."

---

### **9. Which activation function is best for output layer in regression?**

**Options:**
- A) ReLU  
- B) Sigmoid  
- C) Tanh  
- D) Linear  

✅ **Answer:** D  
📘 **Explanation:** In regression, outputs can be unbounded, so the linear function is ideal.

---

### **10. What is a common issue with ReLU in practice?**

**Options:**
- A) Always outputs zero  
- B) Outputs negative values  
- C) Can cause neurons to stop learning  
- D) Is not differentiable at zero  

✅ **Answer:** C  
📘 **Explanation:** If weights drive ReLU into the negative region, the neuron can "die" and stop updating.

---

## 🧠 Quick Comparison Table

| Function    | Range      | Zero-Centered | Use Case                  | Drawback                   |
|-------------|------------|----------------|----------------------------|-----------------------------|
| Linear      | (–∞, ∞)    | ✅ Yes          | Regression output          | No non-linearity            |
| Sigmoid     | (0, 1)     | ❌ No           | Binary classification      | Vanishing gradients         |
| tanh        | (–1, 1)    | ✅ Yes          | Hidden layers              | Vanishing gradients         |
| ReLU        | [0, ∞)     | ❌ No           | Deep networks (hidden)     | Dying neurons (negatives)   |
| Leaky ReLU  | (–∞, ∞)    | ❌ No           | Deep networks (hidden)     | Slight complexity increase  |

---

---
### Sigmoid Activation Function

I am preparing for the data science and machine learning exam on topic 

"""

sigmoid activation function

"""

I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

# 🧠 Sigmoid Activation Function – MCQ Revision Sheet

---

## ✅ Quick Summary

- **Formula:**  
  \[
  \sigma(x) = \frac{1}{1 + e^{-x}}
  \]

- **Range:** (0, 1)  
- **Shape:** S-shaped (sigmoidal curve)  
- **Key Use:** Output layer in binary classification  
- **Limitation:** Vanishing gradient for large/small values of x

---

## 📘 MCQs with Answers and Explanations

---

### **1. What is the output range of the sigmoid function?**

**Options:**
- A) [–1, 1]  
- B) [0, 1]  
- C) [0, ∞)  
- D) (–∞, ∞)  

✅ **Answer:** B  
📘 **Explanation:** The sigmoid function squashes inputs to a range between 0 and 1.

---

### **2. What is the formula for the sigmoid activation function?**

**Options:**
- A) \( \frac{e^x - e^{-x}}{e^x + e^{-x}} \)  
- B) \( \max(0, x) \)  
- C) \( \frac{1}{1 + e^{-x}} \)  
- D) \( x \)  

✅ **Answer:** C  
📘 **Explanation:** This formula defines the sigmoid function.

---

### **3. For large positive values of x, the sigmoid output will be close to:**

**Options:**
- A) 0  
- B) 0.5  
- C) 1  
- D) ∞  

✅ **Answer:** C  
📘 **Explanation:** As x becomes very large, the output of sigmoid approaches 1.

---

### **4. In which layer is the sigmoid activation function most appropriately used?**

**Options:**
- A) Input layer  
- B) Hidden layer in CNN  
- C) Output layer of binary classification  
- D) Output layer of regression  

✅ **Answer:** C  
📘 **Explanation:** Sigmoid is used in binary classification output layers since it returns a probability-like output between 0 and 1.

---

### **5. What is a key disadvantage of using sigmoid in deep networks?**

**Options:**
- A) Exploding gradient  
- B) Vanishing gradient  
- C) Gradient clipping  
- D) It doesn’t normalize output  

✅ **Answer:** B  
📘 **Explanation:** The gradient becomes very small for large positive or negative inputs, leading to slow learning.

---

### **6. What is the derivative of the sigmoid function?**

**Options:**
- A) \( \sigma(x)^2 \)  
- B) \( \sigma(x)(1 - \sigma(x)) \)  
- C) \( 1 - \sigma(x)^2 \)  
- D) \( e^{-x} \)  

✅ **Answer:** B  
📘 **Explanation:**  
\[
\frac{d}{dx} \sigma(x) = \sigma(x)(1 - \sigma(x))
\]

---

### **7. Sigmoid is NOT ideal in hidden layers because:**

**Options:**
- A) It’s slow to compute  
- B) It has no activation  
- C) Its gradients become very small for extreme inputs  
- D) It’s not differentiable  

✅ **Answer:** C  
📘 **Explanation:** For very large or small inputs, sigmoid outputs saturate and gradients vanish, slowing down training.

---

### **8. What is the value of sigmoid(0)?**

**Options:**
- A) 0  
- B) 0.25  
- C) 0.5  
- D) 1  

✅ **Answer:** C  
📘 **Explanation:**  
\[
\sigma(0) = \frac{1}{1 + e^{0}} = 0.5
\]

---

### **9. Which of the following is a property of the sigmoid function?**

**Options:**
- A) Zero-centered output  
- B) Output always positive  
- C) Output only in negative values  
- D) Output always 1  

✅ **Answer:** B  
📘 **Explanation:** Sigmoid outputs are always between 0 and 1 — so always positive.

---

### **10. Why is sigmoid suitable for binary classification outputs?**

**Options:**
- A) Because it outputs negative values  
- B) Because it gives a probability between 0 and 1  
- C) Because it forces sparsity  
- D) Because it’s linear  

✅ **Answer:** B  
📘 **Explanation:** Sigmoid output can be interpreted as the probability of a class — perfect for binary classification.

---

## 🧠 Summary Table – Activation Function Comparison

| Property           | Sigmoid          | tanh             | ReLU             |
|--------------------|------------------|------------------|------------------|
| Output Range       | (0, 1)           | (–1, 1)          | [0, ∞)           |
| Zero-centered?     | ❌ No             | ✅ Yes            | ❌ No             |
| Vanishing Gradient | ✅ Yes            | ✅ Yes            | ❌ Rarely         |
| Use Case           | Binary classification output | Hidden layers | Hidden layers |

---

---
### tanh Activation Function

I am preparing for the data science and machine learning exam on topic 

"""

tanh activation function

"""

I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

# 🧠 tanh Activation Function – MCQ Revision Sheet

---

## ✅ Quick Summary

- **Function:**  
  \[
  \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \]
- **Output Range:** (–1, 1)
- **Zero-centered:** Yes ✅  
- **Use Case:** Often used in **hidden layers** when symmetry around zero helps convergence.
- **Main Drawback:** Can **saturate** for large inputs → leads to **vanishing gradients**.

---

## 📘 MCQs with Answers and Explanations

---

### **1. What is the mathematical range of the tanh function?**

**Options:**
- A) [0, 1]  
- B) (–∞, ∞)  
- C) (–1, 1)  
- D) [0, ∞)

✅ **Answer:** C  
📘 **Explanation:** The `tanh` function outputs values between –1 and 1, unlike sigmoid which is [0, 1].

---

### **2. What is the value of tanh(0)?**

**Options:**
- A) –1  
- B) 0  
- C) 1  
- D) Undefined  

✅ **Answer:** B  
📘 **Explanation:** `tanh(0) = 0`, since the function is symmetric and zero-centered.

---

### **3. What kind of function is tanh?**

**Options:**
- A) Linear  
- B) Non-linear  
- C) Step  
- D) Identity  

✅ **Answer:** B  
📘 **Explanation:** It’s a non-linear function, which allows neural networks to model complex patterns.

---

### **4. Compared to sigmoid, what advantage does tanh offer?**

**Options:**
- A) Higher output range  
- B) Lower computation time  
- C) Zero-centered outputs  
- D) Better at classification  

✅ **Answer:** C  
📘 **Explanation:** `tanh` outputs are centered around 0, which helps **faster convergence** during training compared to sigmoid.

---

### **5. What is a major disadvantage of tanh activation?**

**Options:**
- A) Exploding gradients  
- B) Always positive outputs  
- C) Computational instability  
- D) Vanishing gradient for large inputs  

✅ **Answer:** D  
📘 **Explanation:** Like sigmoid, `tanh` **saturates** for very large positive or negative inputs, causing the gradient to vanish.

---

### **6. tanh is typically used in:**

**Options:**
- A) Regression output layers  
- B) Classification output layers  
- C) Hidden layers  
- D) Input layers  

❌ **Your Answer:** A  
✅ **Correct Answer:** C  
📘 **Explanation:** `tanh` is typically used in **hidden layers**, not output layers (where linear or softmax is used).

---

### **7. The derivative of tanh(x) is:**

**Options:**
- A) \( 1 - \tanh^2(x) \)  
- B) \( \tanh(x) \)  
- C) \( 1 + \tanh(x) \)  
- D) \( x \cdot \tanh(x) \)  

❌ **Your Answer:** Not sure  
✅ **Correct Answer:** A  
📘 **Explanation:**  
\[
\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)
\]

---

### **8. What does it mean for a function to be "zero-centered"?**

**Options:**
- A) Outputs range from 0 to 1  
- B) Output mean is close to 0  
- C) Output range is always negative  
- D) Always returns 0  

✅ **Answer:** B  
📘 **Explanation:** A zero-centered function like `tanh` has outputs symmetrically distributed around zero.

---

### **9. Which of the following statements is TRUE?**

**Options:**
- A) tanh causes sparse activation  
- B) tanh is better than ReLU in deep networks  
- C) tanh performs best for large values of input  
- D) tanh can saturate and slow training  

❌ **Your Answer:** A  
✅ **Correct Answer:** D  
📘 **Explanation:** tanh **can saturate** for large inputs → slows learning due to very small gradients.

---

### **10. Which function is more likely to produce negative outputs: sigmoid or tanh?**

**Options:**
- A) Sigmoid  
- B) tanh  
- C) Both equally  
- D) None  

✅ **Answer:** B  
📘 **Explanation:** Sigmoid ranges [0, 1] → **never negative**. `tanh` ranges (–1, 1) → **can output negatives**.

---

## 🧠 Key Takeaways

| Feature           | tanh        | Sigmoid     | ReLU        |
|------------------|-------------|-------------|-------------|
| Output Range     | (–1, 1)     | (0, 1)      | [0, ∞)      |
| Zero-centered?   | ✅ Yes       | ❌ No        | ❌ No        |
| Derivative Max   | 1           | 0.25        | 1 (x > 0)   |
| Saturation Risk  | High        | High        | Low         |
| Use In           | Hidden Layer| Output (binary)| Hidden Layer |

---


---
### ReLU Activation Function

I am preparing for the data science and machine learning exam on topic 

"""

ReLU activation function

"""

I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

# 🚀 ReLU Activation Function – MCQ Revision Sheet

---

## 🧠 Quick Summary

- **ReLU (Rectified Linear Unit)**:  
  \[
  f(x) = \max(0, x)
  \]
- Introduces **non-linearity** without saturation for positive inputs.
- Helps avoid **vanishing gradient problem** (common with sigmoid/tanh).
- **Used in hidden layers** of most deep neural networks.
- **Variants** include: Leaky ReLU, Parametric ReLU, ReLU6.

---

## ✅ MCQs with Answers & Explanations

---

### **1. What is the formula of the ReLU activation function?**

**Options:**
- A) \( f(x) = \tanh(x) \)  
- B) \( f(x) = \max(0, x) \)  
- C) \( f(x) = \frac{1}{1 + e^{-x}} \)  
- D) \( f(x) = \min(0, x) \)  

✅ **Answer:** B  
📘 **Explanation:** ReLU returns the input if it’s positive, otherwise returns 0.

---

### **2. What is the main advantage of ReLU over sigmoid or tanh in deep networks?**

**Options:**
- A) It's smoother  
- B) It's symmetric  
- C) It avoids vanishing gradient for positive values  
- D) It’s probabilistic  

✅ **Answer:** C  
📘 **Explanation:** ReLU maintains a strong gradient for positive inputs, improving learning speed and depth.

---

### **3. What is the output of ReLU when the input is –2.5?**

**Options:**
- A) –2.5  
- B) 0  
- C) 1  
- D) Undefined  

✅ **Answer:** B  
📘 **Explanation:** Negative values are mapped to 0 by ReLU.

---

### **4. Which of the following is a drawback of ReLU?**

**Options:**
- A) It squashes gradients  
- B) Exploding gradients  
- C) Dying ReLU problem  
- D) It outputs in [0, 1]  

✅ **Answer:** C  
📘 **Explanation:** Neurons can "die" by always outputting 0 if they enter a no-gradient state due to negative inputs.

---

### **5. Which layer is ReLU typically used in?**

**Options:**
- A) Output layer of regression  
- B) Output layer of classification  
- C) Hidden layers  
- D) Input layer  

✅ **Answer:** C  
📘 **Explanation:** ReLU is the most common activation in **hidden layers** due to its simplicity and effectiveness.

---

### **6. What happens to negative input values in ReLU?**

**Options:**
- A) Passed as-is  
- B) Squared  
- C) Become 0  
- D) Multiplied by a small constant  

✅ **Answer:** C  
📘 **Explanation:** For any \( x < 0 \), ReLU outputs 0.

---

### **7. Which activation function is used to solve the "dying ReLU" problem?**

**Options:**
- A) Leaky ReLU  
- B) Softmax  
- C) Tanh  
- D) Identity  

✅ **Answer:** A  
📘 **Explanation:** Leaky ReLU assigns a small slope (e.g., 0.01x) for negative values to prevent neurons from dying.

---

### **8. What is the derivative of ReLU for positive input values?**

**Options:**
- A) 1  
- B) 0  
- C) Undefined  
- D) Input value itself  

✅ **Answer:** A  
📘 **Explanation:** For \( x > 0 \), the derivative of ReLU is constant at 1.

---

### **9. ReLU is a:**

**Options:**
- A) Linear activation function  
- B) Non-linear activation function  
- C) Saturated activation function  
- D) Sigmoid variant  

✅ **Answer:** B  
📘 **Explanation:** ReLU is **non-linear**, despite being piecewise linear—it allows deep networks to learn complex mappings.

---

### **10. What is the range of output values for ReLU?**

**Options:**
- A) [0, 1]  
- B) [–1, 1]  
- C) [0, ∞)  
- D) (–∞, ∞)  

❌ **Your Answer:** A  
✅ **Correct Answer:** C  
📘 **Explanation:** ReLU can return **any non-negative value** from 0 to ∞. It does not limit the upper bound like sigmoid or tanh.

---

## 🔁 Key Takeaways

- ReLU is **computationally efficient**, avoids saturation, and works well with deep networks.
- It **outputs zero for negatives** and **retains positives**, making it sparse and fast.
- Avoid ReLU in output layers for regression—**use linear** instead.
- **Leaky ReLU** is a good fix for dying ReLU neurons.

---


---
### Linear Activation Function

I am preparing for the data science and machine learning exam on topic 

"""

Linear activation function

"""

I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

# 📘 Linear Activation Function – MCQ Revision

---

## ✅ Summary

- A **Linear Activation Function** is simply:  
  \[
  f(x) = x
  \]

- It passes input as-is without transformation.
- Commonly used in **output layers of regression models**.
- Rarely used in **hidden layers**, since it introduces no non-linearity.

---

## 🧠 MCQs with Answers & Explanations

---

### **1. What is the formula for a linear activation function?**

**Options:**
- A) \( \frac{1}{1 + e^{-x}} \)  
- B) \( \max(0, x) \)  
- C) \( x \)  
- D) \( \tanh(x) \)  

✅ **Correct Answer:** C  
📘 **Explanation:** Linear activation is the identity function: \( f(x) = x \).

---

### **2. In which of the following scenarios is a linear activation function typically used?**

**Options:**
- A) Binary classification  
- B) Multi-class classification  
- C) Regression output layer  
- D) Hidden layers in CNNs  

✅ **Correct Answer:** C  
📘 **Explanation:** In regression, continuous values are predicted — so no transformation is needed in the final layer.

---

### **3. What is the main limitation of a linear activation function in hidden layers?**

**Options:**
- A) Too steep gradient  
- B) Exploding output  
- C) No non-linearity introduced  
- D) Output is always negative  

✅ **Correct Answer:** C  
📘 **Explanation:** Without non-linearity, the model cannot learn complex functions — just a stacked linear combination.

---

### **4. What does the derivative of a linear activation function look like?**

**Options:**
- A) Varies with input  
- B) Equal to 0  
- C) Equal to 1  
- D) Equal to input squared  

✅ **Correct Answer:** C  
📘 **Explanation:** Derivative of \( f(x) = x \) is always **1**, which makes backprop simple but not very expressive.

---

### **5. Which of the following statements is TRUE about linear activation functions?**

**Options:**
- A) They are commonly used in hidden layers  
- B) They allow modeling of complex relationships  
- C) They provide non-linear transformation  
- D) They can be useful in the output layer of regression models  

✅ **Correct Answer:** D  
📘 **Explanation:** They are useful in regression models because the target output is continuous.

---

### **6. Using a linear activation function in all layers of a neural network leads to:**

**Options:**
- A) High model accuracy  
- B) Deep representations  
- C) A model that behaves like a linear model  
- D) Better generalization  

✅ **Correct Answer:** C  
📘 **Explanation:** The entire network becomes equivalent to a **single linear transformation**.

---

### **7. Linear activation functions are sometimes referred to as:**

**Options:**
- A) ReLU  
- B) Step function  
- C) Identity function  
- D) Log-sigmoid  

✅ **Correct Answer:** C  
📘 **Explanation:** A linear activation function is also known as the **identity function**.

---

### **8. What is the output range of a linear activation function?**

**Options:**
- A) [0, 1]  
- B) [–1, 1]  
- C) Only positive numbers  
- D) (–∞, ∞)  

✅ **Correct Answer:** D  
📘 **Explanation:** The output of \( f(x) = x \) is unbounded in both directions.

---

### **9. Why are non-linear activation functions preferred in deep networks over linear ones?**

**Options:**
- A) They reduce vanishing gradients  
- B) They introduce sparsity  
- C) They allow learning of complex patterns  
- D) They are easier to com

---
### When to use Softmax over sigmoid activation function?

I am preparing for the data science and machine learning exam on topic 

"""

Softmax, softmax vs sigmoid
"""

I want to score maximum marks. The exam is MCQ based, the exam is conducted by IIT Delhi.
In the last exam I scored 16/24. This time i want to score 24/24. 
Provide me MCQ based questions for practice and later solutions separately to cover the full scope
of questions on this topic.

# 🧠 Softmax vs Sigmoid – Final MCQ Revision (with Answers & Explanations)

---

### **1. What is the primary purpose of the softmax function in a neural network?**

**Options:**
- A) Normalize outputs between 0 and 1 for binary classification  
- B) Convert logits into class probabilities for multi-class classification  
- C) Reduce overfitting in large networks  
- D) Minimize cross-entropy loss  

✅ **Correct Answer:** B  
📘 **Explanation:** Softmax converts raw outputs (logits) into a probability distribution across multiple classes. It’s used in multi-class classification.

---

### **2. The softmax function outputs values that:**

**Options:**
- A) Are always greater than 1  
- B) Can be negative  
- C) Are between 0 and 1 and sum up to 1  
- D) Are always equal  

✅ **Correct Answer:** C  
📘 **Explanation:** Softmax outputs probabilities between 0 and 1 that add up to 1 across all classes.

---

### **3. When is the sigmoid function preferred over softmax?**

**Options:**
- A) When you have more than 3 output classes  
- B) When your model is regression based  
- C) In binary classification pr

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