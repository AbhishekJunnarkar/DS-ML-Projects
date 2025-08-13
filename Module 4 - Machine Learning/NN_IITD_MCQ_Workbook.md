# S20: Introduction to Neural Networks – IIT Delhi MCQ Workbook

A GitHub‑ready study guide with **practice‑first MCQs** and **answers revealed on click**.  
Use this as a repo `README.md` or as a standalone `.md` file.

---

## Table of Contents

- [How to Use](#how-to-use)
- [Introduction to Neural Networks — MCQ Set 1](#introduction-to-neural-networks--mcq-set-1)
- [Introduction to Neural Networks — MCQ Set 2](#introduction-to-neural-networks--mcq-set-2)
- [Perceptron — Summary & MCQs](#perceptron--summary--mcqs)
- [Multi‑Layer Perceptron (MLP) — MCQs](#multi-layer-perceptron-mlp--mcqs)
- [Network Structure Topics](#network-structure-topics)
  - [Input / Hidden / Output Layers — MCQs](#input--hidden--output-layers--mcqs)
  - [Epoch, Batch, Mini‑Batch — MCQs](#epoch-batch-mini-batch--mcqs)
  - [Nodes, Neurons, Weights — Summary & MCQs](#nodes-neurons-weights--summary--mcqs)
- [Activation Functions](#activation-functions)
  - [Classical Activations — MCQs](#classical-activations--mcqs)
  - [Sigmoid — MCQs](#sigmoid--mcqs)
  - [tanh — MCQs](#tanh--mcqs)
  - [ReLU — MCQs](#relu--mcqs)
  - [Linear — MCQs](#linear--mcqs)
  - [Softmax vs Sigmoid — MCQs](#softmax-vs-sigmoid--mcqs)
- [Loss Functions](#loss-functions)
  - [Regression (MAE, MSE, RMSE) — MCQs](#regression-mae-mse-rmse--mcqs)
  - [Classification (Cross‑Entropy) — MCQs](#classification-cross-entropy--mcqs)
- [Training & Tuning](#training--tuning)
  - [Hyperparameter Tuning — MCQs](#hyperparameter-tuning--mcqs)
  - [Backpropagation — MCQs](#backpropagation--mcqs)
  - [Random State (Reproducibility)](#random-state-reproducibility)
- [Deep Learning Landscape](#deep-learning-landscape)
  - [Libraries Overview](#libraries-overview)
  - [CNN Quick MCQs](#cnn-quick-mcqs)
  - [RNN (placeholder)](#rnn-placeholder)
- [NLP & Transformers](#nlp--transformers)
  - [Tokenization / Stemming / Lemmatization — MCQs](#tokenization--stemming--lemmatization--mcqs)
  - [BERT Cheatsheet](#bert-cheatsheet)
  - [Self‑Supervised Learning (SSL)](#self-supervised-learning-ssl)
  - [RLHF — Cheatsheet](#rlhf--cheatsheet)
  - [LLM Prompting (Chain‑of‑Thought)](#llm-prompting-chain-of-thought)
  - [LangChain — MCQs](#langchain--mcqs)
- [Big Gen‑AI Players (Cheatsheet)](#big-gen-ai-players-cheatsheet)

> **Format note:** Each MCQ’s answer/explanation is hidden inside a `<details>` block to support **practice first, reveal later** on GitHub.

---

## How to Use

1. Attempt each MCQ **without** expanding the answer.  
2. Click **“Show answer & explanation”** to check yourself.  
3. Convert this file to PDF from GitHub if needed (Print → Save as PDF).

---

## Introduction to Neural Networks — MCQ Set 1

1) **What is the primary purpose of a neural network?**  
A) Store data  B) Linear regression only  C) Detect complex patterns and relationships  D) Sort inputs  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C. Neural nets learn <i>non‑linear, high‑dimensional</i> patterns across tasks like vision and language.</p>
</details>

2) **The output of a perceptron is:**  
A) Real number  B) Binary value based on threshold  C) Logits  D) Probability vector  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B. Classic perceptron uses a step function to return 0/1 (or −1/+1).</p>
</details>

3) **Activation function commonly used in hidden layers to reduce vanishing gradients:**  
A) Sigmoid  B) tanh  C) ReLU  D) Softmax  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C. ReLU avoids saturation for positive inputs, aiding gradient flow.</p>
</details>

4) **Main role of the activation function:**  
A) Normalize inputs  B) Stabilize training  C) Introduce non‑linearity  D) Reduce parameters  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C. Without activations, stacked layers collapse to a linear map.</p>
</details>

5) **In a neural network, weights are:**  
A) Fixed constants  B) Features  C) Trainable parameters  D) Hyperparameters  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C. Learned with backprop + gradient descent.</p>
</details>

6) **Algorithm commonly used to update weights:**  
A) Backpropagation  B) PCA  C) EM  D) K‑Means  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> A. Backprop computes gradients for weight updates.</p>
</details>

7) **Purpose of backpropagation:**  
A) Data augmentation  B) Feature selection  C) Minimize loss via gradients  D) Increase model depth  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

8) **Common loss for classification:**  
A) MAE  B) Cross‑Entropy  C) MSE  D) Hinge  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

9) **To reduce overfitting:**  
A) Increase depth  B) Lower learning rate  C) Apply dropout  D) Remove bias  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

10) **Bias term lets a neuron:**  
A) Increase model size  B) Reduce variance  C) Shift activation  D) Normalize inputs  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

11) **Not a benefit of MLPs:**  
A) Non‑linear modeling  B) Universal approximation  C) Always interpretable  D) Feature learning  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

12) **Gradient descent is used to:**  
A) Increase loss  B) Select features  C) Minimize loss and find good weights  D) Reduce batch size  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

---

## Introduction to Neural Networks — MCQ Set 2

1) **Forward pass does what?**  
A) Compute gradients  B) Transform inputs to outputs layer‑by‑layer  C) Update weights  D) Minimize loss  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

2) **Backprop gradients rely on:**  
A) Random init  B) Chain rule  C) Taylor series  D) Second‑order only  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

3) **Learning rate is:**  
A) #neurons  B) Dataset size  C) Step size for weight updates  D) #epochs  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

4) **Too‑high learning rate can:**  
A) Slow learning  B) Keep loss flat  C) Overshoot minima / diverge  D) Regularize  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

5) **Binary classification (output+loss) best pair:**  
A) ReLU + MSE  B) Sigmoid + Binary Cross‑Entropy  C) Softmax + CCE  D) tanh + Hinge  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

6) **Xavier/Glorot initialization:**  
A) Regularization  B) Bias setup  C) Maintain variance across layers  D) Normalize inputs  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

7) **Very deep nets may suffer from:**  
A) Always more accurate  B) Faster training  C) Vanishing gradients  D) Fewer parameters  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

8) **Technique that randomly disables neurons:**  
A) Normalization  B) Max Pooling  C) Dropout  D) Clipping  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

9) **Does NOT affect capacity:**  
A) #layers  B) #neurons/layer  C) Activation choice  D) Input data size  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> D. Capacity is architectural; input dim affects representation, not innate capacity.</p>
</details>

10) **Softmax is applied in:**  
A) Hidden layers  B) Input layer  C) Binary output  D) Multi‑class output  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> D.</p>
</details>

11) **L2 (Ridge) helps by:**  
A) Binary weights  B) Sparsity (that’s L1)  C) Penalizing large weights  D) Reducing neurons  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

12) **Softmax output:**  
A) Binary value  B) Probabilities summing to 1  C) One‑hot labels  D) Gradients  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

---

## Perceptron — Summary & MCQs

**Short Summary:** Single‑layer perceptron solves **linearly separable** classification using a step activation. Converges only if data is linearly separable (Perceptron Convergence Theorem).

1) Solvable by single‑layer perceptron:  
A) Non‑linear cls  B) Regression  C) Linearly separable cls  D) Multi‑class (one‑vs‑rest aside)  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

2) Weights update when:  
A) Correct prediction  B) Weights negative  C) Prediction incorrect  D) Bias zero  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

3) Typical activation:  
A) Sigmoid  B) ReLU  C) tanh  D) Step  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> D.</p>
</details>

4) If data not linearly separable:  
A) Slow convergence  B) No convergence  C) Overfits  D) Ignores bias  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

5) Convergence guaranteed only if:  
A) Data normalized  B) Small LR  C) Linearly separable  D) Non‑zero bias  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

6) Role of bias:  
A) Scale output  B) Shift decision boundary  C) Normalize inputs  D) Speed training  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

7) Learning rule uses:  
A) Gradient of loss  B) (y_true − y_pred)  C) Backprop  D) MSE  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B. Update: <code>w ← w + η (y − ŷ) x</code></p>
</details>

8) Cannot be solved by single‑layer perceptron:  
A) AND  B) OR  C) XOR  D) NAND  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

9) Output range:  
A) Continuous  B) [0,1] only  C) {−1,+1} or {0,1}  D) [−1,1]  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

10) Algorithm stops when:  
A) Loss zero  B) All weights positive  C) Converged or max iters  D) All features zero  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

---

## Multi‑Layer Perceptron (MLP) — MCQs

1) MLP can solve (vs single‑layer):  
A) Linearly separable only  B) Regression  C) XOR  D) Binary cls  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

2) Hidden‑layer activations primarily:  
A) Increase size  B) Introduce linearity  C) Reduce time  D) Introduce non‑linearity  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> D.</p>
</details>

3) Common hidden activation:  
A) Softmax  B) ReLU  C) Sigmoid  D) Step  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

4) ReLU preferred because:  
A) Fewer neurons  B) Prevents overfitting  C) Avoids vanishing gradients  D) Computes probabilities  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

5) Multi‑class loss:  
A) MAE  B) MSE  C) Binary CE  D) Categorical CE  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> D.</p>
</details>

6) Purpose of backprop in MLP:  
A) Generate data  B) Prevent overfitting  C) Compute gradients  D) Normalize inputs  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

7) Sets of weights in 2 hidden‑layer MLP:  
A) 1  B) 2  C) 3  D) Depends only on neurons  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C (Input→H1, H1→H2, H2→Out).</p>
</details>

8) Learning type:  
A) Unsupervised  B) Reinforcement  C) Supervised  D) Self‑supervised  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

9) Universal Approximation Theorem:  
A) MLP replicates CNN  B) One hidden layer can approximate any continuous function  C) Only deep nets solve complex tasks  D) Perceptrons outperform MLP  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

10) Not a hyperparameter:  
A) #hidden layers  B) Learning rate  C) Weights  D) Activation  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C (weights are learned parameters).</p>
</details>

---

## Network Structure Topics

### Input / Hidden / Output Layers — MCQs
> *Placeholder for your own questions; keep answers in details blocks to follow the pattern above.*

### Epoch, Batch, Mini‑Batch — MCQs
> *Placeholder for your own questions.*

### Nodes, Neurons, Weights — Summary & MCQs

**Summary:**  
- Neuron computes \( z=\sum w_i x_i + b \) then activation.  
- Weights control influence; bias shifts activation.  
- Parameters per dense layer: \( (\text{in\_features}+1)\times \text{out\_neurons} \).

1) Fully connected layer: 5 inputs, 4 neurons → weights (no bias)?  
A) 9  B) 20  C) 5  D) 4  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B (5×4).</p>
</details>

2) A neuron computes:  
A) Sum only  B) Linear only  C) Weighted sum + activation  D) Bias only  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

3) Total params with 10 inputs, 3 neurons (incl. bias):  
A) 30  B) 33  C) 13  D) 10  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B ((10+1)×3).</p>
</details>

4) Role of a weight:  
A) Activates neuron  B) Stores inputs  C) Input importance  D) Reduce complexity  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

5) 8 neurons, each receives 6 inputs → weights:  
A) 14  B) 48  C) 64  D) 86  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B (6×8).</p>
</details>

6) Shifts activation left/right:  
A) Input  B) Output  C) Weight  D) Bias  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> D.</p>
</details>

7) Weights are learned by:  
A) Manual  B) CV  C) Optimizing loss (e.g., GD)  D) Fixed after init  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

8) Increase neurons in a layer → weights:  
A) Decrease  B) Same  C) Increase  D) Only bias increases  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

9) Weight close to zero after training implies:  
A) Adds non‑linearity  B) Connected input less important  C) Accuracy up  D) Overfits  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

10) 7 inputs → 2 neurons → biases:  
A) 7  B) 2  C) 1  D) 9  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B (one per neuron).</p>
</details>

---

## Activation Functions

### Classical Activations — MCQs

**Covered:** Linear, Sigmoid, tanh, ReLU, Leaky ReLU.

1) Linear activation is:  
A) Sigmoid  B) tanh  C) ReLU  D) Identity  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> D.</p>
</details>

2) Output layer for binary classification:  
A) ReLU  B) tanh  C) Sigmoid  D) Linear  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

3) Avoid sigmoid in deep hidden layers due to:  
A) Cost  B) Exploding grads  C) Vanishing grads  D) Too many params  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

4) tanh range:  
A) [0,1]  B) (−1,1)  C) (0,∞)  D) (−∞,∞)  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

5) ReLU preferred because:  
A) Smoother  B) Avoids saturation for positive x  C) Probabilities  D) Zero‑centered  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

6) ReLU(−5) = ?  
A) −5  B) 0  C) 1  D) Undefined  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> B.</p>
</details>

7) Leaky ReLU differs by:  
A) Adds noise  B) Output layer use  C) Small negative slope  D) More memory  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

8) Sparse activations likely with:  
A) Sigmoid  B) tanh  C) ReLU  D) Linear  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

9) Regression output uses:  
A) ReLU  B) Sigmoid  C) tanh  D) Linear  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> D.</p>
</details>

10) Common ReLU issue:  
A) Always zero  B) Negative outputs  C) Dying neurons  D) Not differentiable at zero (true but not main issue)  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> C.</p>
</details>

### Sigmoid — MCQs

1) Range: [0,1]  
<details><summary>Show answer & explanation</summary>
<p><b>Answer:</b> Yes. \(\sigma(x)=1/(1+e^{-x})\).</p>
</details>

2) Derivative: \(\sigma(x)(1-\sigma(x))\)  
<details><summary>Show explanation</summary>
<p>Used in gradient calculations.</p>
</details>

3) \(\sigma(0)=0.5\)  
<details><summary>Explanation</summary>
<p>Direct from definition.</p>
</details>

### tanh — MCQs

1) Range (−1,1); zero‑centered; derivative \(1-\tanh^2(x)\).  
<details><summary>Notes</summary>
<p>Often better than sigmoid in hidden layers due to zero‑centering, but still can saturate.</p>
</details>

### ReLU — MCQs

Covered above.

### Linear — MCQs

Covered above.

### Softmax vs Sigmoid — MCQs

1) Softmax purpose: multi‑class probabilities (sum to 1).  
<details><summary>Show answer</summary>
<p><b>Answer:</b> Yes, used with categorical cross‑entropy.</p>
</details>

2) Prefer sigmoid when: **binary** (single‑label) classification.  
<details><summary>Show note</summary>
<p>For multi‑label outputs, use sigmoid per class.</p>
</details>

---

## Loss Functions

### Regression (MAE, MSE, RMSE) — MCQs

1) Penalizes large errors most: **MSE**  
<details><summary>Why?</summary>
<p>Squares residuals.</p>
</details>

2) With small errors, MSE vs MAE: **MSE larger**  
<details><summary>Why?</summary>
<p>Squared terms inflate the average.</p>
</details>

3) Most sensitive to outliers: **MSE**  
<details><summary>Why?</summary>
<p>Square magnifies large deviations.</p>
</details>

4) RMSE for preds [3,5,2], truth [4,4,2]:  
<details><summary>Show answer & calc</summary>
<p>Errors = [−1, +1, 0]; squared = [1,1,0]; MSE=2/3; RMSE≈0.8165.</p>
</details>

5) Prefer MAE when: **robust to outliers** desired.  
<details><summary>Why?</summary>
<p>L1 less sensitive to large errors.</p>
</details>

### Classification (Cross‑Entropy) — MCQs

1) Cross‑entropy mainly for **classification**.  
<details><summary>Binary vs Categorical</summary>
<p>Binary CE with sigmoid; categorical CE with softmax.</p>
</details>

2) Low loss when predicted prob ~ true label.  
<details><summary>Numerics</summary>
<p>Log terms approach zero.</p>
</details>

3) Beware <code>log(0)</code> — numerical stability (use eps).

---

## Training & Tuning

### Hyperparameter Tuning — MCQs

1) Hyperparameter example: **Learning rate**  
2) Exhaustive search: **GridSearch**  
3) Grid drawback: **Computationally expensive**  
4) Random Search advantage: **Faster, effective**  
5) Sklearn tool: **GridSearchCV**  
6) Bayesian opt: **Informed by past trials**  
7) Objective: **Improve validation/test performance**  
8) Not a hyperparameter: **Weights**  
9) Early stopping is a **tuning strategy**  
10) Random Search randomly selects **hyperparameter combinations**

### Backpropagation — MCQs

1) Purpose: **Compute gradients to update weights**  
2) Based on: **Chain rule**  
3) Updates: **Weights and biases**  
4) Error propagated **backward**  
5) Vanishing grads often from **sigmoid**  
6) Learning rate controls **update step size**  
7) Mitigate vanishing: **ReLU / BatchNorm / clipping**  
8) Gradients update **opposite** to error gradient  
9) Stop when **converged / early stopping**  
10) Chain rule enables deep training

### Random State (Reproducibility)

`random_state` fixes randomness for reproducible splits, shuffles, and inits:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## Deep Learning Landscape

### Libraries Overview

- **TensorFlow / Keras** — production, deployment, TPUs  
- **PyTorch** — research‑friendly, dynamic graphs  
- **MXNet** — scalable training (AWS)  
- **JAX** — NumPy + autodiff, high‑perf research

### CNN Quick MCQs

1) Stride=2 means: **filter moves 2 px/step → smaller output**  
2) Activation after conv adds **non‑linearity (ReLU)**  
3) Padding **preserves spatial size**  
4) Pooling **downsamples** features  
5) FC layer **flattens & classifies**  
6) Dropout **regularizes** by random deactivation

### RNN (placeholder)

- Time‑series / sequential modeling; LSTM/GRU manage long dependencies.

---

## NLP & Transformers

### Tokenization / Stemming / Lemmatization — MCQs

1) Tokenization goal: **split text into units**  
2) Stemming: **rule‑based crude roots (may be non‑words)**  
3) Lemmatization preferred: **meaningful dictionary forms via POS**  
4) Lemmatization often needs **POS tags**  
5) Order in pipeline: **Tokenize → (Stem/Lemmatize)**

### BERT Cheatsheet

- Bidirectional encoder; MLM + NSP objectives; strong for classification/QA/NER.  
- Variants: DistilBERT, RoBERTa, ALBERT, TinyBERT.

### Self‑Supervised Learning (SSL)

- Create pretext tasks (masking, contrastive); pretrain on unlabeled data; fine‑tune downstream.

### RLHF — Cheatsheet

- SFT → Reward Model → PPO; aligns LLMs with human preferences.

### LLM Prompting (Chain‑of‑Thought)

- Encourage stepwise reasoning (“Let’s think step by step”) for math/logic.

### LangChain — MCQs

- Chains, Agents, Memory, Retrievers, Tools; use for tool‑augmented LLM apps.  

---

## Big Gen‑AI Players (Cheatsheet)

- **OpenAI (ChatGPT/DALL·E)**, **Google (Gemini)**, **Microsoft (Copilot)**, **Meta (Llama)**, **Perplexity**, **Adobe Firefly**, **AWS Bedrock**.

---

> © Prepared by Abhishek — optimized for GitHub Markdown with collapsible answers.
