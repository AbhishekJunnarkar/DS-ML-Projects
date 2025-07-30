# Table of contents

- Deep Learning 

- Deep Feedforward Neural Nets

- Convolutional Neural Nets

- Long short-Term Memory (LSTM) networks

- Introduction to Transformers and Attention Mechanisms

-  Explainable AI

-------------------------------------------

# Introduction

## Deep Learning 

Q. What is Deep Learning?
- Deep learning is a subfield of machine learning that uses algorithms inspired by the structure and function of the human brain, specifically artificial neural networks, to learn from large amounts of data.

Deep learning is a type of machine learning that uses neural networks with many layers to learn complex patterns from data.

Think of deep learning as teaching a machine to learn like a brain — by using many layers of neurons that process data step-by-step.

**Structure:**

A deep learning model is made of:

Input Layer: Takes in the raw data (e.g., pixels of an image)

Hidden Layers: Each layer learns features of increasing complexity

Output Layer: Gives the final result (e.g., "cat" or "dog")

"Deep" just means more than one hidden layer (usually many).

**Analogy:**

Imagine identifying a face:

First layer sees edges

Next layer sees shapes like eyes/nose

Next layer understands a face

Final layer says: “It’s Alice!”

That’s how deep learning learns — simple to complex, one layer at a time.

**Examples:**

📸 Image recognition (e.g., Google Photos recognizing faces)

🎙️ Voice assistants (e.g., Siri, Alexa)

🏎️ Self-driving cars

📝 ChatGPT (yes, me! built using deep learning 🤓)

Q. Why Deep Learning Works So Well:

Can automatically learn features (no manual feature engineering)

Works great with huge amounts of data

Uses powerful GPUs to process large neural networks

--- 

## Smallest building block: Perceptron

A perceptron is the simplest type of neural network — think of it as a basic decision-making unit in a neural network.

A perceptron is like a tiny calculator that takes some inputs, multiplies them by some weights, adds them up, and decides "Yes" or "No" (i.e., 1 or 0) using an activation function.

Let’s say we have inputs: x1, x2, x3

Each input has a weight: w1, w2, w3

#### The perceptron computes:

output = activation(w1*x1 + w2*x2 + w3*x3 + bias)

The activation function (usually a step function or sign function) decides if the neuron "fires" (gives 1) or not (gives 0).

##### Analogy:

Imagine you’re deciding whether to go out:

Input 1: Is it sunny? (Yes=1, No=0)

Input 2: Do you have free time? (Yes=1, No=0)

Input 3: Is your friend free? (Yes=1, No=0)

You assign importance (weights) to each factor, add them up, and if the result is high enough (threshold), you decide “Yes, go out” — that’s what a perceptron does.

##### Key Points:
- One layer, one output
- Used for binary classification (e.g., yes/no, spam/not spam)
- Learns by adjusting weights (using a simple learning rule)

##### Limitations:
- Can only solve linearly separable problems (e.g., it cannot solve the XOR problem)

- That’s why we move to multi-layer perceptrons (MLPs) for more complex tasks.

--- 
## Feed Forward (FF) Neural Network

Q. What is Feedforward (FF) in Neural Networks?

Feedforward means that the data flows only in one direction — from input to output — without looping back.

Think of feedforward as a one-way street:
- 🧩 Input → Hidden Layers → Output

Each neuron passes information forward, and that’s it. No going back, no cycles.

**Example Use Cases of FF Networks:**

- Handwritten digit recognition (like MNIST)
- Spam detection
- Simple regression/classification tasks

---
### Radial Basis Function Network (RBF) 

Q. What is RBF network?

An RBF network is a type of artificial neural network that uses radial basis functions as activation functions. It’s mainly used for function approximation, classification, and regression.

**In Simple Terms:**

- An RBF network is like a pattern-matching system:

- It compares inputs to reference points (centers).

- The closer an input is to a center, the more it activates.

- It then combines these responses to make a decision.

✅ Pros:

Good for interpolation

Fast training for small datasets

Great at handling non-linear data

**❌ Cons:**

Doesn’t scale well with very large datasets

Choosing the right centers and spread (σ) is tricky

**🔍 Example Applications:**

Time series prediction

Control systems

Face recognition

Function approximation

---

### Deep Feed forward

Q. What is Deep Feed Forward Neural Nets?
- A Deep Feedforward Neural Network (also known as a Multilayer Perceptron, or MLP) is the simplest type of deep learning model. It’s a type of artificial neural network where information moves in one direction only — from input to output — without any cycles or loops.

q. What are the key features of Deep Feedforward Beural Nets?

- Feedforward:
  - Data flows only forward through the network.
  - No feedback or loops — unlike Recurrent Neural Networks (RNNs).
- Layers:
  - Input Layer: Receives raw data (e.g., image pixels, text embeddings).
  - Hidden Layers: One or more layers where neurons apply transformations using weights, biases, and activation functions.
  - Output Layer: Produces predictions (e.g., class probabilities, regression values).

- "Deep":
- A network is considered deep when it has more than one hidden layer.
- Deeper networks can learn more complex patterns but are harder to train.

Mathematical Structure:
Each neuron computes:
- y=f(Wx+b)

----
### Forward Pass

Q: What is a forward pass in a neural network?
- A: The forward pass is the process where input data flows through the neural network layer by layer to produce an output (or prediction).

Q: Why is the forward pass important?
- A: It generates the model’s output, which is compared to the true label to compute the loss. This loss is then used in the backward pass to update the weights.

Q: Does the forward pass involve learning or updating weights?
- A: No. The forward pass only computes outputs. Weight updates happen during backpropagation in the training process.

---
### Back Propagation

Q: What is back-propagation?
- A: Back-propagation is the algorithm used to compute gradients of a neural-network’s loss function with respect to its weights and biases. Those gradients tell us how to adjust the parameters to reduce the loss during training.

Q: How does back-propagation work, conceptually?
- A: It applies the chain rule from calculus to propagate the loss gradient backwards through the network, layer by layer, starting at the output and ending at the earliest weights.

Q: Why is back-propagation important?
- A: Without it, training deep networks would require numerically expensive finite-difference methods. Back-prop efficiently reuses intermediate values from the forward pass, making gradient computation feasible for millions of parameters.

Q: Is back-propagation learning by itself?
- A: No. Back-prop only provides gradients. Learning happens when an optimizer uses those gradients to update the weights.


Q: Are there limitations to back-propagation?
- A:

Vanishing / exploding gradients in very deep or recurrent networks.

Requires differentiable operations (discrete choices need tricks like REINFORCE or Gumbel-Softmax).

Needs the entire computational graph in memory for the backward pass, which can be resource-intensive.

---

### Hidden Layers

Q: What are hidden layers in a neural network?
- A: Hidden layers are the intermediate layers between the input layer and the output layer in a neural network. They are called “hidden” because their outputs are not directly observable — they are only used internally by the model.

Q: Why do we need hidden layers?
- A: Hidden layers allow the network to model nonlinear and complex functions. A network with no hidden layers is just a linear model. Adding hidden layers increases the network's capacity to learn.

Q: How many hidden layers should a network have?
- A: It depends on the task:

Simple tasks → 1 or 2 hidden layers may be enough.

Complex tasks (image recognition, language models) → deep networks with many hidden layers are used.

The term deep learning comes from using many hidden layers.

Q: Do all hidden layers use the same size or activation function?
- A: Not necessarily:

Different layers can have different numbers of neurons.

Most use ReLU or variants, but depending on the task, other activations like sigmoid, tanh, or GELU might be used.

---
Q: Are hidden layers trained?
- A: Yes. During training, the network adjusts the weights and biases in hidden layers via backpropagation to minimize the loss.
---

Q: Can a network work without hidden layers?
- A: Only for very simple, linearly separable tasks. For anything involving complex patterns (e.g., images, language), hidden layers are essential.

--- 
### Activation function

Q: What is an activation function?
- A: An activation function adds non-linearity to a neural network. Without it, the network would just behave like a linear model, no matter how many layers you add.

Activation functions decide whether a neuron should "fire" (i.e., pass its signal to the next layer) and how strong that signal should be.

Q: What is the Sigmoid activation function?
A:
The Sigmoid function maps any real-valued input to a value between 0 and 1:
 
Use case: Often used in the output layer for binary classification problems.

Pros:

Smooth output between 0 and 1

Useful when output is a probability

Cons:

Vanishing gradient problem: for large/small inputs, the gradient becomes very small → slows learning

Not zero-centered

Q. Q: What is Softmax?
- A: The Softmax function converts a vector of raw scores (called logits) into a probability distribution — where each value is between 0 and 1, and all values add up to 1.

It is typically used in the output layer of a neural network for multi-class classification.

---

### Hyper parameter Tuning

## Deep Feedforward Neural Nets

## Convolutional Neural Nets

## Recurrent Neural Network (RNN)

A Recurrent Neural Network (RNN) is a type of neural network designed for sequence data – like time series, text, or audio.

### Why Use RNNs?
Traditional neural networks (like feedforward or CNNs) don’t remember past inputs. But RNNs are different – they have memory.

RNNs remember past information using loops, making them suitable for:

Text (sentiment analysis, machine translation)

Time series forecasting (stock prices, weather)

Audio processing (speech recognition)
## Long short-Term Memory (LSTM) networks

## Introduction to Transformers and Attention Mechanisms

## Explainable AI


