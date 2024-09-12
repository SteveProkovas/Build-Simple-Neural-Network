# Build Simple Neural Network

A simple example of building a neural network in Python using TensorFlow and Keras.

## Table of Contents
- [Motivation](#motivation)
- [Usage](#usage)
- [Algorithmic Explanation](#algorithmic-explanation)
- [Mathematical Explanation](#mathematical-explanation)
- [License](#license)

## Motivation
This project aims to provide a basic example of building a neural network with one hidden layer. It demonstrates how to:

- Generate sample input data and target outputs
- Define a neural network architecture with an input layer, hidden layer, and output layer
- Compile the model with an optimizer and loss function
- Train the model on the sample data
- Make predictions using the trained model

## Usage
1. Open the `simple_neural_network.py` file in your preferred code editor.

2. Run the script:
```bash
python simple_neural_network.py
```

This will generate some sample input data, create a neural network with one hidden layer, train the model, and make predictions using the trained model.

3. Customize the code to experiment with different architectures, hyperparameters, or datasets as needed.

## Algorithmic Explanation

1. **Input Layer**:
   - The model receives input data consisting of three features (e.g., \(x_1\), \(x_2\), \(x_3\)) representing each sample in the dataset.

2. **Hidden Layer**:
   - The hidden layer consists of neurons that compute outputs based on the weighted sum of the inputs.
   - Each neuron applies an activation function (e.g., ReLU) to introduce non-linearity:
     \[
     h_j = \text{ReLU}(w_j \cdot X + b_j)
     \]
     where:
     - \(h_j\) is the output of neuron \(j\),
     - \(w_j\) are the weights associated with neuron \(j\),
     - \(X\) is the input vector,
     - \(b_j\) is the bias for neuron \(j\).

3. **Output Layer**:
   - The output layer produces the final prediction using a sigmoid activation function (used for binary classification):
     \[
     y_{\text{pred}} = \sigma(w_o \cdot h + b_o)
     \]
     where:
     - \(y_{\text{pred}}\) is the predicted output,
     - \(w_o\) are the weights for the output layer,
     - \(h\) is the output from the hidden layer,
     - \(b_o\) is the bias for the output neuron.

4. **Loss Function**:
   - The model uses binary cross-entropy as the loss function, which measures the difference between predicted outputs and actual target values.

5. **Optimization**:
   - The Adam optimizer is used to update the weights and biases, minimizing the loss function through backpropagation.

## Pseudocode

```plaintext
BEGIN

  IMPORT necessary libraries
    - Import 'numpy' for numerical operations
    - Import 'keras' from 'tensorflow' for building neural network models

  DEFINE input data (X)
    - X is an array with 3 input values (x1, x2, x3) for each sample

  DEFINE target data (y)
    - y is an array with the corresponding target outputs (0 or 1)

  INITIALIZE model as a Sequential model

  ADD layers to the model:
    - Hidden layer with 4 neurons and ReLU activation function
    - Output layer with 1 neuron and Sigmoid activation function

  COMPILE the model:
    - Use Adam optimizer
    - Use binary_crossentropy as the loss function
    - Measure accuracy

  TRAIN the model:
    - Train on input data X and target output y
    - Run for 1000 epochs with no verbose output

  MAKE predictions on the input data (X)

  PRINT the predicted outputs

END
```

## Mathematical Explanation

1. **Forward Pass**:
   - For each input vector $$X$$:
     - Compute the hidden layer outputs:
       $$ H = \text{ReLU}(W_h \cdot X + B_h) $$
       where:
       - $$W_h$$ is the weight matrix for the hidden layer.
       - $$B_h$$ is the bias vector for the hidden layer.
     - Compute the output:
       $$ y_{\text{pred}} = \sigma(W_o \cdot H + B_o) $$
       where:
       - $$W_o$$ is the weight vector for the output layer.
       - $$B_o$$ is the output bias.

2. **Backward Pass (Backpropagation)**:
   - Compute gradients of the loss with respect to weights and biases using the chain rule.
   - Update weights using gradient descent:
     $$ W \leftarrow W - \eta \cdot \nabla L $$
     where:
     - $$\eta$$ is the learning rate.
     - $$\nabla L$$ is the gradient of the loss function with respect to the weights.

This simple neural network can be expanded with more layers, different activation functions, and various architectures depending on the complexity of the problem you are trying to solve.

## License
This project is licensed under the [Apache License 2.0](LICENSE). You are free to use, modify, and distribute this code as long as you include the original copyright notice and license.
