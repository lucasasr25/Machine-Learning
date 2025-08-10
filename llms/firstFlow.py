import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
from Layer_Dense import Layer_Dense
from relu import Activation_ReLU
from softmax import Activation_Softmax
from cross_entropy import Loss_CategoricalCrossentropy
import numpy as np

nnfs.init()

X, y = vertical_data(samples=100, classes=3)
print(y)


dense1 = Layer_Dense(2 , 3)
activation1 = Activation_ReLU()
dense2 =  Layer_Dense(3 , 3)
activation2 = Activation_Softmax()


loss_function = Loss_CategoricalCrossentropy()

lowest_loss = 9999999

best_dense1_weights = dense1.weights.copy()
best_dense1_biases  = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases  = dense2.biases.copy()

for iteration in range(10000):

    # Generate a new set of weights for this iteration
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate loss
    loss = loss_function.calculate(activation2.output, y)


    # Calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print(f"New set of weights found, iteration: {iteration}, loss: {loss}, acc: {accuracy}")
        
        # Save the current weights and biases
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        
        # Update the lowest recorded loss
        lowest_loss = loss
    else:
        # Revert to the best weights and biases found so far
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
