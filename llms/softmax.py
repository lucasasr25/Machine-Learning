# layer_outputs = [4.8, 1.21, 3.385]

# E = 2.71

# exp_values = []

# for output in layer_outputs:
#     exp_values.append(E ** output)

# print(exp_values)


import numpy as np


class Activation_Softmax:
    def forward(self, inputs):
        # Evitar overflow (fazendo cada item do array menos o maior valor do array, depois E ** i) 
        exp_values = np.exp(inputs -  np.max(inputs, axis = 1, keepdims=True))
        
        # Aqui é feito o valor da probabilidade para cada input
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilities