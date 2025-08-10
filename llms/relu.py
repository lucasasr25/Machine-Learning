import numpy as np
from nnfs.datasets import spiral_data


class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
