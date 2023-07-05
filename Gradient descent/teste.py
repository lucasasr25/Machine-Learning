import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv')

"""x = np.random.randint(0, 51, size=45)
y = np.random.randint(0, 51, size=45)
"""
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
def gradient(m_now, b_now, x, y, L):
    m_gradient = 0
    b_gradient = 0
    n = len(x)
    for i, z in zip(x, y):
        m_gradient += -(2/n) * i * (z - (m_now * i + b_now))
        b_gradient += -(2/n) * (z - (m_now * i + b_now))
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b
m = 0
b = 0
L = 0.001
epochs = 5000
for i in range(epochs):
    m, b = gradient(m, b, x, y, L)
plt.scatter(x, y, color="black")
plt.plot(x, m * x + b, color="purple")
plt.show()
