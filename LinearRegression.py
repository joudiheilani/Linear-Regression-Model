import numpy as np
import matplotlib.pyplot as plt

# this creates 100 evenly spaced values from 0 to 10
x = np.linspace(0, 10, 100) #x values

# this will represent our data, the random is adding noise to make the data imperfect
y = 3 * x + 2 + np.random.randn(100) * 0.5
# if you notice, our w is 3 and b is 2, ignoring the noise. these are our "perfect" values for those two.
# our algorithm result should come close to 3 and 2 for w and b respectively. 

# our initial #guesses", and then the learning rate or step size of the updates
w = 0
b = 0
lr = 0.01 # smaller number means "fine" adjustments, and larger means "coarse" adjustments.

for epoch in range(1000):
    # add the equations from ReadME here
    y_prediction = w * x + b
    error = y_prediction - y
    mse = np.mean((error) ** 2)

    dw = (2/len(x)) * np.dot(error, x)  # how much we should change w by
    db = (2/len(x)) * np.sum(error)     # hown much we should change b by

    w -= lr * dw    # new w guess
    b -= lr * db    # new b guess
    print("Current Epoch = ", epoch, "w = ", w, "b = ", b, "MSE: ", mse)

print("Learned w = ", w, "b = ", b, "With an MSE of: ", mse)

plt.scatter(x, y, color = 'purple', label = "data")
plt.plot(x, w * x + b, color = 'cyan', label = "fitted line")
plt.legend()
plt.show()