# Linear Regression Model
## Overview
This project implements linear regression from scratch in Python using NumPy. This is a foundational project to understand some core mechanics of Machine Learning: 
- Representing data with arrays.
- Defining a model and learning parameters.
- Computing a loss function.
- Updating parameters using gradient descent.

## Goals
- Gain understanding of linear regression.
- Learn gradient descent to optimize learning.
- Develop data visualization skills using MatPlotLib.
- Gain experience in NumPy using arrays.

## Project Implementation 
### 1. Creating Data
  - Generated a simple linear relationship with noise:

$$
y = 3x + 2 + \text{noise}
$$

- Generated 100 points using `numpy.linspace` and `numpy.random.randn`

### 2. Model Definition
  - Model Equation:

$$
\hat{y} = w \cdot x + b
$$

  - Where:
      - $w$ = weight parameter.
      - $b$ = bias parameter.

### 3. Loss Function
  - The Mean Squared Error (MSE) measures the average squared difference between predicted and actual values:

$$
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

### 4. Gradient Descent
  - Minimize loss by iteratively adjusting parameters.
  - Computed gradients manually:

$$
dw = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) \cdot x_i
$$

$$
db = \frac{2}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)
$$

  - Parameters are updated using the learning rate $lr$:

$$
w = w - lr \cdot dw
$$

$$
b = b - lr \cdot db
$$

  - This repeats for all iterations in the code.

### 5. Visualization
  - After training, the noisy data points and the fitted line are plotted using MatPlotLib.

## Results
- Learned parameters:

$$
w \approx 3.0, \quad b \approx 2.0
$$

- Fitted line closely follows all the noisy data points demonstrating that the gradient descent successfully fully minimized the loss.

## Concepts Learned
<div align="center">
  
| Concept | Description |
|---------|-------------|
| Python Fundamentals | Loops, functions, vectorized operations, plotting |
| NumPy | Array manipulation, broadcasting, vectorized computation |
| Linear Regression | Supervised learning, model parameters, error measurement |
| Gradient Descent | Manual implementation of optimization |
| Visualization | Interpreting model predictions graphically |

</div>
