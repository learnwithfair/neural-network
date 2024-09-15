import numpy as np
import matplotlib.pyplot as plt

# Step function for bipolar input
def step_function(x):
    return np.where(x > 0, 1, np.where(x<0,-1,0))

# Perceptron training function
def perceptron_training(inputs, targets, learning_rate=1, epochs=10):
    
    n_samples, n_features = inputs.shape
    weights = np.zeros(n_features)
    bias = 0
    errors = []

    for epoch in range(epochs):
        total_error = 0
        for x, target in zip(inputs, targets):
            # Weighted sum
            linear_output = np.dot(x, weights) + bias
            # Perceptron output using the step function
            predicted = step_function(linear_output)
            # Update rule
            error = target - predicted
            weights += learning_rate * error * x
            bias += learning_rate * error
            total_error += abs(error)
        
        errors.append(total_error)
        print(f'Epoch {epoch+1} of {epochs}, Total Error: {total_error}')
    
    return weights, bias, errors

# Plot decision boundary and convergence curve
def plot_results(inputs, targets, weights, bias, errors):
    # Plot decision boundary
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    for i, input_point in enumerate(inputs):
        if targets[i] == 1:
            plt.scatter(input_point[0], input_point[1], color='blue', marker='o')
        else:
            plt.scatter(input_point[0], input_point[1], color='red', marker='x')
    
    # Equation of decision boundary: w1*x1 + w2*x2 + bias = 0 => x2 = -(w1*x1 + bias)/w2
    x_values = np.linspace(-2, 2, 100)
    if weights[1] != 0:  # To avoid division by zero in case of vertical line
        decision_boundary = -(weights[0] * x_values + bias) / weights[1]
        plt.plot(x_values, decision_boundary, color='green', label='Decision Boundary')
    plt.title('Decision Boundary')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

    # Plot convergence curve
    plt.subplot(1, 2, 2)
    plt.plot(errors, marker='o', color='purple')
    plt.title('Convergence Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Total Error')

    plt.tight_layout()
    plt.show()

# Bipolar inputs for AND function
inputs = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
# Bipolar targets for AND function (-1 for False, 1 for True)
targets = np.array([-1, -1, -1, 1])

# Train the perceptron
learning_rate = 1
epochs = 10
weights, bias, errors = perceptron_training(inputs, targets, learning_rate, epochs)
print("W1 and W2 =",weights,"Bias =",bias)
# Plot results
plot_results(inputs, targets, weights, bias, errors)
