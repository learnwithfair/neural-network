import torch
import torch.nn as nn
import torch.optim as optim

# To Avoid Issues Use This Link 
# https://colab.research.google.com/drive/1u8B2-TPpHR8UBMhi2LumBpqKOw_yxdIu?usp=sharing

# Define the neural network class
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        # Input to Hidden layer (2 inputs to 2 hidden nodes)
        self.hidden = nn.Linear(2, 2)  # 2 input neurons, 2 hidden neurons
        # Hidden to Output layer (2 hidden nodes to 2 output nodes)
        self.output = nn.Linear(2, 2)  # 2 hidden neurons, 2 output neurons
        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass through the network
        h = self.sigmoid(self.hidden(x))  # Hidden layer activation
        y = self.sigmoid(self.output(h))  # Output layer activation
        return y

# Create the network
model = SimpleANN()

# Set the weights and biases based on the diagram
with torch.no_grad():
    model.hidden.weight = torch.nn.Parameter(torch.tensor([[0.15, 0.20], [0.25, 0.30]]))  # w1, w2, w3, w4
    model.hidden.bias = torch.nn.Parameter(torch.tensor([0.35, 0.35]))  # Bias b1 for both hidden neurons
    model.output.weight = torch.nn.Parameter(torch.tensor([[0.40, 0.45], [0.50, 0.55]]))  # w5, w6, w7, w8
    model.output.bias = torch.nn.Parameter(torch.tensor([0.60, 0.60]))  # Bias b2 for both output neurons

# Define input (x1, x2) and target values (T1, T2)
inputs = torch.tensor([[0.05, 0.10]])  # Single input pair
targets = torch.tensor([[0.01, 0.99]])  # Target values for y1 and y2

# Define the loss function (Mean Squared Error Loss)
criterion = nn.MSELoss()

# Define the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Number of epochs (iterations)
epochs = 15000

# Training loop
for epoch in range(epochs):
    # Forward pass: Compute predicted output by passing inputs to the model
    output = model(inputs)
    
    # Compute the loss (Mean Squared Error)
    loss = criterion(output, targets)
    
    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()  # Clear the gradients from the previous step
    loss.backward()  # Backpropagation step
    optimizer.step()  # Update weights

    # Print the loss every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Print final weights and biases after training
print("\nFinal weights and biases:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# Final output after training
final_output = model(inputs)
print(f"\nFinal output (y1, y2): {final_output[0][0]:.2f}, {final_output[0][1]:.2f}")