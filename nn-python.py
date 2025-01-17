import numpy as np

def sigmoid(x):         # activation function
    return 1/ (1 + np.exp(-x))

def sigmoid_derivative(x): # we need this to calculate the gradient during backpropagation
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def mse_loss(y_pred, y_true):
    return np.mean(np.square(y_true - y_pred))

def softmax(y):
    return np.exp(y) / np.sum(np.exp(y))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias1 = np.random.randn(hidden_size)
        self.bias2 = np.random.randn(output_size)
    def forward(self, X):
        self.a1 = np.dot(X, self.weights1) + self.bias1
        self.a2 = sigmoid(self.a1)
        self.a3 = np.dot(self.a2, self.weights2) + self.bias2
        self.a4 = sigmoid(self.a3)
        return self.a4
        
    def backward(self, X, y, output):
        error = output - y
        d_loss = error * sigmoid_derivative(output)
        d_weights2 = np.dot(self.a2.T, d_loss)
        d_bias2 = np.sum(d_loss, axis=0)
        d_hiddenlayer = np.dot(d_loss, self.weights2.T) * sigmoid_derivative(self.a2)
        d_weights1 = np.dot(X.T, d_hiddenlayer)
        d_bias1 = np.sum(d_hiddenlayer, axis=0)
        return d_weights1, d_bias1, d_weights2, d_bias2
        
    def update_params(self, d_weights1, d_bias1, d_weights2, d_bias2, lr):
        self.weights1 -= lr * d_weights1
        self.bias1 -= lr * d_bias1
        self.weights2 -= lr * d_weights2
        self.bias2 -= lr * d_bias2
        
    def train(self, X, y, lr, epochs):
        for epoch in range(epochs):
            forward_output = self.forward(X)
            loss = mse_loss(forward_output, y)
            d_weights1, d_bias1, d_weights2, d_bias2 = self.backward(X, y, forward_output)
            self.update_params(d_weights1, d_bias1, d_weights2, d_bias2, lr)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

if __name__ == "__main__":
    # Input data (4 samples, 3 features)
    X = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    
    # Output data (4 samples, 1 output)
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    # Initialize the neural network
    input_size = 3  # Number of input features
    hidden_size = 4  # Number of neurons in the hidden layer
    output_size = 1  # Number of output neurons
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Train the neural network
    epochs = 10000  # Number of training iterations
    lr = 0.1  # Learning rate for gradient descent
    nn.train(X, y, lr, epochs)
    
    # Test the neural network
    test_output = nn.forward(X)
    print("Final predictions:")
    print(test_output)










