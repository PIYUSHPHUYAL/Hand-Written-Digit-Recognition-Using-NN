import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    
    # Add bias unit to X
    X = np.hstack([np.ones((m, 1)), X])
    
    # Forward propagation
    a1 = X
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    
    # Add bias unit to a2
    a2 = np.hstack([np.ones((m, 1)), a2])
    
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    
    # Get predictions
    predictions = np.argmax(a3, axis=1)
    
    return predictions

# If you want to test the prediction function independently:
if __name__ == "__main__":
    # Load your Theta values
    Theta1 = np.loadtxt('Theta1.txt')
    Theta2 = np.loadtxt('Theta2.txt')
    
    # Create a sample input (this should be a 1x784 vector)
    sample_input = np.random.rand(1, 784)
    
    # Make a prediction
    result = predict(Theta1, Theta2, sample_input)
    print("Predicted digit:", result[0])