import numpy as np
from neuralnetwork import forwardpass, cross_entropy_loss, accuracy

def test_model(test_data, test_y, weights, biases, n_layers, activation="sigmoid"):
    """
    Evaluates the trained model on the test dataset.
    """

    # 1. Forward Pass (No backprop needed here!)
    # x contains the activations, x[n_layers + 1] is the final output layer
    x, _ = forwardpass(test_data, weights, biases, n_layers, activation)
    final_output = x[n_layers + 1]

    # 2. Calculate Metrics
    loss = cross_entropy_loss(final_output, test_y)
    acc = accuracy(final_output, test_y)

    print(f"Test Loss:     {loss:.6f}")
    print(f"Test Accuracy: {acc:.2f}%")
    
    return loss, acc

if __name__ == "__main__":
    # --- PREPARATION ---
    # In a real scenario, you'd load your weights and test data here.
    # For now, let's assume we are using the same config as train.py
    
    HIDDEN_LAYER_SIZES = [32, 32]
    n_layers = len(HIDDEN_LAYER_SIZES)
    
    # MOCK DATA: Replace these with your actual preprocessed test tensors!
    # Remember: x_test must be scaled using the TRAINING set's scaler.
    x_test = np.random.rand(200, 6) 
    y_test = np.random.randint(1, 5, size=(200, 1))

    # You would typically pass the weights/biases returned from your 'fit' function
    # result_weights, result_biases, ... = fit(...)
    # test_model(x_test, y_test, result_weights, result_biases, n_layers)
    
    print("Ready to evaluate. Ensure you pass the 'weights' and 'biases' from your training script.")