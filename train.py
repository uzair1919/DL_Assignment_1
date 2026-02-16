from neuralnetwork import initialise, forwardpass, backprop, cross_entropy_loss, accuracy
import numpy as np


EPOCHS = 20
HIDDEN_LAYER_SIZES = [32, 32]
OUTPUT_SIZE = 4
ACTIVATION = "sigmoid"
LEARNING_RATE = 0.01

def fit(
        train_data, 
        val_data,
        train_y,
        val_y, 
        epochs=EPOCHS, 
        activation=ACTIVATION, 
        hidden_layer_size=HIDDEN_LAYER_SIZES, 
        output_size=OUTPUT_SIZE, 
        learning_rate=LEARNING_RATE
):
    
    training_loss = []
    validation_loss = []
    training_accuracy = []
    val_accuracy = []
    num_features = train_data.shape[1]
    weights, biases = initialise(hidden_layer_size, num_features, output_size)
    n_layers = len(hidden_layer_size)
    d_weights_avg = np.zeros([epochs, n_layers + 1])

    print(f"{'Epoch':<8} | {'Train Loss':<15} | {'Val Loss':<15}")
    print("-" * 45)
    
    for epoch in range(1, epochs+1):
        x, a = forwardpass(train_data, weights, biases, n_layers, activation)

        t_loss = cross_entropy_loss(x[n_layers+1], train_y)
        training_loss.append(t_loss)

        t_acc = accuracy(x[n_layers+1], train_y)
        training_accuracy.append(t_acc)

        weights, biases, d_weights, d_biases = backprop(x, a, train_y, weights, biases, n_layers, learning_rate, activation)

        x_val, a_val = forwardpass(val_data, weights, biases, n_layers, activation)

        v_loss = cross_entropy_loss(x_val[n_layers+1], val_y)
        validation_loss.append(v_loss)
        v_acc = accuracy(x_val[n_layers+1], val_y)
        val_accuracy.append(v_acc)

        for layer in range(1, n_layers +2):
            d_weights_avg[epoch - 1, layer - 1] = np.mean(np.abs(d_weights[layer]))
        
        if epoch % 50 == 0:
            print(f"{epoch:<8} | {t_loss:<15.6f} | {v_loss:<15.6f}")

    return training_loss, validation_loss, d_weights_avg, training_accuracy, val_accuracy, weights, biases


def train_val_split(x, y, val_ratio=0.2):
    n = x.shape[0]
    indices = np.random.permutation(n)
    x = x[indices]
    y = y[indices]
    split = int((1 - val_ratio) * n)
    x_train = x[:split]
    x_val = x[split:]
    y_train = y[:split]
    y_val = y[split:]
    return x_train, x_val, y_train, y_val


if __name__ == "__main__":
    t_losses, v_losses, _, ta, va = fit(
        np.random.rand(1000, 6),
        np.random.rand(200, 6),
        np.random.randint(1, 5, size=(1000,1)),
        np.random.randint(1, 5, size=(200,1))
        )
    
    print("Final training loss:", t_losses[-1])
    print("Final validation loss:", v_losses[-1])
