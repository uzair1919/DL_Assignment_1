import numpy as np

def sigmoid(a):
    return 1/(1+np.exp(-a))

def relu(a):
    return np.maximum(0, a)

def softmax(a):
    exp = np.exp(a)
    sm = exp / np.sum(exp, axis=1, keepdims=True)
    return sm

def delta_l(X, Y):
    n, d = X.shape
    dL_dx = np.zeros([n, d, 1])
    jacobian = np.zeros([n, d, d])
    for row in range(n):
        xi = X[row]
        yi = int(Y[row][0])
        dL_dx[row, yi-1, 0] = -1 / xi[yi-1]
        for i in range(d):
            for j in range(d):
                if i == j:
                    jacobian[row,i,j] = (1-xi[i])*xi[i]
                else:
                    jacobian[row,i,j] = -xi[i]*xi[j]
    delta = jacobian @ dL_dx
    delta = np.squeeze(delta)
    return delta

def initialise(l_sizes, num_feat, output_size):
    n_layers = len(l_sizes)
    weights = {}
    biases = {}
    for l in range(n_layers + 1):
        if l == 0:
            weights[l+1] = np.random.randn(l_sizes[l], num_feat)
            biases[l+1] = np.random.randn(l_sizes[l])
        elif l == n_layers:
            weights[l+1] = np.random.randn(output_size, l_sizes[-1])
            biases[l+1] = np.random.randn(output_size)
        else:
            weights[l+1] = np.random.randn(l_sizes[l], l_sizes[l-1])
            biases[l+1] = np.random.randn(l_sizes[l])
    return weights, biases


def forwardpass(x0, weights, biases, n_layers, activation="sigmoid"):
    a = {}
    x = {0 : x0}
    out_layer = n_layers + 1
    for layer in range(1, out_layer):
        a[layer] = (weights[layer] @ x[layer - 1].T).T + biases[layer]
        if activation == "sigmoid":
            x[layer] = sigmoid(a[layer])
        elif activation == "relu":
            x[layer] = relu(a[layer])
        else:
            raise ValueError("activation layer must be either relu or sigmoid")
    a[out_layer] = (weights[out_layer] @ x[out_layer - 1].T).T + biases[out_layer]
    x[out_layer] = softmax(a[out_layer])
    return x, a

def backprop(x, a, Y, weights, biases, n_layers, lr, activation="sigmoid"):
    out_layer = n_layers + 1
    delta = {
        out_layer : delta_l(x[out_layer], Y)
    }
    d_weights = {}
    d_biases = {}
    n = x[0].shape[0]
    for layer in range(out_layer, 1, -1):
        d_W = (delta[layer].T @ x[layer-1])/n
        d_b = (np.sum(delta[layer], axis=0))/n

        d_weights[layer] = d_W
        d_biases[layer] = d_b

        if activation == "sigmoid":
            g = sigmoid(a[layer-1])
            d_g = (g * (1 - g))
        elif activation == "relu":
            d_g = (a[layer-1] > 0).astype(float)
        else:
            raise ValueError("activation layer must be either relu or sigmoid")
        
        delta[layer-1] = delta[layer] @ weights[layer] * d_g
        weights[layer] -= lr * d_W
        biases[layer] -= lr * d_b

    d_W = (delta[1].T @ x[0])/n
    d_b = (np.sum(delta[1], axis=0))/n
    weights[1] -= lr * d_W
    biases[1] -= lr * d_b

    d_weights[1] = d_W
    d_biases[1] = d_b

    return weights, biases, d_weights, d_biases

def cross_entropy_loss(x, y):
    n, d = x.shape
    y = np.squeeze(y)
    loss = 0
    for row in range(n):
        xi = x[row]
        yi = y[row]
        y_hat = xi[yi-1]
        loss += -np.log(y_hat)
    loss /= n
    return loss

def accuracy(x, y):
    y = np.squeeze(y)
    y_pred = np.argmax(x, axis=1) + 1
    accuracy = np.sum(y_pred == y) / len(y)
    return accuracy

