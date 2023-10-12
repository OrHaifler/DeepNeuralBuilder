def single_forward(self, X: np.ndarray, layer: Layer) -> (np.ndarray, tuple):
    if layer.type in non_linearities:
        out, cache = layer.forward(X)
    else:
        out, cache = layer.forward(X, layer.weights['W'], layer.weights['b'], layer.dropout)
    return out, cache


def update_plot(frame):
    plt.clf()

    # Plot the parameter values at the current frame/epoch
    plt.subplot(221)
    plt.plot(W1_values[frame][:, 0], label='W1[0]')
    plt.plot(W1_values[frame][:, 1], label='W1[1]')
    plt.legend()

    plt.subplot(222)
    plt.plot(b1_values[frame][0, :], label='b1')
    plt.legend()

    plt.subplot(223)
    plt.plot(W2_values[frame][:, 0], label='W2[0]')
    plt.plot(W2_values[frame][:, 1], label='W2[1]')
    plt.legend()

    plt.subplot(224)
    plt.plot(b2_values[frame][0, :], label='b2')
    plt.legend()


for l in range(self.L - 1, -1, -1):
    if self.layers[l].type in non_linearities:
        continue
    dX, dW, db = self.grads[l]
    dW_norm, db_norm = np.linalg.norm(dW), np.linalg.norm(db)
    if dW_norm > clipping_threshold:
        dW *= clipping_threshold / dW_norm
    if db_norm > clipping_threshold:
        db *= clipping_threshold / db_norm
    self.layers[l].weights['W'] -= lr * dW
    self.layers[l].weights['b'] -= lr * db
