import numpy as np

class NeuralNetwork:
    def __init__(self, n_features, n_hidden, n_classes):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.w1 = 0.01 * np.random.randn(n_features, n_hidden)
        self.b1 = np.zeros((1, n_hidden))
        self.w2 = 0.01 * np.random.randn(n_hidden, n_classes)
        self.b2 = np.zeros((1, n_classes))

    def forward(self, X):
        z1 = np.dot(X, self.w1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = np.dot(a1, self.w2) + self.b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return a1, probs

    def loss(self, X, y, reg=1e-3):
        _, probs = self.forward(X)
        N = X.shape[0]
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * reg * np.sum(self.w1**2) + 0.5 * reg * np.sum(self.w2**2)
        return data_loss + reg_loss

    def train(self, X, y, learning_rate=1e-0, reg=1e-3, epochs=10000, verbose=True):
        N = X.shape[0]
        for i in range(epochs):
            # Forward pass
            a1, probs = self.forward(X)

            # Backward pass
            dscores = probs
            dscores[range(N), y] -= 1
            dscores /= N

            # W2 and b2
            dw2 = np.dot(a1.T, dscores)
            db2 = np.sum(dscores, axis=0, keepdims=True)

            # W1 and b1
            da1 = np.dot(dscores, self.w2.T)
            da1[a1 <= 0] = 0  # ReLU gradient
            dw1 = np.dot(X.T, da1)
            db1 = np.sum(da1, axis=0, keepdims=True)

            # Regularization gradient
            dw2 += reg * self.w2
            dw1 += reg * self.w1

            # Update parameters
            self.w1 += -learning_rate * dw1
            self.b1 += -learning_rate * db1
            self.w2 += -learning_rate * dw2
            self.b2 += -learning_rate * db2

            if verbose and i % 1000 == 0:
                print(f"Epoch {i}, Loss: {self.loss(X, y, reg)}")

    def predict(self, X):
        _, probs = self.forward(X)
        return np.argmax(probs, axis=1)