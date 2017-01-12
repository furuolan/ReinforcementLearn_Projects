import numpy as np

class Linear_TD:
    def __init__(self, learning_rate=0.05, lam=0, incremental_updates=False, epsilon=.1):
        self.learning_rate = learning_rate
        self.lam = lam
        self.incremental_updates = incremental_updates
        self.epsilon = epsilon
        self.w = None

    def fit(self, X_train, init_weight=0.5):
        # initialize the weight vector to the same dim as as single instance
        # here, the first instance in the sequence
        self.w = np.full(X_train[0][0].shape, init_weight, dtype=np.float64)
        # the weights of the terminal states are known a-priori to be 0 and 1
        # from the problem definition
        self.w[0] = 0.
        self.w[-1] = 1.

        if self.incremental_updates:
            self.fit_incremental(X_train)
        else:
            batch_norm = 1.
            while batch_norm > self.epsilon:
                batch_dW = self.fit_batch(X_train)
                batch_norm = np.linalg.norm(batch_dW)


    def predict(self, X):
        if self.w is None:
            raise ValueError('Model has not been trained with a call to fit()')

        return np.dot(X, self.w.T)

    def fit_incremental(self, X_train):
        # for each sequence in the training set
        for X in X_train:  # for each seq
            dW = np.zeros_like(self.w)

            for t in range(len(X)):
                lambda_sum = np.zeros_like(self.w)
                for k in range(1, t+1):
                    lambda_sum += (self.lam ** (t - k)) * X[k]
                X_prime = X[t + 1] if t + 1 < len(X) else X[t]
                dW += self.learning_rate * (self.predict(X_prime) - self.predict(X[t])) * lambda_sum
            self.w += dW

    def fit_batch(self, X_train):
        dW = np.zeros_like(self.w)

        # for each sequence in the training set
        for X in X_train:

            for t in range(len(X)):
                lambda_sum = np.zeros_like(self.w)
                for k in range(1, t+1):
                    lambda_sum += (self.lam ** (t - k)) * X[k]
                X_prime = X[t + 1] if t + 1 < len(X) else X[t]
                dW += self.learning_rate * (self.predict(X_prime) - self.predict(X[t])) * lambda_sum
        self.w += dW
        return dW
