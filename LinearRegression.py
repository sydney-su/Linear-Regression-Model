import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Linear_Regression():
    def __init__(self, alpha=1e-3, num_iter=10000, early_stop=100,
                 intercept=True, init_weight=None, penalty=None,
                 lam=1e-3, normalize=False, adaptive=True):
        """
        Linear Regression with gradient descent method.

        Attributes:
        -------------
        alpha: Learning rate.
        num_iter: Number of iterations
        early_stop: Number of steps without improvements
                    that triggers a stop training signal
        intercept: True = with intercept (bias), False otherwise
        init_weight: Optional. The initial weights passed into the model,
                                for debugging
        penalty: {None,l1,l2}. Define regularization type for regression.
                  None: Linear Regression
                  l1: Lasso Regression
                  l2: Ridge Regression
        lam: regularization constant.
        normalize: True = normalize data, False otherwise
        adaptive: True = adaptive learning rate, False = fixed learning rate
        """
        self.alpha = alpha
        self.num_iter = num_iter
        self.early_stop = early_stop
        self.intercept = intercept
        self.init_weight = init_weight
        self.penalty = penalty
        self.lam = lam
        self.normalize = normalize
        self.adaptive = adaptive

    def fit(self, X, y):
        # initialize X, y
        self.X = X
        self.y = np.array([y]).T
        self.maxVal = np.zeros((1, X.shape[1]))
        self.minVal = np.zeros((1, X.shape[1]))

        for i in range(self.X.shape[1]):
            self.maxVal[:, i] = self.X[:, i].max()
            self.minVal[:, i] = self.X[:, i].min()

        self.X = self.X.astype(float)
        # Normalize the data with min-max normalization
        if self.normalize:
           self.X = (self.X - self.minVal) / (self.maxVal - self.minVal)

        # Add bias by concatanating a constant column into X
        if self.intercept:
            ones = np.ones((len(self.X), 1))
            self.X = np.hstack((self.X, ones))

        # initialize coefficient
        self.coef = self.init_weight if self.init_weight is not None \
            else np.array([np.random.uniform(-1, 1, self.X.shape[1])]).T
        # start training, self.loss is used to record losses over iterations
        self.loss = []
        self.gradient_descent()

    def gradient(self):
        cof = -2 / len(self.X)

        # Find prediction and gradient
        pred = self.X @ self.coef
        diff = self.y - pred
        grad = cof * (self.X.T @ (diff))

        # Implement regularization penalty
        if self.penalty == 'l2':
            grad += 2 * self.lam * self.coef
        elif self.penalty == 'l1':
            grad += self.lam * np.sign(self.coef)
            grad[self.coef == 0] = 0
        else:
            pass
        return grad

    def gradient_descent(self):
        print('Start Training')
        for i in range(self.num_iter):
            # calculate prediction y based on current coefficients (self.coef)
            previous_y_hat = self.X @ self.coef
            grad = self.gradient()
            # calculate the new coefficients after incorporating the gradient
            temp_coef = self.coef - self.alpha * grad
            y_pred = self.X @ temp_coef

            # calculate regularization cost (alias: regularization loss) based on
            # self.coef and temp_coef
            if self.penalty == 'l2':
                previous_reg_cost = np.sum(self.coef ** 2)
                current_reg_cost = np.sum(temp_coef ** 2)
            elif self.penalty == 'l1':
                previous_reg_cost = np.sum(np.abs(self.coef))
                current_reg_cost = np.sum(np.abs(temp_coef))
            else:
                previous_reg_cost = 0
                current_reg_cost = 0

            # Calculate error (alias: loss) using sum squared loss
            # and add regularization cost
            pre_error = np.sum(np.square(previous_y_hat - self.y)) + previous_reg_cost
            current_error = np.sum(np.square(y_pred - self.y)) + current_reg_cost

            # Early Stop: early stop is triggered if loss is not decreasing
            # for some number of iterations
            if len(self.loss) > self.early_stop and \
                    self.loss[-1] >= max(self.loss[-self.early_stop:]):
                print('-----------------------')
                print(f'End Training (Early Stopped at iteration {i})')
                return self

            # Implement adaptive learning rate
            # Rules: if current error is smaller than previous error,
            # multiply the current learning rate by 1.3 and update coefficients,
            # otherwise by 0.9 and do nothing with coefficients
            
            if current_error < pre_error:
                self.alpha *= 1.3 if self.adaptive else self.alpha
                self.coef = temp_coef if self.adaptive else self.coef
            else:
                self.alpha *= 0.9 if self.adaptive else self.alpha

            # record stats
            self.loss.append(float(current_error))
            if i % 1000000 == 0:
                print('-----------------------')
                print('Iteration: ' + str(i))
                print('Coef: ' + str(self.coef))
                print('Loss: ' + str(current_error))
        print('-----------------------')
        print('End Training')
        return self

    def predict(self, X):
        X_norm = np.zeros(X.shape)
        for i in range(X.shape[1]):
            X_norm[:, i] = (X[:, i] - self.minVal[:, i]) / (self.maxVal[:, i] - self.minVal[:, i])
        X = X_norm
        
        # add bias 
        if self.intercept:
            ones = np.ones((len(X), 1))
            X = np.hstack((X, ones))

        # Find the model's predictions
        y = X @ self.coef
        return y