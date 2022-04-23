# Import libraries
import numpy as np
import requests
from scipy.stats import multivariate_normal
import re
import copy
np.random.seed(3)


class EM:
    def __init__(self, total_components: int = 2, weights_init = None, features_size: int = 2):
        """Initialize parameter of EM, total_components -> number of latent variables,
        features_size -> number of features/dimensions of data to consider """

        self.n_iter = np.inf
        self.x = None
        self.t_features = features_size
        self.total_components = total_components

        if weights_init is None:
            self.weights_init = np.random.uniform(0, 1, size=total_components)
            self.weights_init = self.weights_init / self.weights_init.sum() # normalize weights to add to 1 in total
        else:
            self.weights_init = weights_init
            self.weights_init = self.weights_init / self.weights_init.sum()

        self.mu_init = []
        self.sigma_init = []

        for i in range(total_components): # initialize mu and sigma for every component
            self.mu_init.append(np.random.uniform(0, 1, size=features_size))
            sigma = np.random.uniform(0, 1, size=(features_size, features_size))
            self.sigma_init.append(sigma @ sigma.T) # to make semi-positive definite

        # final theta values
        self.final_mu = None
        self.final_sigma = None
        self.final_weights = None

    def get_data(self, api_calls: int = 30):
        """Get data through the API"""

        x = []
        for i in range(api_calls):
            r = requests.get("https://24zl01u3ff.execute-api.us-west-1.amazonaws.com/beta")
            x.append(np.array(re.findall(r"\d", r.json()['body']), "int"))

        self.x = np.array(x)

    def init_EM(self):
        """Reinitialize theta parameters"""

        self.weights_init = np.random.uniform(0, 1, size=self.total_components)
        self.weights_init = self.weights_init / self.weights_init.sum()

        self.mu_init = []
        self.sigma_init = []

        for i in range(self.total_components):
            self.mu_init.append(np.random.uniform(0, 1, size=self.t_features))
            sigma = np.random.uniform(0, 1, size=(self.t_features, self.t_features))
            self.sigma_init.append(sigma @ sigma.T)

    def train(self, max_iterations: int = 200):
        """Train model using EM algorithm, max_iterations -> max iterations for which the algorithm should run"""

        if self.x is None:
            self.get_data()

        if self.t_features == 1:
            x_mean = self.x.mean(axis=1, keepdims=True) # take probability of head for every draw

        elif self.t_features == 2:
            one_col = self.x.mean(axis=1, keepdims=True)
            x_mean = np.hstack([one_col, 1 - one_col])      # take probability of head/tail for every draw in two columns

        else:
            x_mean = self.x[:, np.min(self.x.shape[1], self.t_features)]

        last = np.inf
        mu = copy.deepcopy(self.mu_init)
        sigma = copy.deepcopy(self.sigma_init)
        weights = copy.deepcopy(self.weights_init)

        for i in range(max_iterations):

            # E-step
            gamma = []
            for tc in range(self.total_components):
                # gamma_n_k
                gamma.append(weights[tc] * multivariate_normal(mu[tc],
                                                               sigma[tc],
                                                               allow_singular=True).pdf(x_mean))

            gamma = np.array(gamma)
            pyx = np.log(gamma)
            pyx_norm = np.log(np.sum(np.exp(pyx), axis=0, keepdims=True))
            pyx = np.exp(pyx - pyx_norm)
            last_1 = gamma.mean()

            gamma = pyx #gamma / gamma.sum(axis=0, keepdims=True)

            if np.abs(last_1 - last) < 1e-20:   # converge criterai
                print(f"Ended at {i}")
                self.n_iter = i
                break

            # M-step
            for tc in range(self.total_components):
                weights[tc] = gamma[tc].mean()

            for tc in range(self.total_components):
                sigma[tc] = (x_mean - mu[tc]).T @ (gamma[tc][..., np.newaxis] * (x_mean - mu[tc])) / np.sum(gamma[tc])
                mu[tc] = x_mean.T @ gamma[tc] / np.sum(gamma[tc])

            last = last_1

        print(f"""Theta:\n
        mu: {mu}\n
        sigma: {sigma}\n
        weights: {weights}\n""")

        # assign final theta parameters after training
        self.final_mu = mu
        self.final_sigma = sigma
        self.final_weights = weights

    def predict(self, data):
        """Predict class membership"""
        if (self.x is None) or (self.final_mu is None):
            print("Model not trained yet")
            return None

        data = np.array(data)
        if self.t_features == 1:
            x_mean = data.mean(axis=1, keepdims=True)

        elif self.t_features == 2:
            one_col = data.mean(axis=1, keepdims=True)
            x_mean = np.hstack([one_col, 1 - one_col])

        else:
            x_mean = data[:, np.min(self.x.shape[1], self.t_features)]

        gamma = []

        for i in range(self.total_components):
            gamma.append(self.final_weights[i] * multivariate_normal(self.final_mu[i],
                                                                     self.final_sigma[i],
                                                                     allow_singular=True).pdf(x_mean))

        gamma = np.array(gamma)

        return np.argmax(gamma, axis=0) + 1


if __name__ == "__main__":
    em = EM(2, features_size=1)
    em.get_data()
    em.train()
