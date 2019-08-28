import numpy as np


class Controller:
    def __init__(self, input_size, no_actions, params):
        self.input_size = input_size
        self.no_actions = no_actions
        self.bias = np.array(params[:self.no_actions])
        self.weights = np.array(params[self.no_actions:]).reshape(self.input_size, self.no_actions)

    def forward(self, inputs):
        action = np.tanh(np.dot(inputs, self.weights) + self.bias)[0]
        action[1] = (action[1] + 1.0) / 2.0
        action[2] = np.minimum(np.maximum(action[2], 0.0), 1.0)
        return action


def get_random_model_params(param_count, std_dev):
    return np.random.standard_cauchy(param_count) * std_dev