import numpy as np


class Sigmoid:
    def __init__(self):
        self._last_forward_result = None

    def forward(self, X):
        """
        element-wise sigmoid
        :param X: [batch_size, #neuron]
        :return: same shape as X
        """
        self._last_forward_result = 1.0 / (1.0 + np.exp(-X))
        return self._last_forward_result

    def backward(self, prev_grads):
        """
        :param prev_grads:  gradients from loss to "last forward result"
                            must have the same shape as 'last forward result'
        :return:    gradients from loss to X, has same shape as X
        """
        assert prev_grads.shape == self._last_forward_result.shape

        return prev_grads * self._last_forward_result * (1 - self._last_forward_result)


class ReLU:
    def __init__(self):
        self._last_input = None

    def forward(self, X):
        self._last_input = X
        return np.maximum(0, X)

    def backward(self, prev_grads):
        assert prev_grads.shape == self._last_input.shape

        local_grads = np.zeros_like(self._last_input)
        local_grads[self._last_input > 0] = 1.0

        return prev_grads * local_grads
