import numpy as np


class Adagrad:
    def __init__(self, lr):
        self._lr = lr
        # variable name => sum of gradient square (also a vector)
        self._sum_grad2 = {}

    def update(self, variables, gradients):
        for gradname, gradient in gradients.items():
            # ------ update cache
            g2 = gradient * gradient
            if gradname in self._sum_grad2:
                self._sum_grad2[gradname] += g2
            else:
                self._sum_grad2[gradname] = g2

            # ------ calculate delta
            delta = self._lr * gradient / (np.sqrt(self._sum_grad2[gradname]) + 1e-6)

            # ------ update
            if '@' in gradname:
                # 对应着稀疏输入的权重与梯度，gradients中的key遵循着'vocab_name@feat_id'的格式
                varname, row = gradname.split('@')
                row = int(row)

                variable = variables[varname]
                variable[row, :] -= delta
            else:
                variable = variables[gradname]
                variable -= delta






