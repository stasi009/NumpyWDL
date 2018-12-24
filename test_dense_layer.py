from dense_layer import DenseLayer
import numpy as np
from tqdm import tqdm


def calc_numeric_grads(variable, epsilon, loss_fn):
    numeric_grad = np.zeros_like(variable)

    if len(variable.shape) == 2:
        pbar = tqdm(total=variable.shape[0] * variable.shape[1])
        for r in range(variable.shape[0]):
            for c in range(variable.shape[1]):
                variable[r, c] -= epsilon
                # 最终的loss选择用np.sum，从而prev_grads是全1矩阵
                neg_loss = loss_fn()

                variable[r, c] += 2 * epsilon
                # 最终的loss选择用np.sum，从而prev_grads是全1矩阵
                pos_loss = loss_fn()

                numeric_grad[r, c] = (pos_loss - neg_loss) / (2 * epsilon)

                variable[r, c] -= epsilon  # restore to original
                pbar.update(1)

    elif len(variable.shape) == 1:
        pbar = tqdm(total=variable.shape[0])
        for r in range(variable.shape[0]):
            variable[r] -= epsilon
            # 最终的loss选择用np.sum，从而prev_grads是全1矩阵
            neg_loss = loss_fn()

            variable[r] += 2 * epsilon
            # 最终的loss选择用np.sum，从而prev_grads是全1矩阵
            pos_loss = loss_fn()

            numeric_grad[r] = (pos_loss - neg_loss) / (2 * epsilon)

            variable[r] -= epsilon  # restore to original
            pbar.update(1)

    else:
        raise ValueError('unsupported shape')

    return numeric_grad


def test_dense_fc_layer():
    batch_size = 3
    fan_in = 4
    fan_out = 2
    epsilon = 1e-6

    # ---------- forward
    layer = DenseLayer(name='test', shape=[fan_in,fan_out], l2reg=0.01)
    X = np.random.randn(batch_size, fan_in)
    y = layer.forward(X)
    assert y.shape == (batch_size, fan_out)

    # ---------- backward
    # 最终的loss选择用np.sum，从而prev_grads是全1矩阵，得到的derived_grads就是本层自身的gradients
    dX = layer.backward(prev_grads=np.ones((batch_size, fan_out)))

    # ---------- test grads on W
    var_grads = [('W', layer._W, layer._dW), ('b', layer._b, layer._db), ('input', X, dX)]
    for name, variable, grad in var_grads:
        print("\n************* checking numerical gradients on '{}', ......".format(name))
        numeric_grad = calc_numeric_grads(variable=variable,
                                          epsilon=epsilon,
                                          loss_fn=lambda: np.sum(layer.forward(X)) + layer.l2reg_loss)

        print("========== derived gradients = \n{}".format(grad))
        print("========== numeric gradients = \n{}".format(numeric_grad))
        is_equal = np.allclose(grad, numeric_grad)
        assert is_equal
        print("Equal = {}".format(is_equal))


if __name__ == "__main__":
    # np.random.seed(999)

    test_dense_fc_layer()
