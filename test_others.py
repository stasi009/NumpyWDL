import initialization
import numpy as np
import activation
from tqdm import tqdm
import bisect
import utils


def numerical_gradient(variable, loss_fn, epsilon):
    # gradients must have the same shape as variable
    numeric_grad = np.zeros_like(variable)

    pbar = tqdm(total=variable.shape[0] * variable.shape[1])
    for r in range(variable.shape[0]):
        for c in range(variable.shape[1]):
            variable[r, c] -= epsilon
            neg_loss = loss_fn(variable)

            variable[r, c] += 2 * epsilon
            pos_loss = loss_fn(variable)

            numeric_grad[r, c] = (pos_loss - neg_loss) / (2 * epsilon)

            variable[r, c] -= epsilon  # restore to original
            pbar.update(1)

    return numeric_grad


def check_activation(layer):
    # ---------- forward and backward
    X = np.random.randn(3, 4)
    _ = layer.forward(X)
    # 最终的loss选择用np.sum，从而prev_grads是全1矩阵，得到的derived_grads就是本层自身的gradients
    derived_grads = layer.backward(prev_grads=np.ones_like(X))

    # ---------- calculate numeric gradients
    epsilon = 1e-6
    numeric_grads = numerical_gradient(variable=X,
                                       loss_fn=lambda x: np.sum(layer.forward(x)),
                                       epsilon=epsilon)

    # ---------- display
    print("========== derived gradients = \n{}".format(derived_grads))
    print("========== numeric gradients = \n{}".format(numeric_grads))

    # ---------- check
    is_equal = np.allclose(numeric_grads, derived_grads)
    assert is_equal
    print("Equal = {}".format(is_equal))


def test_activations():
    check_activation(activation.Sigmoid())
    check_activation(activation.ReLU())


def test_initializer():
    # ---------------- GlorotUniform
    init_glorot_uniform = initialization.get_global_init('glorot_uniform')
    w = init_glorot_uniform([2, 3])
    print("\nGlorotUniform")
    print(w)
    print(w.sum())

    # ---------------- GlorotNormal
    init_glorot_normal = initialization.get_global_init('glorot_normal')
    w = init_glorot_uniform([20, 50])
    print("\nGlorotNormal")
    # print(w)
    print(w.mean())


# def test_bce_loss_with_logits():
#     bce_layer = BinaryCrossEntropy4Logits()
#
#     # 必须转化成float，否则logits[idx] -= epsilon这种inplace modification时，因为logits本身还是正数
#     # 会导致logits[idx]变化后还是整数，比如1-1e-6会被强制转化成0
#     logits = np.asarray([1, 2, 3], dtype=float)
#     labels = [1, 1, 0]
#
#     loss = bce_layer.forward(logits=logits, labels=labels)
#     print("loss={}".format(loss))
#
#     grad2logits = bce_layer.backward()
#     print("          gradients to logits = {}".format(grad2logits))
#
#     # --------- numeric loss
#     epsilon = 1e-6
#     numeric_grads = np.zeros_like(logits)
#
#     for idx in range(len(logits)):
#         logits[idx] -= epsilon
#         neg_loss = bce_layer.forward(logits=logits, labels=labels)
#
#         logits[idx] += 2 * epsilon
#         pos_loss = bce_layer.forward(logits=logits, labels=labels)
#
#         numeric_grads[idx] = (pos_loss - neg_loss) / (2 * epsilon)
#
#         logits[idx] -= epsilon  # restore to original
#
#     print("numerical gradients to logits = {}".format(numeric_grads))
#
#     # --------- check
#     is_equal = np.allclose(grad2logits, len(logits) * numeric_grads)
#     assert is_equal
#     print("Equal = {}".format(is_equal))


def test_bucket_by_bisect():
    age_boundaries = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]

    for age in [5, 18, 31, 55, 42, 67]:
        idx = bisect.bisect(age_boundaries, age)

        if idx == 0:
            lb = '-inf'
        else:
            lb = age_boundaries[idx - 1]

        if idx == len(age_boundaries):
            hb = 'inf'
        else:
            hb = age_boundaries[idx]

        print("{}<={}<{}".format(lb, age, hb))


def test_split_columns():
    a = np.arange(12).reshape(2, 6)
    print(a)

    splits = utils.split_column(a,[2,1,3])
    for idx, split in enumerate(splits,start=1):
        print("\n--------- {}th split".format(idx))
        print(split)


if __name__ == "__main__":
    np.random.seed(999)

    # test_activations()
    # test_initializer()
    # test_bce_loss_with_logits()
    # test_bucket_by_bisect()
    test_split_columns()
