import numpy as np
from embedding_layer import EmbeddingLayer
from input_layer import SparseInput


def test_embedding_forward():
    layer = EmbeddingLayer(W=np.arange(12).reshape(4, 3), vocab_name=None, field_name=None)

    X = SparseInput(example_indices=[2, 1, 2],
                    feature_ids=[2, 3, 2],
                    feature_values=[1, 2, 2],
                    n_total_examples=5)

    output = layer.forward(X)
    print(output)


def test_embedding_backward():
    layer = EmbeddingLayer(W=np.random.randn(4, 3), vocab_name=None, field_name=None)

    X = SparseInput(example_indices=[1, 1, 2, 3, 3, 3],
                    feature_ids=[0, 3, 1, 2, 1, 0],
                    feature_values=[1, 2, 2, 1, 1, 2],
                    n_total_examples=5)
    output = layer.forward(X)

    grads2W = layer.backward(np.ones((X.n_total_examples, 3)))
    print("========== derived gradients = \n{}".format(grads2W))

    # ----------- calculate numeric gradients
    epsilon = 1e-6
    variable = layer._W
    numeric_grads = np.zeros_like(variable)

    for r in range(variable.shape[0]):
        for c in range(variable.shape[1]):
            variable[r, c] -= epsilon
            neg_delta_loss = np.sum(layer.forward(X))

            variable[r, c] += 2 * epsilon
            pos_delta_loss = np.sum(layer.forward(X))

            numeric_grads[r, c] = (pos_delta_loss - neg_delta_loss) / (2 * epsilon)

            variable[r, c] -= epsilon  # restore to original

    print("========== numeric gradients = \n{}".format(numeric_grads))


if __name__ == "__main__":
    np.random.seed(999)

    # test_embedding_forward()
    test_embedding_backward()
