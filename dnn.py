import numpy as np
from input_layer import DenseInputCombineLayer
from embedding_layer import EmbeddingCombineLayer
from dense_layer import DenseLayer
from activation import ReLU
import utils
import logging
from collections import namedtuple
from base_estimator import BaseEstimator


class DeepNetwork:
    def __init__(self, dense_fields, vocab_infos, embed_fields, hidden_units, L2, optimizer):
        """
        :param dense_fields: a list of tuple (field_name, field's input-dim)
        :param vocab_infos: a list of tuple, each tuple is (vocab_name, vocab_size, embed_size)
        :param embed_fields: a list of tuple (field_name, vocab_name)
        :param hidden_units: a list of ints, n_units for each hidden layer
        :param L2: L2 regularization for hidden dense layer
        :param optimizer: optimizer instance to update the weights
        """
        self._optimizer = optimizer

        # ***************** dense input layer
        self._dense_combine_layer = DenseInputCombineLayer(dense_fields)

        # ***************** embedding layers
        self._embed_combine_layer = EmbeddingCombineLayer(vocab_infos)
        for field_name, vocab_name in embed_fields:
            self._embed_combine_layer.add_embedding(vocab_name=vocab_name, field_name=field_name)

        self._optimize_layers = [self._embed_combine_layer]

        # ***************** MLP
        prev_out_dim = self._dense_combine_layer.output_dim + self._embed_combine_layer.output_dim

        self._hidden_layers = []
        for layer_idx, n_units in enumerate(hidden_units, start=1):
            # ----------- add hidden layer
            hidden_layer = DenseLayer(name="hidden{}".format(layer_idx), shape=[prev_out_dim, n_units], l2reg=L2)
            self._hidden_layers.append(hidden_layer)
            self._optimize_layers.append(hidden_layer)
            logging.info("{}-th hidden layer, weight shape={}".format(layer_idx, hidden_layer.shape))

            # ----------- add activation layer
            self._hidden_layers.append(ReLU())

            # ----------- update previous dimension
            prev_out_dim = n_units

        # final logit layer
        final_logit_layer = DenseLayer(name="final_logit", shape=[prev_out_dim, 1], l2reg=L2)
        logging.info("final logit layer, weight shape={}".format(final_logit_layer.shape))
        self._hidden_layers.append(final_logit_layer)
        self._optimize_layers.append(final_logit_layer)

    def forward(self, features):
        """
        :param features: dict, mapping from field=>dense ndarray or field=>SparseInput
        :return: logits, [batch_size]
        """
        dense_input = self._dense_combine_layer.forward(features)

        embed_input = self._embed_combine_layer.forward(features)

        X = np.hstack([dense_input, embed_input])

        for hidden_layer in self._hidden_layers:
            X = hidden_layer.forward(X)

        return X.flatten()

    def backward(self, grads2logits):
        """
        :param grads2logits: gradients from loss to logits, [batch_size]
        """
        # ***************** 计算所有梯度
        prev_grads = grads2logits.reshape([-1, 1])  # reshape to [batch_size,1]

        # iterate hidden layers backwards
        for hidden_layer in self._hidden_layers[::-1]:
            prev_grads = hidden_layer.backward(prev_grads)

        col_sizes = [self._dense_combine_layer.output_dim, self._embed_combine_layer.output_dim]
        # 抛弃第一个split，因为其对应的是input，无可优化
        _, grads_for_all_embedding = utils.split_column(prev_grads, col_sizes)

        self._embed_combine_layer.backward(grads_for_all_embedding)

        # ***************** 优化
        # 这个操作必须每次backward都调用，这是因为，尽管dense部分的权重是固定的
        # 但是sparse部分，要优化哪个变量，是随着输入不同而不同的
        all_vars, all_grads2var = {}, {}
        for opt_layer in self._optimize_layers:
            all_vars.update(opt_layer.variables)
            all_grads2var.update(opt_layer.grads2var)

        self._optimizer.update(variables=all_vars, gradients=all_grads2var)


DeepHParams = namedtuple("DeepHParams",
                         ['dense_fields', 'vocab_infos', 'embed_fields', 'hidden_units', 'L2', 'optimizer'])


class DeepEstimator(BaseEstimator):
    def __init__(self, hparams, data_source):
        self._dnn = DeepNetwork(dense_fields=hparams.dense_fields,
                                vocab_infos=hparams.vocab_infos,
                                embed_fields=hparams.embed_fields,
                                hidden_units=hparams.hidden_units,
                                L2=hparams.L2,
                                optimizer=hparams.optimizer)
        super().__init__(data_source)

    def train_batch(self, features, labels):
        # ********* forward
        logits = self._dnn.forward(features)
        pred_probas = 1 / (1 + np.exp(-logits))

        # ********* backward
        grads2logits = pred_probas - labels
        self._dnn.backward(grads2logits)

        return pred_probas

    def predict(self, features):
        logits = self._dnn.forward(features)
        return 1 / (1 + np.exp(-logits))
