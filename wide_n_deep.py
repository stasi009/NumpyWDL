import numpy as np
from dnn import DeepNetwork
from wide_layer import WideLayer
from base_estimator import BaseEstimator


class WideDeepEstimator(BaseEstimator):
    def __init__(self, wide_hparams, deep_hparams, data_source):
        self._current_deep_logits = None

        self._wide_layer = WideLayer(field_names=wide_hparams.field_names,
                                     alpha=wide_hparams.alpha,
                                     beta=wide_hparams.beta,
                                     L1=wide_hparams.L1,
                                     L2=wide_hparams.L2,
                                     proba_fn=self._predict_proba)

        self._dnn = DeepNetwork(dense_fields=deep_hparams.dense_fields,
                                vocab_infos=deep_hparams.vocab_infos,
                                embed_fields=deep_hparams.embed_fields,
                                hidden_units=deep_hparams.hidden_units,
                                L2=deep_hparams.L2,
                                optimizer=deep_hparams.optimizer)

        super().__init__(data_source)

    def _predict_proba(self, example_idx, wide_logit):
        deep_logit = self._current_deep_logits[example_idx]
        logit = deep_logit + wide_logit
        return 1 / (1 + np.exp(-logit))

    def train_batch(self, features, labels):
        self._current_deep_logits = self._dnn.forward(features)

        pred_probas = self._wide_layer.train(features, labels)

        self._dnn.backward(grads2logits=pred_probas - labels)

        return pred_probas

    def predict(self, features):
        deep_logits = self._dnn.forward(features)

        wide_logits = self._wide_layer.predict_logits(features)

        logits = deep_logits + wide_logits

        return 1 / (1 + np.exp(-logits))
