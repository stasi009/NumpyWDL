import numpy as np
# from sklearn.metrics import roc_auc_score, log_loss
from metrics import logloss as log_loss, auc as roc_auc_score
from tqdm import tqdm
import logging


class BaseEstimator:

    def __init__(self, data_source):
        self._data_source = data_source

    def get_metrics(self, scores, labels, prefix):
        scores = np.asarray(scores)
        labels = np.asarray(labels)

        metrics = {'{}_logloss'.format(prefix): log_loss(y_true=labels, y_pred=scores),
                   '{}_auc'.format(prefix): roc_auc_score(y_true=labels, y_score=scores)}

        pred_labels = (scores > 0.5).astype(int)
        metrics['{}_accuracy'.format(prefix)] = np.sum(pred_labels == labels) / len(labels)

        return metrics

    def train_batch(self, features, labels):
        """
        :param features: dict, field_name ==> dense matrix or SparseInput
        :param labels: [batch_size] ndarray
        :return: [batch_size] ndarray of predicted probabilities in that batch
        """
        raise NotImplementedError()

    def predict(self, features):
        """
        :param features: dict, field_name ==> dense matrix or SparseInput
        :return: [batch_size] ndarray of predicted probabilities in that batch
        """
        raise NotImplementedError()

    def _train_epoch(self):
        scores = []
        labels = []

        batch_stream = self._data_source.train_batches_per_epoch()
        for batch_features, batch_labels in tqdm(batch_stream):
            pred_probas = self.train_batch(batch_features, batch_labels)

            scores.extend(pred_probas)
            labels.extend(batch_labels)

        return self.get_metrics(scores=scores, labels=labels, prefix='train')

    def _eval_epoch(self):
        scores = []
        labels = []

        batch_stream = self._data_source.test_batches_per_epoch()
        for batch_features, batch_labels in tqdm(batch_stream):
            pred_probas = self.predict(batch_features)

            scores.extend(pred_probas)
            labels.extend(batch_labels)

        return self.get_metrics(scores=scores, labels=labels, prefix='test')

    def train(self, n_epochs):
        metrics_history = []
        for epoch_idx in range(n_epochs):
            logging.info("\n=============== {}-th EPOCH".format(epoch_idx + 1))

            metrics = {}
            metrics.update(self._train_epoch())
            metrics.update(self._eval_epoch())

            logging.info(metrics)
            metrics_history.append(metrics)

        return metrics_history
