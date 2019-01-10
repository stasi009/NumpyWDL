import numpy as np
from collections import defaultdict


class FtrlEstimator:
    """
    每个field对应一个FtrlEstimator。类比于在TensorFlow WDL中，一个feature column对应一个FtrlEstimator
    """

    def __init__(self, alpha, beta, L1, L2):
        self._alpha = alpha
        self._beta = beta
        self._L1 = L1
        self._L2 = L2

        self._n = defaultdict(float)  # n[i]: i-th feature's squared sum of past gradients
        self._z = defaultdict(float)
        self._w = {}  # lazy weights

        self._current_feat_ids = None
        self._current_feat_vals = None

    def predict_logit(self, feature_ids, feature_values):
        """
        :param feature_ids: non-zero feature ids for one example
        :param feature_values: non-zero feature values for one example
        :return: logit for this example
        """
        self._current_feat_ids = feature_ids
        self._current_feat_vals = feature_values

        logit = 0
        self._w.clear() # lazy weights，所以没有必要保留之前的weights

        # 如果当前样本在这个field下所有feature都为0，则feature_ids==feature_values==[]
        # 则没有以下循环，logit=0
        for feat_id, feat_val in zip(feature_ids, feature_values):
            z = self._z[feat_id]
            sign_z = -1. if z < 0 else 1.

            # build w on the fly using z and n, hence the name - lazy weights
            # this allows us for not storing the complete w
            # if abs(z) <= self._L1: self._w[feat_id] = 0.  # w[i] vanishes due to L1 regularization
            if abs(z) > self._L1:
                # apply prediction time L1, L2 regularization to z and get w
                w = (sign_z * self._L1 - z) / ((self._beta + np.sqrt(self._n[feat_id])) / self._alpha + self._L2)
                self._w[feat_id] = w
                logit += w * feat_val

        return logit

    def update(self, pred_proba, label):
        """
        :param pred_proba:  与last_feat_ids/last_feat_vals对应的预测CTR
                            注意pred_proba并不一定等于sigmoid(predict_logit(...))，因为要还要考虑deep侧贡献的logit
        :param label:       与last_feat_ids/last_feat_vals对应的true label
        """
        grad2logit = pred_proba - label

        # 如果当前样本在这个field下所有feature都为0，则没有以下循环，没有更新
        for feat_id, feat_val in zip(self._current_feat_ids, self._current_feat_vals):
            g = grad2logit * feat_val
            g2 = g * g
            n = self._n[feat_id]

            self._z[feat_id] += g

            if feat_id in self._w: # if self._w[feat_id] != 0
                sigma = (np.sqrt(n + g2) - np.sqrt(n)) / self._alpha
                self._z[feat_id] -= sigma * self._w[feat_id]

            self._n[feat_id] = n + g2
