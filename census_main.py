import logging
import pandas as pd
import argparse
import time

from census_data import Dataset, DENSE_FIELDS, CATEGORY_FIELDS
from census_data import VOCAB_LISTS, AGE_BOUNDARIES
from optimization import Adagrad
from wide_n_deep import WideDeepEstimator
from wide_layer import WideHParams, WideEstimator
from dnn import DeepHParams, DeepEstimator
import utils


class DataSource:
    def __init__(self, batch_size):
        self._train_dataset = Dataset('dataset/train.csv')
        self._test_dataset = Dataset('dataset/test.csv')
        self._batch_size = batch_size

    def train_batches_per_epoch(self):
        return self._train_dataset.get_batch_stream(self._batch_size, n_repeat=1)

    def test_batches_per_epoch(self):
        return self._test_dataset.get_batch_stream(self._batch_size, n_repeat=1)

    @property
    def n_train_examples(self):
        return self._train_dataset.n_examples

    @property
    def n_test_examples(self):
        return self._test_dataset.n_examples


def get_deep_hparams(embed_size, hidden_units, L2, learning_rate):
    dense_fields = [(field, 1) for field in DENSE_FIELDS]

    vocab_infos = []
    for vocab_name in CATEGORY_FIELDS:
        if vocab_name == 'age_buckets':
            vocab_size = len(AGE_BOUNDARIES) + 1
        else:
            vocab_size = len(VOCAB_LISTS[vocab_name])
        vocab_infos.append((vocab_name, vocab_size, embed_size))

    # 第一个field代表field name,第二个field代表vocab name
    # 在这个例子中，因为field与vocab是1:1，所以二者同名
    embed_fields = [(field, field) for field in CATEGORY_FIELDS]

    optimizer = Adagrad(learning_rate)

    return DeepHParams(
        dense_fields=dense_fields,
        vocab_infos=vocab_infos,
        embed_fields=embed_fields,
        hidden_units=hidden_units,
        L2=L2,
        optimizer=optimizer)


if __name__ == "__main__":
    # ************ define command-line-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--estimator")
    parser.add_argument('-n', "--n_epoches", type=int, default=10)
    args = parser.parse_args()

    # ************ prepare
    utils.config_logging('log_{}.log'.format(args.estimator))

    data_source = DataSource(batch_size=32)

    deep_hparams = get_deep_hparams(embed_size=16,
                                    hidden_units=[64, 16],
                                    L2=0.01,
                                    learning_rate=0.001)

    wide_hparams = WideHParams(field_names=CATEGORY_FIELDS,
                               alpha=0.1,
                               beta=1,
                               L1=0.1,
                               L2=0.1)

    # ************ run
    if args.estimator == 'wide_n_deep':
        estimator = WideDeepEstimator(wide_hparams=wide_hparams, deep_hparams=deep_hparams, data_source=data_source)
    elif args.estimator == "deep":
        estimator = DeepEstimator(hparams=deep_hparams, data_source=data_source)
    elif args.estimator == 'wide':
        estimator = WideEstimator(hparams=wide_hparams, data_source=data_source)
    else:
        raise ValueError('unknown estimator type={}'.format(args.estimator))

    start_time = time.time()
    metrics_history = estimator.train(args.n_epoches)
    elapsed = time.time() - start_time

    # ************ display result
    logging.info("\n************** TIME COST **************")
    logging.info('{:.2f} seconds for {} epoches'.format(elapsed, args.n_epoches))
    logging.info('{:.2f} examples per second'.format(
        args.n_epoches * (data_source.n_train_examples + data_source.n_test_examples) / elapsed))

    logging.info("\n************** LEARNING CURVE **************")
    metrics_history = pd.DataFrame(metrics_history)
    logging.info(metrics_history)
    metrics_history.to_csv('learn_curve_{}.csv'.format(args.estimator), index=False)
