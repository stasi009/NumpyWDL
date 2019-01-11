import pandas as pd
import sys
import numpy as np
import random
import utils
from input_layer import SparseInput
import bisect
import argparse
from tqdm import tqdm

VOCAB_LISTS = {
    'education': ['Bachelors',
                  'HS-grad',
                  '11th',
                  'Masters',
                  '9th',
                  'Some-college',
                  'Assoc-acdm',
                  'Assoc-voc',
                  '7th-8th',
                  'Doctorate',
                  'Prof-school',
                  '5th-6th',
                  '10th',
                  '1st-4th',
                  'Preschool',
                  '12th'],

    'marital_status': ['Married-civ-spouse',
                       'Divorced',
                       'Married-spouse-absent',
                       'Never-married',
                       'Separated',
                       'Married-AF-spouse',
                       'Widowed'],

    'relationship': ['Husband',
                     'Not-in-family',
                     'Wife',
                     'Own-child',
                     'Unmarried',
                     'Other-relative'],

    'workclass': ['Self-emp-not-inc',
                  'Private',
                  'State-gov',
                  'Federal-gov',
                  'Local-gov',
                  'Self-emp-inc',
                  'Without-pay',
                  'Never-worked'],

    'occupation': ['Tech-support',
                   'Craft-repair',
                   'Other-service',
                   'Sales',
                   'Exec-managerial',
                   'Prof-specialty',
                   'Handlers-cleaners',
                   'Machine-op-inspct',
                   'Adm-clerical',
                   'Farming-fishing',
                   'Transport-moving',
                   'Priv-house-serv',
                   'Protective-serv',
                   'Armed-Forces']
}

VOCAB_MAPPINGS = {field: {featname: idx for idx, featname in enumerate(featnames)} for field, featnames in
                  VOCAB_LISTS.items()}

AGE_BOUNDARIES = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]

DENSE_FIELDS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

DENSE_LOG_MEAN_STD = {'age': (3.6183599219864133, 0.35003117354646957),
                      'education_num': (2.372506496597371, 0.27381608590073075),
                      'capital_gain': (0.7346209104536965, 2.4547377400238553),
                      'capital_loss': (0.35030508122367104, 1.5845809727578963),
                      'hours_per_week': (3.665366478972777, 0.38701441353280025)}

CATEGORY_FIELDS = ['education', 'marital_status', 'relationship', 'workclass', 'occupation', 'age_buckets']


class Dataset:
    def __init__(self, infname):
        with open(infname, "rt") as fin:
            self._field_names = fin.readline().strip().split(',')
            self._lines = [line.strip() for line in fin]

    @property
    def n_examples(self):
        return len(self._lines)

    def parse_line(self, line):
        contents = dict(zip(self._field_names, line.split(',')))
        features = {}

        # ------------- label
        label = int(contents['income_bracket'] == '>50K')

        # ------------- categorical features
        for field in ['education', 'marital_status', 'relationship', 'workclass', 'occupation']:
            vocab_mapping = VOCAB_MAPPINGS[field]
            txt_value = contents[field]
            if txt_value in vocab_mapping:
                # 找不到的，算缺失，不包含进特征
                features[field] = vocab_mapping[txt_value]

        age = int(contents['age'])
        features['age_buckets'] = bisect.bisect(AGE_BOUNDARIES, age)

        # ------------- numeric features
        for field in DENSE_FIELDS:
            raw_value = float(contents[field])
            logmean, logstd = DENSE_LOG_MEAN_STD[field]
            features[field] = (np.log1p(raw_value) - logmean) / logstd

        return features, label

    def get_batch_stream(self, batch_size, n_repeat=1):
        n_repeat = n_repeat if n_repeat > 0 else sys.maxsize

        for _ in range(n_repeat):
            random.shuffle(self._lines)

            for batch_lines in utils.chunk(self._lines, batch_size):
                Xs = {}
                ys = []

                # ------------- allocate for categorical feature
                for field in CATEGORY_FIELDS:
                    Xs[field] = SparseInput(n_total_examples=len(batch_lines),
                                            example_indices=[],
                                            feature_ids=[],
                                            feature_values=[])

                # ------------- allocate for numeric feature
                for field in DENSE_FIELDS:
                    # Xs[field]应该是一个list of list
                    # 外面的list，对应batch中的每个example
                    # 内层的list，对应该样本在field下的值。
                    # 某样本可以在某个field下有多个dense值，比如当你非要用OHE来表示categorical特征的时候
                    # 只不过，这里每个样本在每个field下只有一个值
                    Xs[field] = []

                # ------------- loop and add
                for example_index, line in enumerate(batch_lines):
                    # 顺序遍历，能够保证插入SparseInput中的非零元是按example_index从小到大排好序的
                    current_features, label = self.parse_line(line)
                    ys.append(label)

                    # add categorical feature
                    for field in CATEGORY_FIELDS:
                        if field in current_features:
                            Xs[field].add(example_idx=example_index,
                                          feat_id=current_features[field],
                                          feat_val=1)

                    # add numeric feature
                    for field in DENSE_FIELDS:
                        # wrap into one-element list, since we need to add one row
                        Xs[field].append([current_features[field]])

                yield Xs, np.asarray(ys)


def precompute_log_mean_stddev():
    df = pd.read_csv('dataset/train.csv', usecols=DENSE_FIELDS)
    df = np.log1p(df)  # 数据有长尾, log使之更像正态一些

    means = df.mean()
    stddevs = df.std()
    log_means_stddevs = {field: (means[field], stddevs[field]) for field in DENSE_FIELDS}


def test_standardize(infname):
    print("\n============= standardize '{}'".format(infname))

    df = pd.read_csv(infname, usecols=DENSE_FIELDS)
    df = np.log1p(df)

    means = pd.Series({field: mean for field, (mean, std) in DENSE_LOG_MEAN_STD.items()})
    stddevs = pd.Series({field: std for field, (mean, std) in DENSE_LOG_MEAN_STD.items()})

    df = (df - means) / stddevs
    print(df.describe().loc[['mean', 'std'], :])


def test_batch_stream(infname):
    dataset = Dataset(infname)

    batch_stream = dataset.get_batch_stream(16)

    for batch_idx, (features, labels) in enumerate(batch_stream, start=1):
        print("\n================== {}-th batch".format(batch_idx))
        print("labels: {}\n".format(labels))

        for field in DENSE_FIELDS:
            print("[{}]: {}".format(field, features[field]))

        for field in CATEGORY_FIELDS:
            sp_input = features[field]
            print("\n[{}] example_indices: {}".format(field, sp_input._example_indices))
            print("[{}] feature_ids: {}".format(field, sp_input._feature_ids))
            print("[{}] feature_values: {}".format(field, sp_input._feature_values))


def clean_datas(infname, outfname):
    csv_columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'gender',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
        'income_bracket'
    ]

    with open(infname, 'rt') as fin, open(outfname, 'wt') as fout:
        # write header
        fout.write(",".join(csv_columns) + "\n")

        for line in tqdm(fin):
            line = line.strip()
            line = line.replace(', ', ',')
            if not line or ',' not in line:
                continue
            if line[-1] == '.':
                line = line[:-1]
            line += '\n'
            fout.write(line)
    print("'{}' is cleaned, and re-save to '{}'".format(infname, outfname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', "--job")
    args = parser.parse_args()

    if args.job == "clean":
        clean_datas(infname='dataset/raw_train.txt', outfname='dataset/train.csv')
        clean_datas(infname='dataset/raw_test.txt', outfname='dataset/test.csv')

    else:
        raise ValueError('unknown job={}'.format(args.job))
