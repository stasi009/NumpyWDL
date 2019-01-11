import numpy as np
from initialization import TruncatedNormal
import utils


class EmbeddingLayer:
    """
    简化起见，不支持use_bias和regularization
    不支持regularization的原因是：weight是稠密的，自然L2 Loss的gradient也是稠密的
    为了L2 Loss而破坏稀疏性，增加内容与耗时，有些得不偿失
    一种改进方案是：只正则化本batch中用到的embedding向量
    """

    def __init__(self, W, vocab_name, field_name):
        """
        :param W: dense weight matrix, [vocab_size,embed_size]
        :param b: bias, [embed_size]
        """
        self.vocab_name = vocab_name
        self.field_name = field_name
        self._W = W
        self._last_input = None

    @property
    def output_dim(self):
        return self._W.shape[1]

    def forward(self, X):
        """
        :param X: SparseInput
        :return: [batch_size, embed_size]
        """
        self._last_input = X

        # output: [batch_size, embed_size]
        output = np.zeros((X.n_total_examples, self._W.shape[1]))

        for example_idx, feat_id, feat_val in X.iterate_non_zeros():
            embedding = self._W[feat_id, :]
            output[example_idx, :] += embedding * feat_val

        return output

    def backward(self, prev_grads):
        """
        :param prev_grads: [batch_size, embed_size]
        :return: dw
        """
        dW = {}

        for example_idx, feat_id, feat_val in self._last_input.iterate_non_zeros():
            # [1,embed_size]
            grad_from_one_example = prev_grads[example_idx, :] * feat_val

            if feat_id in dW:
                dW[feat_id] += grad_from_one_example

            else:
                dW[feat_id] = grad_from_one_example

        return dW


class EmbeddingCombineLayer:
    def __init__(self, vocab_infos):
        """
        :param vocab_infos: a list of tuple, each tuple is (vocab_name, vocab_size, embed_size)
        """
        self._weights = {}  # vocab_name ==> weight
        for vocab_name, vocab_size, embed_size in vocab_infos:
            # TruncatedNormal是TF WDL中embedding_column的默认初始化方式
            # These values are similar to values from a `random_normal_initializer`
            # except that values more than two standard deviations from the mean are discarded and re-drawn
            stddev = 1 / np.sqrt(embed_size)
            initializer = TruncatedNormal(mean=0,
                                          stddev=stddev,
                                          lower=-2 * stddev,
                                          upper=2 * stddev)
            self._weights[vocab_name] = initializer(shape=[vocab_size, embed_size])

        # 注意，由于embedding input的稀疏性，一次回代时，不太可能对所有embedding weight有梯度
        # 而是只针对某个field的embedding weight中某feature id对应的行有梯度
        # _grads_to_embed是一个dict,
        # key是"vocab_name@feature_id"的形式，value是一个[embed_size]的ndarray。
        # 因为vocab的weight是多个field所共享的，所以value是每个field对vocab_name@feature_id的梯度的叠加
        self._grads_to_embed = {}
        self._embed_layers = []

    def add_embedding(self, vocab_name, field_name):
        weight = self._weights[vocab_name]
        layer = EmbeddingLayer(W=weight, vocab_name=vocab_name, field_name=field_name)
        self._embed_layers.append(layer)

    @property
    def output_dim(self):
        return sum(layer.output_dim for layer in self._embed_layers)

    def forward(self, sparse_inputs):
        """
        :param sparse_inputs: dict {field_name: SparseInput}
        :return:    每个SparseInput贡献一个embedding vector，返回结果是这些embedding vector的拼接
                    拼接顺序由add_embedding的调用顺序决定
        """
        embedded_outputs = []
        for embed_layer in self._embed_layers:
            sp_input = sparse_inputs[embed_layer.field_name]
            embedded_outputs.append(embed_layer.forward(sp_input))

        # [batch_size, sum of all embed-layer's embed_size]
        return np.hstack(embedded_outputs)

    def backward(self, prev_grads):
        """
        :param prev_grads:  [batch_size, sum of all embed-layer's embed_size]
                            上一层传入的, Loss对本层输出的梯度
        """
        assert prev_grads.shape[1] == self.output_dim

        # 因为output是每列输出的拼接，自然上一层输入的导数也是各层所需要导数的拼接
        # prev_grads_splits是一个数组，存储对应各层的导数
        col_sizes = [layer.output_dim for layer in self._embed_layers]
        prev_grads_splits = utils.split_column(prev_grads, col_sizes)

        self._grads_to_embed.clear()  # reset
        for layer, layer_prev_grads in zip(self._embed_layers, prev_grads_splits):
            # layer_prev_grads: 上一层传入的，Loss对某个layer的输出的梯度
            # layer_grads_to_feat_embed: dict, feat_id==>grads，
            # 由这一个layer造成对某vocab的embedding矩阵的某feat_id对应行的梯度
            layer_grads_to_embed = layer.backward(layer_prev_grads)

            for feat_id, g in layer_grads_to_embed.items():
                # 表示"对某个vocab的embedding weight中的第feat_id行的总导数"
                key = "{}@{}".format(layer.vocab_name, feat_id)

                if key in self._grads_to_embed:
                    self._grads_to_embed[key] += g
                else:
                    self._grads_to_embed[key] = g

    @property
    def variables(self):
        """ 优化变量
        :return: dict from vocab_name to weight matrix
        """
        return self._weights

    @property
    def grads2var(self):
        """ Loss对优化变量的梯度
        :return: dict, key是"vocab_name@feature_id"的形式，value是一个[embed_size]的ndarray
        """
        return self._grads_to_embed

    @property
    def l2reg_loss(self):
        return 0  # 出于保持稀疏的考虑，在embedding层暂不支持正则
