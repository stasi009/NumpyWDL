import numpy as np
import initialization


class DenseLayer:
    def __init__(self, name, shape, l2reg=0, init_method='glorot_uniform'):
        self._name = name
        self._l2reg = l2reg

        self._W = initialization.get_global_init(init_method)(shape)
        self._b = initialization.get_global_init('zero')(shape[1])

        self._dW = None
        self._db = None

        self._last_input = None

    def forward(self, X):
        self._last_input = X

        # last_input: [batch_size, fan_in]
        # W: [fan_in, fan_out]
        # b: [fanout]
        # result: [batch_size, fan_out]
        return np.dot(self._last_input, self._W) + self._b

    def backward(self, prev_grads):
        # prev_grads: [batch_size, fan_out]
        assert prev_grads.shape[1] == self._W.shape[1]

        # self._last_input.T: [fan_in, batch_size]
        # prev_grads: [batch_size, fan_out]
        # dW: [fan_in, fan_out], same shape as W
        self._dW = np.dot(self._last_input.T, prev_grads)

        # 加上l2_loss对W的导数
        self._dW += self._l2reg * self._W

        # 把b想像成特殊的fan_in=1的W，则套用上面的公式
        # db = [1,1,...,1](共batch_size个1,shape=[1,batch_size])*prev_grads([batch_size,fan_out])=各列之和([1,fan_out])
        self._db = np.sum(prev_grads, axis=0)

        # return: dLoss/dX: [batch_size, fan_in]
        # prev_grads: [batch_size, fan_out]
        # self._W.T: [fan_out,fan_in]
        return np.dot(prev_grads, self._W.T)

    @property
    def l2reg_loss(self):
        return 0.5 * self._l2reg * np.sum(self._W ** 2)

    @property
    def shape(self):
        return self._W.shape

    @property
    def output_dim(self):
        return self._W.shape[1]

    @property
    def variables(self):
        return {"{}_W".format(self._name): self._W,
                "{}_b".format(self._name): self._b}

    @property
    def grads2var(self):
        return {"{}_W".format(self._name): self._dW,
                "{}_b".format(self._name): self._db}
