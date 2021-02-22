from tensorflow.keras import Model
from tensorflow.keras.layers import *


class TextDPCNN(object):
    def __init__(self, vocab_size, max_len, num_class, embed_dims=128, num_filters=250, rnn_hidden=256, dropout=0.5, final_activation="softmax"):
        # 词向量维度
        self.embed_dims = embed_dims
        # 词典大小
        self.vocab_size = vocab_size
        # rnn神经元数量
        self.rnn_hidden = rnn_hidden
        # 卷积核数量
        self.num_filters = num_filters
        # dropout
        self.dropout = dropout
        # 最大长度
        self.max_len = max_len
        # 分类数量
        self.num_class = num_class
        # 最后一层的激活函数
        self.final_activation = final_activation

    def embedding(self, x):
        embedding_outputs = Embedding(self.vocab_size, self.embed_dims, input_length=self.max_len)(x)
        return embedding_outputs

    def conv_region(self, x):
        x = Conv1D(self.num_filters, 3, activation="relu")(x)
        return x

    def _block(self, x):
        x = ZeroPadding1D(padding=(0, 1))(x)
        px = MaxPool1D(3, 2)(x)
        x = ZeroPadding1D(padding=1)(px)
        x = ReLU()(x)
        x = Conv1D(self.num_filters, 3)(x)
        x = ZeroPadding1D(padding=1)(x)
        x = ReLU()(x)
        x = Conv1D(self.num_filters, 3)(x)
        x = x + px
        return x

    def get_model(self):
        inputs = Input(shape=(self.max_len, ))

        embedding_outputs = self.embedding(inputs)

        conv_region_outputs = self.conv_region(embedding_outputs)

        conv_region_outputs_padding = ZeroPadding1D(padding=1)(conv_region_outputs)
        conv_region_outputs_padding = ReLU()(conv_region_outputs_padding)

        conv_outputs = Conv1D(self.num_filters, 3, padding="same", activation="relu")(conv_region_outputs_padding)
        conv_outputs = Conv1D(self.num_filters, 3)(conv_outputs)

        res_add_outputs = conv_region_outputs + conv_outputs
        while res_add_outputs.shape[1] > 2:
            res_add_outputs = self._block(res_add_outputs)
        res_add_outputs = Flatten()(res_add_outputs)

        outputs = Dense(self.num_class, activation=self.final_activation)(res_add_outputs)

        model = Model(inputs=inputs, outputs=outputs)

        return model