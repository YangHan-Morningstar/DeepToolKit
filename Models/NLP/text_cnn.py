from tensorflow.keras import Model
from tensorflow.keras.layers import *


class TextCNN(object):
    def __init__(self, vocab_size, max_len, num_class, embed_dims=128, num_filters=256, dropout=0.5, final_activation="softmax"):
        # 词向量维度
        self.embed_dims = embed_dims
        # 词典大小
        self.vocab_size = vocab_size
        # 卷积核尺寸
        self.filter_sizes = [2, 3, 4]
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

        self.convs_block = [
            Conv1D(self.num_filters, filter_size, activation="relu") for filter_size in self.filter_sizes
        ]

    def embedding(self, x):
        embedding_outputs = Embedding(self.vocab_size, self.embed_dims, input_length=self.max_len)(x)
        return embedding_outputs

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = GlobalMaxPool1D()(x)
        return x

    def get_model(self):
        inputs = Input(shape=(self.max_len, ))

        embedding_outputs = self.embedding(inputs)

        conv_pool_cat_outputs = concatenate([self.conv_and_pool(embedding_outputs, conv) for conv in self.convs_block], axis=1)

        dropout_outputs = Dropout(rate=self.dropout)(conv_pool_cat_outputs)

        outputs = Dense(self.num_class, activation=self.final_activation)(dropout_outputs)

        model = Model(inputs=inputs, outputs=outputs)

        return model
