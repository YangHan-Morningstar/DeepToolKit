from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


class TextRCNN(object):

    def __init__(self, vocab_size, max_len, num_class, embed_dims=128, rnn_hidden=256, dropout=0.5, final_activation="softmax"):
        # Bert的隐藏层数量
        self.embed_dims = embed_dims
        # 词典大小
        self.vocab_size = vocab_size
        # RNN隐层层数量
        self.rnn_hidden = rnn_hidden
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

    def get_model(self):
        inputs = Input((self.max_len,))

        embedding_outputs = self.embedding(inputs)

        forward_lstm_outputs = LSTM(self.rnn_hidden, return_sequences=True, go_backwards=False, dropout=self.dropout)(
                embedding_outputs)
        reverse_lstm_outputs = LSTM(self.rnn_hidden, return_sequences=True, go_backwards=True, dropout=self.dropout)(
                embedding_outputs)

        cat_outputs = concatenate([forward_lstm_outputs, embedding_outputs, reverse_lstm_outputs], axis=-1)

        dense_1_outputs = Dense(self.rnn_hidden * 2 + self.embed_dims, activation="tanh")(cat_outputs)

        global_maxpooling1d_outputs = GlobalMaxPool1D()(dense_1_outputs)

        outputs = Dense(self.num_class, activation=self.final_activation)(global_maxpooling1d_outputs)

        model = Model(inputs=inputs, outputs=outputs)

        return model