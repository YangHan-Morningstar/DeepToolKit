from tensorflow.keras import Model
from tensorflow.keras.layers import *


class TextRNN(object):
    def __init__(self, vocab_size, max_len, num_class, embed_dims=128, rnn_hidden=256, dropout=0.5):
        # 词向量维度
        self.embed_dims = embed_dims
        # 词典大小
        self.vocab_size = vocab_size
        # rnn神经元数量
        self.rnn_hidden = rnn_hidden
        # dropout
        self.dropout = dropout
        # 最大长度
        self.max_len = max_len
        # 分类数量
        self.num_class = num_class

    def embedding(self, x):
        embedding_outputs = Embedding(self.vocab_size, self.embed_dims, input_length=self.max_len)(x)
        return embedding_outputs

    def get_model(self):
        inputs = Input(shape=(self.max_len, ))

        embedding_outputs = self.embedding(inputs)

        bilstm_outputs = Bidirectional(LSTM(self.rnn_hidden, return_sequences=True, dropout=self.dropout))(embedding_outputs)
        bilstm_outputs = Bidirectional(LSTM(self.rnn_hidden, return_sequences=True, dropout=self.dropout))(bilstm_outputs)

        dropout_outputs = Dropout(rate=self.dropout)(bilstm_outputs)
        dropout_outputs = dropout_outputs[:, -1, :]

        outputs = Dense(self.num_class, activation="softmax")(dropout_outputs)

        model = Model(inputs=inputs, outputs=outputs)

        return model