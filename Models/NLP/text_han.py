from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


class TextHAN(object):
    def __init__(self, maxlen_sentence, maxlen_word, vocab_size, embedding_dims, num_class, rnn_hidden=128, dropout=0.5, final_activation='softmax'):
        '''
        适用于长文本分类
        # 句子的maxlen必须等于maxlen_sentence * maxlen_word
        :param maxlen_sentence: 长句子分为多少段
        :param maxlen_word: 长句子分成的每个段中有多少个词
        :param vocab_size: 词典大小
        :param embedding_dims: 词向量维度
        :param num_class: 标签数目
        :param rnn_hidden: LSTM或GRU层神经元数目
        :param dropout: dropout
        :param final_activation: 最后一层的激活函数
        '''
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.vocab_size = vocab_size
        self.embedding_dims = embedding_dims
        self.num_class = num_class
        self.rnn_hidden = rnn_hidden
        self.dropout = dropout
        self.final_activation = final_activation

    def embedding(self, x):
        embedding_outputs = Embedding(self.vocab_size, self.embedding_dims)(x)
        return embedding_outputs

    def get_model(self):
        # Word part
        input_word = Input(shape=(self.maxlen_word, ))
        x_word = self.embedding(input_word)
        x_word = Bidirectional(LSTM(self.rnn_hidden, return_sequences=True, dropout=self.dropout))(x_word)
        x_word = Attention(self.maxlen_word)(x_word)
        model_word = Model(input_word, x_word)

        # Sentence part
        input = Input(shape=(self.maxlen_sentence * self.maxlen_word, ))
        input_reshape = Reshape(target_shape=(self.maxlen_sentence, self.maxlen_word))(input)
        x_sentence = TimeDistributed(model_word)(input_reshape)
        x_sentence = Bidirectional(LSTM(self.rnn_hidden, return_sequences=True, dropout=self.dropout))(x_sentence)
        x_sentence = Attention(self.maxlen_sentence)(x_sentence)
        output = Dense(self.num_class, activation=self.final_activation)(x_sentence)
        model = Model(inputs=input, outputs=output)
        return model
