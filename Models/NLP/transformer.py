import tensorflow as tf
from tensorflow.keras import layers


class Transformer(object):

    def __init__(self, maxlen, vocab_size, embed_dim, num_heads, ff_dim, label_num):
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.projection_dim = self.embed_dim // self.num_heads
        self.label_num = label_num

    def token_and_position_embedding(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)(positions)

        x = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)(x)

        output = x + positions
        return output

    def multi_head_self_attention(self, x):
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding dimension = {self.embed_dim} should be divisible by number of heads = {self.num_heads}"
            )

        batch_size = tf.shape(x)[0]
        query_dense = layers.Dense(self.embed_dim)
        key_dense = layers.Dense(self.embed_dim)
        value_dense = layers.Dense(self.embed_dim)
        combine_heads = layers.Dense(self.embed_dim)

        query = query_dense(x)  # (batch_size, seq_len, embed_dim)
        key = key_dense(x)  # (batch_size, seq_len, embed_dim)
        value = value_dense(x)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)

        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)

        return output

    def transformer_block(self, x):
        attn_output = self.multi_head_self_attention(x)
        attn_output = layers.Dropout(0.1)(attn_output, training=True)

        out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = layers.Dropout(0.1)(ffn_output, training=True)

        output = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

        return output

    def ffn(self, x):
        output = tf.keras.Sequential(
            [layers.Dense(self.ff_dim, activation="relu"), layers.Dense(self.embed_dim), ]
        )(x)

        return output

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def get_model(self):
        inputs = layers.Input(shape=(self.maxlen,))

        x = self.token_and_position_embedding(inputs)
        x = self.transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)

        outputs = layers.Dense(self.label_num, activation="softmax")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return model
