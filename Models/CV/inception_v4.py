from keras.layers import *
from keras.models import Model
from keras import backend as K


class InceptionV4:

    def __init__(self, img_rows, img_cols, label_num):

        self.CONV_BLOCK_COUNT = 0
        self.INCEPTION_A_COUNT = 0
        self.INCEPTION_B_COUNT = 0
        self.INCEPTION_C_COUNT = 0

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.label_num = label_num

    def conv_block(self, x, nb_filters, nb_row, nb_col, strides=(1, 1), padding='same', use_bias=False):

        self.CONV_BLOCK_COUNT += 1
        with K.name_scope('conv_block_' + str(self.CONV_BLOCK_COUNT)):
            x = Conv2D(filters=nb_filters, kernel_size=(nb_row, nb_col), strides=strides, padding=padding,
                       use_bias=use_bias)(x)
            x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
            x = Activation("relu")(x)
        return x

    def stem(self, x_input):
        with K.name_scope('stem'):
            x = self.conv_block(x_input, 32, 3, 3, strides=(2, 2), padding='valid')
            x = self.conv_block(x, 32, 3, 3, padding='valid')
            x = self.conv_block(x, 64, 3, 3)

            x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
            x2 = self.conv_block(x, 96, 3, 3, strides=(2, 2), padding='valid')

            x = concatenate([x1, x2], axis=-1)

            x1 = self.conv_block(x, 64, 1, 1)
            x1 = self.conv_block(x1, 96, 3, 3, padding='valid')

            x2 = self.conv_block(x, 64, 1, 1)
            x2 = self.conv_block(x2, 64, 7, 1)
            x2 = self.conv_block(x2, 64, 1, 7)
            x2 = self.conv_block(x2, 96, 3, 3, padding='valid')

            x = concatenate([x1, x2], axis=-1)

            x1 = self.conv_block(x, 192, 3, 3, strides=(2, 2), padding='valid')
            x2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

            merged_vector = concatenate([x1, x2], axis=-1)
        return merged_vector

    def inception_A(self, x_input):

        self.INCEPTION_A_COUNT += 1
        with K.name_scope('inception_A' + str(self.INCEPTION_A_COUNT)):
            averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(
                x_input)
            averagepooling_conv1x1 = self.conv_block(averagepooling_conv1x1, 96, 1, 1)

            conv1x1 = self.conv_block(x_input, 96, 1, 1)

            conv1x1_3x3 = self.conv_block(x_input, 64, 1, 1)
            conv1x1_3x3 = self.conv_block(conv1x1_3x3, 96, 3, 3)

            conv3x3_3x3 = self.conv_block(x_input, 64, 1, 1)
            conv3x3_3x3 = self.conv_block(conv3x3_3x3, 96, 3, 3)
            conv3x3_3x3 = self.conv_block(conv3x3_3x3, 96, 3, 3)

            merged_vector = concatenate([averagepooling_conv1x1, conv1x1, conv1x1_3x3, conv3x3_3x3],
                                        axis=-1)
        return merged_vector

    def inception_B(self, x_input):

        self.INCEPTION_B_COUNT += 1
        with K.name_scope('inception_B' + str(self.INCEPTION_B_COUNT)):
            averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
            averagepooling_conv1x1 = self.conv_block(averagepooling_conv1x1, 128, 1, 1)

            conv1x1 = self.conv_block(x_input, 384, 1, 1)

            conv1x7_1x7 = self.conv_block(x_input, 192, 1, 1)
            conv1x7_1x7 = self.conv_block(conv1x7_1x7, 224, 1, 7)
            conv1x7_1x7 = self.conv_block(conv1x7_1x7, 256, 1, 7)

            conv2_1x7_7x1 = self.conv_block(x_input, 192, 1, 1)
            conv2_1x7_7x1 = self.conv_block(conv2_1x7_7x1, 192, 1, 7)
            conv2_1x7_7x1 = self.conv_block(conv2_1x7_7x1, 224, 7, 1)
            conv2_1x7_7x1 = self.conv_block(conv2_1x7_7x1, 224, 1, 7)
            conv2_1x7_7x1 = self.conv_block(conv2_1x7_7x1, 256, 7, 1)

            merged_vector = concatenate([averagepooling_conv1x1, conv1x1, conv1x7_1x7, conv2_1x7_7x1], axis=-1)
        return merged_vector

    def inception_C(self, x_input):

        self.INCEPTION_C_COUNT += 1
        with K.name_scope('Inception_C' + str(self.INCEPTION_C_COUNT)):
            averagepooling_conv1x1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
            averagepooling_conv1x1 = self.conv_block(averagepooling_conv1x1, 256, 1, 1)

            conv1x1 = self.conv_block(x_input, 256, 1, 1)

            conv3x3_1x1 = self.conv_block(x_input, 384, 1, 1)
            conv3x3_1 = self.conv_block(conv3x3_1x1, 256, 1, 3)
            conv3x3_2 = self.conv_block(conv3x3_1x1, 256, 3, 1)

            conv2_3x3_1x1 = self.conv_block(x_input, 384, 1, 1)
            conv2_3x3_1x1 = self.conv_block(conv2_3x3_1x1, 448, 1, 3)
            conv2_3x3_1x1 = self.conv_block(conv2_3x3_1x1, 512, 3, 1)
            conv2_3x3_1x1_1 = self.conv_block(conv2_3x3_1x1, 256, 3, 1)
            conv2_3x3_1x1_2 = self.conv_block(conv2_3x3_1x1, 256, 1, 3)

            merged_vector = concatenate(
                [averagepooling_conv1x1, conv1x1, conv3x3_1, conv3x3_2, conv2_3x3_1x1_1, conv2_3x3_1x1_2], axis=-1)
        return merged_vector

    def reduction_A(self, x_input):
        with K.name_scope('Reduction_A'):

            maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

            conv3x3 = self.conv_block(x_input, 384, 3, 3, strides=(2, 2), padding='valid')

            conv2_3x3 = self.conv_block(x_input, 192, 1, 1)
            conv2_3x3 = self.conv_block(conv2_3x3, 224, 3, 3)
            conv2_3x3 = self.conv_block(conv2_3x3, 256, 3, 3, strides=(2, 2), padding='valid')

            merged_vector = concatenate([maxpool, conv3x3, conv2_3x3], axis=-1)
        return merged_vector

    def reduction_B(self, x_input):

        with K.name_scope('Reduction_B'):
            maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

            conv3x3 = self.conv_block(x_input, 192, 1, 1)
            conv3x3 = self.conv_block(conv3x3, 192, 3, 3, strides=(2, 2), padding='valid')

            conv1x7_7x1_3x3 = self.conv_block(x_input, 256, 1, 1)
            conv1x7_7x1_3x3 = self.conv_block(conv1x7_7x1_3x3, 256, 1, 7)
            conv1x7_7x1_3x3 = self.conv_block(conv1x7_7x1_3x3, 320, 7, 1)
            conv1x7_7x1_3x3 = self.conv_block(conv1x7_7x1_3x3, 320, 3, 3, strides=(2, 2), padding='valid')

            merged_vector = concatenate([maxpool, conv3x3, conv1x7_7x1_3x3], axis=-1)
        return merged_vector

    def get_model(self):
        x_input = Input(shape=(self.img_rows, self.img_cols, 3,))
        # Stem
        x = self.stem(x_input)

        # 4 x Inception_A
        for i in range(4):
            x = self.inception_A(x)

        # Reduction_A
        x = self.reduction_A(x)

        # 7 x Inception_B
        for i in range(7):
            x = self.inception_B(x)

        # Reduction_B
        x = self.reduction_B(x)

        # Average Pooling
        x = AveragePooling2D(pool_size=(8, 8))(x)

        # dropout
        x = Dropout(0.2)(x)
        x = Flatten()(x)

        # 全连接层
        x = Dense(units=self.label_num, activation='softmax')(x)
        model = Model(inputs=x_input, outputs=x, name='Inception-V4')

        return model