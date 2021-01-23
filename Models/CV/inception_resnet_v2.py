from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class InceptionResnetV2Model:

    def __init__(self, label_num, img_rows=299, img_cols=299):

        self.RESNET_V2_A_COUNT = 0
        self.RESNET_V2_B_COUNT = 0
        self.RESNET_V2_C_COUNT = 0

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.label_num = label_num

    def resnet_v2_stem(self, x_input):

        with K.name_scope("stem"):
            x = Conv2D(32, (3, 3), activation="relu", strides=(2, 2))(x_input)
            x = Conv2D(32, (3, 3), activation="relu")(x)
            x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)

            x1 = MaxPooling2D((3, 3), strides=(2, 2))(x)
            x2 = Conv2D(96, (3, 3), activation="relu", strides=(2, 2))(x)

            x = concatenate([x1, x2], axis=-1)

            x1 = Conv2D(64, (1, 1), activation="relu", padding="same")(x)
            x1 = Conv2D(96, (3, 3), activation="relu")(x1)

            x2 = Conv2D(64, (1, 1), activation="relu", padding="same")(x)
            x2 = Conv2D(64, (7, 1), activation="relu", padding="same")(x2)
            x2 = Conv2D(64, (1, 7), activation="relu", padding="same")(x2)
            x2 = Conv2D(96, (3, 3), activation="relu", padding="valid")(x2)

            x = concatenate([x1, x2], axis=-1)

            x1 = Conv2D(192, (3, 3), activation="relu", strides=(2, 2))(x)

            x2 = MaxPooling2D((3, 3), strides=(2, 2))(x)

            x = concatenate([x1, x2], axis=-1)

            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
        return x

    def inception_resnet_v2_A(self, x_input, scale_residual=True):

        self.RESNET_V2_A_COUNT += 1
        with K.name_scope('inception_resnet_v2_A' + str(self.RESNET_V2_A_COUNT)):
            ar1 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)

            ar2 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
            ar2 = Conv2D(32, (3, 3), activation="relu", padding="same")(ar2)

            ar3 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
            ar3 = Conv2D(48, (3, 3), activation="relu", padding="same")(ar3)
            ar3 = Conv2D(64, (3, 3), activation="relu", padding="same")(ar3)

            merged = concatenate([ar1, ar2, ar3], axis=-1)

            ar = Conv2D(384, (1, 1), activation="linear", padding="same")(merged)
            if scale_residual:
                ar = Lambda(lambda a: a * 0.1)(ar)

            x = add([x_input, ar])
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
        return x

    def inception_resnet_v2_B(self, x_input, scale_residual=True):

        self.RESNET_V2_B_COUNT += 1
        with K.name_scope('inception_resnet_v2_B' + str(self.RESNET_V2_B_COUNT)):
            br1 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)

            br2 = Conv2D(128, (1, 1), activation="relu", padding="same")(x_input)
            br2 = Conv2D(160, (1, 7), activation="relu", padding="same")(br2)
            br2 = Conv2D(192, (7, 1), activation="relu", padding="same")(br2)

            merged = concatenate([br1, br2], axis=-1)

            br = Conv2D(1152, (1, 1), activation="linear", padding="same")(merged)
            if scale_residual:
                br = Lambda(lambda b: b * 0.1)(br)

            x = add([x_input, br])
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
        return x

    def inception_resnet_v2_C(self, x_input, scale_residual=True):

        self.RESNET_V2_C_COUNT += 1
        with K.name_scope('inception_resnet_v2_C' + str(self.RESNET_V2_C_COUNT)):
            cr1 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)

            cr2 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)
            cr2 = Conv2D(224, (1, 3), activation="relu", padding="same")(cr2)
            cr2 = Conv2D(256, (3, 1), activation="relu", padding="same")(cr2)

            merged = concatenate([cr1, cr2], axis=-1)

            cr = Conv2D(2144, (1, 1), activation="linear", padding="same")(merged)
            if scale_residual:
                cr = Lambda(lambda c: c * 0.1)(cr)

            x = add([x_input, cr])
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
        return x

    def reduction_resnet_v2_A(self, x_input):

        with K.name_scope('reduction_resnet_A'):
            ra1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

            ra2 = Conv2D(384, (3, 3), activation='relu', strides=(2, 2), padding='valid')(x_input)

            ra3 = Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
            ra3 = Conv2D(256, (3, 3), activation='relu', padding='same')(ra3)
            ra3 = Conv2D(384, (3, 3), activation='relu', strides=(2, 2), padding='valid')(ra3)

            merged_vector = concatenate([ra1, ra2, ra3], axis=-1)

            x = BatchNormalization(axis=-1)(merged_vector)
            x = Activation('relu')(x)
        return x

    def reduction_resnet_v2_B(self, x_input):

        with K.name_scope('reduction_resnet_v2_B'):
            rbr1 = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x_input)

            rbr2 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
            rbr2 = Conv2D(384, (3, 3), activation="relu", strides=(2, 2))(rbr2)

            rbr3 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
            rbr3 = Conv2D(288, (3, 3), activation="relu", strides=(2, 2))(rbr3)

            rbr4 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
            rbr4 = Conv2D(288, (3, 3), activation="relu", padding="same")(rbr4)
            rbr4 = Conv2D(320, (3, 3), activation="relu", strides=(2, 2))(rbr4)

            merged = concatenate([rbr1, rbr2, rbr3, rbr4], axis=-1)
            rbr = BatchNormalization(axis=-1)(merged)
            rbr = Activation("relu")(rbr)
        return rbr

    def get_model(self, scale=True):

        init = Input((self.img_rows, self.img_cols, 3,))

        # Input shape is 299 * 299 * 3
        x = self.resnet_v2_stem(init)

        # 5 x Inception A
        for i in range(5):
            x = self.inception_resnet_v2_A(x, scale_residual=scale)

        # Reduction A
        x = self.reduction_resnet_v2_A(x)

        # 10 x Inception B
        for i in range(10):
            x = self.inception_resnet_v2_B(x, scale_residual=scale)

        # Reduction B
        x = self.reduction_resnet_v2_B(x)

        # 5 x Inception C
        for i in range(5):
            x = self.inception_resnet_v2_C(x, scale_residual=scale)

        # Average Pooling
        x = AveragePooling2D((8, 8))(x)

        # Dropout
        x = Dropout(0.2)(x)
        x = Flatten()(x)

        # Output layer
        output = Dense(units=self.label_num, activation="softmax")(x)

        model = Model(init, output, name="Inception-ResNet-v2")

        return model
