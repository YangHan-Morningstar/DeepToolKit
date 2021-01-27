from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os
import datetime
import shutil


class CallBack(object):

    def check_target_path(self, target_path):
        if not os.path.exists(target_path):
            os.makedirs(target_path)

    def get_check_point_callback(self, model_target_path, monitor, mode):
        self.check_target_path(model_target_path)

        ck_callback = ModelCheckpoint(model_target_path + '/weights.{epoch:02d}-{' + monitor + ':.4f}.hdf5',
                                                      monitor=monitor,
                                                      mode=mode, verbose=2,
                                                      save_best_only=True,
                                                      save_weights_only=True)

        return ck_callback

    def get_tensorboard_callback(self):
        shutil.rmtree("./logs")
        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        return tensorboard_callback
