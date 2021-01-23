from tensorflow.keras.callbacks import ModelCheckpoint
import os


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
