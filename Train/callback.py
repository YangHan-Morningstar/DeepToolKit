from keras.callbacks import ModelCheckpoint


class CallBack(object):

    # 按照 val_f1 保存模型（与metrics.Metrics搭配使用）
    def get_check_point_callback(self):
        ck_callback = ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_f1:.4f}.hdf5',
                                                      monitor='val_f1',
                                                      mode='max', verbose=2,
                                                      save_best_only=True,
                                                      save_weights_only=True)
