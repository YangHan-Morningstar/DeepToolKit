from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops, math_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.keras.metrics import Metric

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
import tensorflow as tf


class SelfValMetrics(Callback):
    def __init__(self, valid_data):
        super(SelfValMetrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return

class F1Score(Metric):
    def __init__(self,
               thresholds=None,
               top_k=None,
               class_id=None,
               name=None,
               dtype=None):
        super(F1Score, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.precision_true_positives = self.add_weight(
            'precision_true_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.precision_false_positives = self.add_weight(
            'precision_false_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.recall_true_positives = self.add_weight(
            'recall_true_positives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)
        self.recall_false_negatives = self.add_weight(
            'recall_false_negatives',
            shape=(len(self.thresholds),),
            initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.precision_true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.precision_false_positives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.recall_true_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.recall_false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)


    def result(self):
        precision_result = math_ops.div_no_nan(self.precision_true_positives,
                                     self.precision_true_positives + self.precision_false_positives)

        recall_result = math_ops.div_no_nan(self.recall_true_positives,
                                     self.recall_true_positives + self.recall_false_negatives)

        precision_value = precision_result[0] if len(self.thresholds) == 1 else precision_result
        recall_value = recall_result[0] if len(self.thresholds) == 1 else recall_result
        return 2 * (precision_value * recall_value) / (precision_value + recall_value)

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])


class SelfF1(Metric):
    def __init__(self, average="macro", name="self_f1", **kwargs):
        super(SelfF1, self).__init__(name=name, **kwargs)
        self.f1_value = self.add_weight(name="sf", initializer="zeros", dtype="float64")
        self.average = average

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1,))
        if len(y_true.shape) == 2 and y_true.shape[1] != 1:
            y_true = tf.reshape(tf.argmax(y_true, axis=1), shape=(-1,))

        f1_value = f1_score(y_true=y_true.numpy(), y_pred=y_pred.numpy(), average=self.average)

        self.f1_value.assign(f1_value)

    def result(self):
        return self.f1_value

    def reset_states(self):
        self.f1_value.assign(0.0)


class SelfPrecision(Metric):
    def __init__(self, average="macro", name="self_precision", **kwargs):
        super(SelfPrecision, self).__init__(name=name, **kwargs)
        self.precision_value = self.add_weight(name="sp", initializer="zeros", dtype="float64")
        self.average = average

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1,))
        if len(y_true.shape) == 2 and y_true.shape[1] != 1:
            y_true = tf.reshape(tf.argmax(y_true, axis=1), shape=(-1,))

        precision_value = precision_score(y_true=y_true.numpy(), y_pred=y_pred.numpy(), average=self.average)

        self.precision_value.assign(precision_value)

    def result(self):
        return self.precision_value

    def reset_states(self):
        self.precision_value.assign(0.0)


class SelfRecall(Metric):
    def __init__(self, average="macro", name="self_recall", **kwargs):
        super(SelfRecall, self).__init__(name=name, **kwargs)
        self.recall_value = self.add_weight(name="sr", initializer="zeros", dtype="float64")
        self.average = average

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1,))
        if len(y_true.shape) == 2 and y_true.shape[1] != 1:
            y_true = tf.reshape(tf.argmax(y_true, axis=1), shape=(-1,))

        recall_value = recall_score(y_true=y_true.numpy(), y_pred=y_pred.numpy(), average=self.average)

        self.recall_value.assign(recall_value)

    def result(self):
        return self.recall_value

    def reset_states(self):
        self.recall_value.assign(0.0)
