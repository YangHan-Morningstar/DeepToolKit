from tensorflow.keras import backend as K


class SelfCategoricalCrossEntropy(object):
    '''
    修正的交叉熵损失函数
    margin：控制训练边界，正样本大于margin时反向传播梯度为0
    '''
    def __init__(self, margin):
        self.margin = margin
        self.theta = lambda t: (K.sign(t) + 1.) / 2.

    def loss_function(self, y_true, y_pred):
        return - (1 - self.theta(y_true - self.margin) * self.theta(y_pred - self.margin)
                  - self.theta(1 - self.margin - y_true) * self.theta(1 - self.margin - y_pred)
                  ) * (y_true * K.log(y_pred + 1e-8) + (1 - y_true) * K.log(1 - y_pred + 1e-8))
