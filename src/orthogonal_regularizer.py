from __future__ import absolute_import
import six
from keras import backend as K
from theano import tensor as TT
from keras.regularizers import Regularizer
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object

class OrthogonalL1L2(Regularizer):
    """Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        xTx = K.dot(K.transpose(x), x)
        eye_like_xTx = K.zeros_like(xTx)
        eye_like_xTx = TT.fill_diagonal(eye_like_xTx, 1)
        mat_to_reg = xTx - eye_like_xTx
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(mat_to_reg))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(mat_to_reg))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}