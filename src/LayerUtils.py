import cntk as C
import numpy as np


def lrelu(input, leak=0.2, name=""):
    return C.param_relu(C.constant((np.ones(input.shape)*leak).astype(np.float32)), input, name=name)


def bn(input, activation=None, name=""):
    if activation is not None:
        x = C.layers.BatchNormalization(map_rank=1, name=name+"_bn" if name else "")(input)
        x = activation(x, name=name)
    else:
        x = C.layers.BatchNormalization(map_rank=1, name=name)(input)
    return x


def conv(input, filter_shape, num_filters, strides=(1,1), init=C.he_normal(), activation=None, pad=True, name=""):
    return C.layers.Convolution(filter_shape, num_filters, strides=strides, pad=pad, activation=activation, init=init, bias=False, name=name)(input)


def conv_bn(input, filter_shape, num_filters, strides=(1,1), init=C.he_normal(), activation=None, name=""):
    x = conv(input, filter_shape, num_filters, strides, init, name=name+"_conv" if name else "")
    x = bn(x, activation, name=name)
    return x


def conv_bn_lrelu(input, filter_shape, num_filters, strides=(1,1), init=C.he_normal(), name=""):
    return conv_bn(input, filter_shape, num_filters, strides, init, activation=C.leaky_relu, name=name)


def flatten(input, name=""):
    assert (len(input.shape) == 3)
    return C.reshape(input, input.shape[0]*input.shape[1]* input.shape[2], name=name)


def bi_recurrence(input, fwd, bwd, name=""):
    F = C.layers.Recurrence(fwd, go_backwards=False, name='fwd_rnn')(input)
    B = C.layers.Recurrence(bwd, go_backwards=True, name='bwd_rnn')(input)
    h = C.splice(F, B, name=name)
    return h
