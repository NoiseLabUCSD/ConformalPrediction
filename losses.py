#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 11:06:34 2022

@author: ikhurjekar
"""

import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

def gaussian_nll(y_true, y_pred):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix. """
    n_dims = int(int(y_pred.shape[1])/2)
    mu = y_pred[:, 0:n_dims]
    logsigma = y_pred[:, n_dims:]
    mse = -0.5*K.sum(K.square((y_true-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    #log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood = mse + sigma_trace 
    return K.mean(-log_likelihood)


"""Keras implmementations of quantile loss functions. Version 1 is more stable."""
def QuantileLoss(perc, delta=1e-4):
    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = perc.reshape(1, -1)
    def _qloss(y, pred):
        I = tf.cast(y <= pred, tf.float32)
        d = K.abs(y - pred)
        correction = I * (1 - perc) + (1 - I) * perc
        # huber loss
        huber_loss = K.sum(correction * tf.where(d <= delta, 0.5 * d ** 2 / delta, d - 0.5 * delta), -1)
        # order loss
        q_order_loss = K.sum(K.maximum(0.0, pred[:, :-1] - pred[:, 1:] + 1e-6), -1)
        return huber_loss + q_order_loss
    return _qloss


def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.05, 0.95]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)