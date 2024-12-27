#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 14:58:19 2023

@author: jayoxborn
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import mixed_precision

import deepxde as dde
from deepxde.backend import tf

dde.config.enable_xla_jit(mode = False)
dde.config.set_default_float('float16')

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

def get_data(filename):
    nx = 40
    nt = 40
    data = np.load(filename)
    x = data["x"].astype(np.float16)
    t = data["t"].astype(np.float16)
    u = data["u"].astype(np.float16)  # N x nt x nx

    u0 = u[:, 0, :]  # N x nx
    xt = np.vstack((np.ravel(x), np.ravel(t))).T
    u = u.reshape(-1, nt * nx)
    return (u0, xt), u


def main():
    nt = 40
    nx = 40
    x_train, y_train = get_data("train_IC2.npz")
    x_test, y_test = get_data("test_IC2.npz")
    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

    net = dde.maps.DeepONetCartesianProd(
        [nx, 512, 512], [2, 512, 512, 512], "relu", "Glorot normal"
    )

    model = dde.Model(data, net)
    adam = tf.keras.optimizers.Adam(epsilon=1e-04)
    model.compile(
        "adam",
        lr=1e-3,
        decay=("inverse time", 1, 1e-4),
        metrics=["mean l2 relative error"],
        loss = "mean l2 relative error",
    )
    # IC1
    # losshistory, train_state = model.train(epochs=100000, batch_size=None)
    # IC2
    losshistory, train_state = model.train(epochs=250000, batch_size=None)

    y_pred = model.predict(data.test_x)
    np.savetxt("y_pred_deeponet.dat", y_pred[0].reshape(nt, nx))
    np.savetxt("y_true_deeponet.dat", data.test_y[0].reshape(nt, nx))
    np.savetxt("y_error_deeponet.dat", (y_pred[0] - data.test_y[0]).reshape(nt, nx))


if __name__ == "__main__":
    main()
memory_used = tf.config.experimental.get_memory_info('GPU:0')['peak']
print("memory used:", memory_used)
