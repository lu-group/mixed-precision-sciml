#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:23:39 2023

@author: jayoxborn
"""

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import mixed_precision

dde.config.enable_xla_jit(mode = False)
dde.config.set_default_float('float16')

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


# def get_data(path):
#     x_train = np.load(path + "X_train.npy").astype(np.float16)  # (40800, 20, 47)
#     y_train = np.load(path + "Y_train.npy").astype(np.float16)  # (40800, 111, 47)
#     x_test = np.load(path + "X_valid.npy").astype(np.float16)  # (10000, 20, 47)
#     y_test = np.load(path + "Y_valid.npy").astype(np.float16)  # (10000, 111, 47)

#     # rt = np.load(path + "rt.npy")  # (20,) uniform
#     ry = np.load(path + "ry.npy")  # (47,) nonuniform
#     rx = np.load(path + "rx.npy")  # (111,) uniform
#     xx, yy = np.meshgrid(rx, ry, indexing="ij")  # (111, 47)
#     grid = np.vstack((xx.ravel(), yy.ravel())).T

#     x_train = (x_train.reshape(-1, 20 * 47), grid)
#     y_train = y_train.reshape(-1, 111 * 47)
#     x_test = (x_test.reshape(-1, 20 * 47), grid)
#     y_test = y_test.reshape(-1, 111 * 47)
#     return dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

def get_data(path):
    x_train = np.load(path + "X_train.npy").astype(np.float16)  # (40800, 20, 47)
    y_train = np.load(path + "Y_train.npy").astype(np.float16)  # (40800, 111, 47)
    x_test = np.load(path + "X_valid.npy").astype(np.float16)  # (10000, 20, 47)
    y_test = np.load(path + "Y_valid.npy").astype(np.float16)  # (10000, 111, 47)

    # rt = np.load(path + "rt.npy")  # (20,) uniform
    ry = np.load(path + "ry.npy")  # (47,) nonuniform
    rx = np.load(path + "rx.npy")  # (111,) uniform
    xx, yy = np.meshgrid(rx, ry, indexing="ij")  # (111, 47)
    grid = np.vstack((xx.ravel(), yy.ravel())).T

    # Slice x_test and y_test to only one data point
    x_train = (x_train.reshape(-1, 20 * 47), grid)
    y_train = y_train.reshape(-1, 111 * 47)
    x_test = (x_test[0].reshape(1, 20 * 47), grid)  # Keep the first data point and reshape it
    y_test = y_test[0].reshape(1, 111 * 47)  # Keep the first data point and reshape it

    return dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)

class PODDeepONet(dde.maps.NN):
    def __init__(
        self,
        pod_basis,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = dde.maps.activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = dde.maps.activations.get(
                activation
            )

        self.pod_basis = tf.convert_to_tensor(pod_basis, dtype=tf.float16)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = dde.maps.FNN(
                layer_sizes_branch, activation_branch, kernel_initializer
            )
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = dde.maps.FNN(
                layer_sizes_trunk, self.activation_trunk, kernel_initializer
            )
            self.b = tf.Variable(tf.zeros(1))

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]

        x_func = self.branch(x_func)
        if self.trunk is None:
            # POD only
            x = tf.einsum("bi,ni->bn", x_func, self.pod_basis)
        else:
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = tf.einsum("bi,ni->bn", x_func, tf.concat((self.pod_basis, x_loc), 1))
            x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x

def weighted_mse_loss(y_true, y_pred):
    A = tf.math.reduce_max(y_true, axis=1)
    A = tf.expand_dims(A, axis=-1)  # Add a new axis to match y_true - y_pred shape
    
    subtracted = (y_true - y_pred)/10
    
    # Apply loss scaling during computation
    loss_scaled = tf.math.reduce_mean(tf.math.square(subtracted/tf.sqrt(A)), axis=1)
    
    return tf.math.reduce_mean(loss_scaled)*100

def main():
    path = "instability_wave/"
    # path = "/home/lulu/Dropbox/DeepONet_FNO/instability_wave/"
    data = get_data(path)

    # pca = PCA(n_components=0.999).fit(data.train_y)
    w_train = np.load(path + "W_train.npy")[:, None]  # (40800,)
    pca = PCA(n_components=0.99).fit(data.train_y * w_train)
    print("# Components:", pca.n_components_)
    print(np.cumsum(pca.explained_variance_ratio_))
    # plt.figure()
    # plt.imshow(pca.mean_.reshape(111, 47))
    # plt.colorbar()
    # plt.figure()
    # for i in range(3):
    #     plt.subplot(1, 3, i + 1)
    #     plt.imshow(pca.components_[i].reshape(111, 47) * (111 * 47) ** 0.5)
    #     plt.colorbar()
    # plt.show()

    m = 20 * 47
    # activation = "tanh"
    activation = "elu"
    # branch = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.InputLayer(input_shape=(m,)),
    #         tf.keras.layers.Reshape((20, 47, 1)),
    #         tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation=activation),
    #         tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation=activation),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128, activation=activation),
    #         tf.keras.layers.Dense(pca.n_components_),
    #     ]
    # )
    # branch.summary()
    net = PODDeepONet(
        pca.components_.T * (111 * 47) ** 0.5,
        # [m, branch],
        [m, 256, 256, 256, 256, 256, pca.n_components_],
        None,
        activation,
        "Glorot normal",
    )

    def output_transform(inputs, outputs):
        # return outputs / pca.n_components_ + pca.mean_
        return outputs + pca.mean_

    net.apply_output_transform(output_transform)

    save_better_only = dde.callbacks.ModelCheckpoint(filepath = "/home/jhayford/Python_codes/FAIR-COMPARISON", save_better_only = True)
    callbacks = [save_better_only]

   

    model = dde.Model(data, net)
    adam = tf.keras.optimizers.Adam(epsilon=1e-04)
    model.compile(
        optimizer = adam,
        lr=1e-5,
        loss = weighted_mse_loss,
        decay=("inverse time", 1, 1e-6),
        # metrics = [weighted_mse_loss],
    )
    losshistory, train_state = model.train(epochs=5000000, display_every= 5000000, batch_size=1250, callbacks = callbacks)
    
    ################################################################
    # Evaluate tests with different noise amplitudes
    ################################################################
    output = open("deeponet_POD_errs_noise_mix_1e5_bs1250_1.dat", "w")
    amplitudes = [0.0, 0.001, 0.01, 0.1, 0.2]
    Nm = 300
    Na = len(amplitudes)
    acc_err = np.zeros((Nm, Na))
    for ii in range(Nm):
        ev = np.load(path + f"tests/eig_vec_{ii:02}.npy")
        pd = np.load(path + f"tests/prof_data_{ii:02}.npy")

        for ia, amp in enumerate(amplitudes):
            if ia == 0:
                eta0 = np.random.standard_normal(np.shape(ev))
                eta = np.max(ev) * eta0

            noise = amp * eta
            evn = ev + noise

            grid = data.train_x[1]
            x = (evn.astype(np.float32).reshape(1, 20 * 47), grid)
            po = model.predict(x).reshape(1, 111, 47)

            err = np.mean((po - pd) ** 2)
            rel = np.sqrt(err / np.mean(pd ** 2))
            acc_err[ii, ia] = rel

    errs = acc_err.mean(axis=0)
    bars = acc_err.std(axis=0)
    for ia in range(Na):
        print(errs[ia], bars[ia], file=output)
    output.close()


if __name__ == "__main__":
    main()
memory_used = tf.config.experimental.get_memory_info('GPU:0')['peak']
print("memory used:", memory_used)