"""
Created by  
@author Jacob Goldman-Wetzler

Modified by 
@author Joel Hayford
"""

# ================  Load libraries and dependencies  =========================

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
SEED=0xdde
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
dde.config.set_random_seed(SEED)

# ============================================================================
# ===========  Create and train the float32 and float16 models  ==============
# ============================================================================

def f1(x):
    return x * np.sin(5 * x)

dde.config.set_default_float('float32')

geom = dde.geometry.Interval(-1, 1)
data = dde.data.Function(geom, f1, 16, 100)

net = dde.nn.FNN([1] + [10] * 2 + [1], "tanh", "Glorot uniform")

model32 = dde.Model(data, net)

model32.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model32.train(iterations=10000, display_every=1)
dde.utils.external.save_loss_history(losshistory, fname = 'f32 same seed loss')

net32_to_copy = dde.nn.FNN([1] + [10] * 2 + [1], "tanh", "Glorot uniform")
model32_to_copy = dde.Model(data, net32_to_copy)
model32_to_copy.compile("adam", lr=0.001, metrics=["l2 relative error"])
_ = model32_to_copy.data.test()
# do one evaluation to initialize the weights
_ = model32_to_copy.predict([[1]])
loss_history, trainstate = model32_to_copy.train(iterations=0, display_every=1)


dde.config.set_default_float('float16')

geom = dde.geometry.Interval(-1, 1)
data = dde.data.Function(geom, f1, 16, 100)

net = dde.nn.FNN([1] + [10] * 2 + [1], "tanh", "Glorot uniform")
model16_new = dde.Model(data, net)
model16_new.compile("adam", lr=0.001, metrics=["l2 relative error"])

# generate the test arrays
_ = model16_new.data.test()
# do one evaluation to initialize the weights
_ = model16_new.predict([[1]])
# copy the weights
for i, layer in enumerate(model32_to_copy.net.denses):
    model16_new.net.denses[i].set_weights([tf.cast(w, dtype=tf.float16) for w in layer.get_weights()])
losshistory, train_state = model16_new.train(iterations=10000, display_every=1)
dde.utils.external.save_loss_history(losshistory, fname = 'f16 same seed loss')

# ============================================================================
# ==========================  Process the data  ==============================
# ============================================================================

file_path_16 = 'f16 same seed loss'
file_path_32 = 'f32 same seed loss'

# Load data from the file
data_16 = np.genfromtxt(file_path_16, delimiter=' ', skip_header=1)
data_32 = np.genfromtxt(file_path_32, delimiter=' ', skip_header=1)

x = data_16[:, 0]
loss_train_16 = data_16[:, 1]
loss_train_32 = data_32[:, 1]

# ============================================================================
# =======================  Visualize the results  ============================
# ============================================================================

plt.rcParams['font.size'] = 18
plt.xlabel('No. of iterations')
plt.xlim(0,10000)
plt.ylabel('Training loss')
plt.yscale('log')
# plt.plot(x, loss_test_32, label = "Testing float32", color = 'orange', linestyle='dashed')
# plt.plot(x, loss_test_16, label = "Testing float16", color = 'cornflowerblue', linestyle='dashed')
# plt.plot(x, loss_train_32, label = "Training float32", color = 'red')
# plt.plot(x, loss_train_16, label = "Training float16", color = 'midnightblue')
plt.plot(x, loss_train_32, label = "Float32", color = 'red')
plt.plot(x, loss_train_16, label = "Float16", color = 'blue')
plt.legend(loc='upper right', frameon = False)
plt.savefig("phases-" + str(SEED) + ".pdf", bbox_inches='tight')
plt.show()