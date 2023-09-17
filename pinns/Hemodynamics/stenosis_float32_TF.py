#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:09:39 2023

@author: jayoxborn
"""
import numpy as np
import deepxde as dde
from deepxde.backend import tf
import pandas as pd


dde.config.set_default_float('float32')
dde.config.enable_xla_jit(mode = False)
# Import the slice data
data_slice = np.loadtxt("newslice_data.txt")

# If error is high, set data_slice to np.loadtxt("newPDE_data.txt")

eta = 1000

data_slip = np.loadtxt("stenosis_wall_points.txt")

# Import all x,y,z points to use
PDE_data = np.loadtxt("newPDE_data.txt")[:, 0:3]
# Create observable data from imported slices
data_u = data_slice[:, 3:4]
data_v = data_slice[:, 4:5]
data_w = data_slice[:, 5:6]

# Match data_u/v/w with x,y, and z points
data_in = data_slice[:, 0:3]
# Define the PDE
def PDE(t, y):

    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    # p = y[:,3:4]

    du_dx = dde.grad.jacobian(y, t, i=0, j=0)
    du_dy = dde.grad.jacobian(y, t, i=0, j=1)
    du_dz = dde.grad.jacobian(y, t, i=0, j=2)

    dv_dx = dde.grad.jacobian(y, t, i=1, j=0)
    dv_dy = dde.grad.jacobian(y, t, i=1, j=1)
    dv_dz = dde.grad.jacobian(y, t, i=1, j=2)

    dw_dx = dde.grad.jacobian(y, t, i=2, j=0)
    dw_dy = dde.grad.jacobian(y, t, i=2, j=1)
    dw_dz = dde.grad.jacobian(y, t, i=2, j=2)

    dp_dx = dde.grad.jacobian(y, t, i=3, j=0)
    dp_dy = dde.grad.jacobian(y, t, i=3, j=1)
    dp_dz = dde.grad.jacobian(y, t, i=3, j=2)

    du_dxx = dde.grad.hessian(y, t, i=0, j=0, component=0)
    du_dyy = dde.grad.hessian(y, t, i=1, j=1, component=0)
    du_dzz = dde.grad.hessian(y, t, i=2, j=2, component=0)

    dv_dxx = dde.grad.hessian(y, t, i=0, j=0, component=1)
    dv_dyy = dde.grad.hessian(y, t, i=1, j=1, component=1)
    dv_dzz = dde.grad.hessian(y, t, i=2, j=2, component=1)

    dw_dxx = dde.grad.hessian(y, t, i=0, j=0, component=2)
    dw_dyy = dde.grad.hessian(y, t, i=1, j=1, component=2)
    dw_dzz = dde.grad.hessian(y, t, i=2, j=2, component=2)

    Re = 1.0 / eta

    return [
        du_dx + dv_dy + dw_dz,
        (u * du_dx + v * du_dy + w * du_dz)
        + dp_dx
        - 1 / Re * (du_dxx + du_dyy + du_dzz),
        (u * dv_dx + v * dv_dy + w * dv_dz)
        + dp_dy
        - 1 / Re * (dv_dxx + dv_dyy + dv_dzz),
        (u * dw_dx + v * dw_dy + w * dw_dz)
        + dp_dz
        - 1 / Re * (dw_dxx + dw_dyy + dw_dzz),
    ]

idx3 = np.random.choice(PDE_data.shape[0], 2000, replace=False)
geom_data = PDE_data[idx3, 0:3]

geom = dde.geometry.PointCloud(PDE_data)

# Sample the data for no slip
idx = np.random.choice(data_slip.shape[0], 5000, replace=False)
slip = data_slip[idx, 0:3]

# Sample the data for observe
idx2 = np.random.choice(data_slice.shape[0], 5000, replace=False)
ob_u = data_slice[idx2, 3:4]
ob_v = data_slice[idx2, 4:5]
ob_w = data_slice[idx2, 5:6]

ob_xyz = data_slice[idx2,0:3]
 
# Create observable slices in deepxde
observe_u = dde.PointSetBC(ob_xyz, ob_u, component=0)  

observe_v = dde.PointSetBC(ob_xyz, ob_v, component=1)

observe_w = dde.PointSetBC(ob_xyz, ob_w, component=2)

# Create the values our no slip condition should be, cannot just plug in the


# For tf, this may need to be a numpy array of zeros (np.zeros)
# For tf, this may need to be a numpy array of zeros (np.zeros)
zeros = np.zeros((slip.shape[0],1))
noslip_1 = dde.PointSetBC(slip, 0, component=0)
noslip_2 = dde.PointSetBC(slip, zeros, component=1)
noslip_3 = dde.PointSetBC(slip, zeros, component=2)

# We put together all our our observables, and use 10000 domain points
data = dde.data.PDE(
    geom,
    PDE,
    [noslip_1,noslip_2,noslip_3,observe_u, observe_v, observe_w],
    num_domain=10000,
    num_test=1,
)

# The following architecture was found to approximate well. Pressure does not
# Need as large of a network since it does not vary that much with x/y
net = dde.nn.PFNN(
    [3, 128, 128, 128, [64, 64, 64, 16], 4], "swish", "Glorot uniform"
)

# We find the minimums and maximums to normalize x,y, and z.

xmin = np.min(PDE_data[:, 0:1])
xmax = np.max(PDE_data[:, 0:1])
ymin = np.min(PDE_data[:, 1:2])
ymax = np.max(PDE_data[:, 1:2])
zmin = np.min(PDE_data[:, 2:3])
zmax = np.max(PDE_data[:, 2:3])


def feature_transform(inputs):
    x = (inputs[:, 0:1] - xmin) / (xmax - xmin)
    y = (inputs[:, 1:2] - ymin) / (ymax - xmax)
    z = (inputs[:, 2:3] - zmin) / (zmax - zmin)
    return tf.concat((x, y, z), 1)


net.apply_feature_transform(feature_transform)

# Assuming we don't know the true minimums and maximums, the following normalization
# of outputs appears to help with convergence - is applicable since we are only
# using data we can actually see.
umin = np.min(data_u)
umax = np.max(data_u)
vmin = np.min(data_v)
vmax = np.max(data_v)
wmin = np.min(data_w)
wmax = np.max(data_w)
# pmin = np.min(data_p)
# pmax = np.max(data_p)

# We output u, v, w, and p. We also output the derivatives so we can later compute
# WSS. This does not effect the runtime since these derivaties
# already need computed in the PDE
def output_transform(t, y):
    u = (y[:, 0:1] - umin) / (umax - umin)
    v = (y[:, 1:2] - vmin) / (vmax - vmin)
    w = (y[:, 2:3] - wmin) / (wmax - wmin)
    p = y[:, 3:4]
    return tf.concat((u, v, w, p), 1)


net.apply_output_transform(output_transform)

model = dde.Model(data, net)

# We employ a 'pre-training' strategy, of just learning the observable data
# for a few iterations then solving the full PDE system
model.compile("adam", lr=1e-3, loss_weights=[0, 0, 0, 0, 0, 0, 0, 1e2, 1e2, 1e2])
losshistory, train_state = model.train(iterations=10000, display_every = 10000)
model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1, 1, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2])
losshistory, train_state = model.train(iterations=150000, display_every = 150000)
model.compile("adam", lr=1e-4, loss_weights=[1, 1, 1, 1, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2])
losshistory, train_state = model.train(iterations=50000, display_every = 50000)


# dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# We save the model in case we want to train again later or use it
# make sure to change the save_path to your needs

# model.restore(save_path="C:/Users/Mitchell/Downloads/stenosis/model-210000.pt")
model.save(
    save_path="/home/jhayford/Python_codes")

# Creating a pandas dataframe that tracks L2 relative errors.
errors = pd.DataFrame()

# The following can be used to get an estimate of the L2 error
# On our machines it cannot run on the full dataset so we just randomly
# sample half - which should give a decent estimate of the error on v/p.
PDE_test = np.loadtxt("newPDE_data.txt")

size = np.shape(PDE_test)


def operator(t, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    w = y[:, 2:3]
    p = y[:, 3:4]
    dudx = dde.grad.jacobian(u, t, i=0, j=0)
    dudy = dde.grad.jacobian(u, t, i=0, j=1)
    dudz = dde.grad.jacobian(u, t, i=0, j=2)
    dvdx = dde.grad.jacobian(v, t, i=0, j=0)
    dvdy = dde.grad.jacobian(v, t, i=0, j=1)
    dvdz = dde.grad.jacobian(v, t, i=0, j=2)
    dwdx = dde.grad.jacobian(w, t, i=0, j=0)
    dwdy = dde.grad.jacobian(w, t, i=0, j=1)
    dwdz = dde.grad.jacobian(w, t, i=0, j=2)
    return tf.concat(
        (u, v, w, p, dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz), 1
    )


X, y_true = PDE_test[:, 0:3], PDE_test[:, 3:7]

div = 10000
i = 0
y_pred = model.predict(
    X[int(i * div) : int((i + 1) * div), :], operator=operator
)

for i in range(1, size[0] // div - 1):
    # print(torch.cuda.memory_allocated(0))
    # print(i)
    y_dummy = model.predict(
        X[int(i * div) : int((i + 1) * div), :], operator=operator
    )
    y_pred = np.vstack((y_pred, y_dummy))

y_pred = np.vstack(
    (y_pred, model.predict(X[int((i + 1) * div) :, :], operator=operator))
)

# Since p output is just the derivative, we have to adjust it
p_factor = np.mean(y_true[:, 3:4]) - np.mean(y_pred[:, 3:4])

print(
    "L2 relative error of u: ",
    dde.metrics.l2_relative_error(y_true[:, 0:1], y_pred[:, 0:1]),
)
print(
    "L2 relative error of v: ",
    dde.metrics.l2_relative_error(y_true[:, 1:2], y_pred[:, 1:2]),
)
print(
    "L2 relative error of w: ",
    dde.metrics.l2_relative_error(y_true[:, 2:3], y_pred[:, 2:3]),
)
print(
    "L2 relative error of p: ",
    dde.metrics.l2_relative_error(y_true[:, 3:4], y_pred[:, 3:4] + p_factor),
)

Magnitude_velocity_true = (y_true[:, 0:1]**2+y_true[:, 1:2]**2+y_true[:, 2:3]**2)**0.5
Magnitude_velocity_pred = (y_pred[:, 0:1]**2+y_pred[:, 1:2]**2+y_pred[:, 2:3]**2)**0.5

print(
    "L2 relative error of velocity magnitude: ",
    dde.metrics.l2_relative_error(Magnitude_velocity_true, Magnitude_velocity_pred))

# Adding errors to the dataframe.
errors["u"] = [dde.metrics.l2_relative_error(y_true[:, 0:1], y_pred[:, 0:1])]
errors["v"] = [dde.metrics.l2_relative_error(y_true[:, 1:2], y_pred[:, 1:2])]
errors["w"] = [dde.metrics.l2_relative_error(y_true[:, 2:3], y_pred[:, 2:3])]
errors["p"] = [
    dde.metrics.l2_relative_error(y_true[:, 3:4], y_pred[:, 3:4] + p_factor)
]

# Need to compute the magnitude of WSR, get derivatives first
dudx = y_pred[:, 4:5]
dudy = y_pred[:, 5:6]
dudz = y_pred[:, 6:7]
dvdx = y_pred[:, 7:8]
dvdy = y_pred[:, 8:9]
dvdz = y_pred[:, 9:10]
dwdx = y_pred[:, 10:11]
dwdy = y_pred[:, 11:12]
dwdz = y_pred[:, 12:13]

# Evaluating the invariant second order shear tensor in order to obtain the shear rates.
S_xx = dudx
S_xy = (dudy + dvdx) / 2
S_xz = (dudz + dwdx) / 2
S_yx = S_xy
S_yy = dvdy
S_yz = (dvdz + dwdy) / 2
S_zx = S_xz
S_zy = S_yz
S_zz = dwdz

Sxx_Sxx = np.multiply(S_xx, S_xx)
Sxy_Syx = np.multiply(S_xy, S_yx)
Syx_Sxy = np.multiply(S_yx, S_xy)
Syy_Syy = np.multiply(S_yy, S_yy)
Syz_Szy = np.multiply(S_yz, S_zy)
Szy_Syz = np.multiply(S_zy, S_yz)
Szz_Szz = np.multiply(S_zz, S_zz)
Szx_Sxz = np.multiply(S_zx, S_xz)
Sxz_Szx = np.multiply(S_xz, S_zx)

D_II = (
    Sxx_Sxx
    + Sxy_Syx
    + Syx_Sxy
    + Syy_Syy
    + Syz_Szy
    + Szy_Syz
    + Szz_Szz
    + Szx_Sxz
    + Sxz_Szx
)

# Evaluated values of shear rates for each datapoint.
gamma = 2 * np.sqrt(D_II)

WSR_true = PDE_test[:, 7:8]

print(
    "L2 relative error of Wall Shear Rate (WSR): ",
    dde.metrics.l2_relative_error(WSR_true, gamma),
)

errors["WSR"] = [dde.metrics.l2_relative_error(WSR_true, gamma)]

# Saving all the errors to a single .csv file.
errors.to_csv("L2_Relative_Errors_Stenosis_Test_Case_tensorflow.csv")
tosave = np.hstack((X, y_pred[:, 0:3], y_pred[:, 3:4] + p_factor, gamma))

np.savetxt("networkoutput_stenosis_tensorflow.txt", tosave)
memory_used = tf.config.experimental.get_memory_info('GPU:0')['peak']
print("memory used:", memory_used)