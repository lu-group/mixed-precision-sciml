import os
os.environ['DDE_BACKEND'] = 'tensorflow'
import deepxde as dde
import numpy as np
from deepxde.backend import tf
from scipy import io
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import mixed_precision

def solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt):
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h**2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u

# PDE
def pde(x, y, v):
    D = 0.01
    k = 0.01
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    return dy_t - D * dy_xx + k * y**2 - v

policy = mixed_precision.Policy('float16')
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)
mixed_precision.set_global_policy(policy)
dde.config.set_default_float('float16')

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

pde = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=200,
    num_boundary=40,
    num_initial=20,
    num_test=500,
)

# Function space
func_space = dde.data.GRF(length_scale=0.2)

# Data
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=50
)

# Net
net = dde.nn.DeepONetCartesianProd(
    [50, 128, 128, 128],
    [2, 128, 128, 128],
    "tanh",
    "Glorot normal",
)

model = dde.Model(data, net)
adam = tf.keras.optimizers.Adam(
    learning_rate=0.0005,
    epsilon=1e-05
)
model.compile(adam)
losshistory, train_state = model.train(epochs=0)
tf.config.experimental.reset_memory_stats('GPU:0')
tf.keras.backend.clear_session()
losshistory, train_state = model.train(epochs=50000) # 50000
memory_used = tf.config.experimental.get_memory_info('GPU:0')['peak']
print("MEMORY USED: ")
print(memory_used)
dde.utils.plot_loss_history(losshistory)

dde.config.set_random_seed(0)
testing_metric = np.zeros(1000)
for i in range(1000):
    func_feats = func_space.random(1)
    xs = np.linspace(0, 1, num=100)[:, None]
    v = func_space.eval_batch(func_feats, xs)[0]
    x, t, u_true = solve_ADR(
        0,
        1,
        0,
        1,
        lambda x: 0.01 * np.ones_like(x),
        lambda x: np.zeros_like(x),
        lambda u: 0.01 * u**2,
        lambda u: 0.02 * u,
        lambda x, t: np.tile(v[:, None], (1, len(t))),
        lambda x: np.zeros_like(x),
        100,
        100,
    )
    u_true = u_true.T

    v_branch = func_space.eval_batch(func_feats, np.linspace(0, 1, num=50)[:, None])
    xv, tv = np.meshgrid(x, t)
    x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
    u_pred = model.predict((v_branch, x_trunk))
    u_pred = u_pred.reshape((100, 100))
    testing_metric[i] = dde.metrics.l2_relative_error(u_true, u_pred)

print("mean: ")
print(np.mean(testing_metric))
print("standard deviation:")
print(np.std(testing_metric))