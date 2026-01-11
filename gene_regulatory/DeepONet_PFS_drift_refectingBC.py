"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""

import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["DDE_BACKEND"] = "tensorflow"
import deepxde as dde
import tensorflow as tf


def MET_residual(x, u, f):
    sigma = 1
    du_dx = dde.grad.jacobian(u, x, j=0)
    d2u_dx2 = dde.grad.hessian(u, x, j=0)
    return f * du_dx + 0.5 * (sigma**2) * d2u_dx2 + 1


geom = dde.geometry.Interval(0, 2.8709)


def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)  # Left boundary

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 2.8709)  # Right boundary

# Left boundary: Reflecting (Neumann BC)
bc_left = dde.icbc.NeumannBC(geom, lambda _: 0, boundary_left)

# Right boundary: Escaping (Dirichlet BC)
def u_boundary_right(_):
    return 0
bc_right = dde.icbc.DirichletBC(geom, u_boundary_right, boundary_right)


pde = dde.data.PDE(
    geom,
    MET_residual,
    [bc_left, bc_right],
    num_domain=500,
    num_boundary=20,
    num_test=200,
)


degree = 2
space = dde.data.PowerSeries(N=degree + 1)


num_eval_points = 20
evaluation_points = geom.uniform_points(num_eval_points, boundary=True)


pde_op = dde.data.PDEOperatorCartesianProd(
    pde,
    space,
    evaluation_points,
    num_function=200,
    num_test=200,
)


class CustomDeepONet(dde.nn.DeepONetCartesianProd):
    def __init__(self, branches, trunks, activation, kernel_initializer):
        super().__init__(branches, trunks, activation, kernel_initializer)

    def predict(self, inputs):
        p_pred = super().predict(inputs)
        return tf.softplus(p_pred)  # Ensure positive escape time


dim_x = 1
p = 32
net = CustomDeepONet(
    branches=[num_eval_points, 32, p,p,p],
    trunks=[dim_x, 32, p,p,p],
    activation="tanh",
    kernel_initializer="Glorot normal",
)


model = dde.Model(pde_op, net)
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=30000)
dde.utils.plot_loss_history(losshistory)


x_input1 = np.linspace(0, 2.8709, num_eval_points)[:, None]
x_test1 = geom.uniform_points(1000, boundary=True)

Kd_values = np.arange(9.0, 11.701,0.1)
plt.figure(figsize=(10, 6))

predictions = []
for Kd in Kd_values:

    fx1 =(6 * x_input1 ** 2 / (x_input1 ** 2 + Kd)) - 1 * x_input1 + 0.4

    y1 = model.predict((fx1.T, x_test1))

    plt.plot(x_test1, y1.flatten(), label=f'Kd = {Kd:.1f}')
    predictions.append(y1.flatten())

plt.legend()

plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.title('Kd')

plt.grid(True)

plt.show()
import pandas as pd
x_test_df = pd.DataFrame(x_test1, columns=["X"])
x_test_df.to_excel("X.xlsx",index=False )
for i, Kd_val in enumerate(Kd_values):
    y_df = pd.DataFrame(predictions[i], columns=["Y"])
    y_df.to_excel(
        f"Kd_{Kd_val:.1f}_Y.xlsx",
        index=False
    )

x_input1 = np.linspace(0, 2.8709, num_eval_points)[:, None]
x_test1 = geom.uniform_points(1000, boundary=True)
kd_values = np.arange(0.95,1.0801, 0.01)
plt.figure(figsize=(10, 6))
predictions = []

for kd in kd_values:

    fx1 =(6 * x_input1 ** 2 / (x_input1 ** 2 +10)) - kd * x_input1 + 0.4

    y1 = model.predict((fx1.T, x_test1))

    plt.plot(x_test1, y1.flatten(), label=f'kd = {kd:.2f}')
    predictions.append(y1.flatten())

plt.legend()

plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.title('kd')

plt.show()
import pandas as pd
for i, kd_val in enumerate(kd_values):
    y_df = pd.DataFrame(predictions[i], columns=["Y"])  # 指定列名为Y
    y_df.to_excel(
        f"kd_{kd_val:.2f}_Y.xlsx",
        index=False
    )

x_input1 = np.linspace(0, 2.8709, num_eval_points)[:, None]
x_test1 = geom.uniform_points(1000, boundary=True)
kf_values = np.arange(5.5,6.601, 0.05)
plt.figure(figsize=(10, 6))
predictions = []

for kf in kf_values:

    fx1 =(kf * x_input1 ** 2 / (x_input1 ** 2 +10)) - 1 * x_input1 + 0.4

    y1 = model.predict((fx1.T, x_test1))

    plt.plot(x_test1, y1.flatten(), label=f'kf = {kf:.2f}')
    predictions.append(y1.flatten())

plt.legend()

plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.title('kf')

plt.grid(True)

plt.show()
import pandas as pd
for i, kf_val in enumerate(kf_values):
    y_df = pd.DataFrame(predictions[i], columns=["Y"])
    y_df.to_excel(
        f"kf_{kf_val:.2f}_Y.xlsx",
        index=False
    )

x_input1 = np.linspace(0, 2.8709, num_eval_points)[:, None]
x_test1 = geom.uniform_points(1000, boundary=True)
Rbas_values = np.arange(0.16,0.4501, 0.01)
plt.figure(figsize=(10, 6))
predictions = []

for Rbas in Rbas_values:

    fx1 =(6 * x_input1 ** 2 / (x_input1 ** 2 +10)) - 1 * x_input1 + Rbas

    y1 = model.predict((fx1.T,x_test1))

    plt.plot(x_test1, y1.flatten(), label=f'Rbas = {Rbas:.2f}')
    predictions.append(y1.flatten())

plt.legend()

plt.xlabel('$x$')
plt.ylabel('$u(x)$')
plt.title('Rbas')

plt.grid(True)

plt.show()
import pandas as pd
for i, Rbas_val in enumerate(Rbas_values):
    y_df = pd.DataFrame(predictions[i], columns=["Y"])
    y_df.to_excel(
        f"Rbas_{Rbas_val:.2f}_Y.xlsx",
        index=False
    )
