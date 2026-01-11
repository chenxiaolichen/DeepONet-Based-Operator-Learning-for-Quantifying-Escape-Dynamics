import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


dt = 0.0005

n = 101

rng = np.random.default_rng(1234)

sigma = 0.5

x_0 = np.linspace(0, 1.48971, 101)
kf = 6.0
kd = 1.0
Kd = 10.0
Rbas = 0.4

def drift(x):
    return (kf * x**2 / (x**2 + Kd)) - kd * x + Rbas


def diffusivity(x):
    dW = rng.normal(loc=0, scale=np.sqrt(dt), size=x.shape)
    return sigma * dW


t_end = np.zeros((101, 30000))


for i in range(n):
    for j in range(30000):
        t = 0
        x = x_0[i]

        while 0 < x < 1.48971:
            x += dt * drift(x) + diffusivity(x)
            t += dt
        t_end[i][j] = t
    print(f"Initial position {i} completed")


t_need = np.mean(t_end, axis=1)

t_need = pd.DataFrame(t_need, columns=["t"])
t_need.to_excel("sig0.5_T.xlsx")
x_0 = pd.DataFrame(x_0, columns=["x"])
x_0.to_excel("sig0.5_X.xlsx")
