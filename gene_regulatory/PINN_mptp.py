import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
import matplotlib.pyplot as plt
from tqdm import tqdm


kf = 6.0
Kd = 10.0
kd = 1.0
Rbas = 0.4
sigma = 1
T = 2.0
N = 1001
z0 = 0.62685
zT = 4.28343


t = tf.random.uniform((N, 1), minval=0, maxval=T, dtype=tf.float32)
t_boundary = tf.constant([[0.0], [T]], dtype=tf.float32)


class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = [
            layers.Dense(20, activation='tanh'),
            layers.Dense(20, activation='tanh'),
            layers.Dense(20, activation='tanh'),
            layers.Dense(20, activation='tanh')
        ]
        self.output_layer = layers.Dense(1)

    def call(self, t):
        x = t
        for layer in self.hidden:
            x = layer(x)
        return self.output_layer(x)


def loss_function(model, t, t_boundary):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        z = model(t)
        z_t = tape.gradient(z, t)
    z_tt = tape.gradient(z_t, t)
    del tape


    f_z = (kf * z**2 / (z**2 + Kd)) - kd * z + Rbas
    df_dz = (2 * kf * Kd * z) / (z**2 + Kd)**2 - kd
    term1 = f_z * df_dz
    term2 = sigma**2 * (kf * Kd * (Kd - 3 * z**2)) / (z**2 + Kd)**3
    rhs = term1 + term2


    pde_loss = tf.reduce_mean(tf.square(z_tt - rhs))


    z0_pred = model(t_boundary[0:1])
    zT_pred = model(t_boundary[1:2])
    bc_loss = tf.reduce_mean(tf.square(z0_pred - z0) + tf.square(zT_pred - zT))

    return pde_loss + bc_loss


model = PINN()
optimizer = optimizers.Adam(learning_rate=1e-4)
epochs = 20000
pbar = tqdm(range(epochs))

for epoch in pbar:
    with tf.GradientTape() as tape:
        loss = loss_function(model, t, t_boundary)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    pbar.set_description(f"Epoch {epoch+1}, Loss: {loss.numpy():.6f}")


t_test = np.linspace(0, T, 1000).reshape(-1, 1)
z_pred = model.predict(t_test)
import pandas as pd
t_teat_df = pd.DataFrame(t_test, columns=['X'])
z_pred_df = pd.DataFrame(z_pred, columns=['Y'])
t_teat_df.to_excel('sig1_X.xlsx', index=False)
z_pred_df.to_excel('sig1_Y.xlsx', index=False)
plt.figure(figsize=(10, 6))
plt.plot(t_test, z_pred, 'r-', label='PINN')
plt.scatter([0, T], [z0, zT], c='black', s=100)
plt.xlabel(' t')
plt.ylabel('z(t)')
plt.title('Onsager-Machlup')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('sig1_result.png')