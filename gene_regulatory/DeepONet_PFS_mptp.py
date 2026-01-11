import numpy as np
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import trange
import torch.nn.functional as F
import os
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

data_dir = "results"
os.makedirs(data_dir, exist_ok=True)


tmin, tmax = 0, 2.0
Nt = 50000
t_base = np.linspace(tmin, tmax, Nt).astype(np.float64)
m = 10
t_sensor = np.linspace(tmin, tmax, m).astype(np.float64)


Kd = 10.0
kd = 1.0
kf = 6.0
Rbas = 0.4


def solve_mptp(sigma, t):
    """
    z'' = [drift(z) * drift’(z)] + σ² * drift''(z)
    drift(z) = kf*z²/(z²+Kd) - kd*z + Rbas
    drift’(z) = 2*kf*Kd*z/(z²+Kd)² - kd
    drift''(z) = kf*Kd*(Kd - 3z²)/(z²+Kd)³
    z(0)=0.62685, z(T)=4.28343
    """

    def odefun(t, z):
        z1, z2 = z

        drift = (kf * z1 ** 2 / (z1 ** 2 + Kd)) - kd * z1 + Rbas
        drift_prime = (2 * kf * Kd * z1) / (z1 ** 2 + Kd) ** 2 - kd
        drift_double_prime = (kf * Kd * (Kd - 3 * z1 ** 2)) / (z1 ** 2 + Kd) ** 3


        z2_dot = drift * drift_prime + (sigma ** 2) * drift_double_prime
        return [z2, z2_dot]


    def shooting(z2_0):
        sol = solve_ivp(odefun, [tmin, tmax], [0.62685, z2_0], t_eval=t, method='RK45')
        return sol.y[0, -1] - 4.28343


    res = root_scalar(shooting, bracket=[-10.0, 10.0], method='bisect')
    z2_0 = res.root

    sol = solve_ivp(odefun, [tmin, tmax], [0.62685, z2_0], t_eval=t, method='RK45')
    return sol.y[0].astype(np.float32)



class DataGenerator(data.Dataset):
    def __init__(self, sigma_sensor, t, z, batch_size=64):
        self.sigma_sensor = torch.Tensor(sigma_sensor)
        self.t = torch.Tensor(t)
        self.z = torch.Tensor(z)
        self.N = sigma_sensor.shape[0]
        self.batch_size = batch_size

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return (self.sigma_sensor[index], self.t[index]), self.z[index]


def chebyshev_polynomial(t, n, coeffs):

    if isinstance(t, (int, float, np.float64)):
        t = np.array([t])
    t_mapped = 2 * (t - tmin) / (tmax - tmin) - 1
    T = np.zeros((len(t_mapped), n + 1), dtype=np.float64)
    T[:, 0] = 1
    if n >= 1:
        T[:, 1] = t_mapped
    for i in range(2, n + 1):
        T[:, i] = 2 * t_mapped * T[:, i - 1] - T[:, i - 2]
    result = np.dot(T, coeffs)
    return result[0] if len(result) == 1 else result


def generate_chebyshev_coeffs(n, coeff_range=(-1, 1)):
    return np.random.uniform(coeff_range[0], coeff_range[1], size=n + 1)


def generate_one_sample(P, n_cheb, coeff_range, t_grid):

    coeffs = generate_chebyshev_coeffs(n_cheb, coeff_range)
    sigma_fn = lambda t: np.abs(chebyshev_polynomial(t, n_cheb, coeffs)) + 0.01

    sigma_const = sigma_fn(t_grid[0])
    z_true = solve_mptp(sigma_const, t_grid)

    sigma_sensor = np.ones(m) * sigma_const
    N_sample = len(z_true)
    sigma_sensor = np.tile(sigma_sensor, (N_sample, 1))

    max_possible = len(t_grid)
    P = min(P, max_possible)
    idx = np.random.choice(max_possible, size=P, replace=False)
    idx = np.unique(np.concatenate([[0, max_possible - 1], idx]))
    t_sample = t_grid[idx].reshape(-1, 1)
    z_sample = z_true[idx].reshape(-1, 1)

    return sigma_sensor[idx], t_sample, z_sample


def generate_batch_data(N, P, n_cheb, coeff_range, t_grid=t_base):
    pbar = trange(N, desc=f"generate data (MPTP)")
    f_list, t_list, z_list = [], [], []
    for _ in pbar:
        f, t_sample, z_sample = generate_one_sample(P, n_cheb, coeff_range, t_grid)
        f_list.append(f)
        t_list.append(t_sample)
        z_list.append(z_sample)
    return (
        np.concatenate(f_list, axis=0),
        np.concatenate(t_list, axis=0),
        np.concatenate(z_list, axis=0)
    )


def generate_test_data(sigma_const, t_highres, fixed_t_sensor):

    z_true = solve_mptp(sigma_const, t_highres)
    sigma_sensor = np.ones_like(fixed_t_sensor) * sigma_const
    sigma_sensor = sigma_sensor.reshape(1, -1)
    t_test = t_highres.reshape(-1, 1)
    z_test = z_true.reshape(-1, 1)
    return sigma_sensor, t_test, z_test


def save_train_test_data(train_data, test_data):
    save_path = os.path.join(data_dir, "train_test_data_MPTP.npz")
    sigma_train, t_train, z_train = train_data
    sigma_test, t_test, z_test = test_data
    np.savez_compressed(
        save_path,
        sigma_train=sigma_train, t_train=t_train, z_train=z_train,
        sigma_test=sigma_test, t_test=t_test, z_test=z_test,
        t_sensor=t_sensor,
        n_cheb_train=n_cheb_train, n_cheb_test=n_cheb_test
    )



def load_train_test_data():
    load_path = os.path.join(data_dir, "train_test_data_MPTP.npz")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"data not found: {load_path}")
    data = np.load(load_path)
    assert np.allclose(data['t_sensor'], t_sensor), "sensors not match"
    return (
        (data['sigma_train'], data['t_train'], data['z_train']),
        (data['sigma_test'], data['t_test'], data['z_test']),
        data['n_cheb_train'], data['n_cheb_test']
    )


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            torch.nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.0)
            self.layers.append(layer)
        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super(DeepONet, self).__init__()
        self.branch = MLP(branch_layers)
        self.trunk = MLP(trunk_layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.loss_log = []
        self.test_loss_log = []

    def forward(self, sigma_sensor, t):
        B = self.branch(sigma_sensor)
        T = self.trunk(t)
        return torch.sum(B * T, dim=-1, keepdim=True)

    def train_model(self, train_dataset, test_dataset, epochs):
        train_loader = data.DataLoader(train_dataset, batch_size=train_dataset.batch_size, shuffle=True)
        test_loader = data.DataLoader(test_dataset, batch_size=test_dataset.batch_size, shuffle=False)

        pbar = trange(epochs, desc="train DeepONet (MPTP)")
        for epoch in pbar:

            self.train()
            total_train_loss = 0.0
            for (sigma_batch, t_batch), z_batch in train_loader:
                sigma_batch = sigma_batch.to(device)
                t_batch = t_batch.to(device)
                z_batch = z_batch.to(device)

                self.optimizer.zero_grad()
                z_pred = self(sigma_batch, t_batch)
                loss = F.mse_loss(z_pred, z_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()


            self.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for (sigma_batch, t_batch), z_batch in test_loader:
                    sigma_batch = sigma_batch.to(device)
                    t_batch = t_batch.to(device)
                    z_batch = z_batch.to(device)
                    z_pred = self(sigma_batch, t_batch)
                    total_test_loss += F.mse_loss(z_pred, z_batch).item()


            avg_train_loss = total_train_loss / len(train_loader)
            avg_test_loss = total_test_loss / len(test_loader)
            self.loss_log.append(avg_train_loss)
            self.test_loss_log.append(avg_test_loss)

            pbar.set_postfix({
                'train MSE': f'{avg_train_loss:.6e}',
                'test MSE': f'{avg_test_loss:.6e}'
            })

    def predict(self, sigma_sensor, t):
        with torch.no_grad():
            sigma_tensor = torch.Tensor(sigma_sensor).to(device)
            t_tensor = torch.Tensor(t).to(device)
            return self(sigma_tensor, t_tensor).cpu().numpy()

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_log': self.loss_log,
            'test_loss_log': self.test_loss_log
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss_log = checkpoint['loss_log']
        self.test_loss_log = checkpoint['test_loss_log']
        self.eval()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



N_train = 300
N_test = 100
P_per_group = 100
batch_size = 256
epochs = 15000


n_cheb_train = 0
n_cheb_test = 0
train_coeff_range = (0.1, 2.0)
test_coeff_range = (0.1, 2.0)


t_highres = np.linspace(tmin, tmax, 1000).astype(np.float64)
target_sigmas = [0.5, 1.0, 1.5, 2.0]


data_path = os.path.join(data_dir, "train_test_data_MPTP.npz")
model_path = os.path.join(data_dir, "deeponet_MPTP_model.pth")


if os.path.exists(data_path):

    train_data, test_data, loaded_n_train, loaded_n_test = load_train_test_data()
    sigma_train, t_train, z_train = train_data
    sigma_test, t_test, z_test = test_data
else:

    train_data = generate_batch_data(
        N=N_train, P=P_per_group, n_cheb=n_cheb_train,
        coeff_range=train_coeff_range, t_grid=t_base
    )

    test_data = generate_batch_data(
        N=N_test, P=P_per_group, n_cheb=n_cheb_test,
        coeff_range=test_coeff_range, t_grid=t_base
    )
    save_train_test_data(train_data, test_data)
    sigma_train, t_train, z_train = train_data
    sigma_test, t_test, z_test = test_data


train_dataset = DataGenerator(sigma_train, t_train, z_train, batch_size=batch_size)
test_dataset = DataGenerator(sigma_test, t_test, z_test, batch_size=batch_size)


branch_layers = [m, 32, 32, 32, 32]
trunk_layers = [1, 32, 32, 32, 32]
model = DeepONet(branch_layers, trunk_layers).to(device)

if os.path.exists(model_path):
    model.load_model(model_path)
else:
    model.train_model(train_dataset, test_dataset, epochs=epochs)
    model.save_model(model_path)


def test_mptp_sigma(model, sigma_values, t_highres):

    results = []
    for sigma in sigma_values:
        sigma_sensor, t_test, z_true = generate_test_data(sigma, t_highres, t_sensor)
        z_pred = model.predict(sigma_sensor, t_test)

        z_pred = np.clip(z_pred, 0.0, None)

        l2_error = np.linalg.norm(z_pred - z_true) / np.linalg.norm(z_true)
        results.append((t_test.flatten(), z_true.flatten(), z_pred.flatten(), sigma, l2_error))
        print(f"σ={sigma:<4} | L2 error: {l2_error:.6e}")
    return results


mptp_results = test_mptp_sigma(model, target_sigmas, t_highres)

plt.figure(figsize=(16, 10))
for i, (t_vals, z_true, z_pred, sigma, l2_err) in enumerate(mptp_results):
    plt.subplot(2, 2, i + 1)
    plt.plot(t_vals, z_true, 'b-', linewidth=2, label='True MPTP ')
    plt.plot(t_vals, z_pred, 'r--', linewidth=2, label='DeepONet(PFS) ')
    plt.title(f'MPTP (σ={sigma}, L2 error: {l2_err:.6f})', fontsize=12)
    plt.xlabel('Time t', fontsize=10)
    plt.ylabel('TF-A Concentration z(t)', fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'MPTP_different_sigma.png'), dpi=300)
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(model.loss_log, label='Train Loss')
plt.plot(model.test_loss_log, label='Test Loss')
plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training and Test Loss ', fontsize=12)
plt.grid(alpha=0.3)
plt.savefig(os.path.join(data_dir, 'MPTP_loss_curve.png'), dpi=300)
plt.show()