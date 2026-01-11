import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import trange
import torch.nn.functional as F
import os

data_dir = "results2"
os.makedirs(data_dir, exist_ok=True)


tmin, tmax = 0, 2.0
Nt = 50000
t = np.linspace(tmin, tmax, Nt)
m = 10
t_sensor = np.linspace(tmin, tmax, m)

Kd = 10.0
kd = 1.0
kf = 6.0
Rbas = 0.4


def RBF(x1, x2, lengthscales):

    diffs = np.expand_dims(x1 / lengthscales, 1) - np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return np.exp(-0.5 * r2)


def solve_mptp(sigma, t):
    """
    z'' = [ (kf*z²/(z²+Kd) - kd*z + Rbas) * (2*kf*Kd*z/(z²+Kd)² - kd) ] + σ² * [kf*Kd*(Kd-3z²)/(z²+Kd)³]
    z(0)=0.62685, z(T)=4.28343
    """


    def odefun(t, z):
        z1, z2 = z

        drift = (kf * z1 ** 2 / (z1 ** 2 + Kd) - kd * z1 + Rbas) * (2 * kf * Kd * z1 / (z1 ** 2 + Kd) ** 2 - kd)

        diff = sigma ** 2 * (kf * Kd * (Kd - 3 * z1 ** 2) / (z1 ** 2 + Kd) ** 3)

        z2_dot = drift + diff
        return [z2, z2_dot]


    def shooting(z2_0):
        sol = solve_ivp(odefun, [tmin, tmax], [0.62685, z2_0], t_eval=t, method='RK45')
        return sol.y[0, -1] - 4.28343

    from scipy.optimize import root_scalar
    res = root_scalar(shooting, bracket=[-10.0, 10.0], method='bisect')
    z2_0 = res.root

    sol = solve_ivp(odefun, [tmin, tmax], [0.62685, z2_0], t_eval=t, method='RK45')
    return sol.y[0]


def save_train_test_data(f_train, y_train, s_train, f_test, y_test, s_test):
    save_path = os.path.join(data_dir, "train_test_data_MPTP.npz")
    np.savez_compressed(save_path,
                        f_train=f_train, y_train=y_train, s_train=s_train,
                        f_test=f_test, y_test=y_test, s_test=s_test,
                        t_sensor=t_sensor)
    print(f"Train and test data saved to: {save_path}")


def load_train_test_data():
    load_path = os.path.join(data_dir, "train_test_data_MPTP.npz")
    if not os.path.exists(load_path):
        raise FileNotFoundError(load_path)
    data = np.load(load_path)
    return (data['f_train'], data['y_train'], data['s_train'],
            data['f_test'], data['y_test'], data['s_test'],
            data['t_sensor'])


class DataGenerator(data.Dataset):
    def __init__(self, f_sensor, y, s):
        self.f_sensor = torch.Tensor(f_sensor)
        self.y = torch.Tensor(y)
        self.s = torch.Tensor(s)
        self.N = f_sensor.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (self.f_sensor[idx], self.y[idx]), self.s[idx]


def generate_one_sample(P):
    N_grf = 500
    length_scale = 100
    T_grf = np.linspace(tmin, tmax, N_grf)[:, None]
    K = RBF(T_grf, T_grf, length_scale)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(N_grf))
    gp_sample = L @ np.random.randn(N_grf)
    sigma_fn = lambda t: 0.01 + np.abs(np.interp(t, T_grf.flatten(), gp_sample))

    z_true = solve_mptp(sigma_fn(t[0]), t)

    sigma_sensor = sigma_fn(t_sensor)
    sigma_sensor = np.tile(sigma_sensor, (P, 1))

    idx_internal = np.random.randint(1, Nt - 1, size=P - 2)
    idx = np.concatenate([[0], idx_internal, [Nt - 1]])
    y = t[idx].reshape(-1, 1)
    s = z_true[idx].reshape(-1, 1)
    return sigma_sensor, y, s


def generate_batch_data(N, P):
    pbar = trange(N, desc="Generating MPTP data")
    f_list, y_list, s_list = [], [], []
    for _ in pbar:
        f, y, s = generate_one_sample(P)
        f_list.append(f)
        y_list.append(y)
        s_list.append(s)
    f_data = np.concatenate(f_list, axis=0)
    y_data = np.concatenate(y_list, axis=0)
    s_data = np.concatenate(s_list, axis=0)
    return f_data, y_data, s_data

class MLP(nn.Module):
    def __init__(self, layers, use_residual=True):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.use_residual = use_residual
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.0)
            self.layers.append(layer)
        self.activation = nn.Tanh()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x_prev = x
            x = self.activation(layer(x))
            if self.use_residual and x.shape == x_prev.shape:
                x = x + x_prev
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

    def forward(self, f_sensor, t):
        B = self.branch(f_sensor)
        T = self.trunk(t)
        pred = torch.sum(B * T, dim=-1, keepdim=True)
        return pred

    def compute_loss(self, dataloader):
        total_loss = 0.0
        num_batches = 0
        for (f_batch, t_batch), s_batch in dataloader:
            f_batch = f_batch.to(next(self.parameters()).device)
            t_batch = t_batch.to(next(self.parameters()).device)
            s_batch = s_batch.to(next(self.parameters()).device)

            with torch.no_grad():
                s_pred = self(f_batch, t_batch)
                batch_loss = F.mse_loss(s_pred, s_batch)

            total_loss += batch_loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0.0

    def train_model(self, train_dataset, test_dataset=None, epochs=10000, batch_size=256):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        ) if test_dataset else None

        pbar = trange(epochs, desc="Training DeepONet for MPTP")
        for epoch in pbar:
            self.train()
            total_train_loss = 0.0
            num_train_batches = 0

            for (f_batch, t_batch), s_batch in train_loader:
                f_batch = f_batch.to(next(self.parameters()).device)
                t_batch = t_batch.to(next(self.parameters()).device)
                s_batch = s_batch.to(next(self.parameters()).device)

                self.optimizer.zero_grad()
                s_pred = self(f_batch, t_batch)
                mse_loss = F.mse_loss(s_pred, s_batch)
                mse_loss.backward()
                self.optimizer.step()

                total_train_loss += mse_loss.item()
                num_train_batches += 1

            self.scheduler.step()
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            self.loss_log.append(avg_train_loss)
            avg_test_loss = None
            if test_loader:
                self.eval()
                avg_test_loss = self.compute_loss(test_loader)
                self.test_loss_log.append(avg_test_loss)

            pbar.set_postfix({
                'Train MSE': f'{avg_train_loss:.6e}',
                'Test MSE': f'{avg_test_loss:.6e}' if avg_test_loss is not None else 'N/A'
            })

    def predict(self, f_sensor, t):
        with torch.no_grad():
            f_tensor = torch.Tensor(f_sensor).to(next(self.parameters()).device)
            t_tensor = torch.Tensor(t).to(next(self.parameters()).device)
            return self(f_tensor, t_tensor).cpu().numpy()

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_log': self.loss_log,
            'test_loss_log': self.test_loss_log
        }, path)

    def load_model(self, path):
        ckpt = torch.load(path)
        self.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.loss_log = ckpt['loss_log']
        self.test_loss_log = ckpt.get('test_loss_log', [])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

N_train = 300
N_test = 100
P_per_group = 100
batch_size = 256
data_path = os.path.join(data_dir, "train_test_data_MPTP.npz")
model_path = os.path.join(data_dir, "deeponet_MPTP_model.pth")

if os.path.exists(data_path):
    print("Loading train and test data...")
    data = np.load(data_path)
    f_train = data['f_train']
    y_train = data['y_train']
    s_train = data['s_train']
    f_test = data['f_test']
    y_test = data['y_test']
    s_test = data['s_test']
    loaded_t_sensor = data['t_sensor']
    assert np.allclose(loaded_t_sensor, t_sensor), "Sensor positions mismatch!"
else:
    print("Generating training data...")
    f_train, y_train, s_train = generate_batch_data(N_train, P_per_group)
    print("Generating test data...")
    f_test, y_test, s_test = generate_batch_data(N_test, P_per_group)
    save_train_test_data(f_train, y_train, s_train, f_test, y_test, s_test)

train_dataset = DataGenerator(f_train, y_train, s_train)
test_dataset = DataGenerator(f_test, y_test, s_test)

print(f"Total training samples: {len(train_dataset)}")
print(f"Total test samples: {len(test_dataset)}")
print(f"Branch network input dimension: {f_train.shape[1]}")


branch_layers = [m, 32, 32, 32, 32]
trunk_layers = [1, 32, 32, 32, 32]
model = DeepONet(branch_layers, trunk_layers).to(device)


model.train_model(train_dataset, test_dataset, epochs=15000, batch_size=batch_size)
model.save_model(model_path)


sigma_values = [0.1, 0.5, 1, 2]
l2_errors = []
t_test = np.linspace(tmin, tmax, 1000)


def generate_test_data(sigma, t, fixed_t_sensor):
    sigma_fn = lambda t: np.ones_like(t) * sigma  
    z_true = solve_mptp(sigma, t)
    sigma_sensor = sigma_fn(fixed_t_sensor).reshape(1, m)
    y_test = t.reshape(-1, 1)
    s_test = z_true.reshape(-1, 1)
    return sigma_sensor, y_test, s_test


plt.figure(figsize=(12, 8))
for sigma in sigma_values:
    f_test, t_test_batch, z_true = generate_test_data(sigma, t_test, t_sensor)
    z_pred = model.predict(f_test, t_test_batch)
    l2 = np.linalg.norm(z_pred - z_true) / np.linalg.norm(z_true)
    l2_errors.append(l2)
    plt.plot(t_test_batch, z_true, label=f'True MPTP (σ={sigma})', alpha=0.8, linewidth=2)
    plt.plot(t_test_batch, z_pred, '--', label=f'DeepONet Pred (σ={sigma})', alpha=0.8, linewidth=2)

plt.xlabel('Time t', fontsize=12)
plt.ylabel('TF-A Concentration z(t)', fontsize=12)
plt.title('Most Probable Transition Path (MPTP): True vs DeepONet Prediction', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()

print("\nRelative L2 errors for different σ values:")
for sigma, err in zip(sigma_values, l2_errors):
    print(f"σ={sigma:<4} | Relative L2 error: {err:.6e}")

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(model.loss_log, label='Training Loss (MSE)', linewidth=1.5)
if model.test_loss_log:
    plt.plot(model.test_loss_log, label='Test Loss (MSE)', linewidth=1.5)
plt.title('DeepONet Training/Test Loss (MPTP Task)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()