import numpy as np
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import trange
import torch.nn.functional as F
import os
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

data_dir = r"E:\gene\PFS\sig\ep"
os.makedirs(data_dir, exist_ok=True)


xmin, xmax = 0, 1.48971
Nx = 10000
x_base = np.linspace(xmin, xmax, Nx).astype(np.float64)
m = 10
x_sensor = np.linspace(xmin, xmax, m).astype(np.float64)
eps = 1.0


def solve_pde(b_fn, sigma_fn, x):

    N = len(x)
    h = x[1] - x[0]
    sigma_vals = sigma_fn(x)

    sigma_vals = np.clip(sigma_vals, a_min=1e-6, a_max=None)
    b_vals = b_fn(x)

    Chh = (sigma_vals ** 2) / (2 * eps) / (h ** 2)

    A = lil_matrix((N, N), dtype=np.float64)
    rhs = np.zeros(N, dtype=np.float64)

    for i in range(1, N-1):
        A[i, i-1] = Chh[i] - b_vals[i] / (2 * h)
        A[i, i] = -2 * Chh[i]
        A[i, i+1] = Chh[i] + b_vals[i] / (2 * h)


    i = 0
    A[i, i] = 1.0
    rhs[i] = 0.0


    i = N-1
    A[i, i] = 1.0
    rhs[i] = 1.0

    A = csc_matrix(A)
    try:
        u = spsolve(A, rhs)
    except:
        from scipy.sparse.linalg import lsqr
        u = lsqr(A, rhs)[0]


    u = np.clip(u, 0.0, 1.0)
    return u.astype(np.float32)



class DataGenerator(data.Dataset):
    def __init__(self, sigma_sensor, y, s, batch_size=64):
        self.sigma_sensor = torch.Tensor(sigma_sensor)
        self.y = torch.Tensor(y)
        self.s = torch.Tensor(s)
        self.N = sigma_sensor.shape[0]
        self.batch_size = batch_size

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return (self.sigma_sensor[index], self.y[index]), self.s[index]


def chebyshev_polynomial(x, n, coeffs):
    x_mapped = 2 * (x - xmin) / (xmax - xmin) - 1
    T = np.zeros((len(x_mapped), n + 1), dtype=np.float64)
    T[:, 0] = 1
    if n >= 1:
        T[:, 1] = x_mapped
    for i in range(2, n + 1):
        T[:, i] = 2 * x_mapped * T[:, i - 1] - T[:, i - 2]
    return np.dot(T, coeffs)


def generate_chebyshev_coeffs(n, coeff_range=(-1, 1)):
    return np.random.uniform(coeff_range[0], coeff_range[1], size=n + 1)


def generate_one_sample(P, n_cheb, coeff_range, x_grid):
    coeffs = generate_chebyshev_coeffs(n_cheb, coeff_range)
    sigma_fn = lambda x: chebyshev_polynomial(x, n_cheb, coeffs)

    sigma_fn = lambda x: np.abs(chebyshev_polynomial(x, n_cheb, coeffs)) + 0.1
    b_fn = lambda x: (6 * x ** 2) / (x ** 2 + 10) - 1 * x + 0.4

    u_true = solve_pde(b_fn, sigma_fn, x_grid)

    sigma_sensor = sigma_fn(x_sensor)
    N_sample = len(u_true)
    sigma_sensor = np.tile(sigma_sensor, (N_sample, 1))

    max_possible = len(x_grid)
    P = min(P, max_possible)
    idx = np.random.choice(max_possible, size=P, replace=False)
    idx = np.unique(np.concatenate([[0, max_possible - 1], idx]))
    y = x_grid[idx].reshape(-1, 1)
    s = u_true[idx].reshape(-1, 1)

    return sigma_sensor[idx], y, s


def generate_batch_data(N, P, n_cheb, coeff_range, x_grid=x_base):
    pbar = trange(N, desc=f"data generation")
    f_list, y_list, s_list = [], [], []
    for _ in pbar:
        f, y, s = generate_one_sample(P, n_cheb, coeff_range, x_grid)
        f_list.append(f)
        y_list.append(y)
        s_list.append(s)
    return (
        np.concatenate(f_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(s_list, axis=0)
    )


def generate_test_data(sigma_fn, x_highres, fixed_x_sensor):
    b_fn = lambda x: (6 * x ** 2) / (x ** 2 + 10) - 1 * x + 0.4
    u_true = solve_pde(b_fn, sigma_fn, x_highres)
    sigma_sensor = sigma_fn(fixed_x_sensor).reshape(1, -1)
    y_test = x_highres.reshape(-1, 1)
    s_test = u_true.reshape(-1, 1)
    return sigma_sensor, y_test, s_test


def save_train_test_data(train_data, test_data):
    save_path = os.path.join(data_dir, "train_test_data.npz")
    sigma_train, y_train, s_train = train_data
    sigma_test, y_test, s_test = test_data
    np.savez_compressed(
        save_path,
        sigma_train=sigma_train, y_train=y_train, s_train=s_train,
        sigma_test=sigma_test, y_test=y_test, s_test=s_test,
        x_sensor=x_sensor,
        n_cheb_train=n_cheb_train, n_cheb_test=n_cheb_test
    )



def load_train_test_data():
    load_path = os.path.join(data_dir, "train_test_data.npz")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"data file not found: {load_path}")
    data = np.load(load_path)
    assert np.allclose(data['x_sensor'], x_sensor), "sensor do not match"
    return (
        (data['sigma_train'], data['y_train'], data['s_train']),
        (data['sigma_test'], data['y_test'], data['s_test']),
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

    def forward(self, sigma_sensor, x):
        B = self.branch(sigma_sensor)
        T = self.trunk(x)
        return torch.sum(B * T, dim=-1, keepdim=True)

    def train_model(self, train_dataset, test_dataset, epochs):
        train_loader = data.DataLoader(train_dataset, batch_size=train_dataset.batch_size, shuffle=True)
        test_loader = data.DataLoader(test_dataset, batch_size=test_dataset.batch_size, shuffle=False)

        pbar = trange(epochs, desc="train DeepONet")
        for epoch in pbar:

            self.train()
            total_train_loss = 0.0
            for (sigma_batch, x_batch), s_batch in train_loader:
                sigma_batch = sigma_batch.to(device)
                x_batch = x_batch.to(device)
                s_batch = s_batch.to(device)

                self.optimizer.zero_grad()
                s_pred = self(sigma_batch, x_batch)
                loss = F.mse_loss(s_pred, s_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()


            self.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for (sigma_batch, x_batch), s_batch in test_loader:
                    sigma_batch = sigma_batch.to(device)
                    x_batch = x_batch.to(device)
                    s_batch = s_batch.to(device)
                    s_pred = self(sigma_batch, x_batch)
                    total_test_loss += F.mse_loss(s_pred, s_batch).item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_test_loss = total_test_loss / len(test_loader)
            self.loss_log.append(avg_train_loss)
            self.test_loss_log.append(avg_test_loss)

            pbar.set_postfix({
                'train MSE': f'{avg_train_loss:.6e}',
                'test MSE': f'{avg_test_loss:.6e}'
            })

    def predict(self, sigma_sensor, x):
        with torch.no_grad():
            sigma_tensor = torch.Tensor(sigma_sensor).to(device)
            x_tensor = torch.Tensor(x).to(device)
            return self(sigma_tensor, x_tensor).cpu().numpy()

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
train_coeff_range = (0.5, 2.0)
test_coeff_range = (0.5, 2.0)

x = np.linspace(xmin, xmax, 1000).astype(np.float64)
target_sigmas = [0.5, 1.0, 1.5, 2.0]

data_path = os.path.join(data_dir, "train_test_data.npz")
model_path = os.path.join(data_dir, "deeponet_model.pth")

if os.path.exists(data_path):

    train_data, test_data, loaded_n_train, loaded_n_test = load_train_test_data()
    sigma_train, y_train, s_train = train_data
    sigma_test, y_test, s_test = test_data
else:

    train_data = generate_batch_data(
        N=N_train, P=P_per_group, n_cheb=n_cheb_train,
        coeff_range=train_coeff_range, x_grid=x_base
    )

    test_data = generate_batch_data(
        N=N_test, P=P_per_group, n_cheb=n_cheb_test,
        coeff_range=test_coeff_range, x_grid=x_base
    )
    save_train_test_data(train_data, test_data)
    sigma_train, y_train, s_train = train_data
    sigma_test, y_test, s_test = test_data

train_dataset = DataGenerator(sigma_train, y_train, s_train, batch_size=batch_size)
test_dataset = DataGenerator(sigma_test, y_test, s_test, batch_size=batch_size)


branch_layers = [m,32,32,32,32]
trunk_layers = [1, 32,32,32,32]
model = DeepONet(branch_layers, trunk_layers).to(device)

if os.path.exists(model_path):
    model.load_model(model_path)
else:
    model.train_model(train_dataset, test_dataset, epochs=epochs)
    model.save_model(model_path)


def test_pre_sigma(model, sigma_values, x_highres):
    results = []
    for sigma in sigma_values:
        sigma_fn = lambda x: np.ones_like(x) * sigma
        sigma_sensor, y_test, s_test = generate_test_data(
            sigma_fn, x_highres, x_sensor
        )
        s_pred = model.predict(sigma_sensor, y_test)
        s_pred = np.clip(s_pred, 0.0, 1.0)
        l2_error = np.linalg.norm(s_pred - s_test) / np.linalg.norm(s_test)
        results.append((y_test.flatten(), s_test.flatten(), s_pred.flatten(), sigma, l2_error))
        print(f"σ={sigma:<4} | L2 error: {l2_error:.6e}")
    return results


fixed_sigma_results = test_pre_sigma(model, target_sigmas, x)


plt.figure(figsize=(16, 10))
for i, (x_vals, s_true, s_pred, sigma, l2_err) in enumerate(fixed_sigma_results):
    plt.subplot(2, 2, i + 1)
    plt.plot(x_vals, s_true, 'b-', linewidth=2, label='FDM ')
    plt.plot(x_vals, s_pred, 'r--', linewidth=2, label='DeepONet(PFS) ')
    plt.title(f'σ={sigma} (L2 error: {l2_err:.6f})', fontsize=12)
    plt.xlabel('x', fontsize=10)
    plt.ylabel('p(x)', fontsize=10)
    plt.ylim(-0.1, 1.1)
    plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(model.loss_log, label='train')
plt.plot(model.test_loss_log, label='test')
plt.yscale('log')
plt.xlabel('epochs')
plt.ylabel('MSE')
plt.legend()
plt.title('Training and Test Loss ', fontsize=12)
plt.show()