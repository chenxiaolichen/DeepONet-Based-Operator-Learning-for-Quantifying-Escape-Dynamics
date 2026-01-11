import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import trange
import torch.nn.functional as F
import os

data_dir = r"E:\gene\GRF\sig\ep"
os.makedirs(data_dir, exist_ok=True)

xmin, xmax = 0, 1.48971
Nx = 10000
eps = 1.0
x = np.linspace(xmin, xmax, Nx)
m = 10
x_sensor = np.linspace(xmin, xmax, m)


def RBF(x1, x2, lengthscales):
    diffs = np.expand_dims(x1 / lengthscales, 1) - np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return np.exp(-0.5 * r2)


def solve_pde(b_fn, sigma_fn, x):
    """
    PDE：ε/2 * (σ² u'') + b u' = 0
    u(0)=0，u(xmax)=1
    """
    N = len(x)
    h = x[1] - x[0]
    b_vals = b_fn(x)
    sig_vals = sigma_fn(x)

    Chh = (sig_vals ** 2) / (2 * eps) / (h ** 2)
    Cb = b_vals / h

    A = lil_matrix((N, N), dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)

    for i in range(1, N - 1):
        A[i, i - 1] = Chh[i] - 0.5 * Cb[i]
        A[i, i] = -2 * Chh[i]
        A[i, i + 1] = Chh[i] + 0.5 * Cb[i]

    i = 0
    A[i, i] = 1.0
    b[i] = 0.0

    i = N - 1
    A[i, i] = 1.0
    b[i] = 1.0
    A = csc_matrix(A)
    try:
        u = spsolve(A, b)
    except:
        from scipy.sparse.linalg import lsqr
        u = lsqr(A, b)[0]
    return u



def save_train_test_data(f_train, y_train, s_train, f_test, y_test, s_test):
    save_path = os.path.join(data_dir, "train_test_data.npz")
    np.savez_compressed(save_path,
                        f_train=f_train, y_train=y_train, s_train=s_train,
                        f_test=f_test, y_test=y_test, s_test=s_test,
                        x_sensor=x_sensor)



def load_train_test_data():
    load_path = os.path.join(data_dir, "train_test_data.npz")
    if not os.path.exists(load_path):
        raise FileNotFoundError(load_path)
    data = np.load(load_path)
    return (data['f_train'], data['y_train'], data['s_train'],
            data['f_test'], data['y_test'], data['s_test'],
            data['x_sensor'])


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
    X_grf = np.linspace(xmin, xmax, N_grf)[:, None]
    K = RBF(X_grf, X_grf, length_scale)
    L = np.linalg.cholesky(K + 1e-6 * np.eye(N_grf))
    gp_sample = L @ np.random.randn(N_grf)
    sigma_fn = lambda x: 0.5 + np.abs(np.interp(x, X_grf.flatten(), gp_sample))
    b_fn = lambda x: (6 * x ** 2) / (x ** 2 + 10) - 1 * x + 0.4
    u_true = solve_pde(b_fn, sigma_fn, x)
    sigma_sensor = sigma_fn(x_sensor)
    sigma_sensor = np.tile(sigma_sensor, (P, 1))

    idx_internal = np.random.randint(1, Nx - 1, size=P - 2)
    idx = np.concatenate([[0], idx_internal, [Nx - 1]])
    y = x[idx].reshape(-1, 1)
    s = u_true[idx].reshape(-1, 1)
    return sigma_sensor, y, s



def generate_batch_data(N, P):
    pbar = trange(N, desc="Generating data")
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

    def forward(self, f_sensor, x):
        B = self.branch(f_sensor)
        T = self.trunk(x)
        pred = torch.sum(B * T, dim=-1, keepdim=True)
        return pred

    def compute_loss(self, dataloader):
        total_loss = 0.0
        num_batches = 0
        for (f_batch, x_batch), s_batch in dataloader:
            f_batch = f_batch.to(next(self.parameters()).device)
            x_batch = x_batch.to(next(self.parameters()).device)
            s_batch = s_batch.to(next(self.parameters()).device)

            with torch.no_grad():
                s_pred = self(f_batch, x_batch)
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

        pbar = trange(epochs, desc="Training model")
        for epoch in pbar:
            self.train()
            total_train_loss = 0.0
            num_train_batches = 0

            for (f_batch, x_batch), s_batch in train_loader:
                f_batch = f_batch.to(next(self.parameters()).device)
                x_batch = x_batch.to(next(self.parameters()).device)
                s_batch = s_batch.to(next(self.parameters()).device)

                self.optimizer.zero_grad()
                s_pred = self(f_batch, x_batch)
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

    def predict(self, f_sensor, x):
        with torch.no_grad():
            f_tensor = torch.Tensor(f_sensor).to(next(self.parameters()).device)
            x_tensor = torch.Tensor(x).to(next(self.parameters()).device)
            return self(f_tensor, x_tensor).cpu().numpy()

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
data_path = os.path.join(data_dir, "train_test_data.npz")
model_path = os.path.join(data_dir, "deeponet_model.pth")


if os.path.exists(data_path):
    print("Loading train and test data...")
    data = np.load(data_path)
    f_train = data['f_train']
    y_train = data['y_train']
    s_train = data['s_train']
    f_test = data['f_test']
    y_test = data['y_test']
    s_test = data['s_test']
    loaded_x_sensor = data['x_sensor']
    assert np.allclose(loaded_x_sensor, x_sensor), "Sensor positions mismatch!"
else:
    print("Generating training data...")
    f_train, y_train, s_train = generate_batch_data(N_train, P_per_group)
    print("Generating test data...")
    f_test, y_test, s_test = generate_batch_data(N_test, P_per_group)
    save_train_test_data(f_train, y_train, s_train, f_test, y_test, s_test)

train_dataset = DataGenerator(f_train, y_train, s_train)
test_dataset = DataGenerator(f_test, y_test, s_test)



branch_layers = [m, 32, 32, 32, 32]
trunk_layers = [1, 32, 32, 32, 32]
model = DeepONet(branch_layers, trunk_layers).to(device)


model.train_model(train_dataset, test_dataset, epochs=15000, batch_size=batch_size)
model.save_model(model_path)


sigma_values = [0.5, 1.0]
l2_errors = []
x_test = np.linspace(xmin, xmax, 1000)


def generate_test_data(sigma_fn, x, fixed_x_sensor):
    b_fn = lambda x: (6 * x ** 2) / (x ** 2 + 10) - 1 * x + 0.4
    u_true = solve_pde(b_fn, sigma_fn, x)
    sigma_sensor = sigma_fn(fixed_x_sensor).reshape(1, m)
    y_test = x.reshape(-1, 1)
    s_test = u_true.reshape(-1, 1)
    return sigma_sensor, y_test, s_test


plt.figure(figsize=(12, 8))
for sigma in sigma_values:
    sigma_test_fn = lambda x: np.ones_like(x) * sigma
    f_test, y_test, s_test = generate_test_data(sigma_test_fn, x_test, x_sensor)
    s_pred = model.predict(f_test, y_test)
    l2 = np.linalg.norm(s_pred - s_test) / np.linalg.norm(s_test)
    l2_errors.append(l2)
    plt.plot(y_test, s_test, label=f'FDM (σ={sigma}) ', alpha=0.8, linewidth=2)
    plt.plot(y_test, s_pred, '--', label=f'DeepONet(GRF) (σ={sigma}) ', alpha=0.8, linewidth=2)

plt.xlabel('x', fontsize=12)
plt.ylabel('p(x)', fontsize=12)
plt.title('σ=1', fontsize=14)
plt.ylim(-0.1, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()


for sigma, err in zip(sigma_values, l2_errors):
    print(f"σ={sigma:<4} | Relative L2 error: {err:.6e}")


plt.figure(figsize=(10, 6))
plt.plot(model.loss_log, label='Training Loss', linewidth=1.5)
if model.test_loss_log:
    plt.plot(model.test_loss_log, label='Test Loss', linewidth=1.5)
plt.title('DeepONet Training and Test Loss Curves', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()