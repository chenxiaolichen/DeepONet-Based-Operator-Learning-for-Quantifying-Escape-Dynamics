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


data_dir = r"E:\gene\GRF\drift\ep"
os.makedirs(data_dir, exist_ok=True)


xmin, xmax = 0, 2.8709
Nx = 10000
x = np.linspace(xmin, xmax, Nx)
m = 30
D = 0.5


def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def solve_ode(f, x):

    N = len(x)
    h = x[1] - x[0]
    f_vals = f(x)

    A = np.zeros((N, N))
    b = np.zeros(N)

    A[0, 0] = 1.0
    b[0] = 0.0
    A[-1, -1] = 1.0
    b[-1] = 1.0


    for i in range(1, N - 1):
        A[i, i - 1] = D * (1 / h ** 2) - f_vals[i] / (2 * h)
        A[i, i] = -D * (2 / h ** 2)
        A[i, i + 1] = D * (1 / h ** 2) + f_vals[i] / (2 * h)


    A = csc_matrix(A)
    try:
        u = spsolve(A, b)
    except:
        from scipy.sparse.linalg import lsqr
        u = lsqr(A, b)[0]
    return u



def save_train_test_data(f_train, y_train, s_train, f_test, y_test, s_test):

    save_path = os.path.join(data_dir, "train_test_data.npz")
    np.savez_compressed(
        save_path,
        f_train=f_train, y_train=y_train, s_train=s_train,
        f_test=f_test, y_test=y_test, s_test=s_test
    )



def load_train_test_data():

    load_path = os.path.join(data_dir, "train_test_data.npz")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"data file not found: {load_path}")
    data = np.load(load_path)
    return (data['f_train'], data['y_train'], data['s_train'],
            data['f_test'], data['y_test'], data['s_test'])



class DataGenerator(data.Dataset):
    def __init__(self, f_sensor, y, s):
        self.f_sensor = torch.Tensor(f_sensor)
        self.y = torch.Tensor(y)
        self.s = torch.Tensor(s)
        self.N = f_sensor.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return (self.f_sensor[index], self.y[index]), self.s[index]


def generate_one_sample(P):

    N_grf = 500
    length_scale = 0.5
    X = np.linspace(xmin, xmax, N_grf)[:, None]
    K = RBF(X, X, (1.0, length_scale))
    L = np.linalg.cholesky(K + 1e-10 * np.eye(N_grf))
    gp_sample = np.dot(L, np.random.randn(N_grf))
    f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)


    u_true = solve_ode(f_fn, x)


    x_sensor = np.linspace(xmin, xmax, m)
    f_sensor = f_fn(x_sensor)
    f_sensor = np.tile(f_sensor, (P, 1))


    idx = np.random.randint(0, Nx, size=P)
    y = x[idx].reshape(-1, 1)
    s = u_true[idx].reshape(-1, 1)

    return f_sensor, y, s


def generate_batch_data(N, P):

    pbar = trange(N, desc=f"生成{N}组数据")
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



def generate_test_data(f_fn):

    u_true = solve_ode(f_fn, x)
    x_sensor = np.linspace(xmin, xmax, m)
    f_sensor = f_fn(x_sensor).reshape(1, m)
    y_test = x.reshape(-1, 1)
    s_test = u_true.reshape(-1, 1)
    return f_sensor, y_test, s_test


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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.train_loss_log = []
        self.test_loss_log = []

    def forward(self, f_sensor, x):

        B = self.branch(f_sensor)  # [batch, dim]
        T = self.trunk(x)  # [batch, dim]
        return torch.sum(B * T, dim=-1, keepdim=True)

    def compute_loss(self, dataloader, device):

        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for (f_batch, x_batch), s_batch in dataloader:
                f_batch = f_batch.to(device)
                x_batch = x_batch.to(device)
                s_batch = s_batch.to(device)
                s_pred = self(f_batch, x_batch)
                total_loss += F.mse_loss(s_pred, s_batch).item()
        return total_loss / len(dataloader)

    def train_model(self, train_loader, test_loader, epochs, device):

        pbar = trange(epochs, desc="train model")
        for epoch in pbar:

            self.train()
            total_train_loss = 0.0
            for (f_batch, x_batch), s_batch in train_loader:
                f_batch = f_batch.to(device)
                x_batch = x_batch.to(device)
                s_batch = s_batch.to(device)

                self.optimizer.zero_grad()
                s_pred = self(f_batch, x_batch)
                loss = F.mse_loss(s_pred, s_batch)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            self.train_loss_log.append(avg_train_loss)

            avg_test_loss = 0.0
            if test_loader is not None:
                avg_test_loss = self.compute_loss(test_loader, device)
                self.test_loss_log.append(avg_test_loss)

            self.scheduler.step()

            if epoch % 100 == 0:
                pbar.set_postfix({
                    'Train Loss': f'{avg_train_loss:.6e}',
                    'Test Loss': f'{avg_test_loss:.6e}'
                })

    def predict(self, f_sensor, x):
        with torch.no_grad():
            device = next(self.parameters()).device
            f_tensor = torch.Tensor(f_sensor).to(device)
            x_tensor = torch.Tensor(x).to(device)
            return self(f_tensor, x_tensor).cpu().numpy()

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss_log': self.train_loss_log,
            'test_loss_log': self.test_loss_log
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_loss_log = checkpoint['train_loss_log']
        self.test_loss_log = checkpoint.get('test_loss_log', [])



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N_train = 500
    N_test = 100
    P_per_group = 100
    batch_size = 256
    data_path = os.path.join(data_dir, "train_test_data.npz")
    model_path = os.path.join(data_dir, 'deeponet_model.pth')

    if os.path.exists(data_path):
        f_train, y_train, s_train, f_test, y_test, s_test = load_train_test_data()
    else:
        f_train, y_train, s_train = generate_batch_data(N_train, P_per_group)
        f_test, y_test, s_test = generate_batch_data(N_test, P_per_group)
        save_train_test_data(f_train, y_train, s_train, f_test, y_test, s_test)

    train_dataset = DataGenerator(f_train, y_train, s_train)
    test_dataset = DataGenerator(f_test, y_test, s_test)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    branch_layers = [m, 64, 64, 64, 64]
    trunk_layers = [1, 64, 64, 64, 64]
    model = DeepONet(branch_layers, trunk_layers).to(device)
    model.train_model(train_loader, test_loader, epochs=20000, device=device)

    model.save_model(model_path)


    param_configs = {
        'Kd': {
            'range': np.linspace(9.0, 11.7, 50),
            'selected': [9.0, 10.0, 11.0, 11.7],
            'colors': ['b', 'g', 'r', 'c'],
            'styles': ['-', '-', '-', '-']
        },
        'kd': {
            'range': np.linspace(0.95, 1.08, 50),
            'selected': [0.95, 1.0, 1.05, 1.08],
            'colors': ['m', 'y', 'k', 'C1'],
            'styles': ['-', '-', '-', '-']
        },
        'kf': {
            'range': np.linspace(5.5, 6.6, 50),
            'selected': [5.5, 6.0, 6.3, 6.6],
            'colors': ['C2', 'C3', 'C4', 'C5'],
            'styles': ['-', '-', '-', '-']
        },
        'Rbas': {
            'range': np.linspace(0.16, 0.45, 50),
            'selected': [0.16, 0.25, 0.35, 0.45],
            'colors': ['C6', 'C7', 'C8', 'C9'],
            'styles': ['-', '-', '-', '-']
        }
    }

    initial_params = {
        'Kd': 10.0,
        'kd': 1.0,
        'kf': 6.0,
        'Rbas': 0.4
    }

    Nx_test = 1000
    x_test_highres = np.linspace(xmin, xmax, Nx_test)

    for param_name in param_configs.keys():

        plt.figure(figsize=(10, 6))
        plt.title(f'Parameter: {param_name} - True vs Predicted', fontsize=14)

        for idx, param_value in enumerate(param_configs[param_name]['selected']):

            current_params = initial_params.copy()
            current_params[param_name] = param_value

            def f_current(x):
                Kd = current_params['Kd']
                kd = current_params['kd']
                kf = current_params['kf']
                Rbas = current_params['Rbas']
                return (kf * x ** 2) / (x ** 2 + Kd) - kd * x + Rbas

            x_sensor = np.linspace(xmin, xmax, m)
            f_sensor = f_current(x_sensor).reshape(1, m)
            y_test = x_test_highres.reshape(-1, 1)
            s_test = solve_ode(f_current, x_test_highres).reshape(-1, 1)

            s_pred = model.predict(f_sensor, y_test)

            l2_error = np.linalg.norm(s_pred - s_test) / np.linalg.norm(s_test)

            color = param_configs[param_name]['colors'][idx]
            style = param_configs[param_name]['styles'][idx]

            plt.plot(y_test, s_test, f'{color}{style}',
                     label=f'True (${param_name}={param_value:.3f}$, L2={l2_error:.2e})')

            plt.plot(y_test, s_pred, f'{color}--', alpha=0.8)

        plt.xlabel('x', fontsize=12)
        plt.ylabel('u(x)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(model.train_loss_log, label='Train Loss')
    if len(model.test_loss_log) > 0:
        plt.plot(model.test_loss_log, label='Test Loss')
    plt.title('Training and Test Loss Curve', fontsize=12)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('MSE', fontsize=10)
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()