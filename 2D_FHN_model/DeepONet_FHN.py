import numpy as np
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import trange
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.special import chebyt
import os
from matplotlib.ticker import ScalarFormatter

data_dir = "test20"
os.makedirs(data_dir, exist_ok=True)

xmin, xmax = -2, 1
ymin, ymax = -3.664125, 2.335875
Nx, Ny = 50, 50
x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
X, Y = np.meshgrid(x, y)
grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
m = 50
sigma_fixed = 0.3


np.random.seed(42)
fixed_sensor_x = np.random.uniform(xmin, xmax, m)
fixed_sensor_y = np.random.uniform(ymin, ymax, m)
fixed_sensors = np.stack([fixed_sensor_x, fixed_sensor_y], axis=1)



def RBF_2d(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def generate_2d_grf(num_points=300, length_scale=100):
    x_sample = np.random.uniform(xmin, xmax, num_points)
    y_sample = np.random.uniform(ymin, ymax, num_points)
    points = np.stack([x_sample, y_sample], axis=1)

    K = RBF_2d(points, points, (1.0, length_scale)) + 1e-6 * np.eye(num_points)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = (K + K.T) / 2
        L = np.linalg.cholesky(K + 1e-5 * np.eye(num_points))

    grf_sample = np.dot(L, np.random.randn(num_points))
    grf_sample = np.exp(grf_sample)  # 确保为正
    f_grid = griddata(points, grf_sample, (X, Y), method='cubic')

    if np.isnan(f_grid).any():
        f_grid = np.nan_to_num(f_grid, nan=np.nanmean(f_grid))
    return f_grid


def generate_sigma_field(sigma_type="grf", length_scale=100, poly_degree=1, M=0.5):
    if sigma_type == "grf":
        sigma = generate_2d_grf(length_scale=length_scale)
        return np.clip(sigma, 0.5, 2.0)
    elif sigma_type == "chebyshev":
        np.random.seed(np.random.randint(0, 1000))
        coeffs = np.random.uniform(-M, M, poly_degree + 1)
        sigma_grid = np.zeros_like(X)
        for i in range(poly_degree + 1):
            x_norm = 2 * (X - xmin) / (xmax - xmin) - 1
            y_norm = 2 * (Y - ymin) / (ymax - ymin) - 1
            T_i_x = chebyt(i)(x_norm)
            T_i_y = chebyt(i)(y_norm)
            sigma_grid += coeffs[i] * T_i_x * T_i_y
        sigma_grid = np.exp(sigma_grid)
        return np.clip(sigma_grid, 0.5, 2.0)
    elif sigma_type == "constant":
        return np.ones_like(X) * sigma_fixed



def solve_pde_2d(sigma_field, X, Y):
    Ny, Nx = X.shape
    hx = X[0, 1] - X[0, 0]
    hy = Y[1, 0] - Y[0, 0]
    N = Nx * Ny
    A = np.zeros((N, N))
    b = np.ones(N) * (-1)

    def idx(i, j):
        return i * Nx + j

    for i in range(Ny):
        for j in range(Nx):
            if i == 0 or i == Ny - 1 or j == 0 or j == Nx - 1:
                A[idx(i, j), idx(i, j)] = 1.0
                b[idx(i, j)] = 0.0
            else:
                x_ij = X[i, j]
                y_ij = Y[i, j]
                c_x = x_ij - (x_ij ** 3) / 3 - y_ij
                c_y = 0.01 * (x_ij + 1.05)
                sigma_sq = sigma_field[i, j] ** 2
                sigma0_sq = sigma_fixed ** 2

                A[idx(i, j), idx(i, j - 1)] = -c_x / (2 * hx)
                A[idx(i, j), idx(i, j + 1)] = c_x / (2 * hx)
                A[idx(i, j), idx(i - 1, j)] = -c_y / (2 * hy)
                A[idx(i, j), idx(i + 1, j)] = c_y / (2 * hy)

                A[idx(i, j), idx(i, j - 1)] += 0.5 * sigma_sq / (hx ** 2)
                A[idx(i, j), idx(i, j + 1)] += 0.5 * sigma_sq / (hx ** 2)
                A[idx(i, j), idx(i, j)] -= 0.5 * sigma_sq / (hx ** 2) * 2

                A[idx(i, j), idx(i - 1, j)] += 0.5 * sigma0_sq / (hy ** 2)
                A[idx(i, j), idx(i + 1, j)] += 0.5 * sigma0_sq / (hy ** 2)
                A[idx(i, j), idx(i, j)] -= 0.5 * sigma0_sq / (hy ** 2) * 2

                A[idx(i, j), idx(i, j)] += 1e-8

    try:
        u_flat = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        u_flat = np.linalg.lstsq(A, b, rcond=None)[0]
    return u_flat.reshape(Ny, Nx)



def save_training_data(sigma_train, coords_train, u_train, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, sigma_train=sigma_train, coords_train=coords_train, u_train=u_train)



def load_training_data(load_path):
    data = np.load(load_path)
    return data['sigma_train'], data['coords_train'], data['u_train']


def save_test_data(sigma_test, coords_test, u_test, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, sigma_test=sigma_test, coords_test=coords_test, u_test=u_test)



def load_test_data(load_path):
    data = np.load(load_path)
    return data['sigma_test'], data['coords_test'], data['u_test']


def save_visual_test_data(sigma_test, coords_test, u_test, X_grid, Y_grid, u_true, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, sigma_test=sigma_test, coords_test=coords_test, u_test=u_test,
             X_grid=X_grid, Y_grid=Y_grid, u_true=u_true)



def load_visual_test_data(load_path):
    data = np.load(load_path)
    return data['sigma_test'], data['coords_test'], data['u_test'], data['X_grid'], data['Y_grid'], data['u_true']



class DataGenerator2D(data.Dataset):
    def __init__(self, sigma_data, coords_data, u_data, batch_size=64):
        mask = ~np.isnan(sigma_data).any(axis=1) & ~np.isnan(coords_data).any(axis=1) & ~np.isnan(u_data).any(axis=1)
        self.sigma_data = torch.Tensor(sigma_data[mask])
        self.coords_data = torch.Tensor(coords_data[mask])
        self.u_data = torch.Tensor(u_data[mask])
        self.N = self.sigma_data.shape[0]
        self.batch_size = batch_size
        assert self.N > 0, "dataset is empty"

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return (self.sigma_data[index], self.coords_data[index]), self.u_data[index]


def generate_one_data(P, sigma_type="grf"):
    sigma_grid = generate_sigma_field(sigma_type=sigma_type)
    while np.isnan(sigma_grid).any():
        sigma_grid = generate_sigma_field(sigma_type=sigma_type)

    u_true = solve_pde_2d(sigma_grid, X, Y)
    while np.isnan(u_true).any():
        sigma_grid = generate_sigma_field(sigma_type=sigma_type)
        u_true = solve_pde_2d(sigma_grid, X, Y)

    sigma_sensor = griddata(
        (X.flatten(), Y.flatten()),
        sigma_grid.flatten(),
        (fixed_sensor_x, fixed_sensor_y),
        method='cubic'
    )
    sigma_sensor = np.nan_to_num(sigma_sensor, nan=np.nanmean(sigma_sensor))
    sigma_sensor = np.tile(sigma_sensor.reshape(1, m), (P, 1))

    idx = np.random.choice(Nx * Ny, P, replace=False)
    coords = grid_points[idx]
    u_vals = u_true.flatten()[idx].reshape(-1, 1)

    return sigma_sensor, coords, u_vals, sigma_grid, u_true


def generate_dataset_2d(N, P, sigma_type="grf", is_test=False):
    pbar = trange(N, desc=f"generate{'test' if is_test else 'train'}data")
    sigma_list, coords_list, u_list = [], [], []
    for _ in pbar:
        sigma, c, u, _, _ = generate_one_data(P, sigma_type=sigma_type)
        sigma_list.append(sigma)
        coords_list.append(c)
        u_list.append(u)

    sigma_data = np.concatenate(sigma_list, axis=0)
    coords_data = np.concatenate(coords_list, axis=0)
    u_data = np.concatenate(u_list, axis=0)
    return sigma_data, coords_data, u_data


def generate_visual_test_data(sigma_fn):
    sigma_grid = sigma_fn(X, Y)
    u_true = solve_pde_2d(sigma_grid, X, Y)

    sigma_sensor = sigma_fn(fixed_sensor_x, fixed_sensor_y)
    sigma_sensor = np.nan_to_num(sigma_sensor, nan=np.nanmean(sigma_sensor)).reshape(1, m)

    coords_test = grid_points
    u_test = u_true.flatten().reshape(-1, 1)
    return sigma_sensor, coords_test, u_test, X, Y, u_true, sigma_grid



class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            torch.nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.01)
            self.layers.append(layer)
        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class DeepONet2D(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super(DeepONet2D, self).__init__()
        self.branch = MLP(branch_layers)
        self.trunk = MLP(trunk_layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.8
        )
        self.train_loss_log = []
        self.test_loss_log = []
        self.lr_log = []

    def forward(self, sigma_batch, coords_batch):

        branch_out = self.branch(sigma_batch)

        trunk_out = self.trunk(coords_batch)

        y = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
        return y

    def train_model(self, train_dataset, test_dataset, epochs):

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=train_dataset.batch_size,
            shuffle=True
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=test_dataset.batch_size,
            shuffle=False
        )

        pbar = trange(epochs)
        for epoch in pbar:

            self.train()
            total_train_loss = 0.0
            for (sigma_batch, coords_batch), u_batch in train_loader:
                sigma_batch = sigma_batch.to(next(self.parameters()).device)
                coords_batch = coords_batch.to(next(self.parameters()).device)
                u_batch = u_batch.to(next(self.parameters()).device)

                self.optimizer.zero_grad()
                u_pred = self(sigma_batch, coords_batch)
                loss = F.mse_loss(u_pred, u_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_train_loss += loss.item() * sigma_batch.size(0)


            self.eval()
            total_test_loss = 0.0
            with torch.no_grad():
                for (sigma_batch, coords_batch), u_batch in test_loader:
                    sigma_batch = sigma_batch.to(next(self.parameters()).device)
                    coords_batch = coords_batch.to(next(self.parameters()).device)
                    u_batch = u_batch.to(next(self.parameters()).device)

                    u_pred = self(sigma_batch, coords_batch)
                    loss = F.mse_loss(u_pred, u_batch)
                    total_test_loss += loss.item() * sigma_batch.size(0)


            avg_train_loss = total_train_loss / len(train_dataset)
            avg_test_loss = total_test_loss / len(test_dataset)


            self.train_loss_log.append(avg_train_loss)
            self.test_loss_log.append(avg_test_loss)
            self.lr_log.append(self.optimizer.param_groups[0]['lr'])


            self.scheduler.step()


            if epoch % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'Train Loss': f'{avg_train_loss:.6e}',
                    'Test Loss': f'{avg_test_loss:.6e}',
                    'LR': f'{current_lr:.6e}'
                })

    def predict(self, sigma_sensor, coords):

        self.eval()
        with torch.no_grad():
            sigma_tensor = torch.Tensor(sigma_sensor).to(next(self.parameters()).device)
            coords_tensor = torch.Tensor(coords).to(next(self.parameters()).device)
            sigma_tensor = sigma_tensor.repeat(coords_tensor.shape[0], 1)
            return self(sigma_tensor, coords_tensor).cpu().numpy()

    def save_model(self, path):

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.train_loss_log,
            'test_loss': self.test_loss_log
        }, path)


    def load_model(self, path):

        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_loss_log = checkpoint['train_loss']
        self.test_loss_log = checkpoint['test_loss']




if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    N_train = 300
    N_test = 100
    P_per_group = 100
    batch_size = 256
    sigma_type = "grf"


    train_data_path = os.path.join(data_dir, "training_data.npz")
    test_data_path = os.path.join(data_dir, "testing_data.npz")

    if os.path.exists(train_data_path):

        sigma_train, coords_train, u_train = load_training_data(train_data_path)
    else:

        sigma_train, coords_train, u_train = generate_dataset_2d(
            N_train, P_per_group, sigma_type=sigma_type, is_test=False
        )
        save_training_data(sigma_train, coords_train, u_train, train_data_path)

    if os.path.exists(test_data_path):

        sigma_test, coords_test, u_test = load_test_data(test_data_path)
    else:

        sigma_test, coords_test, u_test = generate_dataset_2d(
            N_test, P_per_group, sigma_type=sigma_type, is_test=True
        )
        save_test_data(sigma_test, coords_test, u_test, test_data_path)


    train_dataset = DataGenerator2D(sigma_train, coords_train, u_train, batch_size=batch_size)
    test_dataset = DataGenerator2D(sigma_test, coords_test, u_test, batch_size=batch_size)


    branch_layers = [m, 100, 100, 100, 100, 100]
    trunk_layers = [2, 100, 100, 100, 100, 100]
    model = DeepONet2D(branch_layers, trunk_layers).to(device)

    model.train_model(train_dataset, test_dataset, epochs=20000)


    model_save_path = os.path.join(data_dir, "deeponet_model.pth")
    model.save_model(model_save_path)


    plt.figure(figsize=(15, 6))


    plt.subplot(1, 2, 1)
    plt.plot(model.train_loss_log, label='Training MSE', alpha=0.7)
    plt.plot(model.test_loss_log, label='Testing MSE', alpha=0.7)

    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)


    plt.subplot(1, 2, 2)
    plt.plot(model.lr_log, label='Learning Rate', color='orange')

    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "loss_lr_curves.png"))
    plt.show()





    def integral_2d(f, X, Y):

        Ny, Nx = X.shape
        hx = X[0, 1] - X[0, 0]
        hy = Y[1, 0] - Y[0, 0]
        return np.sum(f * hx * hy)


    sigma_functions = [
        lambda x, y: np.ones_like(x) * 0.5,  # σ=0.5
        lambda x, y: np.ones_like(x) * 1.0,  # σ=1.0
        lambda x, y: np.ones_like(x) * 1.5,  # σ=1.5
        lambda x, y: np.ones_like(x) * 2.0  # σ=2.0
    ]
    sigma_names = ["σ=0.5", "σ=1.0", "σ=1.5", "σ=2.0"]


    for i, (sigma_fn, sigma_name) in enumerate(zip(sigma_functions, sigma_names)):



        sigma_test, coords_test, u_test, X_grid, Y_grid, u_true, sigma_test_grid = generate_visual_test_data(sigma_fn)

        test_save_path = os.path.join(data_dir, f"visual_test_data_{i}.npz")
        save_visual_test_data(sigma_test, coords_test, u_test, X_grid, Y_grid, u_true, test_save_path)


        u_pred = model.predict(sigma_test, coords_test)
        u_pred_grid = u_pred.reshape(Y_grid.shape)


        if np.isnan(u_true).any() or np.isnan(u_pred_grid).any():

            relative_error = np.nan
        else:
            numerator = np.linalg.norm(u_pred_grid - u_true, ord=2)
            denominator = integral_2d(u_true, X_grid, Y_grid)
            relative_error = numerator / (denominator + 1e-10)


        # 结果可视化
        fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(32, 6))


        contour1 = ax2.contourf(X_grid, Y_grid, u_true, levels=100, cmap='viridis')
        ax2.set_title('Reference Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(contour1, ax=ax2)


        contour2 = ax3.contourf(X_grid, Y_grid, u_pred_grid, levels=100, cmap='viridis')
        ax3.set_title('DeepONet Prediction')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(contour2, ax=ax3)


        error = np.abs(u_pred_grid - u_true) / (denominator + 1e-10)  # 相对误差
        contour3 = ax4.contourf(X_grid, Y_grid, error, levels=100, cmap='Reds')
        ax4.set_title(f'Relative Error (Integral-based: {relative_error:.2e})')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        cbar3 = plt.colorbar(contour3, ax=ax4)


        formatter = ScalarFormatter()
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        cbar3.ax.yaxis.set_major_formatter(formatter)

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"results_{i}.png"))
        plt.show()