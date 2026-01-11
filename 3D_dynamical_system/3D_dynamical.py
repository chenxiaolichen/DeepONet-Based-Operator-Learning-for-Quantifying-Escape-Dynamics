import numpy as np
from torch.utils import data
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import trange
import torch.nn.functional as F
from scipy.interpolate import griddata
import os
from matplotlib.ticker import ScalarFormatter

data_dir = "results"
os.makedirs(data_dir, exist_ok=True)


xmin, xmax = -1, 1
ymin, ymax = -1.5, 0
zmin, zmax = -1, 1
Nx, Ny, Nz = 30, 30, 30
x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
z = np.linspace(zmin, zmax, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
m = 50


np.random.seed(42)
fixed_sensor_x = np.random.uniform(xmin, xmax, m)
fixed_sensor_y = np.random.uniform(ymin, ymax, m)
fixed_sensor_z = np.random.uniform(zmin, zmax, m)
fixed_sensors = np.stack([fixed_sensor_x, fixed_sensor_y, fixed_sensor_z], axis=1)


def RBF_3d(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def generate_3d_grf(num_points=500, length_scale=100):
    x_sample = np.random.uniform(xmin, xmax, num_points)
    y_sample = np.random.uniform(ymin, ymax, num_points)
    z_sample = np.random.uniform(zmin, zmax, num_points)
    points = np.stack([x_sample, y_sample, z_sample], axis=1)


    K = RBF_3d(points, points, (1.0, length_scale)) + 1e-6 * np.eye(num_points)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = (K + K.T) / 2
        L = np.linalg.cholesky(K + 1e-5 * np.eye(num_points))

    grf_sample = np.dot(L, np.random.randn(num_points))
    grf_sample = np.exp(grf_sample)

    points_grid = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    f_grid = griddata(points, grf_sample, points_grid, method='linear')
    f_grid = f_grid.reshape(Nx, Ny, Nz)

    if np.isnan(f_grid).any():
        f_grid_nn = griddata(points, grf_sample, points_grid, method='nearest')
        f_grid_nn = f_grid_nn.reshape(Nx, Ny, Nz)
        f_grid = np.where(np.isnan(f_grid), f_grid_nn, f_grid)
    return f_grid


def generate_sigma_field(length_scale=100):
    sigma = generate_3d_grf(length_scale=length_scale)
    return np.clip(sigma, 0.5, 2.0)



# PDE：(y - y³)u_x + z u_y + x u_z + 0.5σ²(u_xx + u_yy + u_zz) = -1
# u|∂D = 0
def solve_pde_3d(sigma_field, X, Y, Z):
    Nx, Ny, Nz = X.shape
    hx = X[1, 0, 0] - X[0, 0, 0]
    hy = Y[0, 1, 0] - Y[0, 0, 0]
    hz = Z[0, 0, 1] - Z[0, 0, 0]
    N = Nx * Ny * Nz
    A = np.zeros((N, N))
    b = np.ones(N) * (-1)


    def idx(i, j, k):
        return i * Ny * Nz + j * Nz + k


    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):

                if i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1 or k == 0 or k == Nz - 1:
                    A[idx(i, j, k), idx(i, j, k)] = 1.0
                    b[idx(i, j, k)] = 0.0
                else:
                    x_ijk = X[i, j, k]
                    y_ijk = Y[i, j, k]
                    z_ijk = Z[i, j, k]


                    c_x = y_ijk - y_ijk ** 3
                    c_y = z_ijk
                    c_z = x_ijk
                    sigma_sq = sigma_field[i, j, k] ** 2


                    if i >= 2 and i <= Nx - 3:
                        A[idx(i, j, k), idx(i - 2, j, k)] += -c_x / (12 * hx)
                        A[idx(i, j, k), idx(i - 1, j, k)] += 8 * c_x / (12 * hx)
                        A[idx(i, j, k), idx(i + 1, j, k)] += -8 * c_x / (12 * hx)
                        A[idx(i, j, k), idx(i + 2, j, k)] += c_x / (12 * hx)
                    else:
                        A[idx(i, j, k), idx(i - 1, j, k)] += -c_x / (2 * hx)
                        A[idx(i, j, k), idx(i + 1, j, k)] += c_x / (2 * hx)


                    if j >= 2 and j <= Ny - 3:
                        A[idx(i, j, k), idx(i, j - 2, k)] += -c_y / (12 * hy)
                        A[idx(i, j, k), idx(i, j - 1, k)] += 8 * c_y / (12 * hy)
                        A[idx(i, j, k), idx(i, j + 1, k)] += -8 * c_y / (12 * hy)
                        A[idx(i, j, k), idx(i, j + 2, k)] += c_y / (12 * hy)
                    else:
                        A[idx(i, j, k), idx(i, j - 1, k)] += -c_y / (2 * hy)
                        A[idx(i, j, k), idx(i, j + 1, k)] += c_y / (2 * hy)


                    if k >= 2 and k <= Nz - 3:
                        A[idx(i, j, k), idx(i, j, k - 2)] += -c_z / (12 * hz)
                        A[idx(i, j, k), idx(i, j, k - 1)] += 8 * c_z / (12 * hz)
                        A[idx(i, j, k), idx(i, j, k + 1)] += -8 * c_z / (12 * hz)
                        A[idx(i, j, k), idx(i, j, k + 2)] += c_z / (12 * hz)
                    else:
                        A[idx(i, j, k), idx(i, j, k - 1)] += -c_z / (2 * hz)
                        A[idx(i, j, k), idx(i, j, k + 1)] += c_z / (2 * hz)


                    if i >= 2 and i <= Nx - 3:
                        coeff = 0.5 * sigma_sq / (12 * hx ** 2)
                        A[idx(i, j, k), idx(i - 2, j, k)] += -coeff
                        A[idx(i, j, k), idx(i - 1, j, k)] += 16 * coeff
                        A[idx(i, j, k), idx(i, j, k)] += -30 * coeff
                        A[idx(i, j, k), idx(i + 1, j, k)] += 16 * coeff
                        A[idx(i, j, k), idx(i + 2, j, k)] += -coeff
                    else:
                        coeff = 0.5 * sigma_sq / (hx ** 2)
                        A[idx(i, j, k), idx(i - 1, j, k)] += coeff
                        A[idx(i, j, k), idx(i + 1, j, k)] += coeff
                        A[idx(i, j, k), idx(i, j, k)] += -2 * coeff


                    if j >= 2 and j <= Ny - 3:
                        coeff = 0.5 * sigma_sq / (12 * hy ** 2)
                        A[idx(i, j, k), idx(i, j - 2, k)] += -coeff
                        A[idx(i, j, k), idx(i, j - 1, k)] += 16 * coeff
                        A[idx(i, j, k), idx(i, j, k)] += -30 * coeff
                        A[idx(i, j, k), idx(i, j + 1, k)] += 16 * coeff
                        A[idx(i, j, k), idx(i, j + 2, k)] += -coeff
                    else:
                        coeff = 0.5 * sigma_sq / (hy ** 2)
                        A[idx(i, j, k), idx(i, j - 1, k)] += coeff
                        A[idx(i, j, k), idx(i, j + 1, k)] += coeff
                        A[idx(i, j, k), idx(i, j, k)] += -2 * coeff


                    if k >= 2 and k <= Nz - 3:
                        coeff = 0.5 * sigma_sq / (12 * hz ** 2)
                        A[idx(i, j, k), idx(i, j, k - 2)] += -coeff
                        A[idx(i, j, k), idx(i, j, k - 1)] += 16 * coeff
                        A[idx(i, j, k), idx(i, j, k)] += -30 * coeff
                        A[idx(i, j, k), idx(i, j, k + 1)] += 16 * coeff
                        A[idx(i, j, k), idx(i, j, k + 2)] += -coeff
                    else:
                        coeff = 0.5 * sigma_sq / (hz ** 2)
                        A[idx(i, j, k), idx(i, j, k - 1)] += coeff
                        A[idx(i, j, k), idx(i, j, k + 1)] += coeff
                        A[idx(i, j, k), idx(i, j, k)] += -2 * coeff


                    A[idx(i, j, k), idx(i, j, k)] += 1e-8


    try:
        u_flat = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:

        u_flat = np.linalg.lstsq(A, b, rcond=None)[0]
    return u_flat.reshape(Nx, Ny, Nz)



def save_training_data(sigma_train, coords_train, u_train, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path,
             sigma_train=sigma_train,
             coords_train=coords_train,
             u_train=u_train)



def load_training_data(load_path):
    data = np.load(load_path)
    return data['sigma_train'], data['coords_train'], data['u_train']


def save_test_dataset(sigma_test, coords_test, u_test, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path,
             sigma_test=sigma_test,
             coords_test=coords_test,
             u_test=u_test)



def load_test_dataset(load_path):

    data = np.load(load_path)
    return data['sigma_test'], data['coords_test'], data['u_test']


def save_test_data(sigma_test, coords_test, u_test, X_grid, Y_grid, Z_grid, u_true, save_path):

    np.savez(save_path,
             sigma_test=sigma_test,
             coords_test=coords_test,
             u_test=u_test,
             X_grid=X_grid,
             Y_grid=Y_grid,
             Z_grid=Z_grid,
             u_true=u_true)



def load_test_data(load_path):
    data = np.load(load_path)
    return data['sigma_test'], data['coords_test'], data['u_test'], data['X_grid'], data['Y_grid'], data['Z_grid'], \
    data['u_true']



class DataGenerator3D(data.Dataset):
    def __init__(self, sigma_sensor, coords, u_vals, batch_size=64):
        # 过滤含NaN的样本
        mask = ~np.isnan(sigma_sensor).any(axis=1) & ~np.isnan(coords).any(axis=1) & ~np.isnan(u_vals).any(axis=1)
        self.sigma_sensor = torch.Tensor(sigma_sensor[mask])
        self.coords = torch.Tensor(coords[mask])
        self.u_vals = torch.Tensor(u_vals[mask])
        self.N = self.sigma_sensor.shape[0]
        self.batch_size = batch_size
        assert self.N > 0, "dataset is empty"

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        return (self.sigma_sensor[index], self.coords[index]), self.u_vals[index]


def generate_one_training_data(P):

    sigma_grid = generate_sigma_field()
    while np.isnan(sigma_grid).any():
        sigma_grid = generate_sigma_field()

    u_true = solve_pde_3d(sigma_grid, X, Y, Z)
    while np.isnan(u_true).any():
        sigma_grid = generate_sigma_field()
        u_true = solve_pde_3d(sigma_grid, X, Y, Z)


    sigma_sensor = griddata(
        (X.flatten(), Y.flatten(), Z.flatten()),
        sigma_grid.flatten(),
        (fixed_sensor_x, fixed_sensor_y, fixed_sensor_z),
        method='linear'
    )

    if np.isnan(sigma_sensor).any():
        sigma_sensor_nn = griddata(
            (X.flatten(), Y.flatten(), Z.flatten()),
            sigma_grid.flatten(),
            (fixed_sensor_x, fixed_sensor_y, fixed_sensor_z),
            method='nearest'
        )
        sigma_sensor = np.where(np.isnan(sigma_sensor), sigma_sensor_nn, sigma_sensor)
    # 扩展维度以匹配查询点数量
    sigma_sensor = np.tile(sigma_sensor.reshape(1, m), (P, 1))

    # 随机选择P个查询点
    idx = np.random.choice(Nx * Ny * Nz, P, replace=False)
    coords = grid_points[idx]
    u_vals = u_true.flatten()[idx].reshape(-1, 1)

    return sigma_sensor, coords, u_vals, sigma_grid


def generate_training_data_3d(N, P):

    pbar = trange(N, desc="training data generation")
    sigma_list, coords_list, u_list, sigma_field_list = [], [], [], []
    for _ in pbar:
        sigma, c, u, sigma_grid = generate_one_training_data(P)
        sigma_list.append(sigma)
        coords_list.append(c)
        u_list.append(u)
        sigma_field_list.append(sigma_grid)

    sigma_train = np.concatenate(sigma_list, axis=0)
    coords_train = np.concatenate(coords_list, axis=0)
    u_train = np.concatenate(u_list, axis=0)
    return sigma_train, coords_train, u_train, sigma_field_list


def generate_test_dataset_3d(N_test, P_per_group):

    pbar = trange(N_test, desc="test data generation")
    sigma_list, coords_list, u_list = [], [], []
    for _ in pbar:
        sigma, c, u, _ = generate_one_training_data(P_per_group)
        sigma_list.append(sigma)
        coords_list.append(c)
        u_list.append(u)

    sigma_test = np.concatenate(sigma_list, axis=0)
    coords_test = np.concatenate(coords_list, axis=0)
    u_test = np.concatenate(u_list, axis=0)
    return sigma_test, coords_test, u_test


def generate_test_data_3d(sigma_fn):

    sigma_grid = sigma_fn(X, Y, Z)
    u_true = solve_pde_3d(sigma_grid, X, Y, Z)
    sigma_sensor = sigma_fn(fixed_sensor_x, fixed_sensor_y, fixed_sensor_z)
    sigma_sensor = np.nan_to_num(sigma_sensor, nan=np.nanmean(sigma_sensor)).reshape(1, m)
    coords_test = grid_points
    u_test = u_true.flatten().reshape(-1, 1)
    return sigma_sensor, coords_test, u_test, X, Y, Z, u_true, sigma_grid



class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            torch.nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.00)
            self.layers.append(layer)
        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class DeepONet3D(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super(DeepONet3D, self).__init__()
        self.branch = MLP(branch_layers)
        self.trunk = MLP(trunk_layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        self.train_loss_log = []
        self.test_loss_log = []
        self.lr_log = []

    def forward(self, sigma_sensor, coords):
        B = self.branch(sigma_sensor)
        T = self.trunk(coords)
        return torch.sum(B * T, dim=-1, keepdim=True)

    def train_model(self, train_dataset, test_dataset, epochs):

        train_loader = data.DataLoader(train_dataset, batch_size=train_dataset.batch_size, shuffle=True)
        test_loader = data.DataLoader(test_dataset, batch_size=test_dataset.batch_size, shuffle=False)

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
                pbar.set_postfix({
                    'Train Loss': f'{avg_train_loss:.6e}',
                    'Test Loss': f'{avg_test_loss:.6e}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.6e}'
                })

    def predict(self, sigma_sensor, coords):

        with torch.no_grad():
            sigma_tensor = torch.Tensor(sigma_sensor).to(next(self.parameters()).device)
            coords_tensor = torch.Tensor(coords).to(next(self.parameters()).device)
            sigma_tensor = sigma_tensor.repeat(coords_tensor.shape[0], 1)  # 扩展维度匹配坐标数量
            return self(sigma_tensor, coords_tensor).cpu().numpy()

    def save_model(self, path):

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss_log': self.train_loss_log,
            'test_loss_log': self.test_loss_log,
            'lr_log': self.lr_log
        }, path)


    def load_model(self, path):

        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_loss_log = checkpoint['train_loss_log']
        self.test_loss_log = checkpoint['test_loss_log']
        self.lr_log = checkpoint['lr_log']



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    N_train = 300
    N_test = 50
    P_per_group = 100
    batch_size = 256
    train_data_path = os.path.join(data_dir, "training_data_3d.npz")
    test_data_path = os.path.join(data_dir, "testing_data_3d.npz")


    if os.path.exists(train_data_path):

        sigma_train, coords_train, u_train = load_training_data(train_data_path)
    else:

        sigma_train, coords_train, u_train, _ = generate_training_data_3d(N_train, P_per_group)
        save_training_data(sigma_train, coords_train, u_train, train_data_path)


    if os.path.exists(test_data_path):

        sigma_test, coords_test, u_test = load_test_dataset(test_data_path)
    else:

        sigma_test, coords_test, u_test = generate_test_dataset_3d(N_test, P_per_group)
        save_test_dataset(sigma_test, coords_test, u_test, test_data_path)


    train_dataset = DataGenerator3D(sigma_train, coords_train, u_train, batch_size=batch_size)
    test_dataset = DataGenerator3D(sigma_test, coords_test, u_test, batch_size=batch_size)

    # 模型定义与训练
    branch_layers = [m,256,256,256,256,256,256]
    trunk_layers = [3, 256,256,256,256,256,256]
    model = DeepONet3D(branch_layers, trunk_layers).to(device)

    model.train_model(train_dataset, test_dataset, epochs=20000)


    model_save_path = os.path.join(data_dir, "deeponet_model_3d.pth")
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
    # 学习率曲线
    plt.subplot(1, 2, 2)
    plt.plot(model.lr_log, label='Learning Rate', color='orange')
    plt.title('Learning Rate Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "loss_lr_curves_4th.png"))
    plt.show()





    def sigma_constant1(x, y, z):
        return np.ones_like(x) * 0.5


    def sigma_constant2(x, y, z):
        return np.ones_like(x) * 1.0


    def sigma_constant3(x, y, z):
        return np.ones_like(x) * 1.5


    def sigma_constant4(x, y, z):
        return np.ones_like(x) * 2.0






    for i, sigma_fn in enumerate(
            [sigma_constant1, sigma_constant2, sigma_constant3, sigma_constant4]):
        sigma_name = ["σ=0.5", "σ=1.0", "σ=1.5", "σ=2.0"][i]


        sigma_test, coords_test, u_test, X_grid, Y_grid, Z_grid, u_true, sigma_test_grid = generate_test_data_3d(
            sigma_fn)


        test_save_path = os.path.join(data_dir, f"test_data_{i}.npz")
        np.savez_compressed(
            test_save_path,
            sigma_test=sigma_test,
            coords_test=coords_test,
            u_test=u_test,
            u_true=u_true,
            sigma_test_grid=sigma_test_grid
        )



        u_pred = model.predict(sigma_test, coords_test)
        u_pred_grid = u_pred.reshape(Nx, Ny, Nz)  # 转换为三维网格


        if np.isnan(u_test).any() or np.isnan(u_pred).any():

            l2_error = np.nan
        else:
            l2_error = np.linalg.norm(u_pred - u_test) / np.linalg.norm(u_test + 1e-10)



        z_slice = Nz // 2
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(24, 6))


        contour1 = ax2.contourf(X_grid[:, :, z_slice], Y_grid[:, :, z_slice], u_true[:, :, z_slice], levels=100,
                                cmap='viridis')

        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(contour1, ax=ax2)

        # 预测解
        contour2 = ax3.contourf(X_grid[:, :, z_slice], Y_grid[:, :, z_slice], u_pred_grid[:, :, z_slice], levels=100,
                                cmap='viridis')
        ax3.set_title('DeepONet Prediction (z-slice)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(contour2, ax=ax3)

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"solution_{i}_4th.png"))
        plt.show()


        fig, ax = plt.subplots(figsize=(8, 6))
        error = np.abs(u_pred_grid[:, :, z_slice] - u_true[:, :, z_slice])
        contour3 = ax.contourf(X_grid[:, :, z_slice], Y_grid[:, :, z_slice], error, levels=100, cmap='Reds')
        ax.set_title(f'Absolute Error (L2: {l2_error:.2e})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        cbar3 = plt.colorbar(contour3, ax=ax)


        formatter = ScalarFormatter()
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        cbar3.ax.yaxis.set_major_formatter(formatter)

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"error_{i}_4th.png"))
        plt.show()