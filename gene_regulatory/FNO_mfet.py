import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve


class Config:
    xmin, xmax = 0.0, 1.48971
    Nx_pde = 10000
    Nx_net = 1000
    x_pde = np.linspace(xmin, xmax, Nx_pde).astype(np.float64)
    sample_indices = np.linspace(0, Nx_pde - 1, Nx_net, dtype=int)
    x_net = x_pde[sample_indices]

    eps = 1.0

    ntrain = 800
    ntest = 200
    length_scale = 100

    fno_modes = 16
    fno_width = 128
    input_dim = 2

    epochs = 15000
    batch_size = 256
    lr = 5e-3

    lr_factor = 0.8
    lr_patience = 300
    lr_min = 1e-6
    lr_cooldown = 50

    save_dir = "FNO_gene"


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def b_fn(x_val):
    return (6 * x_val ** 2) / (x_val ** 2 + 10) - x_val + 0.4


def relative_l2_error(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNO1d(nn.Module):
    def __init__(self, modes, width=128, input_dim=2, lr=5e-4,
                 factor=0.8, patience=200, min_lr=1e-6, cooldown=50):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.input_dim = input_dim

        # 网络层
        self.fc0 = nn.Linear(self.input_dim, self.width)
        self.convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(4)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(4)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            cooldown=cooldown
        )

    def forward(self, x):
        x = self.fc0(x).permute(0, 2, 1)

        for conv, w, bn in zip(self.convs, self.ws, self.bns):
            x1 = conv(x)
            x2 = w(x)
            x = bn(x1 + x2)
            x = F.relu(x)

        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

#FDM
def solve_pde(sigma_vals, x):
    N = len(x)
    h = x[1] - x[0]
    b_vals = b_fn(x)
    sigma_vals = np.array(sigma_vals, dtype=np.float64)

    Chh = (sigma_vals ** 2) / (2 * cfg.eps) / (h ** 2)

    A = lil_matrix((N, N), dtype=np.float64)
    rhs = -np.ones(N, dtype=np.float64)

    for i in range(1, N - 1):
        A[i, i - 1] = Chh[i] - b_vals[i] / (2 * h)
        A[i, i] = -2 * Chh[i]
        A[i, i + 1] = Chh[i] + b_vals[i] / (2 * h)

    i = 0
    A[i, i] = -2 * Chh[i] - 3 * b_vals[i] / (2 * h)
    A[i, i + 1] = -5 * Chh[i] + 4 * b_vals[i] / (2 * h)
    if N >= 3:
        A[i, i + 2] = 4 * Chh[i] - b_vals[i] / (2 * h)
    if N >= 4:
        A[i, i + 3] = -Chh[i]
    rhs[i] = 0.0

    i = N - 1
    A[i, i] = -2 * Chh[i] + 3 * b_vals[i] / (2 * h)
    A[i, i - 1] = -5 * Chh[i] - 4 * b_vals[i] / (2 * h)
    if N >= 3:
        A[i, i - 2] = 4 * Chh[i] + b_vals[i] / (2 * h)
    if N >= 4:
        A[i, i - 3] = -Chh[i]
    rhs[i] = 0.0

    A = csc_matrix(A)
    try:
        u = spsolve(A, rhs)
    except:
        from scipy.sparse.linalg import lsqr
        u = lsqr(A, rhs)[0]

    return u.astype(np.float32)


def generate_data(num_samples, is_train=True):
    input_list = []
    u_list = []
    pbar = trange(num_samples, desc=f"Generating {'Train' if is_train else 'Test'} Data",
                  unit="sample", ncols=100, bar_format="{l_bar}")

    for _ in pbar:
        x_grf = np.linspace(0, cfg.xmax, 500)[:, None]
        K = np.exp(-0.5 * (np.expand_dims(x_grf / cfg.length_scale, 1) - np.expand_dims(x_grf / cfg.length_scale,
                                                                                        0)) ** 2).sum(axis=2)
        K += 1e-6 * np.eye(500)
        L = np.linalg.cholesky(K)
        gp_sample = L @ np.random.randn(500)
        sigma_vals_pde = 0.5 + np.abs(np.interp(cfg.x_pde, x_grf.flatten(), gp_sample))


        sigma_vals_net = sigma_vals_pde[cfg.sample_indices]


        sigma_tensor = torch.tensor(sigma_vals_net, dtype=torch.float32).reshape(cfg.Nx_net, 1)
        x_tensor = torch.tensor(cfg.x_net, dtype=torch.float32).reshape(cfg.Nx_net, 1)
        input_list.append(torch.cat([sigma_tensor, x_tensor], dim=1))


        u_pde = solve_pde(sigma_vals_pde, cfg.x_pde)
        u_net = u_pde[cfg.sample_indices]
        u_list.append(torch.tensor(u_net, dtype=torch.float32).reshape(cfg.Nx_net, 1))

        if _ % 10 == 0:
            pbar.set_postfix({"Processed": f"{_ + 1}/{num_samples}"})

    pbar.close()
    return (torch.stack(input_list, dim=0),
            torch.stack(u_list, dim=0).squeeze(-1))


def generate_sigma_data(sigma_value):
    sigma_vals_pde = np.ones_like(cfg.x_pde) * sigma_value
    sigma_vals_net = sigma_vals_pde[cfg.sample_indices]

    sigma_tensor = torch.tensor(sigma_vals_net, dtype=torch.float32).reshape(cfg.Nx_net, 1)
    x_tensor = torch.tensor(cfg.x_net, dtype=torch.float32).reshape(cfg.Nx_net, 1)
    input_tensor = torch.cat([sigma_tensor, x_tensor], dim=1).unsqueeze(0)

    u_pde = solve_pde(sigma_vals_pde, cfg.x_pde).astype(np.float32)
    u_net = u_pde[cfg.sample_indices]
    return input_tensor, u_net


if __name__ == "__main__":
    train_input_path = os.path.join(cfg.save_dir, "train_input.pt")
    u_train_path = os.path.join(cfg.save_dir, "u_train.pt")
    test_input_path = os.path.join(cfg.save_dir, "test_input.pt")
    u_test_path = os.path.join(cfg.save_dir, "u_test.pt")

    data_exists = os.path.exists(train_input_path) and os.path.exists(u_train_path) and \
                  os.path.exists(test_input_path) and os.path.exists(u_test_path)
    model_exists = os.path.exists(os.path.join(cfg.save_dir, "best_fno_model.pth"))

    if data_exists:
        print("\n=== Loading Existing Data")
        train_input = torch.load(train_input_path)
        u_train = torch.load(u_train_path)
        test_input = torch.load(test_input_path)
        u_test = torch.load(u_test_path)
    else:
        print("\n=== Generating New Data")
        train_input, u_train = generate_data(cfg.ntrain, is_train=True)
        test_input, u_test = generate_data(cfg.ntest, is_train=False)

        torch.save(train_input, train_input_path)
        torch.save(u_train, u_train_path)
        torch.save(test_input, test_input_path)
        torch.save(u_test, u_test_path)


    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_input, u_train),
        batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_input, u_test),
        batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )




    model = FNO1d(
        modes=cfg.fno_modes,
        width=cfg.fno_width,
        input_dim=cfg.input_dim,
        lr=cfg.lr,
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.lr_min,
        cooldown=cfg.lr_cooldown
    ).to(device)

    if model_exists:
        print("\n=== Loading Existing Model===")
        model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "best_fno_model.pth"), map_location=device))
        loss_logs_path = os.path.join(cfg.save_dir, "loss_logs.npz")
        if os.path.exists(loss_logs_path):
            loss_logs = np.load(loss_logs_path)
            train_loss_log = list(loss_logs["train_loss"])
            test_loss_log = list(loss_logs["test_loss"])
            lr_log = list(loss_logs["lr_log"])
            start_epoch = len(train_loss_log)
            print(f"Resuming training from epoch {start_epoch}")
        else:
            train_loss_log, test_loss_log, lr_log = [], [], []
            start_epoch = 0
    else:
        print("\n=== Initializing New Model ===")
        train_loss_log, test_loss_log, lr_log = [], [], []
        start_epoch = 0


    if start_epoch < cfg.epochs:
        pbar = trange(start_epoch, cfg.epochs, desc="Training", bar_format="{l_bar}")
        for epoch in pbar:
            model.train()
            train_total_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                model.optimizer.zero_grad()
                u_pred = model(x_batch).squeeze(-1)
                loss = F.mse_loss(u_pred, y_batch)
                loss.backward()
                model.optimizer.step()
                train_total_loss += loss.item() * x_batch.size(0)

            train_avg_loss = train_total_loss / cfg.ntrain
            train_loss_log.append(train_avg_loss)

            model.eval()
            test_total_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    u_pred = model(x_batch).squeeze(-1)
                    test_total_loss += F.mse_loss(u_pred, y_batch).item() * x_batch.size(0)

            test_avg_loss = test_total_loss / cfg.ntest
            test_loss_log.append(test_avg_loss)

            model.scheduler.step(test_avg_loss)

            current_lr = model.optimizer.param_groups[0]['lr']
            lr_log.append(current_lr)


            if epoch == 0 or test_avg_loss < min(test_loss_log[:-1]):
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_fno_model.pth"))

            pbar.set_postfix({
                "Train MSE": f"{train_avg_loss:.6e}",
                "Test MSE": f"{test_avg_loss:.6e}",
                "Best Test": f"{min(test_loss_log):.6e}",
                "LR": f"{current_lr:.6e}"
            })

        pbar.close()


        torch.save(model.state_dict(), os.path.join(cfg.save_dir, "final_fno_model.pth"))
        np.savez(os.path.join(cfg.save_dir, "loss_logs.npz"),
                 train_loss=np.array(train_loss_log),
                 test_loss=np.array(test_loss_log),
                 lr_log=np.array(lr_log),
                 epochs=np.arange(len(train_loss_log)))
    else:
        print("\n=== Model Already Trained ===")


    if len(train_loss_log) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        ax1.plot(np.arange(len(train_loss_log)), train_loss_log, label="Train MSE", linewidth=2, color="#2E86AB")
        ax1.plot(np.arange(len(test_loss_log)), test_loss_log, label="Test MSE", linewidth=2, linestyle='--',
                 color="#A23B72")
        best_epoch = np.argmin(test_loss_log)
        ax1.axvline(x=best_epoch, color='gray', linestyle=':', label=f'Best Epoch ({best_epoch})')
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("MSE", fontsize=12)
        ax1.set_title("Training and Test Loss Curves ", fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.set_yscale("log")

        ax2.plot(np.arange(len(lr_log)), lr_log, label=f"Learning Rate",
                 linewidth=2, color="#FF6B6B")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Learning Rate", fontsize=12)
        ax2.set_title("Learning Rate Schedule", fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.set_yscale("log")

        plt.tight_layout()
        loss_plot_path = os.path.join(cfg.save_dir, "loss_and_lr_curves.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    else:
        print("\n=== Skipping Loss Visualization")


    model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "best_fno_model.pth"), map_location=device))
    model.eval()


    sigma_test_list = [0.5, 1.0, 1.5, 2.0]
    l2_errors = []
    fdm_results = []
    fno_results = []

    with trange(len(sigma_test_list), desc="Predicting", bar_format="{l_bar}") as pred_pbar:
        for i, sigma in enumerate(sigma_test_list):
            test_input, u_true = generate_sigma_data(sigma)
            fdm_results.append(u_true)

            with torch.no_grad():
                test_input = test_input.to(device)
                u_pred = model(test_input).squeeze(-1).detach().cpu().numpy()[0]
            fno_results.append(u_pred)

            l2_err = relative_l2_error(u_pred, u_true)
            l2_errors.append(l2_err)

            pred_pbar.set_postfix({"σ": sigma, "L2 Error": f"{l2_err:.6e}"})
            pred_pbar.update(1)

    print("\n=== Prediction Errors")
    for sigma, err in zip(sigma_test_list, l2_errors):
        print(f"σ = {sigma:<4} | Relative L2 Error = {err:.6e}")


    plt.figure(figsize=(12, 6))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    for i, (sigma, u_true, u_pred, l2_err) in enumerate(zip(sigma_test_list, fdm_results, fno_results, l2_errors)):
        plt.plot(cfg.x_net, u_true, label=f'FDM (σ={sigma})', linewidth=1.8, color=colors[i], alpha=0.8)
        plt.plot(cfg.x_net, u_pred, '--', label=f'FNO (σ={sigma}, L2={l2_err:.4e})', linewidth=2.2, color=colors[i])

    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title('FNO vs FDM Predictions', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()

