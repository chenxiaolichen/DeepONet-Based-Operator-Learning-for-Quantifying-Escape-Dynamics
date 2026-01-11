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
from scipy.interpolate import interp1d


class Config:
    xmin, xmax = 0.0, 1.48971
    Nx_fdm = 10000
    Nx_unet = 1000
    x_fdm = np.linspace(xmin, xmax, Nx_fdm).astype(np.float64)
    x_unet = np.linspace(xmin, xmax, Nx_unet).astype(np.float64)
    eps = 1.0

    ntrain = 800
    ntest = 200
    length_scale = 100

    unet_channels = [64, 128, 256, 512]
    input_dim = 2

    epochs = 15000
    batch_size = 256
    lr = 5e-3


    lr_factor = 0.8
    lr_patience = 250
    lr_min = 1e-6
    lr_cooldown = 50

    save_dir = "U-Net"


cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def b_fn(x_val):
    return (6 * x_val ** 2) / (x_val ** 2 + 10) - x_val + 0.4


def relative_l2_error(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


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


def downsample_u(u_high, x_high, x_low):
    interp_fun = interp1d(x_high, u_high, kind='linear', fill_value="extrapolate")
    return interp_fun(x_low).astype(np.float32)


def downsample_sigma(sigma_high, x_high, x_low):
    interp_fun = interp1d(x_high, sigma_high, kind='linear', fill_value="extrapolate")
    return interp_fun(x_low).astype(np.float32)



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
        sigma_high = 0.5 + np.abs(np.interp(cfg.x_fdm, x_grf.flatten(), gp_sample))

        u_high = solve_pde(sigma_high, cfg.x_fdm)

        sigma_low = downsample_sigma(sigma_high, cfg.x_fdm, cfg.x_unet)
        u_low = downsample_u(u_high, cfg.x_fdm, cfg.x_unet)

        sigma_tensor = torch.tensor(sigma_low, dtype=torch.float32).reshape(len(cfg.x_unet), 1)
        x_tensor = torch.tensor(cfg.x_unet, dtype=torch.float32).reshape(len(cfg.x_unet), 1)
        input_list.append(torch.cat([sigma_tensor, x_tensor], dim=1))

        u_list.append(torch.tensor(u_low, dtype=torch.float32).reshape(len(cfg.x_unet), 1))

        if _ % 10 == 0:
            pbar.set_postfix({"Processed": f"{_ + 1}/{num_samples}"})

    pbar.close()
    return (torch.stack(input_list, dim=0),
            torch.stack(u_list, dim=0).squeeze(-1))


def generate_sigma_data(sigma_value):
    sigma_high = np.ones_like(cfg.x_fdm) * sigma_value
    u_high = solve_pde(sigma_high, cfg.x_fdm)
    sigma_low = downsample_sigma(sigma_high, cfg.x_fdm, cfg.x_unet)
    sigma_tensor = torch.tensor(sigma_low, dtype=torch.float32).reshape(len(cfg.x_unet), 1)
    x_tensor = torch.tensor(cfg.x_unet, dtype=torch.float32).reshape(len(cfg.x_unet), 1)
    input_tensor = torch.cat([sigma_tensor, x_tensor], dim=1).unsqueeze(0)
    u_low = downsample_u(u_high, cfg.x_fdm, cfg.x_unet)
    return input_tensor, u_high, u_low


class UNet1D(nn.Module):
    def __init__(self, channels=[64, 128, 256], input_dim=2, lr=5e-4,
                 factor=0.8, patience=200, min_lr=1e-6, cooldown=50):
        super(UNet1D, self).__init__()
        self.channels = channels
        self.input_dim = input_dim

        self.encoder = nn.ModuleList()
        in_channels = input_dim
        for out_channels in channels:
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
            self.encoder.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_channels = out_channels

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.ModuleList()
        in_channels = in_channels * 2
        for out_channels in reversed(channels):
            self.decoder.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2))
            self.decoder.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
            in_channels = out_channels

        self.out_conv = nn.Conv1d(in_channels, 1, kernel_size=1)

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
        x = x.permute(0, 2, 1)
        skip_connections = []
        for i in range(0, len(self.encoder), 2):
            conv_block = self.encoder[i]
            pool_block = self.encoder[i + 1]
            x = conv_block(x)
            skip_connections.append(x)
            x = pool_block(x)

        x = self.bottleneck(x)

        skip_idx = len(skip_connections) - 1
        for i in range(0, len(self.decoder), 2):
            up_conv = self.decoder[i]
            conv_block = self.decoder[i + 1]
            x = up_conv(x)
            skip_x = skip_connections[skip_idx]
            if x.shape[-1] != skip_x.shape[-1]:
                x = F.interpolate(x, size=skip_x.shape[-1], mode='linear', align_corners=False)
            x = torch.cat([x, skip_x], dim=1)
            x = conv_block(x)
            skip_idx -= 1

        x = self.out_conv(x)
        return x.permute(0, 2, 1).squeeze(-1)


if __name__ == "__main__":
    data_exists = os.path.exists(os.path.join(cfg.save_dir, "train_input.pt")) and os.path.exists(
        os.path.join(cfg.save_dir, "u_train.pt"))
    model_exists = os.path.exists(os.path.join(cfg.save_dir, "best_unet_model.pth"))

    if data_exists:
        print("\n=== Loading Existing Data ===")
        train_input = torch.load(os.path.join(cfg.save_dir, "train_input.pt"))
        u_train = torch.load(os.path.join(cfg.save_dir, "u_train.pt"))
        test_input = torch.load(os.path.join(cfg.save_dir, "test_input.pt"))
        u_test = torch.load(os.path.join(cfg.save_dir, "u_test.pt"))
    else:
        print("\n=== Generating New Data ===")
        train_input, u_train = generate_data(cfg.ntrain, is_train=True)
        test_input, u_test = generate_data(cfg.ntest, is_train=False)
        torch.save(train_input, os.path.join(cfg.save_dir, "train_input.pt"))
        torch.save(u_train, os.path.join(cfg.save_dir, "u_train.pt"))
        torch.save(test_input, os.path.join(cfg.save_dir, "test_input.pt"))
        torch.save(u_test, os.path.join(cfg.save_dir, "u_test.pt"))

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_input, u_train),
        batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_input, u_test),
        batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )


    model = UNet1D(
        channels=cfg.unet_channels,
        input_dim=cfg.input_dim,
        lr=cfg.lr,
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.lr_min,
        cooldown=cfg.lr_cooldown
    ).to(device)

    if model_exists:
        print("\n=== Loading Existing Model ===")
        model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "best_unet_model.pth"), map_location=device))
        # 加载损失日志
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
                u_pred = model(x_batch)
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
                    u_pred = model(x_batch)
                    test_total_loss += F.mse_loss(u_pred, y_batch).item() * x_batch.size(0)

            test_avg_loss = test_total_loss / cfg.ntest
            test_loss_log.append(test_avg_loss)

            model.scheduler.step(test_avg_loss)
            current_lr = model.optimizer.param_groups[0]['lr']
            lr_log.append(current_lr)

            if epoch == 0 or test_avg_loss < min(test_loss_log[:-1]):
                torch.save(model.state_dict(), os.path.join(cfg.save_dir, "best_unet_model.pth"))

            pbar.set_postfix({
                "Train MSE": f"{train_avg_loss:.6e}",
                "Test MSE": f"{test_avg_loss:.6e}",
                "Best Test": f"{min(test_loss_log):.6e}",
                "LR": f"{current_lr:.6e}"
            })

        pbar.close()

        torch.save(model.state_dict(), os.path.join(cfg.save_dir, "final_unet_model.pth"))
        np.savez(os.path.join(cfg.save_dir, "loss_logs.npz"),
                 train_loss=np.array(train_loss_log),
                 test_loss=np.array(test_loss_log),
                 lr_log=np.array(lr_log))
    else:
        print("\n=== Model Already Trained ===")

    if len(train_loss_log) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

        # 损失曲线
        ax1.plot(np.arange(len(train_loss_log)), train_loss_log, label="Train MSE", linewidth=2, color="#2E86AB")
        ax1.plot(np.arange(len(test_loss_log)), test_loss_log, label="Test MSE", linewidth=2, linestyle='--',
                 color="#A23B72")
        best_epoch = np.argmin(test_loss_log)
        ax1.axvline(x=best_epoch, color='gray', linestyle=':', label=f'Best Epoch ({best_epoch})')
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("MSE", fontsize=12)
        ax1.set_title("Training and Test Loss Curves", fontsize=14, fontweight='bold')
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

        plt.show()

    print("\n=== Loading Best Model for Prediction ===")
    model.load_state_dict(torch.load(os.path.join(cfg.save_dir, "best_unet_model.pth"), map_location=device))
    model.eval()


    sigma_test_list = [0.5, 1.0, 1.5, 2.0]
    l2_errors = []
    fdm_high_results = []
    fdm_low_results = []
    unet_results = []

    with trange(len(sigma_test_list), desc="Predicting", bar_format="{l_bar}") as pred_pbar:
        for i, sigma in enumerate(sigma_test_list):
            test_input, u_high, u_low = generate_sigma_data(sigma)
            fdm_high_results.append(u_high)
            fdm_low_results.append(u_low)

            with torch.no_grad():
                test_input = test_input.to(device)
                u_pred = model(test_input).detach().cpu().numpy()[0]
            unet_results.append(u_pred)

            l2_err = relative_l2_error(u_pred, u_low)
            l2_errors.append(l2_err)

            pred_pbar.set_postfix({"σ": sigma, "L2 Error": f"{l2_err:.6e}"})
            pred_pbar.update(1)

    print("\n=== Prediction Errors ===")
    for sigma, err in zip(sigma_test_list, l2_errors):
        print(f"σ = {sigma:<4} | Relative L2 Error (1000点) = {err:.6e}")

    plt.figure(figsize=(12, 6))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    for i, (sigma, u_high, u_pred, l2_err) in enumerate(
            zip(sigma_test_list, fdm_high_results, unet_results, l2_errors)):

        plt.plot(cfg.x_fdm, u_high, label=f'FDM (σ={sigma})', linewidth=1.8, color=colors[i], alpha=0.6)

        plt.plot(cfg.x_unet, u_pred, '--', label=f'UNet (σ={sigma}, L2={l2_err:.4e})', linewidth=2.2, color=colors[i])

    plt.xlabel('x', fontsize=12)
    plt.ylabel('u(x)', fontsize=12)
    plt.title('UNet  vs FDM Predictions', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    plt.show()

