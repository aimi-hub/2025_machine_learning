
import math
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 目標函數與其導數（Runge function）
def runge(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 25.0 * x**2)

def runge_derivative(x: np.ndarray) -> np.ndarray:
    # f'(x) = -50 x / (1 + 25 x^2)^2
    return -50.0 * x / (1.0 + 25.0 * x**2) ** 2

# 2) 產生資料
N_TRAIN = 200
N_VAL   = 50
N_TEST  = 1000

x_train = np.random.uniform(-1.0, 1.0, size=(N_TRAIN, 1)).astype(np.float32)
y_train = runge(x_train)

x_val   = np.random.uniform(-1.0, 1.0, size=(N_VAL, 1)).astype(np.float32)
y_val   = runge(x_val)

x_test  = np.linspace(-1.0, 1.0, N_TEST, dtype=np.float32).reshape(-1, 1)
y_test  = runge(x_test)
yprime_test = runge_derivative(x_test)

train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(x_val),   torch.from_numpy(y_val))

train_loader = DataLoader(train_ds, batch_size=32,  shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

# 3) MLP
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=64, depth=2, act="tanh"):
        super().__init__()
        Act = nn.Tanh if act == "tanh" else nn.ReLU
        layers = [nn.Linear(in_dim, hidden), Act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), Act()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=1, hidden=64, depth=2, act="tanh").to(DEVICE)

# 4) Loss / Optim
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=50
)

# 5) Train（只以 f(x) 的 MSE 訓練）
EPOCHS = 2000
best_val = float("inf")
best_state = None
patience = 150
since_best = 0
train_losses, val_losses = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    ep_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * xb.size(0)
    ep_loss /= len(train_loader.dataset)
    train_losses.append(ep_loss)

    model.eval()
    with torch.no_grad():
        v_loss = 0.0
        for xb, yb in val_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            pred = model(xb)
            v_loss += criterion(pred, yb).item() * xb.size(0)
        v_loss /= len(val_loader.dataset)
        val_losses.append(v_loss)

    scheduler.step(v_loss)
    if v_loss < best_val - 1e-7:
        best_val = v_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        since_best = 0
    else:
        since_best += 1
        if since_best >= patience:
            print(f"Early stopping @ {epoch} | best val MSE={best_val:.6e}")
            break
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | train {ep_loss:.3e} | val {v_loss:.3e}")

if best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

# 6) 評估：同時計算 f 與 f' 的測試誤差（訓練只用 f）
model.eval()
xt = torch.from_numpy(x_test).to(DEVICE)
xt.requires_grad_(True)  # 讓 autograd 能對輸入求導
yt = torch.from_numpy(y_test).to(DEVICE)

with torch.no_grad():
    yhat = model(xt)
mse_f = criterion(yhat, yt).item()
max_f = torch.max(torch.abs(yhat - yt)).item()

# 導數：用 autograd 求 d/dx of prediction
yhat_for_grad = model(xt)             
grad = torch.autograd.grad(
    outputs=yhat_for_grad.sum(),
    inputs=xt,
    create_graph=False,
    retain_graph=False
)[0]
yprime_true = torch.from_numpy(yprime_test).to(DEVICE)
mse_fprime = criterion(grad, yprime_true).item()
max_fprime = torch.max(torch.abs(grad - yprime_true)).item()

print(f"[Test] f(x):     MSE={mse_f:.6e} | Max|err|={max_f:.6e}")
print(f"[Test] f'(x):    MSE={mse_fprime:.6e} | Max|err|={max_fprime:.6e}")

# 7) 圖
os.makedirs("figs_part1", exist_ok=True)

# f(x)
plt.figure()
plt.plot(x_test, y_test, label="True f(x)")
plt.plot(x_test, yhat.detach().cpu().numpy(), label="NN f̂(x)")
plt.scatter(x_train, runge(x_train), s=10, alpha=0.3, label="train pts")
plt.title("Runge vs NN prediction (Part 1)")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout()
plt.savefig("figs_part1/prediction_fx.png", dpi=160)

# f'(x)
plt.figure()
plt.plot(x_test, yprime_test, label="True f'(x)")
plt.plot(x_test, grad.detach().cpu().numpy(), label="NN d/dx f̂(x)")
plt.title("Derivative comparison (Part 1)")
plt.xlabel("x"); plt.ylabel("y'"); plt.legend(); plt.tight_layout()
plt.savefig("figs_part1/prediction_fprime.png", dpi=160)

# Loss 曲線
plt.figure()
plt.plot(train_losses, label="train MSE (f)")
plt.plot(val_losses, label="val MSE (f)")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("MSE (log)")
plt.title("Training / Validation Loss (Part 1)")
plt.legend(); plt.tight_layout()
plt.savefig("figs_part1/loss_fx.png", dpi=160)

print("Saved to figs_part1/")
