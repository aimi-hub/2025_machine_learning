import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 目標函數與導數（Runge）
def runge(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + 25.0 * x**2)

def runge_derivative(x: np.ndarray) -> np.ndarray:
    # f'(x) = -50 x / (1 + 25 x^2)^2
    return -50.0 * x / (1.0 + 25.0 * x**2) ** 2

# 資料集
N_TRAIN, N_VAL, N_TEST = 200, 50, 1000
x_train = np.random.uniform(-1.0, 1.0, size=(N_TRAIN, 1)).astype(np.float32)
y_train = runge(x_train)
yprime_train = runge_derivative(x_train)

x_val = np.random.uniform(-1.0, 1.0, size=(N_VAL, 1)).astype(np.float32)
y_val = runge(x_val)
yprime_val = runge_derivative(x_val)

x_test = np.linspace(-1.0, 1.0, N_TEST, dtype=np.float32).reshape(-1, 1)
y_test = runge(x_test)
yprime_test = runge_derivative(x_test)

train_ds = TensorDataset(torch.from_numpy(x_train),
                         torch.from_numpy(y_train),
                         torch.from_numpy(yprime_train))
val_ds = TensorDataset(torch.from_numpy(x_val),
                       torch.from_numpy(y_val),
                       torch.from_numpy(yprime_val))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

# MLP 模型
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

mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=50
)

# Loss 權重：總損失 = L_f + LAMBDA * L_fprime
LAMBDA = 1.0

EPOCHS = 2000
best_val = float("inf"); best_state = None
patience, since_best = 150, 0
train_loss_total, val_loss_total = [], []
train_loss_f, train_loss_fp = [], []
val_loss_f, val_loss_fp = [], []

# ======================= Training Loop =======================
for epoch in range(1, EPOCHS + 1):
    # ---------- train ----------
    model.train()
    ep_total = ep_f = ep_fp = 0.0
    for xb, yb, ypb in train_loader:
        # 重要：確保輸入可求導
        xb = xb.clone().detach().to(DEVICE).requires_grad_(True)
        yb = yb.to(DEVICE)
        ypb = ypb.to(DEVICE)

        pred = model(xb)
        grad = torch.autograd.grad(
            outputs=pred.sum(),
            inputs=xb,
            create_graph=True,    # 訓練要 backward，所以建立計算圖
            retain_graph=True
        )[0]

        loss_f  = mse(pred, yb)
        loss_fp = mse(grad, ypb)
        loss = loss_f + LAMBDA * loss_fp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        ep_total += loss.item() * bs
        ep_f     += loss_f.item() * bs
        ep_fp    += loss_fp.item() * bs

    ntrain = len(train_loader.dataset)
    train_loss_total.append(ep_total / ntrain)
    train_loss_f.append(ep_f / ntrain)
    train_loss_fp.append(ep_fp / ntrain)

    # ---------- val ----------
    model.eval()
    vp_total = vp_f = vp_fp = 0.0
    for xb, yb, ypb in val_loader:
        xb = xb.clone().detach().to(DEVICE).requires_grad_(True)
        yb = yb.to(DEVICE)
        ypb = ypb.to(DEVICE)

        pred = model(xb)
        grad = torch.autograd.grad(
            outputs=pred.sum(),
            inputs=xb,
            create_graph=False,   # 只算數值，不需要建立圖
            retain_graph=False
        )[0]

        lf  = mse(pred, yb)
        lfp = mse(grad, ypb)
        l   = lf + LAMBDA * lfp

        bs = xb.size(0)
        vp_total += l.item()  * bs
        vp_f     += lf.item() * bs
        vp_fp    += lfp.item()* bs

    nval = len(val_loader.dataset)
    val_total = vp_total / nval
    val_f     = vp_f     / nval
    val_fp    = vp_fp    / nval

    val_loss_total.append(val_total)
    val_loss_f.append(val_f)
    val_loss_fp.append(val_fp)

    scheduler.step(val_total)

    if val_total < best_val - 1e-7:
        best_val = val_total
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        since_best = 0
    else:
        since_best += 1
        if since_best >= patience:
            print(f"Early stopping @ {epoch} | best val={best_val:.6e}")
            break

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | tr_tot {train_loss_total[-1]:.3e} "
              f"| tr_f {train_loss_f[-1]:.3e} | tr_fp {train_loss_fp[-1]:.3e} "
              f"| va_tot {val_total:.3e} | va_f {val_f:.3e} | va_fp {val_fp:.3e}")

if best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

# ======================= Test 評估 =======================
model.eval()
xt = torch.from_numpy(x_test).to(DEVICE)
xt = xt.clone().detach().requires_grad_(True)  # 測試需要導數 → 不能 no_grad
yt  = torch.from_numpy(y_test).to(DEVICE)
ypt = torch.from_numpy(yprime_test).to(DEVICE)

pred = model(xt)
grad = torch.autograd.grad(
    outputs=pred.sum(),
    inputs=xt,
    create_graph=False,
    retain_graph=False
)[0]

mse_f  = mse(pred, yt).item()
mse_fp = mse(grad, ypt).item()
max_f  = torch.max(torch.abs(pred - yt)).item()
max_fp = torch.max(torch.abs(grad - ypt)).item()

print(f"[Test] f(x):  MSE={mse_f:.6e} | Max|err|={max_f:.6e}")
print(f"[Test] f'(x): MSE={mse_fp:.6e} | Max|err|={max_fp:.6e}")

# ======================= 繪圖 =======================
os.makedirs("figs_part2_nogradfix", exist_ok=True)

# f(x)
plt.figure()
plt.plot(x_test, y_test, label="True f(x)")
plt.plot(x_test, pred.detach().cpu().numpy(), label="NN f̂(x)")
plt.scatter(x_train, runge(x_train), s=10, alpha=0.3, label="train pts")
plt.title("Runge vs NN prediction (Part 2, no-grad fix)")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.tight_layout()
plt.savefig("figs_part2_nogradfix/prediction_fx.png", dpi=160)

# f'(x)
plt.figure()
plt.plot(x_test, yprime_test, label="True f'(x)")
plt.plot(x_test, grad.detach().cpu().numpy(), label="NN d/dx f̂(x)")
plt.title("Derivative comparison (Part 2, no-grad fix)")
plt.xlabel("x"); plt.ylabel("y'"); plt.legend(); plt.tight_layout()
plt.savefig("figs_part2_nogradfix/prediction_fprime.png", dpi=160)

# loss curves（總損失 + 分項）
plt.figure()
plt.plot(train_loss_total, label="train total")
plt.plot(val_loss_total,   label="val total")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss (log)")
plt.title("Total loss (Part 2)"); plt.legend(); plt.tight_layout()
plt.savefig("figs_part2_nogradfix/loss_total.png", dpi=160)

plt.figure()
plt.plot(train_loss_f,  label="train function loss")
plt.plot(val_loss_f,    label="val function loss")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss (log)")
plt.title("Function loss (Part 2)"); plt.legend(); plt.tight_layout()
plt.savefig("figs_part2_nogradfix/loss_f.png", dpi=160)

plt.figure()
plt.plot(train_loss_fp, label="train derivative loss")
plt.plot(val_loss_fp,   label="val derivative loss")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss (log)")
plt.title("Derivative loss (Part 2)"); plt.legend(); plt.tight_layout()
plt.savefig("figs_part2_nogradfix/loss_fprime.png", dpi=160)

print("Saved figures to figs_part2_nogradfix/")
