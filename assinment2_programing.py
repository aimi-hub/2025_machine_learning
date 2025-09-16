#此程式參考至chat GPT 但是有自己看過程式並重新打過
import math
import os
import random
import numpy as np
import torch 
from torch import nn 
#將資料打包
from torch.utils.data import TensorDataset, DataLoader 
#繪圖用
import matplotlib.pyplot as plt 

def set_seed(seed=42):
    random.seed(seed)       #設定py內隨機種子
    np.random.seed(seed)    #設定NumPy隨機數種子    
    torch.manual_seed(seed) # 設定 PyTorch CPU 隨機數種子
    torch.cuda.manual_seed_all(seed)    # 設定所有 GPU 上 PyTorch 的隨機種子
    torch.backends.cudnn.deterministic = True # 讓 cuDNN 每次運算走確定路徑
    torch.backends.cudnn.benchmark = False    # 關閉 cuDNN 的自動最佳化 (避免結果不同)

#確保種子以及跑程式的device
set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1.定義目標函數
def runge(x: np.ndarray) -> np.ndarray: #定義x為任意維度之向量
    return 1.0 / (1.0 + 25.0 * x**2)

#2.產生資料
N_TRAIN = 200
N_VAL   = 50
N_TEST  = 1000  # 用高密度網格評估模型

# 在-1 1之間均勻抽取點產生訓練集並轉換成32-bit浮點數
x_train = np.random.uniform(-1.0, 1.0, size=(N_TRAIN, 1)).astype(np.float32)
y_train = runge(x_train)
# 和訓練集相同邏輯 抽取50點作為驗證集
x_val   = np.random.uniform(-1.0, 1.0, size=(N_VAL, 1)).astype(np.float32)
y_val   = runge(x_val)

# 將(-1, 1)均勻切成1000個點並且重新設定shape方便和網路對接
x_test  = np.linspace(-1.0, 1.0, N_TEST, dtype=np.float32).reshape(-1, 1)
y_test  = runge(x_test)

# 轉成 Tensor + Dataloader
train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(x_val),   torch.from_numpy(y_val))

#建立DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

#3.建立模型 定義MLP來近似Runge function
class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=64, depth=2, act="tanh"):
        super().__init__()
        if act == "tanh":       #選擇激活函數
            A = nn.Tanh
        elif act == "relu":
            A = nn.ReLU
        else:
            raise ValueError("act must be 'tanh' or 'relu'")

        layers = []
        layers.append(nn.Linear(in_dim, hidden))    #輸入層 -> 隱藏層
        layers.append(A())                          #激活函數
        for _ in range(depth - 1):                  #其餘隱藏層
            layers.append(nn.Linear(hidden, hidden)) 
            layers.append(A())                       
        layers.append(nn.Linear(hidden, 1))         #最後輸出層
        self.net = nn.Sequential(*layers)           #打包：將資料串起來形成完整網路

    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=1, hidden=64, depth=2, act="tanh").to(DEVICE)

#4.Loss and 優化
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=50
)

#5. train
EPOCHS = 2000
best_val = float("inf")
best_state = None
patience = 150
since_best = 0

train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # ---- val ----
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val - 1e-7:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        since_best = 0
    else:
        since_best += 1
        if since_best >= patience:
            print(f"Early stopping at epoch {epoch}. Best val MSE = {best_val:.6e}")
            break

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | train MSE {epoch_loss:.6e} | val MSE {val_loss:.6e}")

# 還原到最佳權重
if best_state is not None:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

#6.評估MSE 與最大誤差
model.eval()
with torch.no_grad():
    xt = torch.from_numpy(x_test).to(DEVICE)
    yt = torch.from_numpy(y_test).to(DEVICE)
    yhat = model(xt)
    mse = criterion(yhat, yt).item()
    max_err = torch.max(torch.abs(yhat - yt)).item()

print(f"[Test] MSE = {mse:.6e} | Max |error| = {max_err:.6e}")

#7.graph（函數 vs 預測Loss曲線)
os.makedirs("figs", exist_ok=True)

# 曲線圖
yhat_np = yhat.cpu().numpy()
plt.figure()
plt.plot(x_test, y_test, label="True f(x)")
plt.plot(x_test, yhat_np, label="NN prediction")
plt.scatter(x_train, y_train, s=10, alpha=0.3, label="train pts")
plt.title("Runge function vs NN prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("figs/prediction.png", dpi=160)

# Loss曲線
plt.figure()
plt.plot(train_losses, label="train MSE")
plt.plot(val_losses, label="val MSE")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("MSE (log scale)")
plt.title("Training / Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("figs/loss.png", dpi=160)

print("Saved figures to figs/prediction.png and figs/loss.png")