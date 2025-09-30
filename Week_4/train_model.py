# train_model.py  -- 訓練分類與回歸模型，評估並輸出圖表
import os
import numpy as np
import pandas as pd

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay

# matplotlib（無頭模式）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 檔案路徑設定 ===
base_path = r"c:\Users\user\Desktop\大四上\機器學習\HW4"
cls_file = os.path.join(base_path, "classification_data.csv")
reg_file = os.path.join(base_path, "regression_data.csv")

# === 讀取資料集 ===
df_cls = pd.read_csv(cls_file)
df_reg = pd.read_csv(reg_file)

# === 切分特徵與標籤 ===
X_cls = df_cls[['Longitude', 'Latitude']].values
y_cls = df_cls['Label'].values

X_reg = df_reg[['Longitude', 'Latitude']].values
y_reg = df_reg['Temperature'].values

# === 分割訓練/測試集 (20% 測試) ===
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# === 訓練分類模型 (Decision Tree) ===
clf = DecisionTreeClassifier(random_state=42)
clf.fit(Xc_train, yc_train)
yc_pred = clf.predict(Xc_test)
acc = accuracy_score(yc_test, yc_pred)

# === 訓練回歸模型 (Random Forest) ===
regr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
regr.fit(Xr_train, yr_train)
yr_pred = regr.predict(Xr_test)

# 手動算 RMSE（避免 squared 參數相容性問題）
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))

# === 輸出結果到文字檔 ===
result_file = os.path.join(base_path, "train_results.txt")
with open(result_file, "w", encoding="utf-8") as f:
    f.write("=== 模型評估結果 ===\n")
    f.write(f"分類模型 (Decision Tree) 準確率: {acc:.3f}\n")
    f.write(f"回歸模型 (Random Forest) RMSE: {rmse:.3f}\n")

print("=== 模型評估結果 ===")
print(f"分類模型 (Decision Tree) 準確率: {acc:.3f}")
print(f"回歸模型 (Random Forest) RMSE: {rmse:.3f}")
print(f"\n結果已存成檔案：{result_file}")

# === 圖表輸出 ===

# 1) 分類：混淆矩陣
cm = confusion_matrix(yc_test, yc_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
disp.plot(ax=ax, colorbar=False, values_format="d")
ax.set_title("Classification Confusion Matrix (Decision Tree)")
fig.tight_layout()
cm_path = os.path.join(base_path, "cls_confusion_matrix.png")
fig.savefig(cm_path)
plt.close(fig)

# 2) 回歸：真實值 vs 預測值
min_v = float(np.min([yr_test.min(), yr_pred.min()]))
max_v = float(np.max([yr_test.max(), yr_pred.max()]))
fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=150)
ax.scatter(yr_test, yr_pred, s=8, alpha=0.6)
ax.plot([min_v, max_v], [min_v, max_v], linewidth=1.2)  # 參考線 y=x
ax.set_xlabel("True Temperature (°C)")
ax.set_ylabel("Predicted Temperature (°C)")
ax.set_title("Regression: True vs Predicted (Random Forest)")
ax.grid(True, linewidth=0.4, alpha=0.5)
fig.tight_layout()
pvt_path = os.path.join(base_path, "reg_pred_vs_true.png")
fig.savefig(pvt_path)
plt.close(fig)

# 3) 回歸：殘差直方圖
residuals = yr_pred - yr_test
fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=150)
ax.hist(residuals, bins=30, alpha=0.85)
ax.set_xlabel("Residual (Pred - True) °C")
ax.set_ylabel("Count")
ax.set_title("Regression Residuals Histogram")
ax.grid(True, linewidth=0.4, alpha=0.5)
fig.tight_layout()
res_path = os.path.join(base_path, "reg_residuals_hist.png")
fig.savefig(res_path)
plt.close(fig)

print("圖檔已輸出：")
print(" -", cm_path)
print(" -", pvt_path)
print(" -", res_path)
