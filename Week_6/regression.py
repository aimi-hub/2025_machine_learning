import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class MyQDA:
    """自己實現的二次判別分析(QDA)"""
    
    def __init__(self):
        self.means = {}
        self.covariances = {}
        self.priors = {}
        self.classes = None
        
    def fit(self, X, y):
        """訓練QDA模型"""
        self.classes = np.unique(y)
        n_features = X.shape[1]
        
        for c in self.classes:
            # 獲取當前類別的樣本
            X_c = X[y == c]
            
            # 計算均值
            self.means[c] = np.mean(X_c, axis=0)
            
            # 計算協方差矩陣，添加正則化項避免奇異矩陣
            self.covariances[c] = np.cov(X_c.T) + np.eye(n_features) * 1e-6
            
            # 計算先驗概率
            self.priors[c] = len(X_c) / len(X)
            
        return self
    
    def _calculate_discriminant(self, X, class_label):
        """計算給定類別的判別函數值"""
        mean = self.means[class_label]
        cov = self.covariances[class_label]
        prior = self.priors[class_label]
        
        # 計算協方差矩陣的行列式和逆矩陣
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        
        # 中心化數據
        X_centered = X - mean
        
        # 計算二次型: (x - μ)^T Σ^{-1} (x - μ)
        if X.ndim == 1:
            quadratic = X_centered @ cov_inv @ X_centered
        else:
            quadratic = np.sum(X_centered @ cov_inv * X_centered, axis=1)
        
        # 計算判別函數: -0.5 * [log|Σ| + (x-μ)^T Σ^{-1} (x-μ)] + log(π)
        discriminant = -0.5 * (np.log(cov_det) + quadratic) + np.log(prior)
        
        return discriminant
    
    def predict(self, X):
        """預測類別標籤"""
        # 為每個類別計算判別函數值
        discriminants = []
        for c in self.classes:
            disc = self._calculate_discriminant(X, c)
            discriminants.append(disc)
        
        # 選擇具有最大判別函數值的類別
        discriminants = np.array(discriminants).T
        predictions = np.argmax(discriminants, axis=1)
        
        # 映射回原始類別標籤
        return np.array([self.classes[p] for p in predictions])

class QDARegression:
    """基於QDA的回歸模型"""
    
    def __init__(self):
        self.qda = MyQDA()
        self.regression_models = {}
        self.temperature_threshold = None
        
    def prepare_data(self, df, temperature_threshold=26.0):
        """準備數據：使用QDA分類，然後在特定類別上訓練回歸"""
        X = df[['longitude', 'latitude']].values
        y_temp = df['temperature'].values
        
        # 創建分類標籤（高溫區 vs 低溫區）
        y_class = (y_temp > temperature_threshold).astype(int)
        self.temperature_threshold = temperature_threshold
        
        print(f"數據統計:")
        print(f"  低溫區 (≤ {temperature_threshold}°C): {np.sum(y_class == 0)}")
        print(f"  高溫區 (> {temperature_threshold}°C): {np.sum(y_class == 1)}")
        
        return X, y_temp, y_class
    
    def train(self, X, y_temp, y_class):
        """訓練QDA+回歸組合模型"""
        # 1. 訓練QDA分類器
        print("訓練QDA分類器...")
        self.qda.fit(X, y_class)
        
        # 2. 在各個類別上訓練回歸模型
        self.regression_models = {}
        for class_label in self.qda.classes:
            # 獲取當前類別的數據
            mask = (y_class == class_label)
            X_class = X[mask]
            y_class_temp = y_temp[mask]
            
            if len(X_class) > 0:
                # 訓練線性回歸模型
                reg_model = LinearRegression()
                reg_model.fit(X_class, y_class_temp)
                self.regression_models[class_label] = reg_model
                
                # 評估該類別的回歸性能
                y_pred_class = reg_model.predict(X_class)
                r2 = r2_score(y_class_temp, y_pred_class)
                rmse = np.sqrt(mean_squared_error(y_class_temp, y_pred_class))
                
                class_name = "低溫區" if class_label == 0 else "高溫區"
                print(f"  {class_name}回歸模型 - R²: {r2:.4f}, RMSE: {rmse:.4f}")
    
    def predict(self, X):
        """預測溫度：先分類再回歸"""
        # 1. 使用QDA進行分類
        class_predictions = self.qda.predict(X)
        
        # 2. 根據分類結果使用對應的回歸模型預測溫度
        temperature_predictions = np.zeros(len(X))
        
        for class_label in self.regression_models:
            mask = (class_predictions == class_label)
            if np.any(mask):
                X_class = X[mask]
                temp_pred = self.regression_models[class_label].predict(X_class)
                temperature_predictions[mask] = temp_pred
        
        return temperature_predictions, class_predictions
    
    def evaluate(self, X_test, y_temp_test, y_class_test):
        """評估模型性能"""
        y_temp_pred, y_class_pred = self.predict(X_test)
        
        # 回歸評估
        rmse = np.sqrt(mean_squared_error(y_temp_test, y_temp_pred))
        r2 = r2_score(y_temp_test, y_temp_pred)
        
        # 分類評估
        accuracy = np.mean(y_class_test == y_class_pred)
        
        print(f"回歸性能:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"分類準確率: {accuracy:.4f}")
        
        return rmse, r2, accuracy
    
    def plot_results(self, X, y_temp, y_class, save_path=None):
        """繪製結果圖"""
        y_temp_pred, y_class_pred = self.predict(X)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 真實溫度分布
        sc1 = axes[0,0].scatter(X[:,0], X[:,1], c=y_temp, cmap='RdBu_r', s=20)
        axes[0,0].set_title('True temperature distribution')
        axes[0,0].set_xlabel('longitude')
        axes[0,0].set_ylabel('latitude')
        plt.colorbar(sc1, ax=axes[0,0])
        
        # 2. 預測溫度分布
        sc2 = axes[0,1].scatter(X[:,0], X[:,1], c=y_temp_pred, cmap='RdBu_r', s=20)
        axes[0,1].set_title('Predict temperature distribution')
        axes[0,1].set_xlabel('longitude')
        axes[0,1].set_ylabel('latitude')
        plt.colorbar(sc2, ax=axes[0,1])
        
        # 3. QDA分類結果
        sc3 = axes[1,0].scatter(X[:,0], X[:,1], c=y_class_pred, cmap='coolwarm', s=20)
        axes[1,0].set_title('QDA classification results (red: high temp area, blue: low temp area)')
        axes[1,0].set_xlabel('longitude')
        axes[1,0].set_ylabel('latitude')
        plt.colorbar(sc3, ax=axes[1,0])
        
        # 4. 殘差分布
        residuals = y_temp - y_temp_pred
        sc4 = axes[1,1].scatter(X[:,0], X[:,1], c=residuals, cmap='RdBu_r', s=20)
        axes[1,1].set_title('Prediction residuals')
        axes[1,1].set_xlabel('longitude')
        axes[1,1].set_ylabel('latitude')
        plt.colorbar(sc4, ax=axes[1,1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# 主程式
def main():
    
    # 載入回歸資料集
    df_reg = pd.read_csv("regression_dataset.csv")
    print(f"載入回歸資料集: {len(df_reg)} 筆數據")
    
    # 初始化模型
    model = QDARegression()
    
    # 準備數據
    X, y_temp, y_class = model.prepare_data(df_reg, temperature_threshold=26.0)
    
    # 劃分訓練測試集
    X_train, X_test, y_temp_train, y_temp_test, y_class_train, y_class_test = train_test_split(
        X, y_temp, y_class, test_size=0.3, random_state=42, stratify=y_class
    )
    
    print(f"train set size: {X_train.shape[0]}")
    print(f"test set size: {X_test.shape[0]}")
    
    # 訓練模型
    model.train(X_train, y_temp_train, y_class_train)
    
    # 評估模型
    print("\n=== 測試集評估 ===")
    rmse, r2, accuracy = model.evaluate(X_test, y_temp_test, y_class_test)
    
    # 預測示例
    print("\n=== 溫度預測示例 ===")
    test_points = [
        (121.5, 25.0),  # 台北
        (120.7, 24.1),  # 台中
        (120.3, 22.6),  # 高雄
    ]
    
    for lon, lat in test_points:
        point = np.array([[lon, lat]])
        temp_pred, class_pred = model.predict(point)
        class_name = "high temp area" if class_pred[0] == 1 else "low temp area"
        
        # 找到最近的實際數據點
        distances = np.sqrt((df_reg['longitude'] - lon)**2 + (df_reg['latitude'] - lat)**2)
        nearest_idx = distances.idxmin()
        actual_temp = df_reg.loc[nearest_idx, 'temperature']
        
        print(f"位置 ({lon}, {lat}):")
        print(f"  actual_temp: {actual_temp:.1f}°C")
        print(f"  temp_pred: {temp_pred[0]:.1f}°C")
        print(f"  class_name: {class_name}")
        print()
    
    # 繪製結果
    print("繪製結果圖...")
    model.plot_results(X, y_temp, y_class, "qda_regression_results.png")
    
    print("✅ 回歸分析完成!")

if __name__ == "__main__":
    main()