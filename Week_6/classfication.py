import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    
    def predict_proba(self, X):
        """預測類別概率"""
        # 計算每個類別的判別函數值
        discriminants = []
        for c in self.classes:
            disc = self._calculate_discriminant(X, c)
            discriminants.append(disc)
        
        discriminants = np.array(discriminants).T
        
        # 使用softmax將判別函數值轉換為概率
        max_disc = np.max(discriminants, axis=1, keepdims=True)
        exp_disc = np.exp(discriminants - max_disc)
        probabilities = exp_disc / np.sum(exp_disc, axis=1, keepdims=True)
        
        return probabilities

class QDAClassifier:
    """QDA分類器"""
    
    def __init__(self):
        self.qda = MyQDA()
        self.class_names = None
        
    def prepare_data(self, df):
        """準備分類數據：有效數據 vs 無效數據"""
        X = df[['longitude', 'latitude']].values
        y = df['is_valid'].values  # 0: 無效, 1: 有效
        
        self.class_names = ['無效數據', '有效數據']
        
        print(f"數據統計:")
        print(f"  無效數據: {np.sum(y == 0)}")
        print(f"  有效數據: {np.sum(y == 1)}")
        
        return X, y
    
    def train(self, X, y):
        """訓練QDA分類器"""
        print("訓練QDA分類器...")
        self.qda.fit(X, y)
        print("訓練完成!")
        
        # 訓練集評估
        y_pred = self.qda.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"訓練集準確率: {accuracy:.4f}")
        
        return accuracy
    
    def evaluate(self, X_test, y_test):
        """評估分類器"""
        y_pred = self.qda.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"測試集準確率: {accuracy:.4f}")
        print("\n詳細分類報告:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return accuracy, y_pred
    
    def predict_location(self, longitude, latitude):
        """預測指定位置的数据有效性"""
        X = np.array([[longitude, latitude]])
        prediction = self.qda.predict(X)[0]
        probabilities = self.qda.predict_proba(X)[0]
        
        result = {
            'class': prediction,
            'class_name': self.class_names[prediction],
            'probabilities': {
                self.class_names[0]: probabilities[0],
                self.class_names[1]: probabilities[1]
            }
        }
        
        return result
    
    def plot_decision_boundary(self, X, y, save_path=None):
        """繪製決策邊界"""
        # 創建網格點
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                           np.linspace(y_min, y_max, 200))
        
        # 預測網格點
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.qda.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 繪圖
        plt.figure(figsize=(12, 10))
        
        # 繪製決策邊界
        from matplotlib.colors import ListedColormap
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.3)
        
        # 繪製數據點
        colors = ['red' if label == 0 else 'green' for label in y]
        scatter = plt.scatter(X[:, 0], X[:, 1], c=colors, s=30, 
                            edgecolor='black', alpha=0.7)
        
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('QDA decision boundary - valid data classification (red: invalid, green: valid)')
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='red', markersize=8, label='Invalid data'),
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='green', markersize=8, label='valid data')
        ])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """繪製混淆矩陣"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Valid Data Classification')
        plt.colorbar()
        
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names)
        plt.yticks(tick_marks, self.class_names)
        
        # 在矩陣中顯示數值
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('true category')
        plt.xlabel('predict category')
        plt.show()

# 主程式
def main():
    print("=== 作業六第二題: QDA分類分析 ===")
    
    # 載入分類資料集
    df_cls = pd.read_csv("classification_dataset.csv")
    print(f"載入分類資料集: {len(df_cls)} 筆數據")
    
    # 初始化分類器
    classifier = QDAClassifier()
    
    # 準備數據
    X, y = classifier.prepare_data(df_cls)
    
    # 劃分訓練測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"訓練集大小: {X_train.shape[0]}")
    print(f"測試集大小: {X_test.shape[0]}")
    
    # 訓練模型
    classifier.train(X_train, y_train)
    
    # 評估模型
    print("\n=== 測試集評估 ===")
    test_accuracy, y_pred = classifier.evaluate(X_test, y_test)
    
    # 繪製混淆矩陣
    classifier.plot_confusion_matrix(y_test, y_pred)
    
    # 預測示例
    print("\n=== 數據有效性預測示例 ===")
    test_points = [
        (121.5, 25.0),  # 台北
        (120.7, 24.1),  # 台中
        (120.3, 22.6),  # 高雄
        (119.0, 23.0),  # 可能無效的區域
    ]
    
    for lon, lat in test_points:
        result = classifier.predict_location(lon, lat)
        
        # 找到最近的數據點
        distances = np.sqrt((df_cls['longitude'] - lon)**2 + (df_cls['latitude'] - lat)**2)
        nearest_idx = distances.idxmin()
        actual_class = df_cls.loc[nearest_idx, 'is_valid']
        actual_class_name = "有效數據" if actual_class == 1 else "無效數據"
        
        print(f"位置 ({lon}, {lat}):")
        print(f"  實際類別: {actual_class_name}")
        print(f"  預測類別: {result['class_name']}")
        print(f"  無效概率: {result['probabilities']['無效數據']:.3f}")
        print(f"  有效概率: {result['probabilities']['有效數據']:.3f}")
        print()
    
    # 繪製決策邊界
    print("繪製決策邊界...")
    classifier.plot_decision_boundary(X, y, "qda_classification_results.png")
    
    print("✅ 分類分析完成!")

if __name__ == "__main__":
    main()