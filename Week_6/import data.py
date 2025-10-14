import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re
import os

class TemperatureDataProcessor:
    def __init__(self, base_path: str = None):
        self.base_path = base_path
        self.xml_file_path = None
        self.temperature_data = None
        self.longitude_range = None
        self.latitude_range = None
        self.resolution = 0.03
        self.invalid_value = -999.0
        self.data_time = None
        
    def load_xml_data(self, xml_file_name: str = "O-A0038-003.xml") -> bool:
        """加載XML文件並解析溫度數據"""
        if self.base_path:
            self.xml_file_path = os.path.join(self.base_path, xml_file_name)
        else:
            self.xml_file_path = xml_file_name
            
        print(f"正在加載文件: {self.xml_file_path}")
        
        try:
            tree = ET.parse(self.xml_file_path)
            root = tree.getroot()
            
            # 解析地理信息
            geo_info = root.find('.//{urn:cwa:gov:tw:cwacommon:0.1}GeoInfo')
            self.longitude_range = (
                float(geo_info.find('{urn:cwa:gov:tw:cwacommon:0.1}BottomLeftLongitude').text),
                float(geo_info.find('{urn:cwa:gov:tw:cwacommon:0.1}TopRightLongitude').text)
            )
            self.latitude_range = (
                float(geo_info.find('{urn:cwa:gov:tw:cwacommon:0.1}BottomLeftLatitude').text),
                float(geo_info.find('{urn:cwa:gov:tw:cwacommon:0.1}TopRightLatitude').text)
            )
            
            # 解析時間信息
            data_time = root.find('.//{urn:cwa:gov:tw:cwacommon:0.1}DateTime').text
            self.data_time = data_time
            
            # 解析溫度數據
            content = root.find('.//{urn:cwa:gov:tw:cwacommon:0.1}Content').text
            self._parse_temperature_data(content)
            
            print("數據加載成功!")
            return True
            
        except Exception as e:
            print(f"加載XML數據時出錯: {e}")
            return False
    
    def _parse_temperature_data(self, content: str):
        """解析溫度格點數據"""
        # 使用正則表達式抓取所有科學記號數值
        values_str = re.findall(r"[-+]?\d+\.\d+E[-+]\d+", content)
        values = [float(v) for v in values_str]
        
        # 根據文件說明，網格大小為 67 x 120
        lon_count, lat_count = 67, 120
        expected_len = lon_count * lat_count
        
        if len(values) != expected_len:
            print(f"⚠️ 警告：數據長度 {len(values)} 和預期 {expected_len} 不一致")
        
        # 重塑為網格格式
        grid_data = np.array(values).reshape(lat_count, lon_count)
        
        # 轉置數據，使其符合 (經度, 緯度) 的順序
        self.temperature_data = grid_data.T
        print(f"溫度數據形狀: {self.temperature_data.shape}")
    
    def get_coordinates(self):
        """生成經緯度坐標網格"""
        n_lon, n_lat = self.temperature_data.shape
        
        lon_coords = np.linspace(
            self.longitude_range[0], 
            self.longitude_range[1], 
            n_lon
        )
        lat_coords = np.linspace(
            self.latitude_range[0], 
            self.latitude_range[1], 
            n_lat
        )
        
        return lon_coords, lat_coords
    
    def create_datasets(self):
        """創建回歸和分類資料集"""
        if self.temperature_data is None:
            print("請先加載數據!")
            return None, None
            
        lon_coords, lat_coords = self.get_coordinates()
        
        # 創建完整數據集
        data_list = []
        for i, lon in enumerate(lon_coords):
            for j, lat in enumerate(lat_coords):
                temp = self.temperature_data[i, j]
                data_list.append({
                    'longitude': lon,
                    'latitude': lat,
                    'temperature': temp
                })
        
        df_full = pd.DataFrame(data_list)
        
        # === 1. 回歸資料集：僅保留有效溫度數據 ===
        df_regression = df_full.copy()
        df_regression.replace(self.invalid_value, np.nan, inplace=True)
        df_regression = df_regression.dropna(subset=['temperature']).reset_index(drop=True)
        
        # === 2. 分類資料集：標記有效/無效數據 ===
        df_classification = df_full.copy()
        df_classification['is_valid'] = (df_classification['temperature'] != self.invalid_value).astype(int)
        
        print(f"回歸資料集大小: {len(df_regression)} (有效溫度數據)")
        print(f"分類資料集大小: {len(df_classification)} (包含有效/無效標記)")
        
        return df_regression, df_classification
    
    def save_datasets(self, regression_path="regression_dataset.csv", 
                     classification_path="classification_dataset.csv"):
        """保存回歸和分類資料集"""
        df_reg, df_cls = self.create_datasets()
        
        if df_reg is not None and df_cls is not None:
            df_reg.to_csv(regression_path, index=False)
            df_cls.to_csv(classification_path, index=False)
            
            print(f"✅ 回歸資料集已保存至: {regression_path}")
            print(f"✅ 分類資料集已保存至: {classification_path}")
            
            print("\n回歸資料集前 5 筆：")
            print(df_reg.head())
            print("\n分類資料集前 5 筆：")
            print(df_cls.head())
            
            return True
        return False

# 主程式
if __name__ == "__main__":
    base_path = r"c:\Users\user\Desktop\大四上\機器學習\HW6"
    
    processor = TemperatureDataProcessor(base_path)
    if processor.load_xml_data("O-A0038-003.xml"):
        processor.save_datasets()