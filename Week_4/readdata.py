import xml.etree.ElementTree as ET
import csv
import re
import os

# === 檔案路徑設定 ===
base_path = r"c:\Users\user\Desktop\大四上\機器學習\HW4"
input_file = os.path.join(base_path, "O-A0038-003.xml")

# 輸出檔案
classification_file = os.path.join(base_path, "classification_data.csv")
regression_file = os.path.join(base_path, "regression_data.csv")

# === 讀取 XML ===
tree = ET.parse(input_file)
root = tree.getroot()
ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}

# 取出左下角座標
bl_lon = float(root.find('.//ns:BottomLeftLongitude', ns).text)
bl_lat = float(root.find('.//ns:BottomLeftLatitude', ns).text)

# 取出 <Content> 內的文字 (溫度矩陣)
content = root.find('.//ns:Content', ns).text
vals = re.findall(r'-?\d+\.\d+E[+-]\d+', content)

# 格點大小
ncols = 67
nrows = 120

# === 輸出分類資料集 ===
with open(classification_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Longitude", "Latitude", "Label"])
    for idx, val_str in enumerate(vals):
        lon = bl_lon + (idx % ncols) * 0.03
        lat = bl_lat + (idx // ncols) * 0.03
        label = 0 if float(val_str) == -999.0 else 1
        writer.writerow([lon, lat, label])

# === 輸出回歸資料集 ===
with open(regression_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Longitude", "Latitude", "Temperature"])
    for idx, val_str in enumerate(vals):
        temp = float(val_str)
        if temp != -999.0:
            lon = bl_lon + (idx % ncols) * 0.03
            lat = bl_lat + (idx // ncols) * 0.03
            writer.writerow([lon, lat, temp])

print("已輸出：")
print(" -", classification_file)
print(" -", regression_file)
