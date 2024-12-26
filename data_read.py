import pandas as pd

# 替換為您的 Excel 檔案路徑
file_path = "/Users/zhengqunyao/machine_learning.xlsx"

try:
    # 嘗試讀取 Excel 檔案
    data = pd.read_excel(file_path)
    print("檔案成功讀取！")
    print(data.head())  # 顯示前 5 行資料
except FileNotFoundError:
    print("檔案不存在，請確認路徑！")
except Exception as e:
    print("發生錯誤：", e)