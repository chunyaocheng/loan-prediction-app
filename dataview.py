import pandas as pd

# 讀取資料檔案
#file_path = "/Users/zhengqunyao/machine_learning_with_mapped_ids.xlsx"
#file_path = "/Users/zhengqunyao/machine_learning_v13.xlsx"
file_path = "/Users/zhengqunyao/machine_learning_v24.xlsx"
data = pd.read_excel(file_path)

# 查看資料基本概況
print(data.info())  # 顯示資料型態和非空值數量
print(data.head())  # 查看前 5 行