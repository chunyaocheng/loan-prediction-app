import pandas as pd

# 檔案路徑
file_path = "/Users/zhengqunyao/machine_learning_cleaned.xlsx"

# 讀取資料
data = pd.read_excel(file_path)


# 新增欄位1: 『距今年份』
data['距今年份'] = 113 - data['案件編號'].str[2:5].astype(int)



# 新增欄位2: 『客戶年齡』
data['客戶年齡'] = data['案件編號'].str[2:5].astype(int) + 1911 - data['客戶生日'].astype(str).str[:4].astype(int)

# 確認新增欄位
print(data[['案件編號', '客戶生日', '距今年份', '客戶年齡']].head())

# 儲存為新的檔案
output_path = "/Users/zhengqunyao/machine_learning_with_new_columns.xlsx"
data.to_excel(output_path, index=False)
print(f"新增欄位完成，結果已儲存至：{output_path}")