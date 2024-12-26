import pandas as pd

# 讀取 Excel 檔案
file_path = "/Users/zhengqunyao/machine_learning.xlsx"  # 替換為您的檔案路徑
data = pd.read_excel(file_path)

# 新增案件類型欄位（根據案件編號判斷是 X 或 Y）
data['案件類型'] = data['案件編號'].str[0]  # 提取案件編號第一個字母

# 分離新貸 (X 開頭) 和增貸 (Y 開頭) 資料
x_data = data[data['案件類型'] == 'X']
y_data = data[data['案件類型'] == 'Y']

# 取得增貸案件 (Y 開頭) 的統編清單
y統編清單 = y_data['客戶統編'].unique()  # 唯一增貸統編列表

# 判斷 X 開頭的統編是否在 Y 開頭的統編清單中，新增「增貸記號」欄位
data['增貸記號'] = data.apply(
    lambda row: 1 if row['案件類型'] == 'X' and row['客戶統編'] in y統編清單 else 0,
    axis=1
)

# 將結果儲存到新的 Excel 檔案
output_path = "/Users/zhengqunyao/machine_learning_with_flag.xlsx"  # 替換為輸出的檔案路徑
data.to_excel(output_path, index=False)
print(f"已新增『增貸記號』欄位，結果儲存至：{output_path}")