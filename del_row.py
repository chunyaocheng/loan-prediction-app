import pandas as pd

# 讀取資料
file_path = "/Users/zhengqunyao/machine_learning_v23.xlsx"
data = pd.read_excel(file_path)

# 去除案件類型為 "Y" 的資料
#data = data[data['案件類型'] != 'Y']

#去除欄位 與公司關係 為空值的資料
data = data[data['與公司關係'].notna()]

#去除欄位 月收入 為空值的資料
data = data[data['月收入'].notna()]

# 確認處理結果
print("去除欄位 與公司關係 月收入 後的資料筆數：", len(data))

# 儲存清理後的資料
output_path = "/Users/zhengqunyao/machine_learning_v24.xlsx"
data.to_excel(output_path, index=False)
print(f"清理完成，結果已儲存至：{output_path}")