import pandas as pd

# 載入 Excel 檔案
file_path1 = "/Users/zhengqunyao/machine_learning_v31.xlsx"  # 檔案一路徑
file_path2 = "/Users/zhengqunyao/mlv2.xlsx"  # 檔案二路徑

# 讀取兩個檔案
df1 = pd.read_excel(file_path1)  # 檔案一
df2 = pd.read_excel(file_path2)  # 檔案二

# 根據『案件編號』合併資料
merged_df = pd.merge(df1, df2[['案件編號', '月收入', '借保人總貸款月收支比例']], on='案件編號', how='left')

# 儲存為新的檔案
output_file = "/Users/zhengqunyao/machine_learning_v32.xlsx"
merged_df.to_excel(output_file, index=False)

print(f"合併完成！結果已儲存至 {output_file}")