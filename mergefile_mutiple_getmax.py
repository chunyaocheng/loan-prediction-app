import pandas as pd

# 載入主檔案與加減碼檔案
file_path_main = "/Users/zhengqunyao/merged_file.xlsx"  # 主檔案
file_path_addon = "/Users/zhengqunyao/mlv3.xlsx"  # 加減碼檔案

# 讀取資料
main_df = pd.read_excel(file_path_main)  # 主檔案資料
addon_df = pd.read_excel(file_path_addon)  # 加減碼資料

# 將加減碼資料針對案件編號取最大值
addon_max_df = addon_df.groupby('案件編號', as_index=False)['加減碼'].max()

# 合併主檔案與處理過的加減碼資料
merged_df = pd.merge(main_df, addon_max_df, on='案件編號', how='left')

# 儲存結果到新的檔案
output_file = "/Users/zhengqunyao/merged_with_adjustment.xlsx"
merged_df.to_excel(output_file, index=False)

print(f"合併完成！結果已儲存至 {output_file}")