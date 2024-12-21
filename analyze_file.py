import pandas as pd

# 定義檔案路徑
file_path = "/Users/zhengqunyao/machine_learning_v13.xlsx"

try:
    # 讀取 Excel 檔案
    data = pd.read_excel(file_path)
    print("檔案讀取成功！")
    
    # 計算每個欄位的眾數
    mode_results = data.mode().iloc[0]  # mode() 可能返回多行，只取第一行

    # 顯示欄位名稱與眾數
    print("各欄位的眾數：")
    for column, mode_value in mode_results.items():
        print(f"{column}: {mode_value}")

except Exception as e:
    print(f"檔案讀取失敗或處理失敗：{e}")