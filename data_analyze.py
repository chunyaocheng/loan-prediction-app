import pandas as pd

# 定義檔案路徑
file_path = "/Users/zhengqunyao/machine_learning_V25.xlsx"

try:
    # 讀取 Excel 檔案
    data = pd.read_excel(file_path)
    print("檔案讀取成功！")
    
    # 指定需要計算中位數的欄位
    median_columns = ['Age', 'Income', 'LoanIncomeRatio', 'Adjust']
    
    # 計算中位數（僅限指定欄位）
    median_results = data[median_columns].median()
    
    # 計算眾數（排除指定欄位）
    other_columns = [col for col in data.columns if col not in median_columns]
    mode_results = data[other_columns].mode().iloc[0]  # mode() 返回多行，取第一行

    # 顯示結果
    print("指定欄位的中位數：")
    for column, median_value in median_results.items():
        print(f"{column}: {median_value}")
    
    print("\n其他欄位的眾數：")
    for column, mode_value in mode_results.items():
        print(f"{column}: {mode_value}")

except Exception as e:
    print(f"檔案讀取失敗或處理失敗：{e}")