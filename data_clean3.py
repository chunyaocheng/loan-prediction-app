import pandas as pd

# 讀取資料檔案
file_path = "/Users/zhengqunyao/machine_learning_v1.xlsx"
data = pd.read_excel(file_path)

# 檢查缺失值
print("缺失值情況（處理前）：\n", data.isnull().sum())

# 計算眾數
mode_value = data['與公司關係'].mode()[0]
# 填補缺失值
data['與公司關係'] = data['與公司關係'].fillna(mode_value)



# 再次檢查缺失值
print("缺失值情況（處理後）：\n", data.isnull().sum())

# 儲存清理後的資料
cleaned_path = "/Users/zhengqunyao/machine_learning_v11.xlsx"
data.to_excel(cleaned_path, index=False)
print(f"清理完成，結果已儲存至：{cleaned_path}")