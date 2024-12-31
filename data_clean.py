import pandas as pd

# 讀取資料檔案
file_path = "/Users/zhengqunyao/machine_learning_v41.xlsx"
data = pd.read_excel(file_path)

# 檢查資料基本概況
print("資料概況：")
print(data.info())

# 檢查缺失值
missing_values = data.isnull().sum()
print("缺失值情況：\n", missing_values)


# 填補數值型資料（中位數）
numerical_cols_to_median = ['月收入', '借保人總貸款月收支比例', '加減碼']
for col in numerical_cols_to_median:
    if col in data.columns:
        median_value = data[col].median()
        data[col].fillna(median_value, inplace=True)
        print(f"{col} 缺失值已填補為中位數：{median_value}")

# 填補類別型資料（眾數）
categorical_cols_to_mode = ['與公司關係', '行業別代碼', '職稱代碼']
for col in categorical_cols_to_mode:
    if col in data.columns:
        mode_value = data[col].mode()[0]
        data[col].fillna(mode_value, inplace=True)
        print(f"{col} 缺失值已填補為眾數：{mode_value}")

# 再次檢查缺失值
print("缺失值情況（處理後）：\n", data.isnull().sum())

# 儲存清理後的資料
cleaned_path = "/Users/zhengqunyao/machine_learning_v42.xlsx"
data.to_excel(cleaned_path, index=False)
print(f"清理完成，結果已儲存至：{cleaned_path}")