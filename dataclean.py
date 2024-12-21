import pandas as pd

# 讀取資料檔案
file_path = "/Users/zhengqunyao/machine_learning_with_mapped_ids.xlsx"
data = pd.read_excel(file_path)

# 檢查資料基本概況
print("資料概況：")
print(data.info())

# 檢查缺失值
missing_values = data.isnull().sum()
print("缺失值情況：\n", missing_values)

# 填補類別型資料
categorical_cols_to_unknown = ['公司統編', '公司名稱', '與公司關係', '行業別代碼', '職稱代碼']
for col in categorical_cols_to_unknown:
    data[col].fillna('未知', inplace=True)

# 填補數值型資料（中位數）
median_value = data['申貸性質代碼'].median()
data['申貸性質代碼'].fillna(median_value, inplace=True)

# 填補類別型資料（眾數）
mode_value = data['審核結果代碼'].mode()[0]
data['審核結果代碼'].fillna(mode_value, inplace=True)

# 再次檢查缺失值
print("缺失值情況（處理後）：\n", data.isnull().sum())

# 儲存清理後的資料
cleaned_path = "/Users/zhengqunyao/machine_learning_cleaned.xlsx"
data.to_excel(cleaned_path, index=False)
print(f"清理完成，結果已儲存至：{cleaned_path}")