import pandas as pd

# 讀取資料檔案
file_path = "/Users/zhengqunyao/machine_learning_with_mapped_ids.xlsx"
data = pd.read_excel(file_path)

# 檢查缺失值
print("缺失值情況（處理前）：\n", data.isnull().sum())

# 填補類別型資料
categorical_cols_to_unknown = ['公司名稱', '行業別代碼', '職稱代碼']
for col in categorical_cols_to_unknown:
    data[col] = data[col].fillna('未知')

# 將數值型欄位轉為字串後填補
numeric_to_string_cols = ['公司統編', '與公司關係']
for col in numeric_to_string_cols:
    data[col] = data[col].astype(str).fillna('未知')

# 填補數值型資料（中位數）
median_value = data['申貸性質代碼'].median()
data['申貸性質代碼'] = data['申貸性質代碼'].fillna(median_value)

# 填補類別型資料（眾數）
mode_value = data['審核結果'].mode()[0]
data['審核結果'] = data['審核結果'].fillna(mode_value)

# 再次檢查缺失值
print("缺失值情況（處理後）：\n", data.isnull().sum())

# 儲存清理後的資料
cleaned_path = "/Users/zhengqunyao/machine_learning_cleaned.xlsx"
data.to_excel(cleaned_path, index=False)
print(f"清理完成，結果已儲存至：{cleaned_path}")