import pandas as pd
import joblib

# 1. 載入模型
model_path = "/Users/zhengqunyao/lgb_model_20250318_211840.pkl"



loaded_model = joblib.load(model_path)
print("模型已成功載入！")

# 2. 讀取 Excel 資料
file_path = "/Users/zhengqunyao/test_data44.xlsx"
data = pd.read_excel(file_path)

# 建立預測結果的欄位
data['預測類別'] = None
data['增貸概率'] = None

# 3. 逐筆進行預測
for index, row in data.iterrows():
    # 準備單筆資料
    single_data = {
        "Education": row['Education'],
        "Employment": row['Employment'],
        "Marital": row['Marital'],
        "CompanyRelationship": row['CompanyRelationship'],
        "Industry": row['Industry'],
        "Job": row['Job'],
        "Type": row['Type'],
        "ApprovalResult": row['ApprovalResult'],
        "Years": row['Years'],
        "Age": row['Age'],
        "Income": row['Income'],
        "LoanIncomeRatio": row['LoanIncomeRatio'],
        "Adjust": row['Adjust']
    }
    
    # 將單筆資料轉為 DataFrame
    single_data_df = pd.DataFrame([single_data])
    
    # 類別型特徵進行編碼
    for col in single_data_df.select_dtypes(include=['object']).columns:
        single_data_df[col] = single_data_df[col].astype('category').cat.codes
    
    # 預測類別與概率
    predicted_class = loaded_model.predict(single_data_df)
    predicted_prob = loaded_model.predict_proba(single_data_df)[:, 1]
    
    # 回寫結果到原資料
    data.loc[index, '預測類別'] = predicted_class[0]
    data.loc[index, '增貸概率'] = predicted_prob[0]

# 4. 儲存回寫結果的資料到新的 Excel
output_path = "/Users/zhengqunyao/test_data44_light_normal.xlsx"
data.to_excel(output_path, index=False)
print(f"預測完成，結果已儲存至：{output_path}")