import pandas as pd
import joblib

# 1. 載入模型
model_path = "/Users/zhengqunyao/loan_prediction_xgboost5.pkl"
loaded_model = joblib.load(model_path)
print("模型已成功載入！")

# 2. 準備單筆資料
single_data = {
    "Education": 40,
    "Employment": 10,
    "Marital": 10,
    "CompanyRelationship": 50,
    "Industry": "I",
    "Job": "0",
    "Type": "20",
    "ApprovalResult": "A010",
    "Years": 15,
    "Age": 57
}

single_data_df = pd.DataFrame([single_data])

# 類別型特徵進行編碼
for col in single_data_df.select_dtypes(include=['object']).columns:
    single_data_df[col] = single_data_df[col].astype('category').cat.codes

# 3. 預測
predicted_class = loaded_model.predict(single_data_df)
predicted_prob = loaded_model.predict_proba(single_data_df)[:, 1]

# 4. 輸出結果
print(f"預測類別：{predicted_class[0]}")  # 0 或 1
print(f"增貸概率：{predicted_prob[0]:.2f}")  # 類別 1 的概率