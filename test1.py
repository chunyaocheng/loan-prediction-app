import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# 1. 載入模型
model_path = "/Users/zhengqunyao/loan_prediction_xgboost5.pkl"
loaded_model = joblib.load(model_path)

# 2. 準備測試數據
test_data_path = "/path/to/your/test_data.xlsx"
test_data = pd.read_excel(test_data_path)

features = ["Education", "Employment", "Marital", "CompanyRelationship", "Industry", 
            "Job", "Type", "ApprovalResult", "Years", "Age"]
X_test = test_data[features]

for col in X_test.select_dtypes(include=['object']).columns:
    X_test[col] = X_test[col].astype('category').cat.codes

y_test = test_data["Flag"]

# 3. 預測
y_pred = loaded_model.predict(X_test)
y_prob = loaded_model.predict_proba(X_test)[:, 1]

# 4. 評估
print("分類報告：\n", classification_report(y_test, y_pred))
print("混淆矩陣：\n", confusion_matrix(y_test, y_pred))
print(f"ROC-AUC 分數：{roc_auc_score(y_test, y_prob)}")

# 5. 可視化
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("混淆矩陣")
plt.xlabel("預測值")
plt.ylabel("真實值")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='o', label='XGBoost')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC 曲線")
plt.legend()
plt.show()