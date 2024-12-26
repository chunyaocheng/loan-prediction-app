import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. 讀取資料
file_path = "/Users/zhengqunyao/machine_learning_encoded.xlsx"
data = pd.read_excel(file_path)

# 確認資料
print("資料概況：")
print(data.info())
print(data.head())

# 2. 特徵選擇與目標分離
# 假設目標欄位是 '增貸記號'
X = data.drop(['增貸記號'], axis=1)  # 特徵
y = data['增貸記號']  # 目標

# 3. 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 建立隨機森林模型
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)  # 訓練模型

# 5. 模型測試與評估
y_pred = model.predict(X_test)

# 輸出準確率與分類報告
print("模型準確率：", accuracy_score(y_test, y_pred))
print("分類報告：\n", classification_report(y_test, y_pred))

# 6. 儲存模型
import joblib
model_path = "/Users/zhengqunyao/loan_prediction_model.pkl"
joblib.dump(model, model_path)
print(f"模型已保存至：{model_path}")