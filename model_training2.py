import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 讀取資料
file_path = "/Users/zhengqunyao/machine_learning_v12.xlsx"
data = pd.read_excel(file_path)

# 查看資料基本資訊
# print(data.info())

# 定義特徵欄位與目標欄位
features = ["學歷", "就業狀態", "婚姻狀況", "與公司關係", "行業別代碼", 
            "職稱代碼", "申貸性質代碼", "審核結果", "距今年份", "客戶年齡"]
X = data[features]
y = data["增貸記號"]

# 確認特徵與目標的形狀
# print("X 的形狀：", X.shape)
# print("y 的形狀：", y.shape)

# print(X.dtypes)  # 檢查型別

from sklearn.preprocessing import LabelEncoder

# 對類別型欄位進行 Label Encoding
label_encoders = {}
for col in ["學歷", "就業狀態", "婚姻狀況", "與公司關係", "行業別代碼", "職稱代碼", "申貸性質代碼", "審核結果"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # 保存編碼器以便日後反查

    # 分割資料集，測試集比例為 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 確認訓練集與測試集的大小
print("訓練集大小：", X_train.shape, y_train.shape)
print("測試集大小：", X_test.shape, y_test.shape)

# 建立隨機森林模型
model = RandomForestClassifier(random_state=42)

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 評估模型性能
print("模型準確率：", accuracy_score(y_test, y_pred))
print("分類報告：\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# 獲取特徵重要性
feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

# 繪製特徵重要性圖
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.title("特徵重要性")
plt.show()

import joblib

model_path = "/Users/zhengqunyao/loan_prediction_model2.pkl"
joblib.dump(model, model_path)
print(f"模型已儲存至：{model_path}")