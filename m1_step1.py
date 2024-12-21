import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 讀取資料
file_path = "/Users/zhengqunyao/machine_learning_v12.xlsx"
data = pd.read_excel(file_path)


# 特徵欄位與目標欄位
features = ["學歷", "就業狀態", "婚姻狀況", "與公司關係", "行業別代碼", 
            "職稱代碼", "申貸性質代碼", "審核結果", "距今年份", "客戶年齡"]
X = data[features]
y = data["增貸記號"]

# 確認特徵與目標的形狀
print("X 的形狀：", X.shape)
print("y 的形狀：", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 確認分割後資料集的大小
print("訓練集大小：", X_train.shape, y_train.shape)
print("測試集大小：", X_test.shape, y_test.shape)

# 建立隨機森林模型
model = RandomForestClassifier(random_state=42)

# 訓練模型
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 模型準確率與分類報告
print("模型準確率：", accuracy_score(y_test, y_pred))
print("分類報告：\n", classification_report(y_test, y_pred))