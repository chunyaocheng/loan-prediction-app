import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# 定義資料讀取與檢查函數
def load_and_preprocess_data(file_path, features, target):
    try:
        # 讀取資料
        data = pd.read_excel(file_path)
        print("資料成功讀取！")
        
        # 顯示基本資訊
        print("資料資訊：")
        print(data.info())
        
        # 檢查目標欄位是否存在
        if target not in data.columns:
            raise ValueError(f"目標欄位 '{target}' 不存在於資料中！")
        
        # 選擇特徵欄位與目標欄位
        X = data[features]
        y = data[target]
        
        # Label Encoding
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        print("資料處理完成，類別型欄位已進行編碼！")
        return X, y, label_encoders
    except Exception as e:
        print(f"資料處理失敗：{e}")
        raise

# 定義模型訓練函數
def train_random_forest(X, y, test_size=0.2, random_state=42):
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")
    
    # 建立隨機森林模型
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    # 預測測試集
    y_pred = model.predict(X_test)
    
    # 模型評估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"模型準確率：{accuracy}")
    print("分類報告：\n", report)
    
    return model, y_test, y_pred

# 定義特徵重要性繪圖函數
def plot_feature_importances(model, features, output_path=None):
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title("特徵重要性")
    plt.ylabel("重要性")
    plt.xlabel("特徵")
    if output_path:
        plt.savefig(output_path)
        print(f"特徵重要性圖已儲存至：{output_path}")
    plt.show()

# 定義主程式
if __name__ == "__main__":
    # 定義參數
    file_path = "/Users/zhengqunyao/machine_learning_v13.xlsx"
    features = ["Education", "Employment", "Marital", "CompanyRelationship", "Industry", 
                "Job", "Type", "ApprovalResult", "Years", "Age"]
    target = "Flag"
    
    # 資料讀取與預處理
    X, y, label_encoders = load_and_preprocess_data(file_path, features, target)
    
    # 模型訓練
    model, y_test, y_pred = train_random_forest(X, y)
    
    # 繪製特徵重要性圖
    plot_feature_importances(model, features, output_path="/Users/zhengqunyao/feature_importances.png")
    
    # 儲存模型
    model_path = "/Users/zhengqunyao/loan_prediction_model3.pkl"
    joblib.dump(model, model_path)
    print(f"模型已儲存至：{model_path}")