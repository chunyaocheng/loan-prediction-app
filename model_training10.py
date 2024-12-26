import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

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

    # 測試不同的閥值
def evaluate_threshold(y_true, y_probs, thresholds):
    print("調整閥值並評估結果：")
    for threshold in thresholds:
        y_pred_adj = (y_probs >= threshold).astype(int)
        precision = precision_score(y_true, y_pred_adj)
        recall = recall_score(y_true, y_pred_adj)
        f1 = f1_score(y_true, y_pred_adj)
        print(f"閥值: {threshold:.2f} -> 精確率: {precision:.4f}, 召回率: {recall:.4f}, F1 分數: {f1:.4f}")

# 定義模型訓練與超參數調整函數
def train_random_forest_with_tuning(X, y, test_size=0.2, random_state=42):
    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")
    
    # 處理類別不平衡（SMOTE）
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"過抽樣後訓練集大小：{X_train.shape}, {y_train.shape}")
    
    # 定義隨機森林模型與參數網格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    # 使用 GridSearchCV 尋找最佳參數
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        n_jobs=-1
    )
    
    print("開始參數調整...")
    grid_search.fit(X_train, y_train)
    
    # 最佳參數與模型
    best_model = grid_search.best_estimator_
    print(f"最佳參數：{grid_search.best_params_}")

    
    # 預測測試集
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 類別 1 的機率
    y_pred = (y_prob >= 0.25).astype(int)  # 預設閥值 0.1

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # 找到最佳閾值
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[f1_scores.argmax()]
    print(f"最佳閾值：{best_threshold}")
    
    print("預設閥值 (0.5) 下的分類報告：\n", classification_report(y_test, y_pred))
    print("混淆矩陣：\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC 分數：{roc_auc_score(y_test, y_prob)}")
    
    # 調整不同閥值
    thresholds = np.arange(0.1, 0.9, 0.1)  # 例如從 0.1 到 0.9 閥值
    evaluate_threshold(y_test, y_prob, thresholds)
    
    # 模型評估
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"模型準確率：{accuracy}")
    
    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("confuse matrix")
    plt.xlabel("predict")
    plt.ylabel("actual")
    plt.show()
    
    return best_model, y_test, y_pred, y_prob

# 定義特徵重要性繪圖函數
def plot_feature_importances(model, features, output_path=None):
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title("attr importanc")
    plt.ylabel("importance")
    plt.xlabel("attr")
    if output_path:
        plt.savefig(output_path)
        print(f"特徵重要性圖已儲存至：{output_path}")

    plt.show()

# 定義主程式
if __name__ == "__main__":
    # 定義參數
    file_path = "/Users/zhengqunyao/machine_learning_v25.xlsx"
    features = ["Education", "Employment", "Marital", "CompanyRelationship", "Industry", 
                "Job", "Type", "ApprovalResult", "Years", "Age", "Income", "LoanIncomeRatio", "Adjust"]
    #
    target = "Flag"
    
    # 資料讀取與預處理
    X, y, label_encoders = load_and_preprocess_data(file_path, features, target)
    
    # 模型訓練與超參數調整
    model, y_test, y_pred, y_prob = train_random_forest_with_tuning(X, y)
    
    # 繪製特徵重要性圖
    plot_feature_importances(model, features, output_path="/Users/zhengqunyao/feature_importances_tuned.png")
    
    # 儲存模型
    model_path = "/Users/zhengqunyao/loan_prediction_model_tuned2525.pkl"
    joblib.dump(model, model_path)
    print(f"模型已儲存至：{model_path}")