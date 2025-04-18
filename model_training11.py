import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib


# 資料讀取與預處理函數
def load_and_preprocess_data(file_path, features, target):
    try:
        data = pd.read_excel(file_path)
        print("資料成功讀取！")

        if target not in data.columns:
            raise ValueError(f"目標欄位 '{target}' 不存在於資料中！")

        # 選擇特徵與目標欄位
        X = data[features]
        y = data[target]

        # 類別型特徵進行 Label Encoding
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


# 模型訓練與超參數調整函數 (SVM)
def train_svm_with_tuning(X, y, test_size=0.2, random_state=42):
    # 資料集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")

    # 類別不平衡處理 (SMOTE)
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"過抽樣後訓練集大小：{X_train.shape}, {y_train.shape}")

    # 定義 SVM 模型參數網格
    param_grid = {
        'C': [0.1, 1, 10],               # 正則化參數
        'kernel': ['linear', 'rbf'],     # 核函數
        'gamma': ['scale', 'auto'],      # RBF 核函數的 gamma 參數
    }

    # 使用 GridSearchCV 尋找最佳參數
    grid_search = GridSearchCV(
        estimator=SVC(probability=True, random_state=random_state),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    print("開始參數調整...")
    grid_search.fit(X_train, y_train)

    # 最佳模型與參數
    best_model = grid_search.best_estimator_
    print(f"最佳參數：{grid_search.best_params_}")

    # 測試集預測
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 預測為類別 1 的機率

    # 找出最佳閾值
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[f1_scores.argmax()]
    print(f"最佳閾值：{best_threshold}")

    # 根據最佳閾值進行分類
    y_pred = (y_prob >= best_threshold).astype(int)

    # 評估模型
    print("分類報告：\n", classification_report(y_test, y_pred))
    print("混淆矩陣：\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC 分數：{roc_auc_score(y_test, y_prob)}")

    # 評估不同閾值
    thresholds = np.arange(0.1, 0.9, 0.1)
    for threshold in thresholds:
        y_pred_adj = (y_prob >= threshold).astype(int)
        precision = precision_score(y_test, y_pred_adj)
        recall = recall_score(y_test, y_pred_adj)
        f1 = f1_score(y_test, y_pred_adj)
        print(f"閾值: {threshold:.2f} -> 精確率: {precision:.4f}, 召回率: {recall:.4f}, F1 分數: {f1:.4f}")

    # 繪製混淆矩陣
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return best_model


# 主程式
if __name__ == "__main__":
    # 資料路徑與參數
    file_path = "/Users/zhengqunyao/machine_learning_v44.xlsx"
    features = ["Education", "Employment", "Marital", "CompanyRelationship", "Industry", 
                "Job", "Type", "ApprovalResult", "Years", "Age", "Income", "LoanIncomeRatio", "Adjust"]
    target = "Flag"

    # 資料處理
    X, y, label_encoders = load_and_preprocess_data(file_path, features, target)

    # 模型訓練
    best_model = train_svm_with_tuning(X, y)

    # 儲存模型
    model_path = "/Users/zhengqunyao/svm_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"模型已儲存至：{model_path}")