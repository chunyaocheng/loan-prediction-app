import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from datetime import datetime


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


# 閾值評估函數
def evaluate_threshold(y_true, y_probs, thresholds):
    print("調整閾值並評估結果：")
    for threshold in thresholds:
        y_pred_adj = (y_probs >= threshold).astype(int)
        precision = precision_score(y_true, y_pred_adj)
        recall = recall_score(y_true, y_pred_adj)
        f1 = f1_score(y_true, y_pred_adj)
        print(f"閾值: {threshold:.2f} -> 精確率: {precision:.4f}, 召回率: {recall:.4f}, F1 分數: {f1:.4f}")


# 模型訓練與超參數調整函數
def train_random_forest_with_tuning(X, y, test_size=0.2, random_state=42):
    # 資料集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")

    # 類別不平衡處理 (SMOTE)
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"過抽樣後訓練集大小：{X_train.shape}, {y_train.shape}")

    # 定義隨機森林模型參數網格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample'],

    'n_estimators': [150, 200],  # 減少樹的數量，避免過擬合
    'max_depth': [15, 20],  # 降低 max_depth，減少學習過多細節
    'min_samples_split': [20, 30],  # 限制過小樣本分裂，減少過擬合
    'class_weight': [{0: 1, 1: 1.5}, {0: 1, 1: 2}],  # 減少對 1 類的過補償

    'n_estimators': [100, 300],  # 減少樹的數量，避免過擬合
    'max_depth': [20, 25],  # 降低 max_depth，提升泛化能力
    'min_samples_split': [10, 15],
    'class_weight': [{0: 1, 1: 1.5}, {0: 1, 1: 2}]  # 避免少數類過補償

    #'n_estimators': [50, 100, 150],  # 減少樹的數量，避免過度擬合
    #'max_depth': [5, 10, 15],  # 降低 max_depth，減少過度學習
    #'min_samples_split': [10, 20, 30],  # 提高 min_samples_split，讓分支至少需要更多樣本才能分裂，減少過擬合
    #'class_weight': [{0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 8}]  # 降低 scale_pos_weight，避免對少數類過度補償

    #'n_estimators': [50, 80],  # 減少樹的數量，避免過擬合
    #'max_depth': [5, 7],  # 降低 max_depth，增加泛化能力
    #'min_samples_split': [20, 30, 50],  # 限制過小樣本分裂
    #'max_features': ['sqrt', 'log2'],  # 限制特徵選擇，避免過度學習
    #'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 3}],  # 減少過度補償

    #'n_estimators': [80, 100],  # 稍微提高，但不過度
    #'max_depth': [8, 10],  # 允許學習更細節的模式
    #'min_samples_split': [10, 15, 20],  # 讓模型不會過度受限
    #'max_features': ['sqrt', 'log2'],  # 減少過度擬合
    #'class_weight': [{0: 1, 1: 1.5}, {0: 1, 1: 2}],  # 讓 1 類補償合理化

    #'n_estimators': [80, 100],  # 保持不變
    #'max_depth': [6, 8],  # 降低以提升泛化能力
    #'min_samples_split': [15, 20, 30],  # 降低過擬合
    #'max_features': ['sqrt', 'log2'],  # 保持不變
    #'class_weight': [{0: 1, 1: 1.2}, {0: 1, 1: 1.3}],  # 減少對 1 類的過度補償

    #'n_estimators': [80, 100],  # 保持
    #'max_depth': [8, 10],  # 讓模型學習更深入的結構
    #'min_samples_split': [25, 30],  # 提高，讓決策樹學習更穩定
    #'max_features': ['sqrt', 'log2'],  # 保持
    #'class_weight': [{0: 1, 1: 1.05}, {0: 1, 1: 1.1}]  # 讓 1 類的補償更輕微

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

    # 最佳模型與參數
    best_model = grid_search.best_estimator_
    print(f"最佳參數：{grid_search.best_params_}")

    # 測試集預測
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 預測為類別 1 的機率

    # 找出最佳閾值
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    #best_threshold = thresholds[f1_scores.argmax()]

    #print(f"最佳閾值：{best_threshold}")

    best_threshold_f1 = thresholds[f1_scores.argmax()]
    print(f"原先f1_scores的最佳閥值: {best_threshold_f1}")

    # 找到 Recall <= 82% 的最小 threshold
    for i, r in enumerate(recall):
        if r <= 0.82:
            best_threshold = thresholds[i]
            break

    print(f"選擇的最佳門檻值 (以 Recall <= 82% 為標準): {best_threshold}")
    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_prob >= best_threshold).astype(int)
    print("訓練集 分類報告：\n", classification_report(y_train, y_train_pred))
    print("訓練集 混淆矩陣：\n", confusion_matrix(y_train, y_train_pred))
    print(f"訓練集 ROC-AUC：{roc_auc_score(y_train, y_train_prob)}")
    print(f"訓練集 PR-AUC：{average_precision_score(y_train, y_train_prob)}")

    # 根據最佳閾值進行分類
    y_pred = (y_prob >= best_threshold).astype(int)

    # 評估模型
    print("測試集 分類報告：\n", classification_report(y_test, y_pred))
    print("測試集 混淆矩陣：\n", confusion_matrix(y_test, y_pred))
    print(f"測試集 ROC-AUC 分數：{roc_auc_score(y_test, y_prob)}")
    print(f"測試集 PR-AUC 分數：{average_precision_score(y_test, y_prob)}")

    # 評估不同閾值
    evaluate_threshold(y_test, y_prob, thresholds=np.arange(0.1, 1.0, 0.1))

    # 繪製混淆矩陣
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return best_model, y_test, y_pred, y_prob


# 特徵重要性繪圖函數
def plot_feature_importances(model, features, output_path=None):
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    if output_path:
        plt.savefig(output_path)
        print(f"特徵重要性圖已儲存至：{output_path}")
    plt.show()


# 主程式
if __name__ == "__main__":
    # 資料路徑與參數
    file_path = "/Users/zhengqunyao/machine_learning_v25.xlsx"
    features = ["Education", "Employment", "Marital", "CompanyRelationship", "Industry", 
                "Job", "Type", "ApprovalResult", "Years", "Age", "Income", "LoanIncomeRatio", "Adjust"]
    target = "Flag"

    # 資料處理
    X, y, label_encoders = load_and_preprocess_data(file_path, features, target)

    # 模型訓練
    model, y_test, y_pred, y_prob = train_random_forest_with_tuning(X, y)

    # 特徵重要性繪圖
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_feature_importances(model, features, output_path=f"/Users/zhengqunyao/ml10_{timestamp}.png")

    # 儲存模型
    model_path = f"/Users/zhengqunyao/ml10_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"模型已儲存至：{model_path}")