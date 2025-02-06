import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

# 資料讀取與預處理函數
def load_and_preprocess_data(file_path, features, target):
    try:
        data = pd.read_excel(file_path)
        print("資料成功讀取！")

        if target not in data.columns:
            raise ValueError(f"目標欄位 '{target}' 不存在於資料中！")

        X = data[features]
        y = data[target]

        # 類別型欄位編碼
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes

        print("資料處理完成，類別型欄位已進行編碼！")
        return X, y
    except Exception as e:
        print(f"資料處理失敗：{e}")
        raise

# 調整閥值函數
def evaluate_threshold(y_true, y_probs, thresholds):
    print("調整閥值並評估結果：")
    for threshold in thresholds:
        y_pred_adj = (y_probs >= threshold).astype(int)
        precision = precision_score(y_true, y_pred_adj)
        recall = recall_score(y_true, y_pred_adj)
        f1 = f1_score(y_true, y_pred_adj)
        print(f"閥值: {threshold:.2f} -> 精確率: {precision:.4f}, 召回率: {recall:.4f}, F1 分數: {f1:.4f}")

# 訓練與參數調整函數
def train_xgboost_with_tuning(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")

    # SMOTE 處理類別不平衡
    smote = SMOTE(random_state=random_state, sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"過抽樣後訓練集大小：{X_train.shape}, {y_train.shape}")

    # XGBoost 模型與參數調整
    param_grid = {
    'n_estimators': [600, 1000],  # 增加樹的數量，提高模型學習能力
    'max_depth': [6, 10],  # 增加 max_depth 上限，讓模型學習更複雜的關係
    'learning_rate': [0.01, 0.05],  # 保持較低學習率，但提供稍高的選擇
    'gamma': [0.1, 0.3],  # 避免過擬合
    'scale_pos_weight': [20, 40],  # 增大範圍，適應不同數據不平衡程度
    'subsample': [0.8, 1.0],  # 控制過擬合
    'colsample_bytree': [0.8, 1.0]  # 控制每棵樹的特徵選擇
    }

    xgb_model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='f1', cv=cv, verbose=2, n_jobs=-1)

    print("開始參數調整...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"最佳參數：{grid_search.best_params_}")

    # 測試集評估
    y_prob = best_model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[f1_scores.argmax()]
    print(f"最佳閾值：{best_threshold}")

    # 根據最佳閾值進行分類
    y_pred = (y_prob >= best_threshold).astype(int)
    print("最佳閾值下的分類報告：\n", classification_report(y_test, y_pred))
    print("混淆矩陣：\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC 分數：{roc_auc_score(y_test, y_prob)}")

    # 調整不同閥值
    evaluate_threshold(y_test, y_prob, np.arange(0.1, 1.0, 0.1))

    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("混淆矩陣")
    plt.xlabel("預測值")
    plt.ylabel("實際值")
    plt.show()

    return best_model

# 繪製特徵重要性
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

# 主程式
if __name__ == "__main__":
    # 定義檔案路徑與參數
    file_path = "/Users/zhengqunyao/machine_learning_v46.xlsx"
    features = ["Education", "Employment", "Marital", "CompanyRelationship", "Industry", 
                "Job", "Type", "ApprovalResult", "Years", "Age", "Income", "LoanIncomeRatio", "Adjust"]
    target = "Flag"

    # 資料讀取與處理
    X, y = load_and_preprocess_data(file_path, features, target)

    # 模型訓練
    best_model = train_xgboost_with_tuning(X, y)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 特徵重要性
    plot_feature_importances(best_model, features, output_path=f"/Users/zhengqunyao/ml91_{timestamp}.png")
    
    # 儲存模型
    model_path = f"/Users/zhengqunyao/ml91_{timestamp}.pkl"
    joblib.dump(best_model, model_path)
    print(f"模型已儲存至：{model_path}")