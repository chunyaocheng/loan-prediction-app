import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from datetime import datetime

# ===== 資料讀取與預處理 =====
def load_and_preprocess_data(file_path, features, target):
    data = pd.read_excel(file_path)
    print("資料成功讀取！")
    X = data[features]
    y = data[target]

    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    print("資料處理完成，類別型欄位已編碼！")
    return X, y, label_encoders

# ===== 模型訓練與超參數調整 =====
def train_xgb_with_tuning(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")

    # 類別不平衡處理
    #smote = SMOTE(random_state=random_state)
    smote = SMOTE(sampling_strategy=0.11, random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"過抽樣後訓練集大小：{X_train.shape}")

    # 參數網格
    param_grid = {
        'n_estimators': [200, 300],  # 樹的數量
        'max_depth': [5, 7],  # 樹的最大深度
        'min_child_weight': [1, 5],  # 葉子節點最少樣本
        'learning_rate': [0.02, 0.05],  # 學習率
        'scale_pos_weight': [1, 2, 3],  # 正負樣本不平衡處理
        'subsample': [0.8, 1],  # 子樣本比例
        'colsample_bytree': [0.8, 1]  # 特徵子樣本比例

        ,    'n_estimators': [300, 500],  # 增加樹的數量，搭配低學習率
    'max_depth': [3, 5],  # 減少深度，避免過擬合
    'min_child_weight': [5, 10],  # 增加葉節點最小樣本，讓分支條件更穩定
    'learning_rate': [0.01, 0.02],  # 降低學習率
    'scale_pos_weight': [5, 10, 15],  # 針對不平衡資料增加正類權重
    'subsample': [0.7, 0.8],  # 隨機抽樣樣本
    'colsample_bytree': [0.7, 0.8]  # 
    
    ,    'n_estimators': [100, 200],
    'max_depth': [3, 4],
    'min_child_weight': [10, 20],
    'learning_rate': [0.05, 0.1],
    'scale_pos_weight': [3, 5],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
    ,
            'n_estimators': [200, 300],  # 樹的數量
        'max_depth': [5, 7],  # 樹的最大深度
        'min_child_weight': [1, 5],  # 葉子節點最少樣本
        'learning_rate': [0.02, 0.05],  # 學習率
        'scale_pos_weight': [1, 2, 3],  # 正負樣本不平衡處理
        'subsample': [0.8, 1],  # 子樣本比例
        'colsample_bytree': [0.8, 1]  # 特徵子樣本比例
        ,    'n_estimators': [300],
    'max_depth': [5, 6],  # 減少一點深度防 overfit
    'min_child_weight': [1, 5],  # 維持目前效果
    'learning_rate': [0.03, 0.05],  # 微調學習率
    'scale_pos_weight': [2, 3, 4],  # 微調平衡比例
    'subsample': [0.8],  # 固定不動
    'colsample_bytree': [0.8]  # 固定不動

    ,    'n_estimators': [300, 400],  # 稍微增加樹
    'max_depth': [5, 6],  # 維持適中深度
    'min_child_weight': [5, 10],  # 保守葉子樣本
    'learning_rate': [0.03, 0.05],  # 微調學習率
    'scale_pos_weight': [2, 3],  # 減少正類權重 (避免 Precision 崩潰)
    'subsample': [0.8],  # 固定
    'colsample_bytree': [0.8]  # 固定

    ,    'n_estimators': [300, 400],  # 穩定樹數
    'max_depth': [5, 6],  # 防過擬合
    'min_child_weight': [10, 15],  # 讓模型分裂更謹慎，提升 precision
    'learning_rate': [0.03, 0.05],  # 穩健學習
    'scale_pos_weight': [1.5, 2],  # 微調權重，避免過度補正正類
    'subsample': [0.8],  # 固定穩定抽樣
    'colsample_bytree': [0.7, 0.8]  # 減少特徵雜訊

    ,    'n_estimators': [200, 300, 400],  # 固定，不要太多
    'max_depth': [5, 7],  # 適中深度
    'min_child_weight': [10, 15],  # 再提升葉子樣本，讓分裂更保守
    'learning_rate': [0.03, 0.05],  # 穩定學習
    'scale_pos_weight': [1.5, 2],  # 微調平衡
    'subsample': [0.8, 1],  # 穩定樣本抽取
    'colsample_bytree': [0.7, 0.8]  # 減少特徵抽取避免噪音

        ,    'n_estimators': [200, 300, 400],  # 固定，不要太多
    'max_depth': [5, 7],  # 適中深度
    'min_child_weight': [10, 15],  # 再提升葉子樣本，讓分裂更保守
    'learning_rate': [0.03, 0.05],  # 穩定學習
    'scale_pos_weight': [1.2, 1.5],  # 微調平衡
    'subsample': [0.8, 1],  # 穩定樣本抽取
    'colsample_bytree': [0.7]  # 減少特徵抽取避免噪音

        ,    'n_estimators': [100, 300, 500],  # 固定，不要太多
    'max_depth': [7, 9],  # 適中深度
    'min_child_weight': [1, 5, 10],  # 再提升葉子樣本，讓分裂更保守
    'learning_rate': [0.03, 0.05],  # 穩定學習
    'scale_pos_weight': [1, 3, 5],  # 微調平衡
    'subsample': [0.8, 1],  # 穩定樣本抽取
    'colsample_bytree': [0.8, 1]  # 減少特徵抽取避免噪音

            ,    'n_estimators': [100, 200, 300],  # 固定，不要太多
    'max_depth': [7, 9],  # 適中深度
    'min_child_weight': [1, 5, 10],  # 再提升葉子樣本，讓分裂更保守
    'learning_rate': [0.03, 0.05],  # 穩定學習
    'scale_pos_weight': [1, 3, 5],  # 微調平衡
    'subsample': [0.8, 1],  # 穩定樣本抽取
    'colsample_bytree': [0.8, 1]  # 減少特徵抽取避免噪音
    #PR-AUC： 0.1930809648676122

    }

    # Grid Search
    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        n_jobs=-1
    )
        # ⭐⭐⭐⭐ 針對 PR-AUC 調參
    grid_search = GridSearchCV(
        estimator=XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
        param_grid=param_grid,
        scoring='average_precision',  # 直接優化 PR-AUC
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    print("開始參數調整...")
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"最佳參數：{grid_search.best_params_}")

    # 預測
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 找最佳閾值
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[f1_scores.argmax()]

    print(f"最佳 F1 閾值：{best_threshold}")

            # 自訂 recall 的最小 threshold
    #for i, r in enumerate(recall):
       # if r <= 0.75:
            #best_threshold = thresholds[i]
            #break

    #print(f"選擇的最佳門檻值 (以 Recall <= 82% 為標準): {best_threshold}")

    # 評估
    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    y_train_pred = (y_train_prob >= best_threshold).astype(int)
    y_pred = (y_prob >= best_threshold).astype(int)

    print("\n==== 訓練集 ====")
    print(classification_report(y_train, y_train_pred, digits=4))
    print("混淆矩陣：\n", confusion_matrix(y_train, y_train_pred))
    print("ROC-AUC：", roc_auc_score(y_train, y_train_prob))
    print("PR-AUC：", average_precision_score(y_train, y_train_prob))

    print("\n==== 測試集 ====")
    print(classification_report(y_test, y_pred, digits=4))
    print("混淆矩陣：\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC：", roc_auc_score(y_test, y_prob))
    print("PR-AUC：", average_precision_score(y_test, y_prob))

       # 評估不同閾值
    #evaluate_threshold(y_test, y_prob, thresholds=np.arange(0.1, 1.0, 0.1))

    return best_model, y_test, y_pred, y_prob, best_threshold

# 閾值評估函數
def evaluate_threshold(y_true, y_probs, thresholds):
    print("調整閾值並評估結果：")
    for threshold in thresholds:
        y_pred_adj = (y_probs >= threshold).astype(int)
        precision = precision_score(y_true, y_pred_adj)
        recall = recall_score(y_true, y_pred_adj)
        f1 = f1_score(y_true, y_pred_adj)
        print(f"閾值: {threshold:.2f} -> 精確率: {precision:.4f}, 召回率: {recall:.4f}, F1 分數: {f1:.4f}")

# ===== 特徵重要性視覺化 =====
def plot_feature_importances(model, features, output_path=None):
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index, palette='coolwarm')
    plt.title("Feature Importances", fontsize=16)
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"特徵重要性圖已儲存至：{output_path}")
    plt.show()

# ===== 分類報告表格美化輸出 =====
def print_classification_report(y_true, y_pred, title="分類報告"):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose().round(5)
    print(f"\n【{title}】\n")
    print(df_report)

# ===== 主程式 =====
if __name__ == "__main__":
    file_path = "/Users/zhengqunyao/train_data44.xlsx"
    features = ["Education", "Employment", "Marital", "CompanyRelationship", "Industry",
                "Job", "Type", "ApprovalResult", "Years", "Age", "Income", "LoanIncomeRatio", "Adjust"]
    target = "Flag"

    # 資料處理
    X, y, label_encoders = load_and_preprocess_data(file_path, features, target)

    # 模型訓練
    model, y_test, y_pred, y_prob, threshold = train_xgb_with_tuning(X, y)

    # 特徵重要性圖
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_feature_importances(model, features, output_path=f"/Users/zhengqunyao/xgb_importance_{timestamp}.png")

    # 儲存模型
    model_path = f"/Users/zhengqunyao/xgb_model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"模型已儲存至：{model_path}")