import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime
from imblearn.over_sampling import ADASYN


# 資料讀取與預處理函數
def load_and_preprocess_data(file_path, features, target):
    try:
        data = pd.read_excel(file_path)
        print("資料成功讀取！")
        
        if target not in data.columns:
            raise ValueError(f"目標欄位 '{target}' 不存在於資料中！")
        
        X = data[features]
        y = data[target]
        
        # 處理類別不平衡 (Label Encoding)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        print("資料處理完成，類別型欄位已進行編碼！")
        return X, y
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

# 修改主訓練函數
def train_xgboost_with_tuning(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")
    
    # SMOTE 過抽樣
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)


    #adasyn = ADASYN(random_state=42)
    #X_train, y_train = adasyn.fit_resample(X_train, y_train)
    print(f"過抽樣後訓練集大小：{X_train.shape}, {y_train.shape}")
    
    # XGBoost 訓練及參數調整 (略)
    param_grid = {
    #'n_estimators': [200, 300],       # 減少樹數
    #'max_depth': [4, 5, 6],              # 降低深度
    #'learning_rate': [0.03, 0.1],     # 保留
    #'scale_pos_weight': [20, 30, 40],     # 適中即可，不用太極端
    #'gamma': [0.5, 1, 2] ,           # 調高
    #'min_child_weight':  [5, 7, 10],   # 調高
    #'subsample': [0.6, 0.8],          # 降低
    #'colsample_bytree': [0.6, 0.8]    # 降低

    #'n_estimators': [200, 300],
    #'max_depth': [4, 5, 6],
    #'learning_rate': [0.03, 0.1],
    #'scale_pos_weight': [30, 40],  # 收斂範圍
    #'gamma': [1, 2, 3],            # 保守切割
    #'min_child_weight': [7, 10, 12], # 降低過擬合
    #'subsample': [0.6, 0.8],
    #'colsample_bytree': [0.6, 0.8]

    #'n_estimators': [200, 300],
    #'max_depth': [4, 5, 6],
    #'learning_rate': [0.03, 0.1],
    #'scale_pos_weight': [30, 40],  # 不動
    #'gamma': [2, 3, 5],            # **提高，讓模型更謹慎預測 1**
    #'min_child_weight': [10, 12, 15], # **提高，防止過擬合**
    #'subsample': [0.6, 0.8],
    #'colsample_bytree': [0.6, 0.8]

    'n_estimators': [150, 200],  # 減少樹的數量
    'max_depth': [4, 5],         # 降低模型深度
    'learning_rate': [0.03, 0.1],
    'scale_pos_weight': [10, 20],
    'gamma': [3, 5, 7],          # **提高 gamma**
    'min_child_weight': [12, 15, 20],  # **讓模型學得更保守**
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],

    #'n_estimators': [100, 150],  # 減少樹的數量，避免學習太多
    #'max_depth': [4, 5],         # 控制模型深度
    #'learning_rate': [0.03, 0.1],
    #'scale_pos_weight': [20, 30],  # **降低類別 1 的重要性，減少誤報**
    #'gamma': [5, 7, 10],          # **提高 gamma，讓模型更謹慎**
    #'min_child_weight': [12, 15, 20],  # **讓模型更保守**
    #'subsample': [0.6, 0.8],
    #'colsample_bytree': [0.6, 0.8]

    'n_estimators': [500, 800],  # 增加樹的數量，提高模型學習能力
    'max_depth': [3, 4],  # 增加 max_depth 上限，讓模型學習更複雜的關係
    'learning_rate': [0.01, 0.05],  # 保持較低學習率，但提供稍高的選擇
    'scale_pos_weight': [3, 5, 8],
    'gamma': [5, 7, 10],  # 避免過擬合
    'min_child_weight': [10, 15, 20],  # **讓模型學得更保守**
    'subsample': [0.6, 0.8],  # 控制過擬合
    'colsample_bytree': [0.6, 0.8],  # 控制每棵樹的特徵選擇

    'n_estimators': [500, 800],  # 維持樹的數量
    'max_depth': [3, 4],  # 控制模型複雜度
    'learning_rate': [0.03, 0.07],  # 稍微提升學習率，加快收斂
    'scale_pos_weight': [10, 15, 20],  # 增加對 1 類的關注度
    'gamma': [3, 5, 7],  # 保持適度的正則化
    'min_child_weight': [5, 10, 15],  # 讓模型稍微靈活
    'subsample': [0.6, 0.8],  # 保持
    'colsample_bytree': [0.6, 0.8], # 保持

    'n_estimators': [500, 800],  # 保持
    'max_depth': [3, 4],  # 保持
    'learning_rate': [0.03, 0.07],  # 保持
    'scale_pos_weight': [5, 8, 10],  # 降低偏向少數類別的程度
    'gamma': [2, 3, 5],  # 降低 gamma 限制
    'min_child_weight': [3, 5, 10],  # 讓模型更靈活
    'subsample': [0.6, 0.8],  # 保持
    'colsample_bytree': [0.6, 0.8], # 保持

    #'n_estimators': [500, 800],  # 保持
    #'max_depth': [3, 4],  # 保持
    #'learning_rate': [0.03, 0.07, 0.1],  # 提高學習率，加快收斂
    #'scale_pos_weight': [3, 5, 8],  # 降低對 1 類的強調，減少錯誤預測
    #'gamma': [5, 7, 10],  # 增加 gamma 限制，讓模型更保守
    #'min_child_weight': [5, 10, 15],  # 提高 min_child_weight，防止過擬合
    #'subsample': [0.6, 0.8],  # 保持
    #'colsample_bytree': [0.6, 0.8]  # 保持
    }
    xgb_model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='f1', cv=3, verbose=2, n_jobs=-1)
    
    print("開始參數調整...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"最佳參數：{grid_search.best_params_}")
    
    # 測試集評估
    y_prob = best_model.predict_proba(X_test)[:, 1]  # 預測機率

    # 找出最佳閾值
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold_f1 = thresholds[f1_scores.argmax()]
    print(f"原先f1_scores的最佳閥值: {best_threshold_f1}")

    # 找到 Recall >= 88% 的最小 threshold
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

    #print(f"最佳閾值：{best_threshold}")
    # 根據最佳閾值進行分類
    y_pred = (y_prob >= best_threshold).astype(int)
    print("測試集 分類報告：\n", classification_report(y_test, y_pred))
    print("測試集 混淆矩陣：\n", confusion_matrix(y_test, y_pred))
    print(f"測試集 ROC-AUC 分數：{roc_auc_score(y_test, y_prob)}")
    print(f"測試集 PR-AUC 分數：{average_precision_score(y_test, y_prob)}")
    
    # 調整不同閥值
    thresholds = np.arange(0.1, 1.0, 0.1)  # 例如從 0.1 到 0.9 閥值
    evaluate_threshold(y_test, y_prob, thresholds)
    
    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("confuse matrix")
    plt.xlabel("predict")
    plt.ylabel("actual")
    plt.show()
    
    return best_model

# 繪製特徵重要性
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

# 主程式
if __name__ == "__main__":
    # 定義檔案路徑與參數
    file_path = "/Users/zhengqunyao/machine_learning_v25.xlsx"
    features = ["Education", "Employment", "Marital", "CompanyRelationship", "Industry", 
                "Job", "Type", "ApprovalResult", "Years", "Age", "Income", "LoanIncomeRatio", "Adjust"]
    target = "Flag"
    
    # 資料讀取與處理
    X, y = load_and_preprocess_data(file_path, features, target)
    
    # 模型訓練
    best_model = train_xgboost_with_tuning(X, y)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 特徵重要性
    plot_feature_importances(best_model, features, output_path=f"/Users/zhengqunyao/ml9_{timestamp}.png")
    
    # 儲存模型
    model_path = f"/Users/zhengqunyao/ml9_{timestamp}.pkl"
    joblib.dump(best_model, model_path)
    print(f"模型已儲存至：{model_path}")