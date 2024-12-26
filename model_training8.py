
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


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

# 更新後的模型訓練與最佳閾值選擇
def train_and_optimize_xgboost(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")
    
    # SMOTE 過抽樣
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"過抽樣後訓練集大小：{X_train_resampled.shape}, {y_train_resampled.shape}")

    param_grid = {
    'n_estimators': [300, 500],           # 增加樹的數量，讓模型有更多學習能力
    'max_depth': [6, 8, 10],             # 增大樹的深度以捕捉更複雜的模式
    'learning_rate': [0.03, 0.05, 0.1],  # 增加學習率，使模型更快學習
    'scale_pos_weight': [20, 30, 50],    # 提高少數類別的權重
    'subsample': [0.7, 0.8, 0.9],        # 減少過擬合
    'colsample_bytree': [0.7, 0.8]       # 控制每棵樹使用的特徵比例
    }
    
    # 隨機搜索調參
    xgb_model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        scoring='recall',
        cv=3,
        n_iter=50,
        verbose=2,
        n_jobs=-1,
        random_state=random_state
    )
    
    print("開始參數調整...")
    random_search.fit(X_train_resampled, y_train_resampled)
    
    best_model = random_search.best_estimator_
    print(f"最佳參數：{random_search.best_params_}")
    
    # 測試集評估
    y_prob = best_model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # 找到最佳閾值
    f1_scores = 2 * (precision * recall) / (precision + recall)
    best_threshold = thresholds[f1_scores.argmax()]
    print(f"最佳閾值：{best_threshold}")
    
    # 使用最佳閾值進行分類
    #y_pred = (y_prob >= best_threshold).astype(int)
    y_pred = (y_prob >= 0.9).astype(int)  # 預設閥值 0.1
    
    print("分類報告：\n", classification_report(y_test, y_pred))
    print("混淆矩陣：\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC 分數：{roc_auc_score(y_test, y_prob)}")

        # 調整不同閥值
    thresholds = np.arange(0.1, 0.9, 0.1)  # 例如從 0.1 到 0.9 閥值
    evaluate_threshold(y_test, y_prob, thresholds)
    
    # 保存測試結果
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted_Prob': y_prob, 'Predicted_Label': y_pred})
    results_df.to_excel('/Users/zhengqunyao/test_results.xlsx', index=False)
    print("測試集預測結果已保存！")
    
    return best_model, best_threshold

# 繪製特徵重要性
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
    # 定義檔案路徑與參數
    file_path = "/Users/zhengqunyao/machine_learning_v13.xlsx"
    features = ["Education", "Employment", "Marital", "CompanyRelationship", "Industry", 
                "Job", "Type", "ApprovalResult", "Years", "Age"]
    #, "Income", "LoanIncomeRatio", "Adjust"
    target = "Flag"
    
    # 資料讀取與處理
    X, y = load_and_preprocess_data(file_path, features, target)
    
    # 模型訓練與最佳閾值選擇
    best_model, best_threshold = train_and_optimize_xgboost(X, y)
    
    # 特徵重要性
    plot_feature_importances(best_model, features, output_path="/Users/zhengqunyao/feature_importances_xgboost_optimized251.png")
    
    # 儲存模型
    model_path = "/Users/zhengqunyao/loan_prediction_xgboost_optimized251.pkl"
    joblib.dump(best_model, model_path)
    print(f"模型已儲存至：{model_path}")