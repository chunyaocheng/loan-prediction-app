import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

# 模型訓練與超參數調整
def train_xgboost_with_tuning(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")
    
    # SMOTE 過抽樣
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"過抽樣後訓練集大小：{X_train.shape}, {y_train.shape}")
    
    # 定義參數網格
    param_grid = {

    #'n_estimators': [100, 300],
    #'max_depth': [5, 10],
    #'learning_rate': [0.05],
    #'scale_pos_weight': [10, 15],
    #'subsample': [0.8],
    #'colsample_bytree': [0.8]

    # 'n_estimators': [100, 300],
    # 'max_depth': [5, 10],
    # 'learning_rate': [0.05],
    # 'scale_pos_weight':  [20, 30, 50],
    # 'subsample': [0.8],
    #'colsample_bytree': [0.8]

    'n_estimators': [100, 300],
    'max_depth': [5, 7],
    'learning_rate': [0.05],
    'scale_pos_weight': [5, 10, 15],  # 減小類別 1 的權重
    'gamma': [0, 0.1, 0.3],  # 增加正則化
    'min_child_weight': [1, 3, 5],  # 控制葉節點最小權重
    'subsample': [0.8],
    'colsample_bytree': [0.8]
    }
    
    # XGBoost 模型
    xgb_model = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='precision',
        cv=3,
        verbose=2,
        n_jobs=-1
    )
    
    print("開始參數調整...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"最佳參數：{grid_search.best_params_}")
    
    # 測試集評估
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    print("分類報告：\n", classification_report(y_test, y_pred))
    print("混淆矩陣：\n", confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC 分數：{roc_auc_score(y_test, y_prob)}")
    
    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title("confusion matrix")
    plt.xlabel("predict")
    plt.ylabel("actual")
    plt.show()
    
    return best_model

# 繪製特徵重要性
def plot_feature_importances(model, features, output_path=None):
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title("attr mportanc")
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
    
    # 特徵重要性
    plot_feature_importances(best_model, features, output_path="/Users/zhengqunyao/feature_importances_xgboost253.png")
    
    # 儲存模型
    model_path = "/Users/zhengqunyao/loan_prediction_xgboost253.pkl"
    joblib.dump(best_model, model_path)
    print(f"模型已儲存至：{model_path}")