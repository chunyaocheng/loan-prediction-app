import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, average_precision_score,
                             precision_recall_curve)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
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
def train_rf_with_tuning(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")

    # 類別不平衡處理
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"過抽樣後訓練集大小：{X_train.shape}")

    # 參數網格
    param_grid = {
        'n_estimators': [100, 200],  # 樹的數量
        'max_depth': [5, 10],  # 樹的最大深度
        'min_samples_leaf': [20, 50],  # 葉子節點最少樣本數
        'class_weight': [None, 'balanced'],  # 類別權重
        'max_features': ['sqrt', 'log2']  # 分割時考慮的最大特徵數
    }

    # Grid Search
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
    best_model = grid_search.best_estimator_
    print(f"最佳參數：{grid_search.best_params_}")

    # 預測
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 找最佳閾值
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # 防止除以0
    best_threshold = thresholds[f1_scores.argmax()]
    print(f"最佳 F1 閾值：{best_threshold}")

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

    return best_model, y_test, y_pred, y_prob, best_threshold

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
    model, y_test, y_pred, y_prob, threshold = train_rf_with_tuning(X, y)

    # 特徵重要性圖
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_feature_importances(model, features, output_path=f"/Users/zhengqunyao/rf_importance_{timestamp}.png")

    # 儲存模型
    model_path = f"/Users/zhengqunyao/rf_model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"模型已儲存至：{model_path}")