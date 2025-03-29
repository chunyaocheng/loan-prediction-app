import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, average_precision_score,
                             precision_recall_curve)
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
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

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    print("資料處理完成，類別型欄位已編碼並標準化！")
    return X, y, label_encoders, scaler

# ===== 模型訓練與超參數調整 =====
def train_svm_with_tuning(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"訓練集大小：{X_train.shape}, 測試集大小：{X_test.shape}")

    # 類別不平衡處理
    smote = SMOTE(random_state=random_state)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"過抽樣後訓練集大小：{X_train.shape}")

    # 參數網格
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto'],
        'class_weight': [None, 'balanced']
    }

    # Grid Search
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
    best_model = grid_search.best_estimator_
    print(f"最佳參數：{grid_search.best_params_}")

    # 預測
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # 找最佳閾值
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
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
    X, y, label_encoders, scaler = load_and_preprocess_data(file_path, features, target)

    # 模型訓練
    model, y_test, y_pred, y_prob, threshold = train_svm_with_tuning(X, y)

    # 儲存模型
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"/Users/zhengqunyao/svm_model_{timestamp}.pkl"
    joblib.dump({'model': model, 'scaler': scaler, 'label_encoders': label_encoders}, model_path)
    print(f"模型已儲存至：{model_path}")