import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 讀取檔案
file_path = "/Users/zhengqunyao/predicted.xlsx"  # 修改為你的檔案路徑
data = pd.read_excel(file_path)

# 假設 'Flag' 是真實值，'預測類別' 是模型的預測結果
y_true = data['實際類別']  # 真實值
y_pred = data['預測類別']  # 預測結果

# 計算混淆矩陣
conf_matrix = confusion_matrix(y_true, y_pred)

# 確保混淆矩陣為 2x2 並展開為 TN, FP, FN, TP
if conf_matrix.shape == (2, 2):
    TN, FP, FN, TP = conf_matrix.ravel()
else:
    raise ValueError("混淆矩陣形狀不是 2x2，請檢查輸入數據！")

# (1) 計算總體精確率 (Overall Accuracy)
overall_accuracy = accuracy_score(y_true, y_pred)

# (2) 類別 1 的精確率與召回率
precision_1 = precision_score(y_true, y_pred, pos_label=1)
recall_1 = recall_score(y_true, y_pred, pos_label=1)

# (3) 類別 0 的精確率與召回率
precision_0 = precision_score(y_true, y_pred, pos_label=0)
recall_0 = recall_score(y_true, y_pred, pos_label=0)

# 輸出結果
print(f"總體精確率 (Overall Accuracy)：{overall_accuracy:.4f}")
print(f"1 精確率 (Precision for class 1)：{precision_1:.4f}")
print(f"1 召回率 (Recall for class 1)：{recall_1:.4f}")
print(f"0 精確率 (Precision for class 0)：{precision_0:.4f}")
print(f"0 召回率 (Recall for class 0)：{recall_0:.4f}")
print("\n混淆矩陣結果：")
print(f"TP (True Positive)：{TP}")
print(f"TN (True Negative)：{TN}")
print(f"FP (False Positive)：{FP}")
print(f"FN (False Negative)：{FN}")