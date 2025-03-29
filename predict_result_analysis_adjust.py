import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 讀取資料
file_path = "/Users/zhengqunyao/test_data44_light_normal.xlsx"  # 修改為你的檔案路徑
data = pd.read_excel(file_path)

# 假設 '實際類別' 是真實值，'增貸概率' 是模型預測概率
y_true = data['Flag']  # 真實值
y_prob = data['增貸概率']  # 模型預測概率

# 將增貸概率 >= 0.2 視為 1，否則為 0
threshold = 0.33
y_pred_adjusted = (y_prob >= threshold).astype(int)

# 計算精確率、召回率和混淆矩陣
accuracy = accuracy_score(y_true, y_pred_adjusted)
precision_1 = precision_score(y_true, y_pred_adjusted, pos_label=1)
recall_1 = recall_score(y_true, y_pred_adjusted, pos_label=1)
precision_0 = precision_score(y_true, y_pred_adjusted, pos_label=0)
recall_0 = recall_score(y_true, y_pred_adjusted, pos_label=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_adjusted).ravel()

# 輸出結果
print(f"使用閥值 (threshold)：{threshold:.4f}")
print(f"總體精確率 (Overall Accuracy)：{accuracy:.4f}")
print(f"1 精確率 (Precision for class 1)：{precision_1:.4f}")
print(f"1 召回率 (Recall for class 1)：{recall_1:.4f}")
print(f"0 精確率 (Precision for class 0)：{precision_0:.4f}")
print(f"0 召回率 (Recall for class 0)：{recall_0:.4f}")
print("\n混淆矩陣結果：")
print(f"TP (True Positive)：{tp}")
print(f"TN (True Negative)：{tn}")
print(f"FP (False Positive)：{fp}")
print(f"FN (False Negative)：{fn}")