import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 讀取資料檔案
file_path = "/Users/zhengqunyao/machine_learning_cleaned.xlsx"
data = pd.read_excel(file_path)

# 檢查重要數值型特徵的分布變化（如 申貸性質代碼）
plt.figure(figsize=(10, 5))
sns.histplot(data['申貸性質代碼'], kde=True, bins=20)
plt.title('填補後的申貸性質代碼分布')
plt.show()

# 類別型資料 One-Hot Encoding（如 行業別代碼 和 職稱代碼）
encoded_data = pd.get_dummies(data, columns=['行業別代碼', '職稱代碼'])

# 儲存清理後和處理的資料
encoded_data.to_excel("/Users/zhengqunyao/machine_learning_encoded.xlsx", index=False)