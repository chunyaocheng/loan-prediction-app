import pandas as pd
import random
import string

# 讀取 Excel 檔案
file_path = "/Users/zhengqunyao/machine_learning_with_flag.xlsx"
data = pd.read_excel(file_path)

# 建立唯一的隨機統編生成函數
def generate_unique_id(existing_ids, length=10):
    while True:
        new_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
        if new_id not in existing_ids:  # 確保唯一性
            existing_ids.add(new_id)
            return new_id

# 創建映射表，將舊統編對應到新的唯一值
existing_ids = set()
id_mapping = {}

for customer_id in data['客戶統編'].unique():
    id_mapping[customer_id] = generate_unique_id(existing_ids)

# 替換客戶統編為新的唯一值
data['客戶統編'] = data['客戶統編'].map(id_mapping)

# 儲存新的 Excel 檔案
output_path = "/Users/zhengqunyao/machine_learning_with_mapped_ids.xlsx"
data.to_excel(output_path, index=False)

# 輸出映射表到 CSV 文件
mapping_path = "/Users/zhengqunyao/id_mapping.csv"
pd.DataFrame(list(id_mapping.items()), columns=['原統編', '新統編']).to_csv(mapping_path, index=False)

print(f"已完成客戶統編替換，結果儲存至：{output_path}")
print(f"統編映射表已儲存至：{mapping_path}")