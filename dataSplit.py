import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据集
data = pd.read_csv(r'dataset\mydataset\final.csv')

# 将数据集划分为训练集（train）、验证集（val）和测试集（test）
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)  # 70% 训练集，30% 剩余数据
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 15% 验证集，15% 测试集

# 保存分割后的数据集
train_data.to_csv(r'dataset\mydataset\train.csv', index=False)
val_data.to_csv(r'dataset\mydataset\val.csv', index=False)
test_data.to_csv(r'dataset\mydataset\test.csv', index=False)

print("数据集划分完成！")
