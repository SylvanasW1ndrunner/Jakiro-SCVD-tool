import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 输入文件路径
input_files = {
    'RE': 're_data.csv',
    'IO': 'io_data.csv',
    'TO': 'to_data.csv',
    'SAFE': 'safe_data.csv'
}

# 输出目录
output_train_dir = 'train'
output_test_dir = 'test'

# 创建目录
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# 处理 RE, IO, TO 文件
for label, input_file in input_files.items():
    data = pd.read_csv(input_file)

    if label == 'SAFE':
        # SAFE 数据随机选取 2w 行并划分 4:1
        sampled_data = data.sample(n=20000, random_state=42)
        train, test = train_test_split(sampled_data, test_size=0.2, random_state=42)
    else:
        # RE, IO, TO 数据按 80% 训练集和 20% 测试集划分
        train, test = train_test_split(data, test_size=0.2, random_state=42)

    # 保存到对应目录
    train.to_csv(os.path.join(output_train_dir, f'{label}_train.csv'), index=False)
    test.to_csv(os.path.join(output_test_dir, f'{label}_test.csv'), index=False)

print(f"数据处理完成！生成的训练和测试数据已保存到以下目录：\n- {output_train_dir}\n- {output_test_dir}")
