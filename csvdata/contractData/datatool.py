import pandas as pd

RETraincsv = "train/RE_train.csv"
RETestcsv = "test/RE_test.csv"
SAFETraincsv = "train/SAFE_train.csv"
SAFETestcsv = "test/SAFE_test.csv"
IOTraincsv = "train/IO_train.csv"
IOTestcsv = "test/IO_test.csv"
TOTraincsv = "train/TO_train.csv"
TOTestcsv = "test/TO_test.csv"

RETrain = pd.read_csv(RETraincsv)
RETest = pd.read_csv(RETestcsv)
SAFETrain = pd.read_csv(SAFETraincsv)
SAFETest = pd.read_csv(SAFETestcsv)
IOTrain = pd.read_csv(IOTraincsv)
IOTTest = pd.read_csv(IOTestcsv)
TOTrain = pd.read_csv(TOTraincsv)
TOTTest = pd.read_csv(TOTestcsv)

# 函数：从 SAFETrain 中抽取指定数量的样本并修改标签为 0
def extract_and_modify_safe_data(sample_size):
    # 从 SAFETrain 中随机抽取样本
    samples = SAFETrain.sample(n=int(sample_size), random_state=42)
    samples['label'] = 0  # 修改标签为0
    return samples

# 对每个数据集进行处理
RETrain_samples = extract_and_modify_safe_data(len(RETrain) * 1.5)
RETest_samples = extract_and_modify_safe_data(len(RETest) * 1.5)
IOTrain_samples = extract_and_modify_safe_data(len(IOTrain) * 1.5)
IOTTest_samples = extract_and_modify_safe_data(len(IOTTest) * 1.5)
TOTrain_samples = extract_and_modify_safe_data(len(TOTrain) * 1.5)
TOTTest_samples = extract_and_modify_safe_data(len(TOTTest) * 1.5)

# 将 SAFE 的样本添加到相应的训练集和测试集
RETrain = pd.concat([RETrain, RETrain_samples], ignore_index=True)
RETest = pd.concat([RETest, RETest_samples], ignore_index=True)
IOTrain = pd.concat([IOTrain, IOTrain_samples], ignore_index=True)
IOTTest = pd.concat([IOTTest, IOTTest_samples], ignore_index=True)
TOTrain = pd.concat([TOTrain, TOTrain_samples], ignore_index=True)
TOTTest = pd.concat([TOTTest, TOTTest_samples], ignore_index=True)

# 将修改后的数据集保存回 CSV 文件
RETrain.to_csv("RE_train_files.csv", index=False)
RETest.to_csv("RE_test_files.csv", index=False)
IOTrain.to_csv("IO_train_files.csv", index=False)
IOTTest.to_csv("IO_test_files.csv", index=False)
TOTrain.to_csv("TO_train_files.csv", index=False)
TOTTest.to_csv("TO_test_files.csv", index=False)

print("数据已成功添加并保存。")