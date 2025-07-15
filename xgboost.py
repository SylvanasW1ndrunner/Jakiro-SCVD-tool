from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BitsAndBytesConfig
from transformers import LlamaForCausalLM,LlamaTokenizer
import torch
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import StandardScaler
import scipy
import json
from sklearn.preprocessing import normalize
import lightgbm as lgb
from SmartContractDetect.DataLoader import CustomDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def bertembedding(texts, bert_model,tokenizer):
    inputs = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(**inputs)

    text_vectors = outputs.last_hidden_state[:, 0, :].numpy()
    print(text_vectors)
    # kernel,bias = compute_kernel_bias(text_vectors)
    # text_vectors = transform_and_normalize(text_vectors, kernel, bias)
    normalized_vectors = normalize(text_vectors, norm='l2')
    return text_vectors

def compute_kernel_bias(vecs, n_components=1024):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    scaler = StandardScaler()
    vecs = scaler.fit_transform(vecs)
    cov = np.cov(vecs.T)
    u, s, vh = scipy.linalg.svd(cov, full_matrices=False)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :n_components], -mu

def transform_and_normalize(vecs, kernel=None, bias=None):
    """ 最终向量标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def llmembedding(texts,model,tokenizer):
    inputs = tokenizer(texts, return_tensors='pt',padding=True, truncation=True)
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    text_vectors = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
    # print(text_vectors)
    kernel,bias = compute_kernel_bias(text_vectors)
    text_vectors = transform_and_normalize(text_vectors, kernel, bias)
    text_vectors = normalize(text_vectors, norm='l2')
    # print(text_vectors)
    # kernel,bias = compute_kernel_bias(text_vectors)
    # text_vectors = transform_and_normalize(text_vectors, kernel, bias)
    del inputs,outputs
    return text_vectors

def similiarityCal():
    ### code_path = "/kaggle/input/codejson/reWriitten.json"
    code_path = "/kaggle/input/codejson3/reWritten3.json"

    with open(code_path, "r") as file:
        data = json.load(file)

    rewrite_code = [entry['rewritten'] for entry in data.values()]

    origin_code ="library SafeMath { function sub(uint256 a, uint256 b) internal pure returns (uint256) { assert(b <= a); return a - b; } } contract BountyHunt { using SafeMath for uint; mapping(address => uint) public bountyAmount; uint public totalBountyAmount; function claimBounty() { uint balance = bountyAmount[msg.sender]; if (msg.sender.call.value(balance)()) { totalBountyAmount = totalBountyAmount.sub(balance); bountyAmount[msg.sender] = 0; } } }"
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    llm_model = LlamaForCausalLM.from_pretrained('/kaggle/input/llama2-7b-hf/Llama2-7b-hf')
    llm_tokenizer = LlamaTokenizer.from_pretrained('/kaggle/input/llama2-7b-hf/Llama2-7b-hf')
    origin_bert_code = bertembedding(origin_code,bert_model,bert_tokenizer)
    origin_llm_code = llmembedding(origin_code,llm_model,llm_tokenizer)

    cosine_similarity_bert = 0
    cosine_similarity_llm = 0

    for code in rewrite_code:
        tmp1 = bertembedding(code,bert_model,bert_tokenizer)
        tmp2 = llmembedding(code,llm_model,llm_tokenizer)

        # 将向量转换为一维
        flattened_origin_bert_code = origin_bert_code.flatten()
        flattened_tmp1 = tmp1.flatten()

        flattened_origin_llm_code = origin_llm_code.flatten()
        flattened_tmp2 = tmp2.flatten()

        # 计算点积
        dot_product1 = np.dot(flattened_origin_bert_code, flattened_tmp1)
        dot_product2 = np.dot(flattened_origin_llm_code, flattened_tmp2)

        # 计算余弦相似度
        cosine_similarity_bert += dot_product1 / (np.linalg.norm(flattened_tmp1) * np.linalg.norm(flattened_origin_bert_code))
        cosine_similarity_llm += dot_product2 / (np.linalg.norm(flattened_tmp2) * np.linalg.norm(flattened_origin_llm_code))



    print("average cosine_similarity for bert is :",cosine_similarity_bert/10)
    print("average cosine_similarity for llm is :",cosine_similarity_llm/10)

# similiarityCal()
DatasetA_path = r"dataset\mydataset"
dataFlag = 'A'
if dataFlag == 'A':
    train_path = DatasetA_path + r"\train.csv"
    test_path = DatasetA_path + r"\test.csv"
    val_path = DatasetA_path + r"\val.csv"

# bert_model = BertModel.from_pretrained('bert-base-uncased')
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda:0")
model_path = r'/SmartContractDetect/finaltemp/png/llama7B-hf'
llm_model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto',torch_dtype=torch.float16)
llm_tokenizer = LlamaTokenizer.from_pretrained(model_path)

# 检查是否已经有 pad_token，没有的话添加
llm_tokenizer.pad_token = llm_tokenizer.eos_token


    # 确保模型词汇表的大小和分词器一
# 如果模型没有 pad_token，添加一个新的 pad_token
print("加载完毕！")

TrainList = []
TrainLabelList = []
TestList = []
TestLabelList = []
ValList = []
ValLabelList = []

trainDataset = CustomDataset(train_path, llm_tokenizer)
testDataset = CustomDataset(test_path, llm_tokenizer)
valDataset  = CustomDataset(val_path, llm_tokenizer)

trainDataLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
testDataLoader = DataLoader(testDataset, batch_size=32, shuffle=True)
valDataLoader = DataLoader(valDataset, batch_size=32,shuffle=True)

for batch in tqdm(trainDataLoader, desc='Processing  Embeddings'):
    # print("this is : ", batch)
    texts = batch['text']
    labels = batch['label']
    print(texts)
    #    text_vectors = bertembedding(texts, bert_model,bert_tokenizer)
    text_vectors = llmembedding(texts, llm_model,llm_tokenizer)
    #     kernel, bias = compute_kernel_bias(text_vectors)
    #     text_vectors = transform_and_normalize(text_vectors, kernel, bias)
    TrainList.append(text_vectors)
    TrainLabelList.append(labels)

TrainArray = np.concatenate(TrainList, axis=0)
print(TrainArray.shape)
TrainLabelArray = np.concatenate(TrainLabelList, axis=0).tolist()
print(TrainLabelArray)
#print(TrainLabelArray.shape)
##TrainLabelArray = TrainLabelList


for batch in tqdm(testDataLoader, desc='Processing test  Embeddings'):
    texts = batch['text']
    labels = batch['label']

    ##text_vectors = bertembedding(texts, bert_model,bert_tokenizer)
    text_vectors = llmembedding(texts, llm_model,llm_tokenizer)
    #     kernel,bias = compute_kernel_bias(text_vectors)
    #     text_vectors = transform_and_normalize(text_vectors, kernel, bias)
    #     Testid.append(id)
    TestList.append(text_vectors)
    TestLabelList.append(labels)

TestArray = np.concatenate(TestList, axis=0)
TestLabelArray = np.concatenate(TestLabelList, axis=0).tolist()

for batch in tqdm(valDataLoader, desc='Processing val  Embeddings'):
    texts = batch['text']
    labels = batch['label']

    #     text_vectors = bertembedding(texts, bert_model,bert_tokenizer)
    text_vectors = llmembedding(texts, llm_model,llm_tokenizer)
    #     kernel,bias = compute_kernel_bias(text_vectors)
    #     text_vectors = transform_and_normalize(text_vectors, kernel, bias)
    ValList.append(text_vectors)
    ValLabelList.append(labels)
ValArray = np.concatenate(ValList,axis = 0)
ValLabelArray = np.concatenate(ValLabelList, axis=0).tolist()

from sklearn.metrics import classification_report

def train():
    # 使用你手动构建的训练和验证集
    X_train = TrainArray  # 你构建的训练数据
    y_train = TrainLabelArray  # 你构建的训练标签
    X_valid = TestArray  # 你构建的验证数据
    y_valid = TestLabelArray  # 你构建的验证标签

    # class_counts = Counter(y_train)
    # total_samples = len(y_train)
    # class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    # sample_weights = np.array([class_weights[label] for label in y_train])

    trainData = lgb.Dataset(X_train, label=y_train)
    validData = lgb.Dataset(X_valid, label=y_valid)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 5,
        'learning_rate': 0.01,  # 降低学习率
        'num_leaves': 20,  # 降低num_leaves，避免过拟合
        'max_depth': 5,  # 限制树的深度
        'feature_fraction': 0.9,  # 使用所有特征
        'bagging_fraction': 0.9,  # 增加bagging_fraction
        'bagging_freq': 5,
        'verbose': 0,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }





    num_round = 10
    with tqdm(total=num_round, desc='Training LightGBM') as pbar:
        def progress_callback(env):
            pbar.update(1)

        gbm = lgb.train(params, trainData, num_boost_round=num_round, valid_sets=[validData],
                        callbacks=[progress_callback])

    gbm.save_model('model.txt')





def predict():
    gbm = lgb.Booster(model_file='model.txt')
    y_pred = gbm.predict(TestArray)
    y_pred = np.argmax(y_pred, axis=1)
    result_df = testDataset.data.copy()
    result_df['pre_label'] = y_pred
    result_df.to_csv('/kaggle/working/TestResult.csv', index=False)
    y_true = testDataset.data['label']

    report = classification_report(y_true, y_pred)

    with open('classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)


def validate(model_path):
    gbm = lgb.Booster(model_file=model_path)
    y_val = gbm.predict(ValArray)
    y_pred = np.argmax(y_val, axis=1)
    y_true = valDataset.data['label']  # 真实标签
    report = classification_report(y_true, y_pred)

    # 将报告写入文本文件
    with open('val_report2.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    print("分类报告已导出至 'val_report2.txt'")

if __name__ == '__main__':
    train()
    validate("model.txt")