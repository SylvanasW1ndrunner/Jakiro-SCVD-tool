import os

import pandas as pd

delegate_code = r"O:\SmartContractDetect\dataset\mydataset\delegatecall\sourcecode"
integer_code = r"O:\SmartContractDetect\dataset\mydataset\Integeroverflow\sourcecode"
reentrancy_code = r"O:\SmartContractDetect\dataset\mydataset\reentrancy\sourcecode"
timestamp_code = r"O:\SmartContractDetect\dataset\mydataset\timestamp\sourcecode"

df = pd.DataFrame(columns=["code", "label"])


def csvGen():
    index = 0
    with open(r"O:\SmartContractDetect\dataset\mydataset\delegatecall\final_delegatecall_label.txt", "r") as f:
        with open(r"O:\SmartContractDetect\dataset\mydataset\delegatecall\final_delegatecall_name.txt", "r") as f2:
            for line in f:
                line = line.strip()
                line2 = f2.readline().strip()
                if line == "0":
                    label = 0;
                else:
                    label = 1;
                codepath = r"O:/SmartContractDetect/dataset/mydataset/delegatecall/sourcecode/" + line2
                code = open(codepath, "r").read()
                df.loc[index] = [code, label]
                index += 1

    with open(r"O:\SmartContractDetect\dataset\mydataset\Integeroverflow\final_integeroverflow_label.txt", "r") as f:
        with open(r"O:\SmartContractDetect\dataset\mydataset\Integeroverflow\final_integeroverflow_name.txt",
                  "r") as f2:
            for line in f:
                line = line.strip()
                line2 = f2.readline().strip()
                if line == "0":
                    label = 0;
                else:
                    label = 2;
                codepath = r"O:/SmartContractDetect/dataset/mydataset/Integeroverflow/sourcecode/" + line2
                code = open(codepath, "r").read()
                df.loc[index] = [code, label]
                index += 1

    with open(r"O:\SmartContractDetect\dataset\mydataset\reentrancy\final_reentrancy_label.txt", "r") as f:
        with open(r"O:\SmartContractDetect\dataset\mydataset\reentrancy\final_reentrancy_name.txt", "r") as f2:
            for line in f:
                line = line.strip()
                line2 = f2.readline().strip()
                if line == "0":
                    label = 0;
                else:
                    label = 3;
                codepath = r"O:/SmartContractDetect/dataset/mydataset/reentrancy/sourcecode/" + line2
                code = open(codepath, "r").read()
                df.loc[index] = [code, label]
                index += 1

    with open(r"O:\SmartContractDetect\dataset\mydataset\timestamp\final_timestamp_label.txt", "r") as f:
        with open(r"O:\SmartContractDetect\dataset\mydataset\timestamp\final_timestamp_name.txt", "r") as f2:
            for line in f:
                line = line.strip()
                line2 = f2.readline().strip()
                if line == "0":
                    label = 0;
                else:
                    label = 4;
                codepath = r"O:/SmartContractDetect/dataset/mydataset/timestamp/sourcecode/" + line2
                code = open(codepath, "r").read()
                df.loc[index] = [code, label]
                index += 1

    df.to_csv(r"O:\SmartContractDetect\dataset\mydataset\final.csv", index=False)


import re


def preprocess_solidity_code(code):
    # 去掉注释
    code = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.MULTILINE | re.DOTALL)

    # 去掉字符串字面量（可选，视具体需求而定）
    code = re.sub(r'"(.*?)"|\'(.*?)\'', '', code)

    # 去掉多余的空白字符
    code = re.sub(r'\s+', ' ', code).strip()

    return code


