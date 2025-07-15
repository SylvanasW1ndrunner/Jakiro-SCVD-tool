import os
import re
from solcx import compile_standard, install_solc

# 安装指定的 Solidity 编译器版本
install_solc('0.4.26')

# 设置文件夹路径
contracts_dir = r'finaldata/RE/sourcecode'
bytecode_dir = r'finaldata/RE/bytecode'
wronglist = []
# 确保 bytecode 文件夹存在
os.makedirs(bytecode_dir, exist_ok=True)
cnt = 0
# 遍历所有 .sol 文件
for filename in os.listdir(contracts_dir):
    if filename.endswith('.sol'):
        # 完整的文件路径
        contract_path = os.path.join(contracts_dir, filename)

        # 读取合约文件
        with open(contract_path , 'r', encoding='utf-8', errors='ignore') as f:
            contract_source = f.read()

        # 修改编译器版本，支持0.4.25及以上


        # 使用正则表达式去掉注释（包括中文注释）
        ##cleaned_source = re.sub(r'//.*?$|/\*.*?\*/', '', contract_source, flags=re.DOTALL | re.MULTILINE)

        # 将清理后的源代码写回文件
        ##print(contract_source)
        try:
            # 使用 solcx 编译合约并获取所有合约的字节码
            compiled_sol = compile_standard({
                "language": "Solidity",
                "sources": {
                    filename: {
                        "content": contract_source
                    }
                },
                "settings": {
                    "outputSelection": {
                        "*": {
                            "*": ["evm.bytecode.object"]  # 只选择字节码
                        }
                    }
                }
            }, solc_version='0.4.26')
            contracts = compiled_sol['contracts'][filename]
            for contract_name, contract_info in contracts.items():
                bytecode = contract_info['evm']['bytecode']['object']
                if len(bytecode) > 0:
                    path = os.path.join(bytecode_dir, f'{filename}_{contract_name}.txt')
                    with open(path, 'w') as f:
                        f.write(bytecode)
        except Exception as e:
            wronglist.append(filename)
            print(f'编译失败: {filename}，错误: {e}')
            cnt+=1
with open ('wronglist.txt', 'w') as f:
    for item in wronglist:
        f.write(item + '\n')
print(cnt)