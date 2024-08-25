from web3 import Web3
import json

# 设置以太坊节点的 URL
#ethereum_node_url = "https://mainnet.infura.io/v3/16beff21e2bb4871b0b95fd5e2908cfa"
ethereum_node_url = "https://mainnet.infura.io/v3/70405274746d49fbb3e2f8860fddd881"
web3 = Web3(Web3.HTTPProvider(ethereum_node_url))

# Uniswap V2 工厂合约地址
uniswap_factory_address = "0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f"

# 加载 Uniswap V2 工厂合约 ABI
# 注意：你需要替换这里的 abi_path 为你保存 Uniswap V2 工厂合约 ABI 的文件路径
abi_path = "abi.json"
with open(abi_path) as f:
    uniswap_factory_abi = json.load(f)

# 创建 Uniswap V2 工厂合约对象
uniswap_factory_contract = web3.eth.contract(address=uniswap_factory_address, abi=uniswap_factory_abi)

# 获取所有 Token 对
all_pairs_length = uniswap_factory_contract.functions.allPairsLength().call()
all_pairs = [uniswap_factory_contract.functions.allPairs(i).call() for i in range(all_pairs_length)]

# 获取 Token 合约地址
token_addresses = []
for pair_address in all_pairs:
    pair_contract = web3.eth.contract(address=pair_address, abi=uniswap_pair_abi)
    token0_address = pair_contract.functions.token0().call()
    token1_address = pair_contract.functions.token1().call()
    token_addresses.extend([token0_address, token1_address])

# 去重
unique_token_addresses = list(set(token_addresses))

print("Token 合约地址列表:")
for token_address in unique_token_addresses:
    print(token_address)
