from web3 import Web3
chainApi = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/16beff21e2bb4871b0b95fd5e2908cfa'))

# 获取最新区块数据
block = chainApi.eth.get_block("latest")
print(block)
