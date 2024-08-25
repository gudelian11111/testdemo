from web3 import Web3, HTTPProvider
#web3 = Web3(HTTPProvider('https://mainnet.infura.io/v3/d6243bb783b44485ad6636b6c3411377'))
#res = web3.isConnected()
#print(res)

from web3 import Web3

# Infura URL
infura_url = "https://mainnet.infura.io/v3/d6243bb783b44485ad6636b6c3411377"

# 创建 Web3 实例
web3 = Web3(Web3.HTTPProvider(infura_url))

# 检查连接状态
if web3.isConnected():
    print("Connected to Ethereum node")
else:
    print("Not connected to Ethereum node")


