import scipy.io as scio
from pgmpy.readwrite import BIFReader
import pandas as pd
from scipy.io import savemat

# 读取 BIF 文件
reader = BIFReader("asia.bif")  # 替换为你的文件
model = reader.get_model()

# 获取所有变量（节点）
nodes = model.nodes()
print("Nodes:", nodes)

# 获取所有边（因果关系）
edges = model.edges()
print("Edges:", edges)

# 生成 DataFrame (可选，若想存储网络结构)
df_edges = pd.DataFrame(edges, columns=["Parent", "Child"])
print(df_edges)
from pgmpy.sampling import BayesianModelSampling

# 生成样本数据（采样 1000 行）
inference = BayesianModelSampling(model)
sampled_data = inference.forward_sample(size=1000)

# 转换为 DataFrame
df = pd.DataFrame(sampled_data)

file_name = r"dataset/asia1000.csv"
# scio.savemat(file_name, {'data': df.values})


# 将数据保存为 .mat 文件
import pandas as pd

df.to_csv(file_name, index=False)

# 加载 CSV
dfnew = pd.read_csv(file_name)
print(dfnew)