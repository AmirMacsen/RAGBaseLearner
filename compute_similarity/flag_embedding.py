################
# 1. 一般情况，用户输入和文档都比较短
################
import os

## 使用显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from FlagEmbedding import FlagModel

sentence1 = ["周末我准备去海边度假，享受阳光和沙滩", "研究表明，适量运动有助于提升工作效率和创造力"]
sentence2 = ["我期待的假期是在沙滩上，听着海浪声放松", "科技公司近期发布了一款新的智能手机，引起了广泛关注"]

### 北京智远的模型
model = FlagModel("D:\\ai\\modelscope_models\\BAAI\\bge-large-zh-v1___5", use_fp16= True)
embedding1 = model.encode(sentence1)
print(embedding1.shape)
embedding2 = model.encode(sentence2)
print(embedding2.shape)


# 计算相似性 ，通过矩阵的点积运算
"""
[[0.717  0.2744]
 [0.3281 0.3274]]
 sentence1[0] 与 sentence2中的每一句话计算相似性 得到  [0.717  0.2744]
 sentence1[1] 与 sentence2中的每一句话计算相似性 得到  [0.3281 0.3274]
"""
similarity = embedding1 @ embedding2.T
print(similarity)



################
# 2. query比较短，文档比较长
################

queries = ["最新的AI研究成果", "健康饮食的重要性"]
docs = ["根据最新研究，AI模型已开始应用到各种领域，如医疗、金融、交通等。",
        "健康饮食是指通过balanced diet（均衡饮食）来提高身体健康，从而 prevent disease（预防疾病）。"]


# encode_queries 为每个查询自动添加指令，从而优化查询的嵌入表示
q_embedding = model.encode_queries(queries)

# 文档向量不需要添加指令
d_embedding = model.encode(docs)

similarity = q_embedding @ d_embedding.T
print(similarity)