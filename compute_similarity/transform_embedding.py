################
# 1. 一般情况，用户输入和文档都比较短
################

## 使用显卡

from sentence_transformers import SentenceTransformer

sentence1 = ["周末我准备去海边度假，享受阳光和沙滩", "研究表明，适量运动有助于提升工作效率和创造力"]
sentence2 = ["我期待的假期是在沙滩上，听着海浪声放松", "科技公司近期发布了一款新的智能手机，引起了广泛关注"]

### 北京智远的模型
model = SentenceTransformer("D:\\ai\\modelscope_models\\BAAI\\bge-large-zh-v1___5")
embedding1 = model.encode(sentence1, normalize_embeddings=True)
print(embedding1.shape)
embedding2 = model.encode(sentence2, normalize_embeddings=True)
print(embedding2.shape)


# 计算相似性 ，通过矩阵的点积运算
"""
[[0.7168557  0.27448064]
 [0.3281418  0.3274266 ]]
 sentence1[0] 与 sentence2中的每一句话计算相似性 得到  [0.7168557  0.27448064]
 sentence1[1] 与 sentence2中的每一句话计算相似性 得到  [0.3281418  0.3274266 ]
"""
similarity = embedding1 @ embedding2.T
print(similarity)



################
# 2. query比较短，文档比较长
################

queries = ["最新的AI研究成果", "健康饮食的重要性"]
docs = ["根据最新研究，AI模型已开始应用到各种领域，如医疗、金融、交通等。",
        "健康饮食是指通过balanced diet（均衡饮食）来提高身体健康，从而 prevent disease（预防疾病）。"]


#### 添加一个指令，弥补短文档的不足
instruction = "为这个文本生成用于检索的向量："
q_embedding = model.encode([instruction + query for query in queries], normalize_embeddings=True)

# 文档向量不需要添加指令
d_embedding = model.encode(docs, normalize_embeddings=True)

similarity = q_embedding @ d_embedding.T
print(similarity)