"""
基于rerank模型计算句子的相关度
"""

from BCEmbedding.BCEmbedding import RerankerModel

model = RerankerModel("D:\\ai\\modelscope_models\\maidalun\\bce-reranker-base_v1")

query = "小猫在屋顶上，眺望深蓝色的天空"
docs = [
    "小猫在屋顶上",
    "我和小猫是好朋友"
]

# 构造语句对
pairs = [(query, doc) for doc in docs]

# 计算相似度得分
scores = model.compute_score(pairs)
print(scores)

# 重排序
rerank_result = model.rerank(query, docs)
print(rerank_result)