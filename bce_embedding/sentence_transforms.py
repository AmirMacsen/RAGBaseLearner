"""
使用sentence transforms
"""

from sentence_transformers import SentenceTransformer,CrossEncoder

model = SentenceTransformer('D:/ai/modelscope_models/maidalun/bce-embedding-base_v1')
passages = [
    "小猫站在屋顶上",
    "河流滚滚向前，奔赴东海"
]

embeddings = model.encode(passages, normalize_embeddings= True)
print(embeddings.shape)

query = "小猫在哪里"
pairs = [(query, doc) for doc in passages]

cross_encoder = CrossEncoder('D:/ai/modelscope_models/maidalun/bce-reranker-base_v1')
scores = cross_encoder.predict(pairs)
print(scores)
