"""
使用bce-embedding计算文档向量
"""

from BCEmbedding.BCEmbedding import EmbeddingModel

model = EmbeddingModel(model_name_or_path="D:\\ai\\modelscope_models\\maidalun\\bce-embedding-base_v1")


sentences = ["今天天气不错", "待会儿一起散步"]

embeddings = model.encode(sentences)
print(embeddings.shape)

