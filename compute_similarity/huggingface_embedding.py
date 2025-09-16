import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("D:\\ai\\modelscope_models\\BAAI\\bge-large-zh-v1___5")
model = AutoModel.from_pretrained("D:\\ai\\modelscope_models\\BAAI\\bge-large-zh-v1___5")

# 设置模型为评估状态,非训练模型
model.eval()


documents = [
    "深度学习技术再计算机视觉领域非常重要",
    "使用深度学习模型可以理解文档的深层语义",
    "密集检索的优势是通过学习文档和查询的表示来提高检索的准确性"
]

query = "密集检索的优势"

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取最后一层隐藏层的数据
    # 这里没有使用平均池化，因为效果不是很好
    # 使用CLS令牌的嵌入作为句子嵌入的表示，CLS令牌位于每个序列的开头，经常用于句子级任务的表示
    embeddings = outputs[0][:,0]
    # 对句子嵌入进行L2标准化
    normalize_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalize_embeddings

query_embedding = get_embedding(query)
print(query_embedding.shape)

# 计算文档
document_embeddings = torch.stack([get_embedding(doc) for doc in documents]).squeeze()
# 如果不压缩，会多出一个维度torch.Size([3, 1, 1024])
# document_embeddings = torch.stack([get_embedding(doc) for doc in documents])
print(document_embeddings.shape)

# 计算相似性
similarity = torch.matmul(query_embedding, document_embeddings.T)
print(similarity)

# 也可以使用cosine相似度
similarity = torch.nn.functional.cosine_similarity(query_embedding, document_embeddings, dim=1)
print(similarity)

# 或者使用scikit-learn中的函数
similarity = cosine_similarity(query_embedding, document_embeddings)
print(similarity)
