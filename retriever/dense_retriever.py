import torch
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "深度学习技术再计算机视觉领域非常重要",
    "使用深度学习模型可以理解文档的深层语义",
    "密集检索的优势是通过学习文档和查询的表示来提高检索的准确性"
]

query = "密集检索的优势"


from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained("D:\\ai\\modelscope_models\\dienstag\\chinese-macbert-base")
model = BertModel.from_pretrained("D:\\ai\\modelscope_models\\dienstag\\chinese-macbert-base")


def get_embedding(text):
    """
    通过分词模型分词，通过model获取向量表示

    为什么要对最后一层获取平均池化的概率：
    首先分词后会对每一个词进行向量化的表示，平均池化就是对每一个向量的对应维度的值相加再除以词的数量

    平均池化就是对一个句子中所有分词的向量进行了一种统一处理，最终考虑了所有词的信息，输出一个固定维度的向量。
    :param text:
    :return:
    """
    # 输入预处理 max_length 限制文档输入的最大长度，超出部分进行截断
    # padding 对短序列进行填充
    # truncation 对长序列进行截断
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # 获取模型的最后一层隐藏状态 shape: [batch_size, sequence_length, hidden_size]
    with torch.no_grad():
        outputs = model(**inputs)
    # 使用平均池化获取句子表示
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    return sentence_embedding

query_embedding = get_embedding(query)
print(query_embedding.shape)

# 需要压缩文档向量
dock_embedding = torch.stack([get_embedding(doc) for doc in documents]).squeeze()
print(dock_embedding.shape)


# 计算相似度
similarity = cosine_similarity(query_embedding, dock_embedding)
print(similarity)

# 获取最相似的文档
index = similarity.argmax()
print(documents[index])
