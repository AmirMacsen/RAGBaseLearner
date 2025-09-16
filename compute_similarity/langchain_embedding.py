import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "D:\\ai\\modelscope_models\\BAAI\\bge-large-zh-v1___5"

kwargs = {
    "device": "cuda"
}

encode_kwargs = {
    "normalize_embeddings": True
}

hf = HuggingFaceBgeEmbeddings(model_name=model_name,
                            model_kwargs=kwargs,
                            encode_kwargs=encode_kwargs,
                            query_instruction="为这个文本生成用于检索的向量：")

queries = ["最新的AI研究成果", "健康饮食的重要性"]
docs = ["根据最新研究，AI模型已开始应用到各种领域，如医疗、金融、交通等。",
        "健康饮食是指通过balanced diet（均衡饮食）来提高身体健康，从而 prevent disease（预防疾病）。"]

q_embedding = torch.stack([torch.tensor(hf.embed_query(query)) for query in queries])
print(q_embedding.shape)


d_embedding = torch.tensor(hf.embed_documents(docs))

similarity = q_embedding @ d_embedding.T
print(similarity)