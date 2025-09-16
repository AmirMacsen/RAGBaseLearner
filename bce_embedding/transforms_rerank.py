import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("D:\\ai\\modelscope_models\\maidalun\\bce-embedding-base_v1")
model = AutoModelForSequenceClassification.from_pretrained("D:\\ai\\modelscope_models\\maidalun\\bce-reranker-base_v1")
# 推理
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

passages = [
    "小猫站在屋顶上",
    "河流滚滚向前，奔赴东海"
]

# 分词
# encoded_inputs = tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=512)
# # 将输入数据移到与模型相同的设备上
# encode_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
# outputs = model(**encode_inputs, return_dict=True)
# embeddings = outputs.last_hidden_state[:, 0]  # cls
# print(embeddings.shape)
# # 归一化
# embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
# print(embeddings.shape)

# 接收用户查询
query = "小猫在哪里"
# pairs
pairs = [(query, passage) for passage in passages]

# tokenizer
encode_inputs = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True, max_length=512)
# 将输入数据移到与模型相同的设备上
encode_inputs = {k: v.to(device) for k, v in encode_inputs.items()}
# 使用模型进行前向传播的计算
outputs = model(**encode_inputs, return_dict=True).logits.view(-1, ).float()

# 归一化
scores = torch.sigmoid(outputs)
print( scores)



