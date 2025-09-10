import math
from collections import Counter


### 计算逆文档频率
def compute_idf(doc_list):
    idf_score = {}
    total_len = len(doc_list)

    doc_freq = Counter([word for doc in doc_list for word in set(doc)])
    for word, freq in doc_freq.items():
        idf_score[word] = math.log(total_len / (1 + freq))
    return idf_score

### 计算bm25_score
def compute_bm25(doc, query, idf_score, avgdl, k1=1.5, b=0.75):
    """
    计算文档相对于一个查询的BM25得分
    :param doc: 文档
    :param query: 查询
    :param idf_score: 计算出来的idf得分
    :param avgdl: 文档集合中文档的平得分
    :param k1:
    :param b:
    :return:
    """
    score = 0.0
    doc_len = len(doc)
    doc_freqs = Counter(doc)
    for word in query:
        if word not in doc_freqs:
            continue
        score += idf_score[word] * doc_freqs[word] * (k1 + 1) / (doc_freqs[word] + k1 * (1 - b + b * doc_len / avgdl))

    return score


def compute_avgdl(doc_list):
    return sum(len(doc) for doc in doc_list) / len(doc_list)

doc_list = [
    ["小猫", "在", "屋顶", "上"],
    ["小狗", "和", "小狗", "是", "好朋友"],
    ["我", "喜欢", "读书"]
]

idf_score = compute_idf(doc_list)
avgdl = compute_avgdl(doc_list)

score = []

query = ["小猫", "在", "哪里"]
for doc in doc_list:
    score.append(compute_bm25(doc, query, idf_score, avgdl))

print(score)
