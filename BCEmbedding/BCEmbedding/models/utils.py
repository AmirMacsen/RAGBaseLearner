'''
@Description: 
@Author: shenlei
@Date: 2024-01-15 13:06:56
@LastEditTime: 2024-04-09 17:33:35
@LastEditors: shenlei
'''
from typing import List, Dict, Tuple, Type, Union
from copy import deepcopy

def reranker_tokenize_preproc(
    query: str, 
    passages: List[str],
    tokenizer=None,
    max_length: int=512,
    overlap_tokens: int=80,
    ):
    """
    对查询和段落进行token化预处理，生成模型可直接输入的键值对（含input_ids等），并记录每个输入对应的原始段落ID

    参数:
        query: 检索查询文本（与段落进行匹配的核心文本）
        passages: 待处理的候选段落列表（需要与查询计算相关性的文本集合）
        tokenizer: 分词器（用于将文本转换为模型可识别的token ID）
        max_length: 模型允许的最大序列长度（query+passage+分隔符的总长度不能超过此值）
        overlap_tokens: 长段落分块时的重叠token数量（用于保持分块间的上下文连贯性）

    返回:
        Tuple[List[Dict], List[int]]:
            - 第一个元素：预处理后的模型输入列表，每个元素为包含input_ids、attention_mask等的字典
            - 第二个元素：每个输入对应的原始段落ID列表（长段落分块后会对应相同ID）
    """
    assert tokenizer is not None, "Please provide a valid tokenizer for tokenization!"
    # 获取分词器的分隔符token ID（用于分隔查询和段落，通常为[SEP]的ID）
    sep_id = tokenizer.sep_token_id

    def _merge_inputs(chunk1_raw, chunk2):
        """
        内部辅助函数：合并两个token化后的片段（通常是查询和段落/段落分块）

        参数:
            chunk1_raw: 第一个片段的token化结果（通常是查询的token信息）
            chunk2: 第二个片段的token化结果（通常是段落/段落分块的token信息）

        返回:
            合并后的token化结果，包含input_ids、attention_mask，若有token_type_ids也会合并
        """
        chunk1 = deepcopy(chunk1_raw)

        chunk1['input_ids'].append(sep_id)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(sep_id)

        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])

        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids'])+2)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1

    # 对查询进行token化（不截断、不填充，保留原始长度）
    query_inputs = tokenizer.encode_plus(query, truncation=False, padding=False)
    # 计算段落部分允许的最大长度：总长度 - 查询长度 - 2个分隔符（query [SEP] passage [SEP]）
    max_passage_inputs_length = max_length - len(query_inputs['input_ids']) - 2
    # 断言确保段落可用长度足够（至少100token，否则查询过长，需缩短）
    assert max_passage_inputs_length > 100, "Your query is too long! Please make sure your query less than 400 tokens!"
    # 计算实际使用的重叠token数：取设定值与段落最大长度1/4的较小值（避免重叠过多导致效率低）
    overlap_tokens_implt = min(overlap_tokens, max_passage_inputs_length//4)

    # 存储处理后的模型输入和对应的段落ID
    res_merge_inputs = []  # 合并后的模型输入列表
    res_merge_inputs_pids = []  # 每个输入对应的原始段落ID（用于后续分数合并）

    # 遍历每个段落，处理后添加到结果中
    for pid, passage in enumerate(passages):
        # 对段落进行token化（不截断、不填充，不加特殊符号，保留原始token序列）
        passage_inputs = tokenizer.encode_plus(passage, truncation=False, padding=False, add_special_tokens=False)
        # 获取段落的token长度
        passage_inputs_length = len(passage_inputs['input_ids'])

        # 情况1：段落长度 <= 允许的最大段落长度，直接与查询合并
        if passage_inputs_length <= max_passage_inputs_length:
            # 合并查询和当前段落，生成模型输入
            qp_merge_inputs = _merge_inputs(query_inputs, passage_inputs)
            res_merge_inputs.append(qp_merge_inputs)
            res_merge_inputs_pids.append(pid)  # 记录当前段落ID

        # 情况2：段落过长，需要分块处理（保持重叠以维持上下文）
        else:
            start_id = 0  # 分块起始位置（token索引）
            # 循环分块，直到覆盖整个段落
            while start_id < passage_inputs_length:
                # 计算当前分块的结束位置（不超过段落总长度）
                end_id = start_id + max_passage_inputs_length
                # 截取当前分块的token信息（input_ids、attention_mask等）
                sub_passage_inputs = {k: v[start_id:end_id] for k, v in passage_inputs.items()}
                # 更新下一次分块的起始位置：若未到结尾，向前回退overlap_tokens_implt（保持重叠）
                start_id = end_id - overlap_tokens_implt if end_id < passage_inputs_length else end_id

                # 合并查询和当前分块，生成模型输入
                qp_merge_inputs = _merge_inputs(query_inputs, sub_passage_inputs)
                res_merge_inputs.append(qp_merge_inputs)
                res_merge_inputs_pids.append(pid)  # 同一长段落的分块共享相同ID

    # 返回预处理后的模型输入列表和对应的段落ID列表
    return res_merge_inputs, res_merge_inputs_pids