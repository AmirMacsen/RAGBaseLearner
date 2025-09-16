'''
@Description: 
@Author: shenlei
@Date: 2023-11-28 14:04:27
@LastEditTime: 2024-05-13 17:04:41
@LastEditors: shenlei
'''
import logging
import torch

import numpy as np

from tqdm import tqdm
from typing import List, Dict, Tuple, Type, Union
from copy import deepcopy

from .utils import reranker_tokenize_preproc

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('BCEmbedding.models.RerankerModel')


class RerankerModel:
    def __init__(
            self,
            model_name_or_path: str='maidalun1020/bce-reranker-base_v1',
            use_fp16: bool=False,
            device: str=None,
            **kwargs
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
        logger.info(f"Loading from `{model_name_or_path}`.")
        
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device
        
        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")

        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16}")

        # for advanced preproc of tokenization
        self.max_length = kwargs.get('max_length', 512)
        self.overlap_tokens = kwargs.get('overlap_tokens', 80)

    # 计算相似度得分
    def compute_score(
            self, 
            sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], 
            batch_size: int = 256,
            max_length: int = 512,
            enable_tqdm: bool=True,
            **kwargs
        ):
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus
        
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        # 推理阶段，不用记录梯度
        with torch.no_grad():
            scores_collection = []
            # 按批次处理句子对，使用tqdm显示进度条
            # range(0, len(sentence_pairs), batch_size)：以batch_size为步长生成索引，实现分批
            # desc='Calculate scores'：进度条描述文字
            # disable=not enable_tqdm：控制是否显示进度条（enable_tqdm为True时显示）
            for sentence_id in tqdm(range(0, len(sentence_pairs), batch_size), desc='Calculate scores', disable=not enable_tqdm):
                # 获取当前批次的句子对（从sentence_id到sentence_id+batch_size，避免越界）
                # 每一批都是二维数组，每一项都是一个句子对，一般是用户查询与文档的句子对比如 [["xxxx", "xxxxx"], ["xcxxxxx", "xxxxvxxxx"]]
                sentence_pairs_batch = sentence_pairs[sentence_id:sentence_id+batch_size]
                # 使用tokenizer对当前批次的句子对进行预处理
                # padding=True：自动填充到批次中最长序列的长度
                # truncation=True：超过max_length的序列自动截断
                # max_length：最大序列长度限制
                # return_tensors="pt"：返回PyTorch张量格式的结果
                inputs = self.tokenizer(
                            sentence_pairs_batch, 
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                        )

                # 将预处理后的输入张量转移到模型所在设备（如GPU/CPU）
                # 确保输入与模型在同一设备上，避免计算错误
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                # 调用模型进行推理：
                # **inputs_on_device：将字典形式的输入解包传入模型
                # return_dict=True：返回模型输出的字典对象（包含logits等）
                # .logits：获取模型输出的原始logits（未经过激活函数的输出）
                # .view(-1,)：将logits重塑为一维张量（展平）
                # .float()：转换为float类型，确保数值类型一致
                scores = self.model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
                # 分数归一化，使之保存在[0, 1]范围内
                scores = torch.sigmoid(scores)
                # 将计算得到的分数从设备（如GPU）转移到CPU，转换为numpy数组，再转为Python列表
                # 最后添加到scores_collection中，收集所有批次的结果
                scores_collection.extend(scores.cpu().numpy().tolist())
        
        if len(scores_collection) == 1:
            return scores_collection[0]
        return scores_collection

    def rerank(
            self,
            query: str,
            passages: List[str],
            batch_size: int=256,
            **kwargs
        ):
        # remove invalid passages
        # 对文档进行处理：
        # 1. 删除无效的文档（长度为0的文档）
        # 2. 截断文档长度为128000字符
        passages = [p[:128000] for p in passages if isinstance(p, str) and 0 < len(p)]
        if query is None or len(query) == 0 or len(passages) == 0:
            return {'rerank_passages': [], 'rerank_scores': []}
        
        # preproc of tokenization
        # 预处理：将查询与段落组合成句子对，并进行token化前处理
        # 生成sentence_pairs（查询-段落对）和sentence_pairs_pids（记录每个句子对对应的原始段落ID）
        # 内部会处理长段落截断、token重叠等逻辑（基于传入的tokenizer和长度参数）
        sentence_pairs, sentence_pairs_pids = reranker_tokenize_preproc(
            query, passages, 
            tokenizer=self.tokenizer, # 用于token化的分词器
            max_length=self.max_length, # 最大序列长度
            overlap_tokens=self.overlap_tokens, # 长段落截断时的重叠token数（保持上下文连贯性）
            )

        # batch inference
        # 批量推理配置：若使用多GPU，按GPU数量放大batch_size（充分利用多卡资源）
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus

        tot_scores = []
        with torch.no_grad():
            # 按批次处理句子对
            for k in range(0, len(sentence_pairs), batch_size):
                # 对批次内句子对进行padding（填充到批次内最长序列长度）
                # 转换为PyTorch张量格式
                batch = self.tokenizer.pad(
                        sentence_pairs[k:k+batch_size],
                        padding=True,
                        max_length=None,
                        pad_to_multiple_of=None,
                        return_tensors="pt"
                    )
                # 将批次数据转移到模型所在设备（GPU/CPU），确保数据与模型在同一设备
                batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
                # 模型推理：
                # **batch_on_device：解包字典参数传入模型
                # return_dict=True：返回字典格式的模型输出
                # .logits：获取模型原始输出（未经过激活函数）
                # .view(-1,)：将输出展平为一维张量
                # .float()：转换为float类型确保数值兼容性
                scores = self.model(**batch_on_device, return_dict=True).logits.view(-1,).float()
                # 应用sigmoid激活函数，将logits转换为0-1之间的相关性概率分数
                scores = torch.sigmoid(scores)
                # 将分数从设备转移到CPU，转为numpy数组后再转为Python列表，添加到总分数列表
                tot_scores.extend(scores.cpu().numpy().tolist())

        # ranking
        # 分数合并：由于长段落可能被拆分为多个句子对，需合并同一原始段落的分数
        # 初始化与段落数量对应的分数列表（初始值为0）
        merge_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(sentence_pairs_pids, tot_scores):
            # 遍历每个句子对的分数及其对应的原始段落ID，取同一段落的最大分数作为最终分数
            # （理由：长段落的多个片段中，与查询最相关的片段代表整体相关性）
            merge_scores[pid] = max(merge_scores[pid], score)

        # 排序：对合并后的分数进行降序排序，获取排序索引
        merge_scores_argsort = np.argsort(merge_scores)[::-1]
        sorted_passages = [] # 排序后的段落
        sorted_scores = [] # 对应的分数
        for mid in merge_scores_argsort:
            sorted_scores.append(merge_scores[mid])
            sorted_passages.append(passages[mid])
        
        return {
            'rerank_passages': sorted_passages,
            'rerank_scores': sorted_scores,
            'rerank_ids': merge_scores_argsort.tolist()  # 原始段落的排序ID（用于追溯原始顺序）
        }
