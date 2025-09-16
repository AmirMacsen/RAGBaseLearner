'''
@Description: 
@Author: shenlei
@Date: 2023-11-28 14:04:27
@LastEditTime: 2024-09-02 22:32:01
@LastEditors: Chengjie Guo i@guoch.xyz
'''
import logging
import torch
from tqdm import tqdm
from numpy import ndarray
import numpy as np
from typing import List, Dict, Tuple, Type, Union

from transformers import AutoModel, AutoTokenizer
from BCEmbedding.utils import logger_wrapper
logger = logger_wrapper('BCEmbedding.models.EmbeddingModel')


class EmbeddingModel:
    """
    文本嵌入模型类：用于将文本转换为固定维度的向量表示（嵌入向量），支持预训练模型加载、多设备运行及多种池化方式

    核心功能：通过预训练语言模型提取文本语义特征，生成可用于检索、相似度计算等任务的嵌入向量
    """

    def __init__(
            self,
            model_name_or_path: str = 'maidalun1020/bce-embedding-base_v1',
            pooler: str = 'cls',
            use_fp16: bool = False,
            device: str = None,
            **kwargs
    ):
        """
        初始化嵌入模型，加载预训练模型和分词器，配置运行设备和精度

        参数:
            model_name_or_path: 预训练模型的名称（HuggingFace Hub）或本地路径
            pooler: 池化方式，用于从模型输出的隐藏状态中提取句子级嵌入
                    - 'cls': 使用[CLS] token的隐藏状态（推荐，适用于多数预训练模型）
                    - 'mean': 对所有有效token的隐藏状态进行平均（需结合attention mask）
            use_fp16: 是否使用FP16半精度计算（可加速推理并减少内存占用）
            device: 运行设备，可选值：'cpu'、'cuda'、'cuda:0'、'0'（数字表示GPU编号），默认自动检测
            **kwargs: 传递给AutoTokenizer.from_pretrained和AutoModel.from_pretrained的额外参数
                     （如trust_remote_code=True用于加载自定义模型）
        """
        # 加载分词器和预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        logger.info(f"Loading from `{model_name_or_path}`.")

        # 验证池化方式有效性
        assert pooler in ['cls', 'mean'], f"`pooler` should be in ['cls', 'mean']. 'cls' is recommended!"
        self.pooler = pooler

        # 自动检测可用GPU数量
        num_gpus = torch.cuda.device_count()

        # 配置运行设备（优先使用用户指定，否则自动选择）
        if device is None:
            # 无指定设备时，有GPU则用cuda，否则用cpu
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            # 处理用户指定的设备（如数字'0'转换为'cuda:0'）
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device

        # 确定GPU使用数量（用于后续批量处理优化）
        if self.device == "cpu":
            self.num_gpus = 0  # CPU模式下无GPU
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1  # 单GPU指定（如cuda:0）
        elif self.device == "cuda":
            self.num_gpus = num_gpus  # 多GPU模式（使用所有可用GPU）
        elif self.device == "xpu":
            # 支持Intel XPU设备（需依赖intel_extension_for_pytorch）
            import intel_extension_for_pytorch as ipex
            self.num_gpus = 0  # XPU不归类为GPU计数
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")

        # 启用FP16半精度计算（仅在GPU上有效）
        if use_fp16:
            self.model.half()

        # 设置模型为评估模式（关闭dropout等训练相关层）
        self.model.eval()
        # 将模型移动到目标设备
        self.model = self.model.to(self.device)

        # XPU设备优化（Intel扩展）
        if self.device == "xpu":
            self.model = ipex.optimize(self.model)

        # 多GPU数据并行（当使用多个GPU时）
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        # 日志输出配置信息
        logger.info(
            f"Execute device: {self.device};\t "
            f"gpu num: {self.num_gpus};\t "
            f"use fp16: {use_fp16};\t "
            f"embedding pooling type: {self.pooler};\t "
            f"trust remote code: {kwargs.get('trust_remote_code', False)}"
        )

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 256,
            max_length: int = 512,
            normalize_to_unit: bool = True,
            return_numpy: bool = True,
            enable_tqdm: bool = True,
            query_instruction: str = "",
            **kwargs
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        将输入文本转换为嵌入向量

        参数:
            sentences: 输入文本，可为单句字符串或句子列表
            batch_size: 批量处理大小（多GPU时会自动乘以GPU数量）
            max_length: 文本最大token长度（超过会被截断）
            normalize_to_unit: 是否将嵌入向量归一化到单位向量（L2归一化，常用于相似度计算）
            return_numpy: 是否返回numpy数组（False则返回torch张量）
            enable_tqdm: 是否显示进度条
            query_instruction: 查询指令前缀（用于构造查询嵌入，如"检索查询："）
            **kwargs: 传递给tokenizer的额外参数

        返回:
            文本嵌入向量，形状为[样本数, 嵌入维度]，类型为numpy数组或torch张量
        """
        # 多GPU时按GPU数量放大批次大小（充分利用多卡算力）
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus

        # 处理单句输入：转换为列表格式
        if isinstance(sentences, str):
            sentences = [sentences]

        # 禁用梯度计算（推理阶段无需更新模型，节省内存并加速）
        with torch.no_grad():
            # 收集所有批次的嵌入向量
            embeddings_collection = []

            # 按批次处理文本
            for sentence_id in tqdm(
                    range(0, len(sentences), batch_size),
                    desc='Extract embeddings',  # 进度条描述
                    disable=not enable_tqdm  # 控制是否显示进度条
            ):
                # 截取当前批次的文本
                batch_sentences = sentences[sentence_id:sentence_id + batch_size]

                # 若指定查询指令，为批次内所有文本添加前缀（用于查询嵌入的场景）
                if isinstance(query_instruction, str) and len(query_instruction) > 0:
                    batch_sentences = [query_instruction + sent for sent in batch_sentences]

                # 文本token化处理
                inputs = self.tokenizer(
                    batch_sentences,
                    padding=True,  # 自动填充到批次内最长序列
                    truncation=True,  # 超过max_length则截断
                    max_length=max_length,  # 最大长度限制
                    return_tensors="pt"  # 返回PyTorch张量
                )

                # 将token化结果转移到模型所在设备
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}

                # 模型前向推理，获取隐藏状态
                outputs = self.model(**inputs_on_device, return_dict=True)

                # 根据指定的池化方式提取句子嵌入
                if self.pooler == "cls":
                    # CLS池化：取最后一层隐藏状态的第一个token（[CLS]）
                    embeddings = outputs.last_hidden_state[:, 0]
                elif self.pooler == "mean":
                    # 平均池化：基于attention mask对有效token的隐藏状态求平均（排除padding）
                    attention_mask = inputs_on_device['attention_mask']  # 形状：[batch_size, seq_len]
                    last_hidden = outputs.last_hidden_state  # 形状：[batch_size, seq_len, hidden_dim]
                    # 加权求和（padding部分mask为0，不参与计算）
                    embeddings = (last_hidden * attention_mask.unsqueeze(-1).float()).sum(1)
                    # 除以有效token数量（避免长度影响）
                    embeddings = embeddings / attention_mask.sum(-1).unsqueeze(-1)
                else:
                    raise NotImplementedError(f"Pooler `{self.pooler}` is not implemented!")

                # 归一化到单位向量（L2归一化，常用于提升相似度计算稳定性）
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

                # 将当前批次的嵌入向量转移到CPU并添加到收集列表
                embeddings_collection.append(embeddings.cpu())

            # 拼接所有批次的嵌入向量，形成最终结果
            embeddings = torch.cat(embeddings_collection, dim=0)

        # 转换为numpy数组（如果需要）
        if return_numpy and not isinstance(embeddings, np.ndarray):
            embeddings = embeddings.numpy()

        return embeddings
