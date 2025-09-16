from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain.schema import Document
from BCEmbedding import RerankerModel  # 使用底层重排序模型更可靠
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import Field, ConfigDict

# -------------------------- 1. 初始化并测试嵌入模型--------------------------
print("初始化嵌入模型并测试...")
embedding_model_name = "D:\\ai\\modelscope_models\\maidalun\\bce-embedding-base_v1"
embedding_model_kwargs = {"device": "cuda:0"}
embedding_encode_kwargs = {"batch_size": 32}

try:
    embed_model = HuggingFaceBgeEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=embedding_model_kwargs,
        encode_kwargs=embedding_encode_kwargs,
    )

    # 测试嵌入模型是否能正常生成向量
    test_text = "测试嵌入模型生成向量"
    test_embedding = embed_model.embed_query(test_text)
    print(f"嵌入模型测试成功：生成向量维度为 {len(test_embedding)}（正常应为768或1024）")
    print(f"向量前5位：{test_embedding[:5]}")  # 验证是否为数值向量
except Exception as e:
    print(f"嵌入模型初始化或测试失败: {e}")
    exit()

# -------------------------- 2. 初始化重排序模型 --------------------------
print("\n初始化重排序模型...")
try:
    rerank_model = RerankerModel(
        model_name_or_path="D:\\ai\\modelscope_models\\maidalun\\bce-reranker-base_v1",
        device="cuda:0"
    )
    print("重排序模型初始化成功")
except Exception as e:
    print(f"重排序模型初始化失败: {e}")
    exit()

# -------------------------- 3. 自定义压缩器 --------------------------
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor


class BCE_Rerank_Compressor(BaseDocumentCompressor):
    """适配BCEmbedding重排序模型的压缩器（符合LangChain接口规范）"""
    # Pydantic配置：允许未知类型（解决Reranker模型类型报错）
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 字段声明（Pydantic要求）
    rerank_model: RerankerModel = Field(..., description="BCEmbedding重排序模型实例")
    top_n: int = Field(5, description="重排序后保留的Top N结果")

    def compress_documents(
            self,
            documents: list[Document],  # 基础检索返回的文档列表
            query: str,  # 用户查询
            callbacks=None  # LangChain回调参数（无需使用，仅兼容）
    ) -> list[Document]:
        """核心方法：对基础检索结果重排序并筛选Top N"""
        # 容错：无输入文档直接返回
        if not documents:
            print("无待重排序的文档")
            return []

        # 步骤1：提取文档文本（重排序模型需要纯文本列表）
        passages = [doc.page_content for doc in documents]

        # 步骤2：调用重排序模型（返回格式：{"rerank_passages": [], "rerank_scores": [], ...}）
        try:
            rerank_result = self.rerank_model.rerank(query, passages)
            # 提取重排序后的段落和对应分数
            sorted_passages = rerank_result["rerank_passages"]
            sorted_scores = rerank_result["rerank_scores"]
        except Exception as e:
            print(f"重排序模型调用失败: {e}")
            return []

        # 步骤3：组合段落与分数，按分数降序排序（确保Top N最相关）
        score_passage_pairs = list(zip(sorted_scores, sorted_passages))
        score_passage_pairs.sort(reverse=True, key=lambda x: x[0])  # 按分数倒序

        # 步骤4：筛选Top N，匹配原始文档（保留元数据）
        top_docs = []
        for idx, (score, passage_text) in enumerate(score_passage_pairs[:self.top_n], 1):
            try:
                # 匹配原始文档（通过文本内容完全匹配）
                matched_doc = next(
                    doc for doc in documents if doc.page_content == passage_text
                )
                # 给文档添加重排序分数元数据
                matched_doc.metadata["rerank_score"] = score
                top_docs.append(matched_doc)
                print(f"重排序匹配成功 - 结果{idx}（分数：{score:.4f}）")
            except StopIteration:
                print(f"重排序匹配失败 - 结果{idx}（未找到对应原始文档）")
            except Exception as e:
                print(f"重排序处理失败 - 结果{idx}：{e}")

        return top_docs


rerank_compressor = BCE_Rerank_Compressor(rerank_model=rerank_model, top_n=5)

# -------------------------- 4. 文档预处理（增加内容校验）--------------------------
print("\n处理文档并校验内容...")
try:
    loader = PyMuPDFLoader("P020250220355436997535.pdf")
    raw_docs = loader.load()
    print(f"加载PDF成功，共 {len(raw_docs)} 页")


    # 清洗文本并过滤空内容
    def clean_text(text: str) -> str:
        text = text.replace("\n", " ").strip()
        return " ".join(text.split())


    valid_docs = []
    for i, doc in enumerate(raw_docs):
        cleaned = clean_text(doc.page_content)
        if len(cleaned) < 10:  # 过滤过短的无效内容
            print(f"警告：第 {i + 1} 页清洗后内容过短（{len(cleaned)}字符），已跳过")
            continue
        doc.page_content = cleaned
        valid_docs.append(doc)

    print(f"文档清洗完成，有效文档页数：{len(valid_docs)}（过滤掉空或过短内容）")

    # 分割文本并再次校验
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = text_splitter.split_documents(valid_docs)

    # 过滤分割后过短的片段
    split_docs = [doc for doc in split_docs if len(doc.page_content) > 50]
    print(f"文本分割并过滤后，得到 {len(split_docs)} 个有效片段")

    # 打印部分片段内容确认
    if split_docs:
        print(f"\n随机抽查3个片段内容：")
        for i in [0, min(5, len(split_docs) - 1), min(10, len(split_docs) - 1)]:
            print(f"片段{i}：{split_docs[i].page_content[:150]}...")
    else:
        print("错误：分割后无有效文档片段！请检查PDF内容是否正常")
        exit()

except Exception as e:
    print(f"文档处理失败: {e}")
    exit()

# -------------------------- 5. FAISS检索器 --------------------------
print("\n创建FAISS索引...")
try:
    # 1. 生成文档向量并检查（关键调试）
    print("正在生成文档向量并检查...")
    sample_doc = split_docs[0].page_content
    sample_embedding = embed_model.embed_query(sample_doc)
    print(f"样本文档向量维度：{len(sample_embedding)}")
    print(f"样本向量前5位：{sample_embedding[:5]}")

    # 2. 创建索引时强制使用内积距离（显式设置）
    faiss_db = FAISS.from_documents(
        split_docs,
        embed_model,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    print(f"FAISS索引创建完成，包含 {faiss_db.index.ntotal} 个片段")

    # 3.不设置score_threshold，只通过k控制返回数量
    base_retriever = faiss_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}  # 直接返回前10个结果，不做阈值过滤
    )
    print("检索器创建完成（已移除阈值过滤，确保返回结果）")

except Exception as e:
    print(f"索引/检索器创建失败: {e}")
    exit()

# -------------------------- 6. 测试基础检索器（增加向量相似度计算）--------------------------
print("\n测试基础检索器（带向量相似度计算）...")
query = "AI幻觉是什么"
try:
    # 生成查询向量并检查
    query_embedding = embed_model.embed_query(query)
    print(f"查询向量维度：{len(query_embedding)}")

    # 方法1：使用检索器获取结果
    base_results = base_retriever.get_relevant_documents(query)
    print(f"基础检索器返回 {len(base_results)} 个结果（预期10个）")

    # 方法2：直接用FAISS的similarity_search_with_score验证
    # （确保检索器逻辑与底层FAISS一致）
    faiss_direct_results = faiss_db.similarity_search_with_score(query, k=10)
    print(f"FAISS底层直接检索返回 {len(faiss_direct_results)} 个结果（预期10个）")

    # 打印底层检索的分数（内积值，越大越相关）
    print("\nFAISS底层检索分数（内积值）：")
    for i, (doc, score) in enumerate(faiss_direct_results, 1):
        print(f"  结果{i}：分数={score:.4f}，内容：{doc.page_content[:100]}...")

    # 如果两种方法结果数量不一致，说明检索器参数有问题
    if len(base_results) != len(faiss_direct_results):
        print("\n警告：检索器与FAISS底层结果数量不一致！")
        print("可能原因：langchain检索器内部做了额外过滤")
        # 强制使用底层结果作为备选
        base_results = [doc for doc, _ in faiss_direct_results]

except Exception as e:
    print(f"基础检索测试失败: {e}")
    exit()

# -------------------------- 7. 最终检索与重排序 --------------------------
print("\n使用压缩检索器查询...")
try:
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=rerank_compressor,
        base_retriever=base_retriever
    )

    final_results = compression_retriever.get_relevant_documents(query)

    print(f"\n=== 最终重排序结果（共 {len(final_results)} 个）===")
    for i, doc in enumerate(final_results, 1):
        print(f"  结果{i}（分数：{doc.metadata['rerank_score']:.4f}）：{doc.page_content[:200]}...")

    if not final_results:
        print("\n警告：重排序后无结果！可能原因：")
        print("1. 基础检索结果与查询完全无关（重排序分数过低）")
        print("2. 文档中确实没有与'AI幻觉'相关的内容")
        print("建议：尝试文档中存在的关键词（如文档标题相关词汇）重新测试")

except Exception as e:
    print(f"压缩检索失败: {e}")
    import traceback

    traceback.print_exc()
