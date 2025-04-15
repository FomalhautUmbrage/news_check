import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# 初始化模型和向量存储
def initialize_system():
    # 加载预训练模型（轻量且高效）
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # 创建 FAISS 索引，维度为 384（由模型决定）
    index = faiss.IndexFlatL2(384)
    return model, index


# 添加网页内容到向量存储
def add_webpages_to_store(model, index, webpages):
    # 将网页内容转换为向量
    vectors = model.encode(webpages)
    # 添加到 FAISS 索引
    index.add(np.array(vectors))
    return webpages


# 执行搜索
def perform_search(query, model, index, webpages, k=5):
    # 将查询转换为向量
    query_vector = model.encode([query])[0]
    # 在索引中搜索最相似的 k 个结果
    D, I = index.search(np.array([query_vector]), k)
    # 返回对应的网页内容
    return [webpages[i] for i in I[0]]


# 示例使用
if __name__ == "__main__":
    # 初始化系统
    model, index = initialize_system()

    # 模拟全网数据（示例网页内容）
    webpages = [
        "Python 是一种流行的编程语言。",
        "机器学习是人工智能的一个子集。",
        "Web 开发涉及创建网站和应用程序。",
        "数据科学结合统计学和计算机科学从数据中提取洞察。",
        "云计算提供按需访问的计算资源。"
    ]

    # 将网页内容添加到向量存储
    webpages = add_webpages_to_store(model, index, webpages)

    # 用户查询
    query = "机器学习是什么？"

    # 执行搜索并返回结果
    results = perform_search(query, model, index, webpages)

    # 输出结果
    print("搜索结果：")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")