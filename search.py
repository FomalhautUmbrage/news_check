import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
from bs4 import BeautifulSoup
import json

# 初始化模型和向量存储
def initialize_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.IndexFlatL2(384)  # 向量维度为384
    return model, index

# 抓取新闻内容
def fetch_news_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    except Exception as e:
        print(f"无法提取 {url}: {e}")
        return ""

# 将新闻添加到向量存储
def add_webpages_to_store(model, index, urls):
    contents = [fetch_news_content(url) for url in urls]
    vectors = model.encode([c for c in contents if c])  # 仅编码非空内容
    if vectors.size > 0:
        index.add(np.array(vectors))
    return contents

# 将查询转换为向量
def encode_query(model, query):
    return model.encode([query])[0]

# 执行搜索
def perform_search(index, query_vector, k=5):
    D, I = index.search(np.array([query_vector]), min(k, index.ntotal))
    return I[0]

# 使用本地DeepSeek模型判断真实性
def check_authenticity(content, query):
    try:
        prompt = f"请判断以下新闻内容的真实性，查询是: '{query}'\n内容: {content[:1000]}"
        url = "http://localhost:11435/api/generate"
        payload = {
            "model": "deepseek-r1:7b",
            "prompt": prompt,
            "max_tokens": 150,
            "stream": False
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print(f"API Raw Response: {response.text}")
            lines = response.text.strip().split('\n')
            full_text = ""
            for line in lines:
                if line:
                    try:
                        data = json.loads(line)
                        full_text += data.get("response", data.get("text", ""))
                    except json.JSONDecodeError as e:
                        print(f"JSON 解析错误: {e} 在行: {line}")
            return full_text or "无有效响应内容"
        else:
            return f"真实性判断失败: {response.status_code} - {response.text}"
    except Exception as e:
        return f"真实性判断失败: {e}"

# 主函数
def main(urls):
    model, index = initialize_system()
    contents = add_webpages_to_store(model, index, urls)
    query = input("请输入您的查询: ")
    query_vector = encode_query(model, query)
    indices = perform_search(index, query_vector)
    print("搜索结果：")
    for i, idx in enumerate(indices, 1):
        content = contents[idx]
        authenticity = check_authenticity(content, query)
        print(f"{i}. {content[:100] if content else '无内容'}...\n真实性判断: {authenticity}\n")

# 示例使用
if __name__ == "__main__":
    urls = [
        "https://www.bbc.com/news/war-in-ukraine"
    ]
    main(urls)