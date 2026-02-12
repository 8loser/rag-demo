"""
Demo 1: Sentence Transformers 基本使用
展示如何載入 embedding model 並取得文字的向量值

功能：
- 載入的 embedding model (all-MiniLM-L6-v2)
- 將文字轉換為向量特徵 (384 維)
- 輸出向量形狀和部分數值

前置需求：
- pip install sentence-transformers
"""

# 加上 huggingface 的鏡像站點環境變數，避免下載模型太慢
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer

# 載入模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 準備文字
sentences = ["這是一段測試文字", "EndeavourOS 開發環境"]

# 取得 Feature Vector
embeddings = model.encode(sentences)

# 查看 Shape [句子數量, 每個句子的維度]
# 輸出會是 (2, 384)，代表 2 個句子，每個句子 384 維, 是 all-MiniLM-L6-v2 模型的特徵維度。
print(f"向量形狀 (embeddings.shape): {embeddings.shape}") 

# 查看具體數值 (只印出第一句的前 5 個向量值作為範例)
print(f"第一句: `{sentences[0]}`")
print(f"前 5 個向量值: {embeddings[0][:5]}")
print(f"第二句: `{sentences[1]}`")
print(f"前 5 個向量值: {embeddings[1][:5]}")

# 執行結果
# 向量形狀 (embeddings.shape): (2, 384)
# 第一句的前 5 個向量值: [0.03097137 0.09432378 0.04806018 0.02745436 0.02937624]