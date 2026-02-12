"""
Demo2: Qdrant 向量資料庫語義搜尋範例

功能：
- 使用 paraphrase-multilingual-MiniLM-L12-v2 多語言模型將文字轉換為向量
- 將向量存入 Qdrant 向量資料庫
- 執行語義搜尋，找出與查詢問題最相關的內容

前置需求：
啟動 Qdrant 容器
```
podman run --name qdrant -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    docker.io/qdrant/qdrant
```
Web 界面：
http://localhost:6333/dashboard
http://127.0.0.1:6333/dashboard
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# 載入 Embed Model, 改用 paraphrase-multilingual-MiniLM-L12-v2 對中文理解程度比較好
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 建立 Qdrant 客戶端
client = QdrantClient("127.0.0.1", port=6333)

# collection 不存在則建立
# size 為 384, 必須跟 embed model 維度搭配
collection_name = "multilingual_notes"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# 準備寫入的資料 - AI 產品知識庫
documents = [
    {"id": 1, "text": "ClaudeAI 提供三種方案：Free (每月 50 則訊息)、Pro ($20/月無限對話)、Team ($30/月含協作功能)"},
    {"id": 2, "text": "API 調用限制：Free tier 每分鐘 5 次請求，Pro 每分鐘 60 次，Enterprise 無限制"},
    {"id": 3, "text": "支援的模型：Claude 3.5 Sonnet (最佳平衡)、Claude 3 Opus (最強推理)、Claude 3 Haiku (最快速度)"},
    {"id": 4, "text": "Claude 的 context window 支援 200K tokens，約等於 150,000 個英文單字或 500 頁文件"},
    {"id": 5, "text": "所有方案都支援多語言對話，包含繁體中文、簡體中文、英文、日文等 50+ 種語言"},
    {"id": 6, "text": "企業方案提供 SSO 單一登入、自定義保留政策、優先技術支援和 99.9% SLA 保證"},
    {"id": 7, "text": "Claude API 採用 RESTful 設計，使用 Bearer Token 認證，需在 Header 加入 x-api-key"},
    {"id": 8, "text": "資料隱私：對話內容不會用於訓練模型，企業客戶可選擇完全不留存對話紀錄"},
    {"id": 9, "text": "快速開始三步驟：1) 註冊帳號 2) 取得 API Key 3) 安裝 SDK (pip install anthropic)"},
    {"id": 10, "text": "Claude 擅長程式碼生成、文件摘要、多輪對話、結構化資料處理和複雜推理任務"},
]

# 批次產生每個內容的向量, 利用 GPU 加速批次處理省效能
texts = [doc["text"] for doc in documents]
vectors = model.encode(texts)

# 建立 vector db (Qdrant) 可用向量與內容資料點
points = []
for i, doc in enumerate(documents):
    points.append(PointStruct(
        id=doc["id"],
        vector=vectors[i].tolist(),
        payload={"page_content": doc["text"]}
    ))

# 新增資料點到 Qdrant
client.upsert(collection_name=collection_name, points=points)

# 語義搜尋測試 (展示語義檢索能力)
queries = [
    "我的預算有限，有免費方案嗎？",
    "API 會不會太慢影響使用者體驗？",
    "我們公司需要符合資安規範，資料會外洩嗎？"
]

for query_text in queries:
    query_vector = model.encode(query_text).tolist()

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3  # 抓相似度最高的 3 筆資料
    ).points

    # 顯示搜尋結果
    print(f"\n問題：{query_text}")
    for i, res in enumerate(search_result):
        content = res.payload['page_content']
        score = res.score
        print(f"  {i+1}. [相似度: {score:.3f}] {content[:100]}")

# 執行結果
# 問題：我的預算有限，有免費方案嗎？
#   1. [相似度: 0.485] ClaudeAI 提供三種方案：Free (每月 50 則訊息)、Pro ($20/月無限對話)、Team ($30/月含協作功能)
#   2. [相似度: 0.291] API 調用限制：Free tier 每分鐘 5 次請求，Pro 每分鐘 60 次，Enterprise 無限制
#   3. [相似度: 0.176] 企業方案提供 SSO 單一登入、自定義保留政策、優先技術支援和 99.9% SLA 保證

# 問題：API 會不會太慢影響使用者體驗？
#   1. [相似度: 0.607] API 調用限制：Free tier 每分鐘 5 次請求，Pro 每分鐘 60 次，Enterprise 無限制
#   2. [相似度: 0.375] Claude API 採用 RESTful 設計，使用 Bearer Token 認證，需在 Header 加入 x-api-key
#   3. [相似度: 0.362] 快速開始三步驟：1) 註冊帳號 2) 取得 API Key 3) 安裝 SDK (pip install anthropic)

# 問題：我們公司需要符合資安規範，資料會外洩嗎？
#   1. [相似度: 0.414] 資料隱私：對話內容不會用於訓練模型，企業客戶可選擇完全不留存對話紀錄
#   2. [相似度: 0.397] 企業方案提供 SSO 單一登入、自定義保留政策、優先技術支援和 99.9% SLA 保證
#   3. [相似度: 0.391] API 調用限制：Free tier 每分鐘 5 次請求，Pro 每分鐘 60 次，Enterprise 無限制