# RAG Demo

從 Embedding 基礎到完整 RAG Pipeline 的漸進式範例。

## 範例說明

| 檔案 | 說明 |
|------|------|
| `01_embedding_basics.py` | Sentence Transformers 基本使用, 將句子轉為 384 維的向量 |
| `02_vectordb_semantic_search.py` | Vector DB 應用, Qdrant 儲存、檢索 |
| `03_rag_pipeline.py` | 完整 RAG 流程：Retrieval + Augmentation + Generation |

## 環境設置

### 1. 建立並進入 venv 環境
```bash
python -m venv venv
source venv/bin/activate
```

### 2. 安裝套件
```bash
# 基礎 (01, 02)
pip install sentence-transformers qdrant-client

# RAG pipeline (03)
pip install langchain-huggingface langchain-qdrant langchain-ollama langchain-core
```

### 3. 啟動服務

**Qdrant 向量資料庫** (02, 03 需要)
```bash
podman run --name qdrant -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    docker.io/qdrant/qdrant
```
Web 界面：http://localhost:6333/dashboard

**Ollama 本地 LLM** (03 需要)
```bash
ollama serve
ollama pull llama3
```

## 執行順序

```bash
# Step 1: 了解 Embedding 向量化
python 01_embedding_basics.py

# Step 2: 建立向量資料庫並測試語義搜尋
python 02_vectordb_semantic_search.py

# Step 3: 執行完整 RAG 問答
python 03_rag_pipeline.py
```
