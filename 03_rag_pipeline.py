"""
Demo3: RAG (Retrieval-Augmented Generation) 完整演示

展示重點：
Retrieval   - 語義檢索相關文檔
Augmentation - 動態組裝 prompt
Generation   - LLM 生成答案

使用場景：企業內部知識庫問答系統

前置需求：
1. 啟動 Ollama: ollama serve && ollama pull llama3
2. 啟動 Qdrant: docker run -p 6333:6333 qdrant/qdrant
3. 安裝依賴: pip install langchain-{huggingface,qdrant,ollama,core}
4. 建立知識庫: python demo2.py

後續可完善項目：
- 避免幻覺機制 (驗證答案是否基於參考資料）
- 相似度閾值判定 (過濾低相關文檔，如 score < 0.7)
- 參考資料不足處理 (檢索結果質量太低時的應對策略)
- Prompt 參考資料使用指引 (明確要求 LLM 僅基於提供資料回答)
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Embedding 模型（需與 demo2.py 一致）
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# [R] Retrieval - 連接向量資料庫
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="multilingual_notes",
    url="http://127.0.0.1:6333"
)


# [A] Augmentation - 設計 prompt 模板
template = """你是企業知識庫助手，請根據以下參考資料回答問題。

參考資料：
{context}

問題：{question}

請用繁體中文簡潔回答。
"""
prompt = ChatPromptTemplate.from_template(template)

# 檢索設定：取最相關的 3 筆文檔
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# [G] Generation - 本地 LLM
llm = ChatOllama(model="llama3", base_url="http://localhost:11434")

# 構建 RAG Chain：Retrieval -> Augmentation -> Generation
rag_chain = (
    {
        "context": retriever | format_docs,  # 檢索 + 格式化
        "question": RunnablePassthrough()    # 傳遞原問題
    }
    | RunnablePassthrough.assign(
        # Prompt -> LLM -> 文字
        # prompt - 接收 {context, question}，生成格式化的提示詞
        # llm - 將提示詞發送給 Llama3 模型，得到 AI 回答
        # StrOutputParser() - 將 LLM 的輸出轉換成純文字字符串
        # | 是 LangChain 的 LCEL 語法
        answer=prompt | llm | StrOutputParser()
    )
)


# 完整 RAG 流程 demo
if __name__ == "__main__":
    query = "我是小團隊，預算不多但想試用 AI，有適合的方案嗎？"

    print(f"\n{'='*60}")
    print(f"問題「{query}」")
    print(f"{'='*60}")

    # [R] Retrieval - 語義檢索
    print(f"\nRetrieval - 從 Vector DB 檢索關聯最高的 3 筆資料")
    print(f"{'-'*60}")
    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    docs = [doc for doc, _ in docs_with_scores]
    formatted_context = format_docs(docs)
    print(f"檢索到 {len(docs)} 筆相關資料：")
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"  {i}. [相似度: {score:.4f}] {doc.page_content[:80]}...")

    # [A] Augmentation - 增強 Prompt
    print(f"\nAugmentation - 增強 Prompt")
    print(f"{'-'*60}")
    augmented_prompt = prompt.invoke({"context": formatted_context, "question": query})
    print(f"增強後 Prompt 內容：")
    print(augmented_prompt.to_string())

    # [G] Generation - LLM 回答
    print(f"\nGeneration - LLM 生成答案")
    print(f"{'-'*60}")
    result = rag_chain.invoke(query)
    print(f"LLM 回答：")
    print(f"{result['answer']}")
    print(f"\n{'='*60}")