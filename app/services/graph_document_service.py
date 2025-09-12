from typing import Any, Dict, List
import hashlib
import asyncio

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services import neo4j_service
from app.utils.api_key_manager import (
    get_embedding_model,
    get_graph_extraction_model,
    reset_key_index,
    switch_to_next_key,
    get_keys_count,
)
from app.utils.pdf_utils import extract_text_by_page


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)


async def _invoke_model(model, prompt: str) -> Dict[str, Any]:
    return await model.ainvoke(prompt)


async def process_and_create_graph(*, filename: str, content: bytes, size: int, mimetype: str) -> Dict[str, Any]:
    print(f"[Graph] 開始處理檔案: {filename}")
    file_hash = hashlib.sha256(content).hexdigest()

    existing = neo4j_service.find_document_by_hash(file_hash)
    if existing:
        print(f"[Graph] 檔案 {filename} 已存在。跳過處理。")
        return {"message": "檔案已存在，跳過處理", "fileId": existing["id"], "isNew": False}

    pages_text = await extract_text_by_page(content)
    docs = [
        {"pageContent": t, "metadata": {"source": filename, "pageNumber": i + 1}}
        for i, t in enumerate(pages_text)
        if t.strip() and len(t.strip()) > 10
    ]

    if not docs:
        print(f"[Graph] 檔案 {filename} 沒有有效內容。")
        return {"message": "檔案沒有可處理的內容", "isNew": True}

    chunks = []
    for d in docs:
        parts = text_splitter.split_text(d["pageContent"])
        for chunk in parts:
            chunks.append({"pageContent": chunk, "metadata": d["metadata"]})
    print(f"[Graph] 文件分塊完成，共 {len(chunks)} 個塊。")

    if get_keys_count() == 0:
        raise RuntimeError("設定錯誤：請在 .env 中提供 GOOGLE_API_KEY_LIST。")

    BATCH_SIZE = 30
    all_vectors: List[List[float]] = []

    print(f"[Graph] 開始分批生成向量，共找到 {get_keys_count()} 個可用的 API Key。")
    reset_key_index()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_texts = [c["pageContent"] for c in batch_chunks]
        print(f"[Graph] 正在準備生成向量批次: {i // BATCH_SIZE + 1}")
        batch_success = False
        while not batch_success:
            embeddings_model = get_embedding_model()
            if not embeddings_model:
                raise RuntimeError("所有 API Key 的配額都已耗盡，無法生成向量。")
            try:
                batch_vectors = await embeddings_model.aembed_documents(batch_texts)
                if not batch_vectors or len(batch_vectors) != len(batch_texts) or any((not v or len(v) == 0) for v in batch_vectors):
                    raise RuntimeError("API 回傳了無效或空的向量結果，將嘗試重試。")
                all_vectors.extend(batch_vectors)
                batch_success = True
                print(f"[Graph] 向量批次 {i // BATCH_SIZE + 1} 生成成功。")
            except Exception as err:
                print(f"[Graph] 向量生成批次遇到錯誤: {err}")
                if not switch_to_next_key():
                    raise RuntimeError("所有 API Key 均已嘗試，向量生成最終失敗。")
                await asyncio.sleep(1)

    print(f"[Graph] 所有向量均已成功生成，共 {len(all_vectors)} 個。")

    print("[Graph] 開始分批提取實體與關係...")
    reset_key_index()
    GRAPH_BATCH_SIZE = 5
    all_extractions: List[Dict[str, Any]] = []

    for i in range(0, len(chunks), GRAPH_BATCH_SIZE):
        batch_chunks = chunks[i : i + GRAPH_BATCH_SIZE]
        print(f"[Graph] 正在準備處理圖譜批次: {i // GRAPH_BATCH_SIZE + 1}")
        batch_success = False
        while not batch_success:
            model = get_graph_extraction_model()
            if not model:
                raise RuntimeError("所有 API Key 的配額都已耗盡，無法繼續處理。")
            try:
                prompt_template = """
                您是一位通用資訊提取 AI。您的任務是分析任何類型的文本，準確地提取出其中的核心實體（Entities）和它們之間的明確關係（Relationships），並以嚴格的 JSON 格式輸出。

                ### 核心規則:

                1.  **實體 (Entity) 提取**:
                    -   識別文本中代表真實世界物體或核心概念的名詞或名詞片語。
                    -   **實體的類型 (type)** 應該是根據文本上下文推斷出的通用、單一的名詞。例如：'人物', '組織', '地點', '產品', '技術', '日期', '概念', '事件'。請不要使用預設的固定列表，而是根據內容靈活判斷。

                2.  **關係 (Relationship) 提取**:
                    -   關係必須是文本中**清晰、明確陳述的直接聯繫**。
                    -   **關係的類型 (type)** 應該用一個簡潔的、描述性的動詞或動詞片語來表示，以準確反映實體間的互動。例如：'位於', '發明了', '收購了', '擁有', '合作夥伴是', '發佈於'。

                3.  **完全基於文本 (Strictly Text-Based)**:
                    -   您的所有輸出都**必須嚴格基於**所提供的文本。
                    -   **請勿推斷**文本中未提及的關係，也**不要使用**任何您自身的外部知識。

                4.  **格式與一致性**:
                    -   關係中的 'source' 和 'target' 名稱，必須與 'entities' 列表中對應的實體 'name' **完全一致**。
                    -   如果沒有找到任何實體或關係，必須返回包含空陣列的 JSON: {"entities": [], "relationships": []}。
                    -   絕對不要在 JSON 物件之外添加任何解釋、註解或對話。

                請根據以上規則，分析以下文本:
                ---
                <<CHUNK>>
                ---
                """
                prompts = [prompt_template.replace("<<CHUNK>>", chunk["pageContent"]) for chunk in batch_chunks]
                results = await asyncio.gather(*[_invoke_model(model, p) for p in prompts])
                all_extractions.extend(results)
                batch_success = True
                print(f"[Graph] 圖譜批次 {i // GRAPH_BATCH_SIZE + 1} 處理成功。")
            except Exception as err:
                print(f"[Graph] 圖譜批次處理錯誤: {err}")
                if not switch_to_next_key():
                    raise RuntimeError("所有 API Key 均已達到速率限制，處理終止。")
                await asyncio.sleep(1)

    chunks_with_graph = [
        {
            "text": c["pageContent"],
            "metadata": c["metadata"],
            "embedding": all_vectors[i],
            "entities": (all_extractions[i].get("entities") if i < len(all_extractions) and all_extractions[i] else []) or [],
            "relationships": (all_extractions[i].get("relationships") if i < len(all_extractions) and all_extractions[i] else []) or [],
        }
        for i, c in enumerate(chunks)
    ]

    document_data = {"name": filename, "size": size, "hash": file_hash, "mimetype": mimetype}
    result = neo4j_service.create_graph_from_document(document_data, chunks_with_graph)
    print(f"[Graph] 檔案 {filename} 已成功建立圖譜，文件節點 ID: {result['fileId']}")
    return {
        "message": f"成功為 {filename} 建立圖譜",
        "fileId": result["fileId"],
        "chunksCount": len(chunks),
        "entitiesCount": sum(len(c.get("entities", [])) for c in chunks_with_graph),
        "relationshipsCount": sum(len(c.get("relationships", [])) for c in chunks_with_graph),
        "isNew": True,
    }
