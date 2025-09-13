from typing import Any, Dict, List, Optional
import hashlib
import asyncio
import os
import re
import contextlib
from urllib.parse import urlparse

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
from app.services.progress_bus import publish_progress
from app.services.crawler import load_markdown


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)


async def _invoke_model(model, prompt: str) -> Dict[str, Any]:
    return await model.ainvoke(prompt)


async def process_and_create_graph(*, filename: str, content: bytes, size: int, mimetype: str, client_id: Optional[str] = None) -> Dict[str, Any]:
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
    total_tasks = len(chunks)
    total_with_db = total_tasks + 1  # 多加 1 代表入庫步驟
    await publish_progress(client_id, {"type": "progress", "done": 0, "total": total_with_db})

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

    print("[Graph] 開始併發提取實體與關係...")
    reset_key_index()

    # 速率與併發設定：可用環境變數覆蓋，預設保守以避免 429
    RPM = int(os.getenv("GRAPH_RPM", "8"))  # 每分鐘請求數（專案級）
    MAX_CONCURRENCY = min(int(os.getenv("GRAPH_MAX_CONCURRENCY", "8")), max(1, RPM))
    MAX_RETRIES_PER_TASK = int(os.getenv("GRAPH_MAX_RETRIES", "3"))
    all_extractions: List[Dict[str, Any]] = [None] * len(chunks)

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

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Token Bucket 速率限制器（專案級 RPM）
    token_bucket = asyncio.Queue(maxsize=RPM)
    for _ in range(RPM):
        token_bucket.put_nowait(1)

    async def _rate_limiter_worker(bucket: asyncio.Queue, rpm: int):
        interval = max(0.05, 60.0 / max(1, rpm))
        try:
            while True:
                await asyncio.sleep(interval)
                if not bucket.full():
                    bucket.put_nowait(1)
        except asyncio.CancelledError:
            pass

    refill_task = asyncio.create_task(_rate_limiter_worker(token_bucket, RPM))

    _completed = 0
    _completed_lock = asyncio.Lock()

    async def extract_one(idx: int, chunk: Dict[str, Any]) -> None:
        nonlocal _completed
        chunk_text = chunk.get("pageContent", "")
        meta = chunk.get("metadata", {}) or {}
        page_no = meta.get("pageNumber")
        print(f"[Graph] [Start] 任務 {idx + 1}/{total_tasks} (頁 {page_no}) len={len(chunk_text)}")
        attempts = 0
        backoff_base = 1.0
        while attempts < MAX_RETRIES_PER_TASK:
            model = get_graph_extraction_model()
            if not model:
                raise RuntimeError("所有 API Key 的配額都已耗盡，無法繼續處理。")
            prompt = prompt_template.replace("<<CHUNK>>", chunk_text)
            try:
                # 先消耗一個 Token，保證不超過 RPM
                await token_bucket.get()
                async with semaphore:
                    res = await _invoke_model(model, prompt)
                all_extractions[idx] = res or {"entities": [], "relationships": []}
                async with _completed_lock:
                    _completed += 1
                    print(f"[Graph] [OK] 任務 {idx + 1}/{total_tasks} (頁 {page_no}) -> 完成 {_completed}/{total_tasks}")
                    await publish_progress(client_id, {"type": "progress", "done": _completed, "total": total_with_db})
                return
            except Exception as err:
                err_text = str(err)
                retry_delay_match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_text)
                delay_seconds = backoff_base * (2 ** attempts)
                if retry_delay_match:
                    try:
                        delay_seconds = max(delay_seconds, float(retry_delay_match.group(1)))
                    except Exception:
                        pass
                print(f"[Graph] [Retry] 任務 {idx + 1}/{total_tasks} (頁 {page_no}) 第 {attempts + 1} 次，等待 {delay_seconds:.1f}s; 錯誤: {err}")
                if not switch_to_next_key():
                    attempts += 1
                await asyncio.sleep(delay_seconds)

        all_extractions[idx] = {"entities": [], "relationships": []}
        print(f"[Graph] [Fail] 任務 {idx + 1}/{total_tasks} (頁 {page_no})，已達最大重試")
        async with _completed_lock:
            _completed += 1
            await publish_progress(client_id, {"type": "progress", "done": _completed, "total": total_with_db})

    try:
        tasks = [asyncio.create_task(extract_one(i, c)) for i, c in enumerate(chunks)]
        await asyncio.gather(*tasks)
    finally:
        refill_task.cancel()
        with contextlib.suppress(Exception):
            await refill_task
    print(f"[Graph] 圖譜抽取完成：{sum(1 for e in all_extractions if e)}/{len(chunks)}")

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
    # 入庫完成，補發最終進度（達 100%）
    await publish_progress(client_id, {"type": "progress", "done": total_with_db, "total": total_with_db})
    await publish_progress(client_id, {"type": "finished", "fileId": result["fileId"], "chunks": len(chunks)})
    return {
        "message": f"成功為 {filename} 建立圖譜",
        "fileId": result["fileId"],
        "chunksCount": len(chunks),
        "entitiesCount": sum(len(c.get("entities", [])) for c in chunks_with_graph),
        "relationshipsCount": sum(len(c.get("relationships", [])) for c in chunks_with_graph),
        "isNew": True,
    }


async def process_website_and_create_graph(*, url: str, client_id: Optional[str] = None) -> Dict[str, Any]:
    """抓取網站並以 Markdown 轉換處理，建立圖譜。

    產生單一 Document，name=網站網域或最後一段 path，pageNumber 以分塊序數表示。
    """
    print(f"[Graph] 開始抓取網站: {url}")

    # 1) 抓取並轉為 Markdown
    docs_markdown = await load_markdown(url)
    if not docs_markdown:
        raise RuntimeError("抓取結果為空，無法處理")
    markdown = "\n\n".join([(getattr(docs_markdown[i], "page_content", "") or "") for i in range(len(docs_markdown))])
    if not markdown.strip():
        raise RuntimeError("抓取結果為空，無法處理")

    # 2) 準備單一 Document 的原始文件資訊
    size = len(markdown.encode("utf-8"))
    file_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()

    # 如果相同內容已存在，直接返回（避免重複建立）
    existing = neo4j_service.find_document_by_hash(file_hash)
    if existing:
        print(f"[Graph] 網頁 {url} 已存在。跳過處理。")
        return {"message": "內容已存在，跳過處理", "fileId": existing["id"], "isNew": False}

    # 3) 切分為 chunks
    docs = [{"pageContent": markdown, "metadata": {"source": url, "pageNumber": 1}}]
    chunks = []
    for d in docs:
        parts = text_splitter.split_text(d["pageContent"])
        for idx, chunk in enumerate(parts):
            chunks.append({"pageContent": chunk, "metadata": {"source": url, "pageNumber": idx + 1}})
    print(f"[Graph] 網頁內容分塊完成，共 {len(chunks)} 個塊。")

    if not chunks:
        return {"message": "內容沒有可處理的分塊", "isNew": True}

    total_tasks = len(chunks)
    total_with_db = total_tasks + 1
    await publish_progress(client_id, {"type": "progress", "done": 0, "total": total_with_db})

    if get_keys_count() == 0:
        raise RuntimeError("設定錯誤：請在 .env 中提供 GOOGLE_API_KEY_LIST。")

    # 4) 生成向量（分批，含換 key 重試）
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

    # 5) 圖譜抽取（沿用並發、速率限制與重試機制）
    print("[Graph] 開始併發提取實體與關係...")
    reset_key_index()

    RPM = int(os.getenv("GRAPH_RPM", "8"))
    MAX_CONCURRENCY = min(int(os.getenv("GRAPH_MAX_CONCURRENCY", "8")), max(1, RPM))
    MAX_RETRIES_PER_TASK = int(os.getenv("GRAPH_MAX_RETRIES", "3"))
    all_extractions: List[Dict[str, Any]] = [None] * len(chunks)

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

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    token_bucket = asyncio.Queue(maxsize=RPM)
    for _ in range(RPM):
        token_bucket.put_nowait(1)

    async def _rate_limiter_worker(bucket: asyncio.Queue, rpm: int):
        interval = max(0.05, 60.0 / max(1, rpm))
        try:
            while True:
                await asyncio.sleep(interval)
                if not bucket.full():
                    bucket.put_nowait(1)
        except asyncio.CancelledError:
            pass

    refill_task = asyncio.create_task(_rate_limiter_worker(token_bucket, RPM))

    _completed = 0
    _completed_lock = asyncio.Lock()

    async def extract_one(idx: int, chunk: Dict[str, Any]) -> None:
        nonlocal _completed
        chunk_text = chunk.get("pageContent", "")
        meta = chunk.get("metadata", {}) or {}
        page_no = meta.get("pageNumber")
        print(f"[Graph] [Start] 任務 {idx + 1}/{total_tasks} (頁 {page_no}) len={len(chunk_text)}")
        attempts = 0
        backoff_base = 1.0
        while attempts < MAX_RETRIES_PER_TASK:
            model = get_graph_extraction_model()
            if not model:
                raise RuntimeError("所有 API Key 的配額都已耗盡，無法繼續處理。")
            prompt = prompt_template.replace("<<CHUNK>>", chunk_text)
            try:
                await token_bucket.get()
                async with semaphore:
                    res = await _invoke_model(model, prompt)
                all_extractions[idx] = res or {"entities": [], "relationships": []}
                async with _completed_lock:
                    _completed += 1
                    print(f"[Graph] [OK] 任務 {idx + 1}/{total_tasks} (頁 {page_no}) -> 完成 {_completed}/{total_tasks}")
                    await publish_progress(client_id, {"type": "progress", "done": _completed, "total": total_with_db})
                return
            except Exception as err:
                err_text = str(err)
                retry_delay_match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err_text)
                delay_seconds = backoff_base * (2 ** attempts)
                if retry_delay_match:
                    try:
                        delay_seconds = max(delay_seconds, float(retry_delay_match.group(1)))
                    except Exception:
                        pass
                print(f"[Graph] [Retry] 任務 {idx + 1}/{total_tasks} (頁 {page_no}) 第 {attempts + 1} 次，等待 {delay_seconds:.1f}s; 錯誤: {err}")
                if not switch_to_next_key():
                    attempts += 1
                await asyncio.sleep(delay_seconds)

        all_extractions[idx] = {"entities": [], "relationships": []}
        print(f"[Graph] [Fail] 任務 {idx + 1}/{total_tasks} (頁 {page_no})，已達最大重試")
        async with _completed_lock:
            _completed += 1
            await publish_progress(client_id, {"type": "progress", "done": _completed, "total": total_with_db})

    try:
        tasks = [asyncio.create_task(extract_one(i, c)) for i, c in enumerate(chunks)]
        await asyncio.gather(*tasks)
    finally:
        refill_task.cancel()
        with contextlib.suppress(Exception):
            await refill_task

    # 6) 組裝與入庫
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

    document_data = {"name": url, "size": size, "hash": file_hash, "mimetype": "text/markdown"}
    result = neo4j_service.create_graph_from_document(document_data, chunks_with_graph)
    print(f"[Graph] 檔案 {url} 已成功建立圖譜，文件節點 ID: {result['fileId']}")
    await publish_progress(client_id, {"type": "progress", "done": total_with_db, "total": total_with_db})
    await publish_progress(client_id, {"type": "finished", "fileId": result["fileId"], "chunks": len(chunks)})
    return {
        "message": f"成功為 {url} 建立圖譜",
        "fileId": result["fileId"],
        "chunksCount": len(chunks),
        "entitiesCount": sum(len(c.get("entities", [])) for c in chunks_with_graph),
        "relationshipsCount": sum(len(c.get("relationships", [])) for c in chunks_with_graph),
        "isNew": True,
    }
