from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, List
import json
from datetime import datetime, timezone

from app.services.ai_service import generate_answer_with_langchain
from app.services import neo4j_service
from app.utils.api_key_manager import get_query_entity_extraction_model, get_embedding_model

router = APIRouter(prefix="", tags=["query"]) 


async def sse_event_stream(question: str, selected_file_ids: List[str]) -> AsyncGenerator[bytes, None]:
    def make_progress(message: str, data=None, event_type: str = "progress") -> bytes:
        payload = {"type": event_type, "message": message, "data": data, "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
        return (json.dumps(payload) + "\n").encode("utf-8")

    yield make_progress("開始處理查詢...", {"question": question})

    if not selected_file_ids:
        yield make_progress("⚠️ 未選擇任何文件，將使用全部文件進行檢索。")
        return

    # 圖譜檢索（實體提取）
    try:
        yield make_progress("🔍 [圖譜檢索] 正在提取實體...", event_type="graphProgress")
        entity_model = get_query_entity_extraction_model()
        if entity_model is None:
            # No API key configured for Google models; skip entity extraction to avoid ADC fallback
            yield make_progress("⚠️ [圖譜檢索] 已跳過：未配置 Google API Key，無法執行實體提取。", event_type="graph")
            graph_results = []
        else:
            extraction_prompt = f"從以下問題中提取出最關鍵的人物、地點、組織或概念等實體。只返回實體名稱。\n問題: \"{question}\""
            extraction_result = await entity_model.ainvoke(extraction_prompt)
            entities = [e for e in (extraction_result.get("entities") or []) if e]
            yield make_progress(f"[圖譜檢索] 提取到實體: [{', '.join(entities)}]，正在檢索...", event_type="progress")
            graph_results = neo4j_service.retrieve_context_by_entities(entities, selected_file_ids)
            yield make_progress(f"✅[圖譜檢索] 完成！找到 {len(graph_results)} 個相關結果", len(graph_results), event_type="graph")
    except Exception as e:
        yield make_progress(f"❌[圖譜檢索] 發生錯誤: {str(e)}", 0, event_type="graph")
        graph_results = []

    # 向量檢索
    try:
        yield make_progress("🔍 [向量檢索] 正在生成查詢向量...", event_type="vectorProgress")
        embeddings = get_embedding_model()
        query_vector = await embeddings.aembed_query(question.strip())
        yield make_progress("[向量檢索] 正在檢索圖譜中的向量...")
        vector_results = neo4j_service.retrieve_graph_context(query_vector, 20, selected_file_ids)
        yield make_progress(f"✅[向量檢索] 完成！找到 {len(vector_results)} 個相關結果", len(vector_results), event_type="vector")
    except Exception as e:
        print(e)
        yield make_progress(f"❌[向量檢索] 發生錯誤: {str(e)}", 0, event_type="vector")
        vector_results = []

    # 全文檢索
    try:
        yield make_progress("🔍 [全文檢索] 正在執行全文檢索...", event_type="fulltextProgress")
        fulltext_results = neo4j_service.retrieve_context_by_keywords(question, selected_file_ids)
        yield make_progress(f"✅[全文檢索] 完成！找到 {len(fulltext_results)} 個相關結果", len(fulltext_results), event_type="fulltext")
    except Exception as e:
        yield make_progress(f"❌[全文檢索] 發生錯誤: {str(e)}", 0, event_type="fulltext")
        fulltext_results = []

    combined = {}
    for doc in [*graph_results, *vector_results, *fulltext_results]:
        if doc and doc.get("chunkId"):
            combined[doc["chunkId"]] = doc
    initial_docs = list(combined.values())
    yield make_progress(f"✅ 融合去重完成！共得到 {len(initial_docs)} 個候選文檔", len(initial_docs), event_type="merge")

    if not initial_docs:
        result_payload = {"type": "result", "question": question, "answer": "抱歉，在您指定的文件中找不到任何相關資訊。", "answer_with_citations": [], "raw_sources": [], "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
        yield (json.dumps(result_payload) + "\n").encode("utf-8")
        return

    formatted_context = []
    file_ids = []
    chunk_ids = []
    for idx, doc in enumerate(initial_docs, start=1):
        md = doc
        content = doc.get("content") or doc.get("text")
        source_info = f"source_file: \"{md.get('source')}\", page_number: \"{md.get('page')}\", file_id: \"{md.get('fileId')}\", file_chunk_id: \"{md.get('chunkId')}\""
        formatted_context.append(f"[上下文來源 {idx}]\n{source_info} \n內容: \"\"\"\n{content}\n\"\"\"")
        file_ids.append(md.get("fileId"))
        chunk_ids.append(md.get("chunkId"))

    ai_response = await generate_answer_with_langchain("\n\n".join(formatted_context), question.strip(), file_ids, chunk_ids)

    sources = [
        {
            "content": (d.get("content") or d.get("text")),
            "source": d.get("source") or "未知來源",
            "pageNumber": d.get("page") or "未知頁碼",
            "score": d.get("score") or d.get("relevance_score"),
            "fileId": d.get("fileId"),
            "chunkId": d.get("chunkId"),
        }
        for d in initial_docs
    ]

    result_payload = {"type": "result", "question": question.strip(), "answer": ai_response.get("answer"), "answer_with_citations": ai_response.get("answer_with_citations") or [], "raw_sources": sources, "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    yield (json.dumps(result_payload) + "\n").encode("utf-8")


@router.post("/query-stream")
async def query_stream(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    selected_file_ids = body.get("selectedFileIds") or []
    if not question:
        raise HTTPException(status_code=400, detail={"error": "請提供查詢問題"})
    if len(selected_file_ids) == 0:
        raise HTTPException(status_code=400, detail={"error": "請至少選擇一個文件進行檢索"})

    return StreamingResponse(sse_event_stream(question, selected_file_ids), media_type="text/plain; charset=utf-8")
