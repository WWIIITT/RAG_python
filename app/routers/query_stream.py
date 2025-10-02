from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, List
import json
from datetime import datetime, timezone
import asyncio

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

    # --- 並行執行三種檢索（透過事件佇列轉發進度） ---
    event_queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def enqueue_progress(message: str, data=None, event_type: str = "progress") -> None:
        await event_queue.put(make_progress(message, data, event_type))

    async def run_graph_search() -> List[dict]:
        # Graph search disabled - only using vector search now
        try:
            await enqueue_progress("⚠️ [圖譜檢索] 已跳過：圖譜檢索功能已停用，僅使用向量檢索。", 0, event_type="graph")
            return []
            
            # Original graph search code (disabled):
            # await enqueue_progress("🔍 [圖譜檢索] 正在提取實體...", event_type="graphProgress")
            # entity_model = get_query_entity_extraction_model()
            # if entity_model is None:
            #     await enqueue_progress("⚠️ [圖譜檢索] 已跳過：未配置 Google API Key，無法執行實體提取。", 0, event_type="graph")
            #     return []
            # extraction_prompt = f"從以下問題中提取出最關鍵的人物、地點、組織或概念等實體。只返回實體名稱。\n問題: \"{question}\""
            # extraction_result = await entity_model.ainvoke(extraction_prompt)
            # print(f"[Graph] [OK] 提取到實體: [{', '.join(extraction_result.get('entities') or [])}]")
            # entities = [e for e in (extraction_result.get("entities") or []) if e]
            # await enqueue_progress(f"[圖譜檢索] 提取到實體: [{', '.join(entities)}]，正在檢索...", event_type="progress")
            # graph_results_local = await asyncio.to_thread(
            #     neo4j_service.retrieve_context_by_entities, entities, selected_file_ids
            # )
            # await enqueue_progress(
            #     f"✅[圖譜檢索] 完成！找到 {len(graph_results_local)} 個相關結果",
            #     len(graph_results_local),
            #     event_type="graph",
            # )
            # return graph_results_local
        except Exception as e:
            await enqueue_progress(f"❌[圖譜檢索] 發生錯誤: {str(e)}", 0, event_type="graph")
            return []

    async def run_vector_search() -> List[dict]:
        try:
            await enqueue_progress("🔍 [向量檢索] 正在生成查詢向量...", event_type="vectorProgress")
            embeddings = get_embedding_model()
            query_vector = await embeddings.aembed_query(question.strip())
            await enqueue_progress("[向量檢索] 正在檢索圖譜中的向量...")
            vector_results_local = await asyncio.to_thread(
                neo4j_service.retrieve_graph_context, query_vector, 20, selected_file_ids
            )
            await enqueue_progress(
                f"✅[向量檢索] 完成！找到 {len(vector_results_local)} 個相關結果",
                len(vector_results_local),
                event_type="vector",
            )
            return vector_results_local
        except Exception as e:
            await enqueue_progress(f"❌[向量檢索] 發生錯誤: {str(e)}", 0, event_type="vector")
            return []

    async def run_fulltext_search() -> List[dict]:
        try:
            await enqueue_progress("🔍 [全文檢索] 正在執行全文檢索...", event_type="fulltextProgress")
            fulltext_results_local = await asyncio.to_thread(
                neo4j_service.retrieve_context_by_keywords, question, selected_file_ids
            )
            await enqueue_progress(
                f"✅[全文檢索] 完成！找到 {len(fulltext_results_local)} 個相關結果",
                len(fulltext_results_local),
                event_type="fulltext",
            )
            return fulltext_results_local
        except Exception as e:
            await enqueue_progress(f"❌[全文檢索] 發生錯誤: {str(e)}", 0, event_type="fulltext")
            return []

    graph_task = asyncio.create_task(run_graph_search())
    vector_task = asyncio.create_task(run_vector_search())
    fulltext_task = asyncio.create_task(run_fulltext_search())

    # 持續轉發子任務的進度事件，直到三任務結束且佇列清空
    while True:
        if graph_task.done() and vector_task.done() and fulltext_task.done() and event_queue.empty():
            break
        try:
            event = event_queue.get_nowait()
            yield event
            event_queue.task_done()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.02)

    # 取得子任務結果
    graph_results = graph_task.result() if not graph_task.cancelled() else []
    vector_results = vector_task.result() if not vector_task.cancelled() else []
    fulltext_results = fulltext_task.result() if not fulltext_task.cancelled() else []

    # 使用 RRF (Reciprocal Rank Fusion) 算法融合多個檢索結果
    def reciprocal_rank_fusion(results_list: List[List[dict]], k: int = 60) -> List[dict]:
        """
        使用 RRF 算法融合多個檢索結果
        
        RRF 公式: score(d) = Σ 1/(k + rank_r(d))
        - 在多個檢索源中都出現的文檔會得到更高分數
        - 不需要歸一化不同檢索源的分數
        
        Args:
            results_list: 多個檢索源的結果列表（已按相關性排序）
            k: RRF 常數，用於平滑分數（默認 60）
        
        Returns:
            按 RRF 分數排序的文檔列表
        """
        rrf_scores = {}
        
        # 遍歷每個檢索源
        for results in results_list:
            if not results:
                continue
            # 遍歷該檢索源的每個文檔及其排名（從 1 開始）
            for rank, doc in enumerate(results, start=1):
                chunk_id = doc.get("chunkId")
                if not chunk_id:
                    continue
                
                # 計算 RRF 分數：1 / (k + rank)
                score = 1.0 / (k + rank)
                
                # 累加到該文檔的總 RRF 分數
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = {
                        "doc": doc,
                        "rrf_score": 0.0,
                        "sources": []
                    }
                rrf_scores[chunk_id]["rrf_score"] += score
                rrf_scores[chunk_id]["sources"].append({"rank": rank, "score": score})
        
        # 按 RRF 分數降序排序
        sorted_docs = sorted(
            rrf_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )
        
        # 返回文檔列表（添加 RRF 分數到文檔中）
        return [
            {**item["doc"], "rrf_score": round(item["rrf_score"], 4)}
            for item in sorted_docs
        ]
    
    # 使用 RRF 融合三種檢索結果
    initial_docs = reciprocal_rank_fusion([
        graph_results,
        vector_results,
        fulltext_results
    ], k=60)
    
    yield make_progress(
        f"✅ RRF 融合完成！共得到 {len(initial_docs)} 個候選文檔（已按相關性重排序）", 
        len(initial_docs), 
        event_type="merge"
    )

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

    yield make_progress("🔍 [AI 回應] 正在生成回答...", event_type="aiProgress")
    ai_response = await generate_answer_with_langchain("\n\n".join(formatted_context), question.strip(), file_ids, chunk_ids)
    print(ai_response)

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

    yield make_progress("✅ [AI 回應] 完成！", event_type="aiProgress")
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
