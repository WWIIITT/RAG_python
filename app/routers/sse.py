from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
import asyncio

from app.services.progress_bus import get_queue, remove_queue


router = APIRouter(prefix="/sse", tags=["sse"])


@router.get("/progress")
async def sse_progress(request: Request, clientId: str):
    if not clientId:
        raise HTTPException(status_code=400, detail={"error": "clientId 缺失"})

    queue = await get_queue(clientId)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # keep-alive
                    yield "data: {\"type\": \"keepalive\"}\n\n"
        finally:
            await remove_queue(clientId)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


