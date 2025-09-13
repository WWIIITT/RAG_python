import asyncio
from typing import Dict, Any, Optional


_queues: Dict[str, asyncio.Queue] = {}
_lock = asyncio.Lock()


async def get_queue(client_id: str) -> asyncio.Queue:
    async with _lock:
        q = _queues.get(client_id)
        if q is None:
            q = asyncio.Queue()
            _queues[client_id] = q
        return q


async def remove_queue(client_id: str) -> None:
    async with _lock:
        _queues.pop(client_id, None)


async def publish_progress(client_id: Optional[str], data: Dict[str, Any]) -> None:
    if not client_id:
        return
    async with _lock:
        q = _queues.get(client_id)
    if q is None:
        return
    try:
        q.put_nowait(data)
    except asyncio.QueueFull:
        # 忽略無法投遞
        pass


