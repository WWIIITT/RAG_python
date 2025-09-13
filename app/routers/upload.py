from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

from app.services.graph_document_service import process_and_create_graph, process_website_and_create_graph

router = APIRouter(prefix="", tags=["upload"])


@router.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...), clientId: Optional[str] = Query(default=None)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail={"error": "沒有檔案被上傳"})

        results = []
        for f in files:
            data = await f.read()
            try:
                res = await process_and_create_graph(filename=f.filename, content=data, size=f.size or 0, mimetype=f.content_type or "application/octet-stream", client_id=clientId)
            except Exception as e:
                res = {"error": True, "originalname": f.filename, "message": str(e)}
            results.append(res)

        return {"message": f"處理完成 {len(results)} 個檔案。", "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "伺服器內部錯誤", "details": str(e)})


class UploadLinkBody(BaseModel):
    url: str


@router.post("/upload-link")
async def upload_link(body: UploadLinkBody, clientId: Optional[str] = Query(default=None)):
    try:
        if not body or not body.url or not body.url.strip():
            raise HTTPException(status_code=400, detail={"error": "url 缺失"})

        url = body.url.strip()

        res = await process_website_and_create_graph(url=url, client_id=clientId)


        return {"message": "連結處理完成", "result": res}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "伺服器內部錯誤", "details": str(e)})
