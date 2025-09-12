from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

from app.services.graph_document_service import process_and_create_graph

router = APIRouter(prefix="", tags=["upload"])


@router.post("/upload-multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    try:
        if not files:
            raise HTTPException(status_code=400, detail={"error": "沒有檔案被上傳"})

        results = []
        for f in files:
            data = await f.read()
            try:
                res = await process_and_create_graph(filename=f.filename, content=data, size=f.size or 0, mimetype=f.content_type or "application/octet-stream")
            except Exception as e:
                res = {"error": True, "originalname": f.filename, "message": str(e)}
            results.append(res)

        return {"message": f"處理完成 {len(results)} 個檔案。", "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "伺服器內部錯誤", "details": str(e)})
