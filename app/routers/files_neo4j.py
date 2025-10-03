from fastapi import APIRouter, HTTPException

from app.services import neo4j_service

router = APIRouter(prefix="", tags=["files"])


@router.get("/files")
async def get_files():
    try:
        files = neo4j_service.get_files_list()
        return {"message": "檔案清單獲取成功", "files": files, "total": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "獲取檔案清單失敗", "details": str(e)})


@router.delete("/files/{file_id}")
async def delete_file(file_id: str):
    try:
        result = neo4j_service.delete_file(file_id)
        return {"message": result["message"], "success": True, "deletedFile": result["deletedFile"]}
    except Exception as e:
        if str(e) == "檔案不存在":
            raise HTTPException(status_code=404, detail={"error": "檔案不存在", "details": str(e)})
        raise HTTPException(status_code=500, detail={"error": "刪除檔案失敗", "details": str(e)})


@router.get("/files/{file_id}")
async def get_file_details(file_id: str):
    try:
        details = neo4j_service.get_specific_file(file_id)
        return {"message": "檔案詳細資訊獲取成功", "file": details["file"], "chunks": details["chunks"]}
    except Exception as e:
        if str(e) == "檔案不存在":
            raise HTTPException(status_code=404, detail={"error": "檔案不存在", "details": str(e)})
        raise HTTPException(status_code=500, detail={"error": "獲取檔案詳細資訊失敗", "details": str(e)})


@router.get("/chunks/{chunk_id}/source-details")
async def get_chunk_source_details(chunk_id: str):
    try:
        details = neo4j_service.get_source_details_by_chunk_id(chunk_id)
        return {"message": "來源資訊獲取成功", "details": details}
    except Exception as e:
        if str(e) == "找不到 chunk":
            raise HTTPException(status_code=404, detail={"error": "找不到對應的 chunk", "details": str(e)})
        raise HTTPException(status_code=500, detail={"error": "獲取來源資訊失敗", "details": str(e)})