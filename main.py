from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.config import get_settings
from app.services import neo4j_service
from app.routers.upload import router as upload_router
from app.routers.files_neo4j import router as files_router
from app.routers.query_stream import router as query_router
from app.routers.sse import router as sse_router


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title="RAG FastAPI", version="0.1.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers (/neo4j/* to be compatible with old Express base path)
    app.include_router(upload_router)
    app.include_router(files_router , prefix="/neo4j")
    app.include_router(query_router, prefix="/api")
    app.include_router(sse_router)

    # Optional root (keep existing behavior for direct calls)
    app.include_router(query_router)

    @app.on_event("startup")
    def on_startup():
        # 檢查/建立 Neo4j 索引
        neo4j_service.setup_vector_index()

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, reload=True)