from fastapi import FastAPI
from contextlib import asynccontextmanager
from adapters.secondary.vector_store.vector_store_singleton import VectorStoreManager
from application.services.document_loader import load_documents
from adapters.primary.api.chat_controller import router
import logging
import time

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_time = time.perf_counter()
    logger.info("🚀 Iniciando aplicación...")
    try:
        # Cargar documentos
        docs_start = time.perf_counter()
        documents = load_documents()
        docs_time = time.perf_counter() - docs_start
        logger.info(f"📚 Documentos cargados en {docs_time:.2f} segundos")
        
        if documents:
            store_start = time.perf_counter()
            vector_store = VectorStoreManager().vector_store
            store_time = time.perf_counter() - store_start
            logger.info(f"✅ Vector store inicializado en {store_time:.2f} segundos")
        else:
            logger.warning("⚠️ No se inicializó el vector store porque no hay documentos")
        
        total_time = time.perf_counter() - start_time
        logger.info(f"🎉 Aplicación iniciada en {total_time:.2f} segundos")
        yield
    finally:
        logger.info("👋 Limpiando recursos...")

app = FastAPI(lifespan=lifespan)

# Include the router for the chat endpoints
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
