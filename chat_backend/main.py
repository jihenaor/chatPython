from fastapi import FastAPI
from contextlib import asynccontextmanager
from adapters.secondary.vector_store.vector_store_singleton import VectorStoreManager
from adapters.primary.api.chat_controller import router
import logging
import time

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_time = time.perf_counter()
    logger.info("🚀 Iniciando aplicación...")
    try:
        # Inicializar el VectorStoreManager
        logger.info("📚 Inicializando Vector Store Manager...")
        await VectorStoreManager.initialize()
        
        # Ahora podemos obtener la instancia inicializada
        vector_store = VectorStoreManager().vector_store
        logger.info("✅ Vector Store inicializado correctamente")
        
        total_time = time.perf_counter() - start_time
        logger.info(f"🎉 Aplicación iniciada en {total_time:.3f}s")
        yield
    finally:
        logger.info("👋 Limpiando recursos...")

app = FastAPI(lifespan=lifespan)

# Include the router for the chat endpoints
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
