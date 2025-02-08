from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os
import logging
import time
from typing import Optional
from application.services.document_loader import load_documents

# Configurar logging con formato que incluye timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    _instance = None
    _vector_store = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    async def initialize(cls):
        """Inicialización asíncrona al inicio de la aplicación"""
        if cls._vector_store is None:
            try:
                logger.info("🔄 Inicializando vector store persistente...")
                start_time = time.perf_counter()
                
                # 1. Carga de documentos (una sola vez)
                docs_start = time.perf_counter()
                documents = load_documents()
                docs_time = time.perf_counter() - docs_start
                logger.info(f"📚 Documentos cargados en {docs_time:.3f}s")
                
                # 2. Configuración de embeddings
                embed_start = time.perf_counter()
                embeddings = OllamaEmbeddings(model="llama3.2")
                
                # 3. Configuración de Chroma persistente
                persist_directory = os.path.join(os.path.dirname(__file__), '../../../vector_store')
                os.makedirs(persist_directory, exist_ok=True)
                
                # 4. Inicialización de Chroma con persistencia
                cls._vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=persist_directory
                )
                embed_time = time.perf_counter() - embed_start
                
                total_time = time.perf_counter() - start_time
                logger.info(f"""✅ Vector store persistente inicializado:
                    - Carga de documentos: {docs_time:.3f}s
                    - Creación de embeddings: {embed_time:.3f}s
                    - Tiempo total: {total_time:.3f}s
                    - Documentos procesados: {len(documents)}
                    - Directorio de persistencia: {persist_directory}""")
                
            except Exception as e:
                logger.error(f"❌ Error inicializando vector store: {str(e)}")
                raise
    
    @property
    def vector_store(self):
        """Retorna la instancia de la base de datos vectorial"""
        if self._vector_store is None:
            raise RuntimeError("❌ Vector store no está inicializado. Llame a initialize() primero.")
        return self._vector_store 