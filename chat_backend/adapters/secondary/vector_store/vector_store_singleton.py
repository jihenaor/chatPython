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
    _instance: Optional['VectorStoreManager'] = None
    _vector_store = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("🚀 Creando nueva instancia de VectorStoreManager")
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
        return cls._instance
    
    def initialize_with_documents(self, documents):
        """Inicializa la base de datos vectorial con documentos"""
        if not documents:
            logger.warning("⚠️ No hay documentos para inicializar el vector store")
            return

        if not self._initialized:
            start_time = time.perf_counter()
            try:
                logger.info("🔄 Iniciando proceso de inicialización del vector store...")
                
                # Inicializar embeddings
                embed_start = time.perf_counter()
                logger.info("📊 Configurando Ollama Embeddings...")
                embeddings = OllamaEmbeddings(
                    model="llama3.2",
                    base_url="http://localhost:11434"
                )
                embed_time = time.perf_counter() - embed_start
                logger.info(f"✅ Embeddings configurados en {embed_time:.2f} segundos")
                
                # Configurar directorio de persistencia
                persist_directory = os.path.join(os.path.dirname(__file__), '../../../vector_store')
                logger.info(f"📁 Usando directorio de persistencia: {persist_directory}")
                os.makedirs(persist_directory, exist_ok=True)
                
                # Inicializar Chroma
                chroma_start = time.perf_counter()
                logger.info("🔨 Inicializando base de datos Chroma...")
                self._vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=embeddings
                )
                chroma_time = time.perf_counter() - chroma_start
                logger.info(f"✅ Chroma inicializado en {chroma_time:.2f} segundos")
                
                # Agregar documentos
                docs_start = time.perf_counter()
                logger.info(f"📝 Agregando {len(documents)} documentos al vector store...")
                self._vector_store.add_documents(documents)
                docs_time = time.perf_counter() - docs_start
                logger.info(f"✅ Documentos agregados en {docs_time:.2f} segundos")
                
                self._initialized = True
                total_time = time.perf_counter() - start_time
                logger.info(f"🎉 Vector store inicializado exitosamente en {total_time:.2f} segundos")
                logger.info(f"""📊 Desglose de tiempos:
                    - Configuración de embeddings: {embed_time:.2f}s
                    - Inicialización de Chroma: {chroma_time:.2f}s
                    - Carga de documentos: {docs_time:.2f}s
                    - Total: {total_time:.2f}s""")
                
            except Exception as e:
                logger.error(f"❌ Error en la inicialización: {str(e)}")
                raise
    
    @property
    def vector_store(self):
        """Retorna la instancia de la base de datos vectorial"""
        if not self._initialized:
            start_time = time.perf_counter()
            logger.info("🔄 Iniciando carga lazy del vector store...")
            documents = load_documents()
            self.initialize_with_documents(documents)
            total_time = time.perf_counter() - start_time
            logger.info(f"✅ Carga lazy completada en {total_time:.2f} segundos")
            
        if self._vector_store is None:
            raise RuntimeError("❌ Vector store no está inicializado")
            
        return self._vector_store 