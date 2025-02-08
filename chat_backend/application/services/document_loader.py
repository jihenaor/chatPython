import os
import logging
from langchain_community.document_loaders import PyPDFLoader
import shutil

logger = logging.getLogger(__name__)

def load_documents():
    """Carga los documentos PDF desde el directorio docs"""
    try:
        # Obtener la ruta base del proyecto
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        docs_directory = os.path.join(base_dir, 'docs')
        
        if not os.path.exists(docs_directory):
            logger.warning(f"El directorio {docs_directory} no existe")
            return []
            
        documents = []
        pdf_files = [f for f in os.listdir(docs_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No se encontraron archivos PDF en {docs_directory}")
            return documents
            
        for file in pdf_files:
            file_path = os.path.join(docs_directory, file)
            logger.info(f"📚 Cargando documento: {file}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
        logger.info(f"✅ Total de documentos cargados: {len(documents)}")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Error cargando documentos: {str(e)}")
        raise 