from langchain_community.document_loaders import PyPDFLoader
import os
import logging
import shutil

logger = logging.getLogger(__name__)

def load_documents():
    """Carga los documentos PDF una sola vez al inicio"""
    try:
        # Obtener la ruta base del proyecto
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        docs_directory = os.path.join(base_dir, 'docs')
        
        # Crear el directorio docs si no existe
        if not os.path.exists(docs_directory):
            logger.info(f"Creando directorio docs en: {docs_directory}")
            os.makedirs(docs_directory)
            
            # Mover el archivo doc2.pdf al directorio docs si existe
            old_path = os.path.join(base_dir, 'doc2.pdf')
            if os.path.exists(old_path):
                new_path = os.path.join(docs_directory, 'doc2.pdf')
                logger.info(f"Moviendo {old_path} a {new_path}")
                shutil.move(old_path, new_path)
        
        documents = []
        pdf_files = [f for f in os.listdir(docs_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No se encontraron archivos PDF en {docs_directory}")
            return documents
            
        for file in pdf_files:
            file_path = os.path.join(docs_directory, file)
            logger.info(f"Cargando documento: {file}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
        logger.info(f"Total de documentos cargados: {len(documents)}")
        return documents
        
    except Exception as e:
        logger.error(f"Error cargando documentos: {str(e)}")
        raise 