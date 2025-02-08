from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
import os
from dotenv import load_dotenv
import time
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Define adapters for different LLMs
class ChatAdapterFactory:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY no está configurado.")
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")

        self.adapters = {
            "ollama": ChatOllama(
                model="llama3.2",
                temperature=0,
            ),
            "openai": ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=api_key
            )
        }

    def get_adapter(self, model_name: str):
        adapter = self.adapters.get(model_name, self.adapters["ollama"])
        logger.info(f"🔄 Usando adaptador para modelo: {model_name}")
        return adapter

    async def invoke(self, messages):
        start_time = time.perf_counter()
        try:
            logger.info("🔄 Iniciando llamada al modelo...")
            
            # 1. Preparación del adaptador y configuración
            prep_start = time.perf_counter()
            adapter = self.get_adapter("ollama")
            logger.info("�� Preparando llamada al modelo")
            prep_time = time.perf_counter() - prep_start
            
            # 2. Llamada a la API
            api_start = time.perf_counter()
            
            # 2.1 Preparación de mensajes para Ollama
            format_start = time.perf_counter()
            formatted_messages = self._format_messages(messages)
            format_time = time.perf_counter() - format_start
            logger.info(f"📝 Formato de mensajes: {format_time:.3f}s")
            
            # 2.2 Conexión con el servidor Ollama
            connect_start = time.perf_counter()
            # La conexión HTTP se establece aquí
            connect_time = time.perf_counter() - connect_start
            logger.info(f"🔌 Conexión establecida: {connect_time:.3f}s")
            
            # 2.3 Generación de respuesta
            generate_start = time.perf_counter()
            response = await adapter.agenerate([messages])
            generate_time = time.perf_counter() - generate_start
            logger.info(f"⚡ Generación de respuesta: {generate_time:.3f}s")
            
            api_time = time.perf_counter() - api_start
            
            # 3. Procesamiento de la respuesta
            process_start = time.perf_counter()
            content = response.generations[0][0].text
            ai_message = AIMessage(content=content)
            process_time = time.perf_counter() - process_start
            
            total_time = time.perf_counter() - start_time
            logger.info(f"""✅ Llamada al modelo completada:
                - Preparación del adaptador: {prep_time:.3f}s
                - Formato de mensajes: {format_time:.3f}s
                - Conexión al servidor: {connect_time:.3f}s
                - Generación de respuesta: {generate_time:.3f}s
                - Procesamiento de respuesta: {process_time:.3f}s
                - Tiempo total API: {api_time:.3f}s
                - Tiempo total: {total_time:.3f}s
                
                Métricas adicionales:
                - Tamaño de entrada: {len(str(messages))} caracteres
                - Tamaño de respuesta: {len(content)} caracteres
                """)
            
            return ai_message
            
        except Exception as e:
            error_time = time.perf_counter() - start_time
            logger.error(f"""❌ Error en llamada al modelo:
                - Tiempo hasta error: {error_time:.3f}s
                - Tipo de error: {type(e).__name__}
                - Mensaje: {str(e)}
                - Contexto: Modelo Ollama, tamaño entrada: {len(str(messages))} caracteres""")
            raise

    def _format_messages(self, messages):
        """Formatea los mensajes para el modelo de Ollama"""
        # Implementar formateo específico si es necesario
        return messages