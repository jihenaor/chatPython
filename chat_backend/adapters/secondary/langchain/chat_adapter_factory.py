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
            
            # Tiempo de preparación de la llamada
            prep_start = time.perf_counter()
            adapter = self.get_adapter("ollama")  # Por defecto usamos ollama
            logger.info("📝 Preparando llamada al modelo")
            prep_time = time.perf_counter() - prep_start
            
            # Tiempo de la llamada API
            api_start = time.perf_counter()
            response = await adapter.agenerate([messages])
            api_time = time.perf_counter() - api_start
            
            total_time = time.perf_counter() - start_time
            logger.info(f"""✅ Llamada al modelo completada:
                - Preparación: {prep_time:.3f}s
                - Llamada API: {api_time:.3f}s
                - Total: {total_time:.3f}s""")
            
            # Extraer el contenido de la respuesta
            content = response.generations[0][0].text
            return AIMessage(content=content)
            
        except Exception as e:
            error_time = time.perf_counter() - start_time
            logger.error(f"❌ Error en llamada al modelo (tiempo: {error_time:.3f}s): {str(e)}")
            raise