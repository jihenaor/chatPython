from adapters.secondary.langchain.chat_adapter_factory import ChatAdapterFactory
from typing import Dict, Optional
import logging
import time
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from adapters.secondary.vector_store.vector_store_singleton import VectorStoreManager

logger = logging.getLogger(__name__)

class AskQuestionUseCase:
    def __init__(self):
        self.vector_store = VectorStoreManager().vector_store
        self.chat_adapter = ChatAdapterFactory()
        
    async def execute(self, query: str, model: str = "ollama", 
                     max_tokens: int = 2048) -> str:
        """
        Ejecuta el caso de uso de pregunta-respuesta usando RAG
        
        Args:
            query: Pregunta del usuario
            model: Modelo a utilizar (default: ollama)
            max_tokens: Límite de tokens para la respuesta
        """
        start_time = time.perf_counter()
        metrics = {}
        
        try:
            # 1. Recuperación de documentos relevantes
            retrieval_start = time.perf_counter()
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 2},
                search_type="similarity"
            )
            retrieved_docs = await retriever.ainvoke(query)
            retrieval_time = time.perf_counter() - retrieval_start
            
            metrics["retrieval"] = {
                "time": retrieval_time,
                "docs_count": len(retrieved_docs),
                "total_chars": sum(len(doc.page_content) for doc in retrieved_docs)
            }
            
            # 2. Preparación del prompt
            prompt_start = time.perf_counter()
            context = "\n".join(doc.page_content for doc in retrieved_docs)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Eres un asistente experto que responde en español a preguntas basándose en el contexto proporcionado.
                    Usa solo la información del contexto para responder.
                    Si la información no está en el contexto, indica que no puedes responder.
                    Mantén las respuestas concisas y directas.
                    
                    Contexto: {context}"""),
                ("human", "{question}")
            ])
            
            messages = prompt.invoke({
                "context": context,
                "question": query
            }).messages
            
            prompt_time = time.perf_counter() - prompt_start
            metrics["prompt"] = {
                "time": prompt_time,
                "context_length": len(context),
                "total_messages": len(messages)
            }
            
            # 3. Generación de respuesta
            generation_start = time.perf_counter()
            response = await self.chat_adapter.invoke(messages)

            generation_time = time.perf_counter() - generation_start

            metrics["generation"] = {
                "time": generation_time,
                "response_length": len(response.content)
            }
            
            # 4. Métricas finales
            total_time = time.perf_counter() - start_time
            logger.info(f"""📊 Métricas de ejecución:
                Recuperación de documentos ({metrics['retrieval']['time']:.3f}s):
                - Documentos recuperados: {metrics['retrieval']['docs_count']}
                - Caracteres totales: {metrics['retrieval']['total_chars']}
                
                Preparación del prompt ({metrics['prompt']['time']:.3f}s):
                - Longitud del contexto: {metrics['prompt']['context_length']}
                - Total de mensajes: {metrics['prompt']['total_messages']}
                
                Generación de respuesta ({metrics['generation']['time']:.3f}s):
                - Longitud de respuesta: {metrics['generation']['response_length']}
                
                Tiempo total: {total_time:.3f}s
            """)
            
            return response.content
            
        except Exception as e:
            error_time = time.perf_counter() - start_time
            logger.error(f"""❌ Error en el caso de uso:
                - Tiempo hasta error: {error_time:.3f}s
                - Tipo: {type(e).__name__}
                - Mensaje: {str(e)}
                - Query: {query}
                - Modelo: {model}""")
            raise
