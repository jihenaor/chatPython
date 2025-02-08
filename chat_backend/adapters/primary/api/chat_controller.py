from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from typing import Dict, Optional
import os
import logging
import time

from application.use_cases.ask_question import AskQuestionUseCase

logger = logging.getLogger(__name__)

class QuestionRequest(BaseModel):
    session_id: str  # Unique identifier for the session
    query: str  # The actual content of the query
    model: str = "ollama"  # Optional parameter to specify the LLM
    additional_params: Optional[Dict[str, str]] = None  # Additional parameters for context (e.g., input_language, output_language)


router = APIRouter()

# Define a mapping of models to their max token limits
MODEL_MAX_TOKENS = {
    "ollama": 2048,
    "gpt-3.5": 4096,
    "gpt-4": 8192,
}

@router.post("/chat")
async def ask_question(request: QuestionRequest):
    start_time = time.perf_counter()
    logger.info(f"""📝 Nueva solicitud de chat:
        - Session ID: {request.session_id}
        - Modelo: {request.model}
        - Query: {request.query[:50]}...
        - Params adicionales: {request.additional_params}""")
    
    try:
        # Configuración
        config_start = time.perf_counter()
        use_trim = os.getenv("USE_TRIM", "false").lower() == "true"
        max_tokens = MODEL_MAX_TOKENS.get(request.model, 2048)
        logger.info(f"""⚙️ Configuración:
            - Use trim: {use_trim}
            - Max tokens: {max_tokens}""")
        config_time = time.perf_counter() - config_start
        
        # Ejecución del caso de uso
        use_case_start = time.perf_counter()
        use_case = AskQuestionUseCase()
        logger.info("🔄 Ejecutando caso de uso...")
        
        response = use_case.execute(
            session_id=request.session_id,
            query=request.query,
            model=request.model,
            additional_params=request.additional_params,
            use_trim=use_trim,
            max_tokens=max_tokens
        )
        use_case_time = time.perf_counter() - use_case_start
        
        # Métricas finales
        total_time = time.perf_counter() - start_time
        logger.info(f"""✅ Respuesta generada exitosamente:
            - Tiempo de configuración: {config_time:.2f}s
            - Tiempo de ejecución: {use_case_time:.2f}s
            - Tiempo total: {total_time:.2f}s
            - Longitud de respuesta: {len(str(response))} caracteres""")
        
        return {"response": response}
    
    except Exception as e:
        error_time = time.perf_counter() - start_time
        logger.error(f"""❌ Error en el endpoint de chat:
            - Tiempo hasta error: {error_time:.2f}s
            - Error: {str(e)}
            - Session ID: {request.session_id}""")
        
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "session_id": request.session_id,
                "model": request.model,
                "processing_time": f"{error_time:.2f}s"
            }
        )
