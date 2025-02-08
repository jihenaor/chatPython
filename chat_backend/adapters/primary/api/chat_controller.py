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
    logger.info(f"📝 Nueva consulta: {request.query[:50]}...")
    
    try:
        use_case = AskQuestionUseCase()
        response = await use_case.execute(
            query=request.query,
            model=request.model,
            max_tokens=MODEL_MAX_TOKENS.get(request.model, 2048)
        )
        
        total_time = time.perf_counter() - start_time
        logger.info(f"✅ Respuesta generada en {total_time:.3f}s")
        
        return {"response": response}
        
    except Exception as e:
        error_time = time.perf_counter() - start_time
        logger.error(f"❌ Error: {str(e)} ({error_time:.3f}s)")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
