from fastapi import APIRouter
from pydantic import BaseModel

from typing import Dict, Optional
import os
import logging

from application.use_cases.ask_question import AskQuestionUseCase

# Configurar logging test
logging.basicConfig(level=logging.INFO)
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
    logger.info(f"📝 Received chat request with session_id: {request.session_id}")
    logger.info(f"🤖 Requested model: {request.model}")
    logger.info(f"❓ Query: {request.query}")
    
    try:
        use_trim = os.getenv("USE_TRIM", "false").lower() == "true"
        max_tokens = MODEL_MAX_TOKENS.get(request.model, 2048)
        
        logger.info(f"📊 Configuration - use_trim: {use_trim}, max_tokens: {max_tokens}")
        
        logger.info("🔄 Initializing AskQuestionUseCase...")
        use_case = AskQuestionUseCase()
        
        logger.info("🚀 Executing use case...")
        response = use_case.execute(
            session_id=request.session_id,
            query=request.query,
            model=request.model,
            additional_params=request.additional_params,
            use_trim=use_trim,
            max_tokens=max_tokens
        )
        
        logger.info("✅ Successfully generated response")
        return {"response": response}
        
    except Exception as e:
        logger.error(f"❌ Error processing request: {str(e)}", exc_info=True)
        return {"error": str(e)}
